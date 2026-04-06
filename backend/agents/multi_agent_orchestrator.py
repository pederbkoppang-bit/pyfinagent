"""
Multi-Agent Orchestrator — Matching Anthropic's multi-agent research diagram.

Flow (from the diagram at 1:24:58 and docs.anthropic.com):

  1. User sends query
  2. Communication Agent classifies (Sonnet 4.6)
  3. Ford (LeadResearcher equivalent, Opus 4.6):
     a. think(plan approach) — Ford plans before acting
     b. save plan → Memory (episodic)
     c. retrieve context from Memory (harness state)
     d. create subagents for aspect A + B in parallel
     e. each subagent: search → think(evaluate) → complete_task
     f. think(synthesize results)
     g. "More research needed?" — if incomplete, create more subagents
     h. exit loop when sufficient
  4. Quality Gate — SEPARATE agent evaluates (never self-evaluate)
  5. persist results → Memory (episodic)
  6. return to user

Critical principle: "Separating generation from evaluation is the strongest lever."
Ford NEVER evaluates his own output. A different agent does.

Models:
  Communication: claude-sonnet-4-6 (routing + quality gate)
  Main / Ford:   claude-opus-4-6   (planning + synthesis)
  Q&A / Analyst: claude-opus-4-6   (quantitative analysis)
  Researcher:    claude-sonnet-4-6 (literature + evidence)

References:
  https://www.anthropic.com/engineering/multi-agent-research-system
  https://www.anthropic.com/engineering/harness-design-long-running-apps
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

from backend.agents.agent_definitions import (
    AGENT_CONFIGS,
    AgentConfig,
    AgentType,
    ClassificationResult,
    QueryComplexity,
    classify_trivial,
    parse_llm_classification,
    parse_delegation_request,
    parse_harness_trigger,
    strip_delegation_tag,
)
from backend.agents.mas_events import MASEvent, get_event_bus, make_run_id

logger = logging.getLogger(__name__)

# Max iterations for "more research needed?" loop
MAX_RESEARCH_ITERATIONS = 3

# Max tool-use turns per subagent (search → evaluate → search again)
MAX_TOOL_TURNS = 5

# ═══════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS (for subagent tool loops)
#
# From the diagram: each subagent runs web_search → think(evaluate)
# → search again in a loop. Our tools give agents access to harness
# state so they can ground their analysis in real experiment data.
# ═══════════════════════════════════════════════════════════════════

AGENT_TOOLS = [
    {
        "name": "read_evaluator_critique",
        "description": "Read the latest evaluator critique — verdict (PASS/FAIL/CONDITIONAL), scores for Statistical Validity, Robustness, Simplicity, Reality Gap, and weak periods.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_experiment_results",
        "description": "Read the latest experiment results — what changes were made, Sharpe before/after, parameters tried.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_research_plan",
        "description": "Read the current research plan — planner's hypothesis, proposed directions, success criteria.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_experiment_log",
        "description": "Read a summary of all experiments — total count, recent keep/discard rates, last N experiment details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "last_n": {"type": "integer", "description": "Number of recent experiments to show (default 20)"},
            },
            "required": [],
        },
    },
    {
        "name": "read_best_params",
        "description": "Read current best parameters — Sharpe, DSR, and all parameter values from optimizer_best.json.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_contract",
        "description": "Read the current sprint contract — hypothesis, success criteria, fail conditions.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_harness_log",
        "description": "Read recent harness cycle summaries — verdict, scores, hypothesis, duration for each cycle.",
        "input_schema": {
            "type": "object",
            "properties": {
                "last_n": {"type": "integer", "description": "Number of recent cycles (default 5)"},
            },
            "required": [],
        },
    },
]
MAX_RESEARCH_ITERATIONS = 3


class MultiAgentOrchestrator:
    """
    Orchestrator matching Anthropic's multi-agent research diagram.

    Key differences from naive orchestration:
    1. PLANNING STEP: Ford thinks about approach before delegating
    2. ITERATIVE LOOP: "More research needed?" after synthesis
    3. QUALITY GATE: Separate agent evaluates (Ford never self-evaluates)
    4. MEMORY PERSIST: Results saved to episodic memory after each interaction
    """

    def __init__(self):
        self._client = None
        self._masker = None

    def _get_masker(self):
        """Lazy-init ObservationMasker for context compression."""
        if self._masker is None:
            try:
                from backend.agents.harness_memory import (
                    ObservationMasker, HarnessMemory, create_masker, init_session_memory,
                )
                memory, _ = init_session_memory()
                self._masker = create_masker(
                    model_name="claude-opus-4-6",
                    memory=memory,
                )
            except Exception as e:
                logger.debug(f"ObservationMasker init failed: {e}")
        return self._masker

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                from backend.config.settings import get_settings
                settings = get_settings()
                api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not configured.")
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("pip install anthropic")
        return self._client

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════

    def classify_message_sync(self, message: str) -> ClassificationResult:
        """Classify via Communication Agent (Sonnet 4.6). Trivial check first."""
        trivial = classify_trivial(message)
        if trivial:
            return trivial
        try:
            import asyncio as _aio
            loop = _aio.new_event_loop()
            try:
                return loop.run_until_complete(self._classify_via_llm(message))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassificationResult(
                agent_type=AgentType.MAIN, complexity=QueryComplexity.SIMPLE,
                confidence=0.4, reasoning=f"Error: {e}",
            )

    def execute_classified_sync(self, message, classification, sender="user", context=None):
        """Execute pre-classified message. Full diagram flow."""
        import asyncio as _aio
        loop = _aio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._execute_full_flow(message, classification, sender, context)
            )
        except Exception as e:
            return {
                "response": f"⚠️ Error: {str(e)[:200]}",
                "agent_type": classification.agent_type.value,
                "classification": classification,
                "processing_time_ms": 0,
                "token_usage": {"input": 0, "output": 0},
            }
        finally:
            loop.close()

    async def handle_message(self, message, sender="user", source="slack", context=None):
        """Full flow: classify → execute → quality gate → persist."""
        start = time.time()

        # Step 1: Classify
        classification = classify_trivial(message)
        if not classification:
            try:
                classification = await self._classify_via_llm(message)
            except Exception:
                classification = ClassificationResult(
                    agent_type=AgentType.MAIN, complexity=QueryComplexity.SIMPLE,
                    confidence=0.4, reasoning="Classification error",
                )

        if classification.agent_type == AgentType.DIRECT:
            response = self._handle_direct(message)
            return self._build_result(response, classification, start, {"input": 0, "output": 0})

        return await self._execute_full_flow(message, classification, sender, context)

    def call_single_agent_sync(self, agent_type, message, is_subtask=False,
                                classification=None, context=None):
        """Synchronous single-agent call for Slack task streaming."""
        start = time.time()
        config = AGENT_CONFIGS[agent_type]
        if not classification:
            classification = ClassificationResult(
                agent_type=agent_type, complexity=QueryComplexity.SIMPLE,
                confidence=0.8, reasoning="Direct call",
            )
        task = self._build_subtask_prompt(message, agent_type, classification, context) \
            if is_subtask else self._build_task_prompt(message, classification, context)
        try:
            # Use tool loop so agents can read harness data iteratively
            text, usage = self._call_agent_with_tools(config, task)
            clean = strip_delegation_tag(text)
            return {"response": clean, "agent_type": agent_type.value,
                    "token_usage": usage, "processing_time_ms": round((time.time()-start)*1000, 1)}
        except Exception as e:
            return {"response": f"⚠️ Error: {str(e)[:150]}", "agent_type": agent_type.value,
                    "token_usage": {"input": 0, "output": 0},
                    "processing_time_ms": round((time.time()-start)*1000, 1), "error": str(e)}

    # ═══════════════════════════════════════════════════════════════
    # FULL FLOW (matching Anthropic's diagram)
    # ═══════════════════════════════════════════════════════════════

    async def _execute_full_flow(self, message, classification, sender="user", context=None):
        """
        The complete orchestrator-worker flow from Anthropic's diagram:

        1. think(plan approach) — Ford plans before acting
        2. save plan → Memory
        3. retrieve context → harness state
        4. create subagents in parallel
        5. synthesize results
        6. "More research needed?" iterative loop
        7. Quality Gate — separate agent evaluates
        8. persist results → Memory
        """
        start = time.time()
        total_usage = {"input": 0, "output": 0}
        loop = asyncio.get_event_loop()
        bus = get_event_bus()
        run_id = make_run_id()

        # ── Emit: classify ──────────────────────────────────────
        bus.emit(MASEvent(
            event_type="classify", agent="Communication", run_id=run_id,
            data={
                "primary": classification.agent_type.value,
                "complexity": classification.complexity.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning[:200],
                "parallel_agents": [a.value for a in (classification.parallel_agents or [])],
                "query_preview": message[:100],
            },
        ))

        # ── Step 1: think(plan approach) ────────────────────────
        plan = None
        if classification.complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX):
            plan_start = time.time()
            plan, plan_usage = await self._think_plan(message, classification, context)
            total_usage["input"] += plan_usage.get("input", 0)
            total_usage["output"] += plan_usage.get("output", 0)
            logger.info(f"📋 Ford planned approach: {plan[:100]}...")

            bus.emit(MASEvent(
                event_type="plan", agent="Ford", run_id=run_id,
                duration_ms=round((time.time() - plan_start) * 1000, 1),
                tokens=plan_usage,
                data={"plan_preview": plan[:300]},
            ))

            # ── Step 2: save plan → Memory ──────────────────────
            self._save_to_memory(f"PLAN for query: {message[:80]}\n{plan}")
            bus.emit(MASEvent(
                event_type="memory_save", agent="Ford", run_id=run_id,
                data={"type": "plan", "target": "episodic"},
            ))

        # ── Step 3: retrieve context → harness state ────────────
        harness_ctx = self._get_harness_context(classification.agent_type)

        # ── Step 4-5-6: Execute with iterative loop ─────────────
        exec_start = time.time()
        if classification.complexity == QueryComplexity.COMPLEX and classification.parallel_agents:
            response, exec_usage = await self._iterative_parallel_research(
                message, classification, plan, context, run_id=run_id,
            )
        else:
            response, exec_usage = await self._single_with_delegation(
                message, classification, plan, context, run_id=run_id,
            )

        total_usage["input"] += exec_usage.get("input", 0)
        total_usage["output"] += exec_usage.get("output", 0)

        # ── Step 7: Quality Gate — SEPARATE agent evaluates ─────
        if classification.complexity != QueryComplexity.TRIVIAL:
            gate_start = time.time()
            checked_response, gate_usage = await self._quality_gate(
                message, response, classification
            )
            total_usage["input"] += gate_usage.get("input", 0)
            total_usage["output"] += gate_usage.get("output", 0)

            bus.emit(MASEvent(
                event_type="quality_gate", agent="Quality Gate", run_id=run_id,
                duration_ms=round((time.time() - gate_start) * 1000, 1),
                tokens=gate_usage,
                data={"passed": checked_response is None},
            ))
            if checked_response:
                response = checked_response

        # ── Step 8: CitationAgent — add source citations ──────────
        cite_start = time.time()
        cited_response, cite_usage = await self._add_citations(
            response, classification
        )
        total_usage["input"] += cite_usage.get("input", 0)
        total_usage["output"] += cite_usage.get("output", 0)
        if cited_response:
            bus.emit(MASEvent(
                event_type="citation", agent="CitationAgent", run_id=run_id,
                duration_ms=round((time.time() - cite_start) * 1000, 1),
                tokens=cite_usage,
            ))
            response = cited_response

        # ── Step 9: persist results → Memory ────────────────────
        self._save_to_memory(
            f"RESPONSE for: {message[:60]}\n"
            f"Agent: {classification.agent_type.value}\n"
            f"Response: {response[:300]}"
        )

        # ── Emit: complete ──────────────────────────────────────
        bus.emit(MASEvent(
            event_type="complete", agent="Ford", run_id=run_id,
            duration_ms=round((time.time() - start) * 1000, 1),
            tokens=total_usage,
            data={
                "response_length": len(response),
                "agent_type": classification.agent_type.value,
                "complexity": classification.complexity.value,
            },
        ))

        return self._build_result(response, classification, start, total_usage)

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: PLANNING (Ford thinks before acting)
    # ═══════════════════════════════════════════════════════════════

    async def _think_plan(self, message, classification, context=None):
        """
        Ford's planning step — thinks about approach before delegating.

        From diagram: "think(plan approach)" happens before creating subagents.
        This ensures Ford decomposes the question properly and assigns
        each subagent a clear, non-overlapping task.
        """
        plan_prompt = (
            f"USER QUERY:\n{message}\n\n"
            f"CLASSIFIED AS: {classification.agent_type.value} "
            f"({classification.complexity.value})\n\n"
            f"PLAN YOUR APPROACH (2-3 sentences):\n"
            f"1. What are the key aspects of this question?\n"
            f"2. Which agent(s) should handle each aspect?\n"
            f"3. What specific output do you need from each?\n"
            f"4. How will you synthesize their findings?\n\n"
            f"Be concrete. Don't say 'research this topic' — say exactly what to look for."
        )

        ford_config = AGENT_CONFIGS[AgentType.MAIN]
        loop = asyncio.get_event_loop()
        plan_text, usage = await loop.run_in_executor(
            None, self._call_agent, ford_config, plan_prompt,
        )
        return plan_text, usage

    # ═══════════════════════════════════════════════════════════════
    # STEP 4-6: ITERATIVE PARALLEL RESEARCH
    # ═══════════════════════════════════════════════════════════════

    async def _iterative_parallel_research(self, message, classification, plan=None, context=None, run_id=""):
        """
        Iterative research loop from the diagram:
          → spawn subagents → synthesize → "more research needed?" → loop or exit

        Max MAX_RESEARCH_ITERATIONS rounds to prevent runaway.
        """
        agents = classification.parallel_agents or [classification.agent_type]
        all_findings = []
        total_usage = {"input": 0, "output": 0}
        loop = asyncio.get_event_loop()

        bus = get_event_bus()

        for iteration in range(1, MAX_RESEARCH_ITERATIONS + 1):
            logger.info(f"🔄 Research iteration {iteration}/{MAX_RESEARCH_ITERATIONS}")

            # Spawn subagents in parallel
            tasks = []
            for agent_type in agents:
                config = AGENT_CONFIGS[agent_type]
                sub_task = self._build_subtask_prompt(
                    message, agent_type, classification, context,
                    plan=plan, previous_findings=all_findings,
                )
                tasks.append(
                    loop.run_in_executor(None, self._call_agent_with_tools, config, sub_task)
                )
                bus.emit(MASEvent(
                    event_type="delegate", agent="Ford", run_id=run_id,
                    iteration=iteration,
                    data={
                        "target_agent": agent_type.value,
                        "target_name": config.name,
                        "model": config.model,
                    },
                ))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect findings
            iteration_findings = []
            for i, result in enumerate(results):
                agent_type = agents[i]
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent_type.value} failed: {result}")
                    iteration_findings.append(f"⚠️ {agent_type.value}: Error - {str(result)[:100]}")
                    bus.emit(MASEvent(
                        event_type="error", agent=agent_type.value, run_id=run_id,
                        iteration=iteration,
                        data={"error": str(result)[:200]},
                    ))
                else:
                    text, usage = result
                    clean = strip_delegation_tag(text)
                    total_usage["input"] += usage.get("input", 0)
                    total_usage["output"] += usage.get("output", 0)
                    iteration_findings.append(clean)

            all_findings.extend(iteration_findings)

            # ── "More research needed?" decision ────────────────
            if iteration < MAX_RESEARCH_ITERATIONS:
                needs_more, decision_usage = await self._check_research_complete(
                    message, all_findings
                )
                total_usage["input"] += decision_usage.get("input", 0)
                total_usage["output"] += decision_usage.get("output", 0)

                bus.emit(MASEvent(
                    event_type="loop_check", agent="Ford", run_id=run_id,
                    iteration=iteration,
                    data={"needs_more": needs_more, "findings_count": len(all_findings)},
                    tokens=decision_usage,
                ))

                if not needs_more:
                    logger.info(f"✅ Research complete after {iteration} iteration(s)")
                    break
                else:
                    logger.info(f"🔄 More research needed — continuing to iteration {iteration + 1}")
            else:
                logger.info(f"⏱️ Max iterations reached ({MAX_RESEARCH_ITERATIONS})")

        # Synthesize all findings
        synth_start = time.time()
        synthesis, synth_usage = await self._synthesize(message, agents, all_findings)
        total_usage["input"] += synth_usage.get("input", 0)
        total_usage["output"] += synth_usage.get("output", 0)

        bus.emit(MASEvent(
            event_type="synthesize", agent="Ford", run_id=run_id,
            duration_ms=round((time.time() - synth_start) * 1000, 1),
            tokens=synth_usage,
            data={"findings_count": len(all_findings), "iterations": iteration},
        ))

        return synthesis, total_usage

    async def _check_research_complete(self, original_query, findings_so_far):
        """
        The "More research needed?" decision from the diagram.
        Ford (LeadResearcher) evaluates whether the findings are sufficient.
        """
        check_prompt = (
            f"ORIGINAL QUERY:\n{original_query}\n\n"
            f"FINDINGS SO FAR ({len(findings_so_far)} pieces):\n"
            + "\n---\n".join(f[:200] for f in findings_so_far[-4:])  # Last 4 findings
            + "\n\nAre these findings SUFFICIENT to answer the query comprehensively?\n"
            f"Consider: Are there gaps? Missing perspectives? Unanswered aspects?\n\n"
            f"Respond with ONLY 'COMPLETE' or 'NEEDS_MORE: [what's missing]'"
        )

        ford_config = AGENT_CONFIGS[AgentType.MAIN]
        loop = asyncio.get_event_loop()
        decision_text, usage = await loop.run_in_executor(
            None, self._call_agent, ford_config, check_prompt,
        )

        needs_more = "NEEDS_MORE" in decision_text.upper()
        logger.info(f"📋 Research check: {'NEEDS_MORE' if needs_more else 'COMPLETE'}")
        return needs_more, usage

    async def _synthesize(self, query, agent_types, all_findings):
        """Ford synthesizes all subagent findings into a cohesive response."""
        emoji_map = {AgentType.QA: "📊", AgentType.RESEARCH: "🔬", AgentType.MAIN: "⚙️"}

        # Build labeled sections
        sections = []
        for i, finding in enumerate(all_findings):
            agent_type = agent_types[i % len(agent_types)]
            config = AGENT_CONFIGS.get(agent_type)
            label = config.name if config else agent_type.value
            emoji = emoji_map.get(agent_type, "🤖")
            sections.append(f"{emoji} *{label}:*\n{finding}")

        synth_prompt = (
            f"ORIGINAL QUERY:\n{query}\n\n"
            f"SUBAGENT FINDINGS:\n" + "\n\n---\n\n".join(sections) +
            f"\n\nSYNTHESIZE these findings into a clear, concise answer.\n"
            f"- Lead with the key insight\n"
            f"- Preserve specific data points from each agent\n"
            f"- Note any contradictions between agents\n"
            f"- Keep under 400 words for Slack\n"
        )

        ford_config = AGENT_CONFIGS[AgentType.MAIN]
        loop = asyncio.get_event_loop()
        synthesis, usage = await loop.run_in_executor(
            None, self._call_agent, ford_config, synth_prompt,
        )
        return strip_delegation_tag(synthesis), usage

    # ═══════════════════════════════════════════════════════════════
    # SINGLE AGENT WITH DELEGATION
    # ═══════════════════════════════════════════════════════════════

    async def _single_with_delegation(self, message, classification, plan=None, context=None, run_id=""):
        """Single agent with parent-driven delegation via [DELEGATE:xxx]."""
        config = AGENT_CONFIGS[classification.agent_type]
        task = self._build_task_prompt(message, classification, context, plan=plan)

        loop = asyncio.get_event_loop()
        # Primary agent uses tool loop — can read harness data iteratively
        response_text, usage = await loop.run_in_executor(
            None, self._call_agent_with_tools, config, task,
        )
        total_usage = dict(usage)

        # Check for delegation
        delegate_to = parse_delegation_request(response_text)
        if delegate_to and delegate_to != classification.agent_type:
            sub_config = AGENT_CONFIGS.get(delegate_to)
            if sub_config and delegate_to.value in config.can_delegate_to:
                logger.info(f"🔀 {config.name} delegated to {sub_config.name}")

                parent_clean = strip_delegation_tag(response_text)
                sub_task = (
                    f"ORIGINAL QUESTION:\n{message}\n\n"
                    f"PARENT ({config.name}) ANALYSIS:\n{parent_clean}\n\n"
                    f"Add depth from your expertise. Don't repeat the parent's analysis.\n"
                    f"You have tools to read harness data — use them to ground your analysis."
                )

                # Sub-agent also uses tool loop
                sub_response, sub_usage = await loop.run_in_executor(
                    None, self._call_agent_with_tools, sub_config, sub_task,
                )
                total_usage["input"] += sub_usage.get("input", 0)
                total_usage["output"] += sub_usage.get("output", 0)

                emoji_map = {AgentType.QA: "📊", AgentType.RESEARCH: "🔬", AgentType.MAIN: "⚙️"}
                merged = (
                    f"{emoji_map.get(classification.agent_type, '🤖')} *{config.name}:*\n{parent_clean}"
                    f"\n\n---\n\n"
                    f"{emoji_map.get(delegate_to, '🤖')} *{sub_config.name}:*\n{sub_response}"
                )
                return merged, total_usage

        # Check harness trigger
        if parse_harness_trigger(response_text):
            classification.triggers_harness = True

        return strip_delegation_tag(response_text), total_usage

    # ═══════════════════════════════════════════════════════════════
    # STEP 7: QUALITY GATE (separate evaluation — never self-evaluate)
    # ═══════════════════════════════════════════════════════════════

    async def _quality_gate(self, original_query, response, classification):
        """
        CRITICAL: Ford NEVER evaluates his own output.

        The Communication Agent (Sonnet, different model) does a quality check
        using a 0.0–1.0 scoring rubric (Anthropic's recommendation).

        From the article: "A single LLM call with a single prompt outputting
        scores from 0.0-1.0 and a pass-fail grade was the most consistent
        and aligned with human judgements."

        Criteria (adapted from Anthropic's research eval rubric):
        - Accuracy: Do claims match harness data / sources?
        - Completeness: Are all aspects of the question addressed?
        - Groundedness: Does it reference real data, not hallucinate?
        - Conciseness: Is it appropriate length for Slack (<400 words)?

        Threshold: Any criterion < 0.6 = FAIL. Overall < 0.7 = FAIL.

        Returns (improved_response, usage) or (None, usage) if pass.
        """
        gate_prompt = (
            f"QUALITY CHECK — You are reviewing another agent's response.\n\n"
            f"USER ASKED:\n{original_query}\n\n"
            f"AGENT ({classification.agent_type.value}) RESPONDED:\n{response[:1500]}\n\n"
            f"SCORING RUBRIC (0.0 to 1.0 each):\n"
            f"1. ACCURACY (0.0-1.0): Do the specific claims match reality? Are numbers correct?\n"
            f"   0.0 = fabricated claims, 0.5 = some claims unverifiable, 1.0 = all claims grounded\n"
            f"2. COMPLETENESS (0.0-1.0): Does it answer ALL aspects of the question?\n"
            f"   0.0 = ignores the question, 0.5 = partial answer, 1.0 = fully addresses everything\n"
            f"3. GROUNDEDNESS (0.0-1.0): Are claims backed by data/sources, or hand-wavy?\n"
            f"   0.0 = pure opinion, 0.5 = some data cited, 1.0 = every claim has evidence\n"
            f"4. CONCISENESS (0.0-1.0): Is it right-sized for Slack (<400 words)?\n"
            f"   0.0 = massive wall of text, 0.5 = could be tighter, 1.0 = perfectly concise\n\n"
            f"FEW-SHOT CALIBRATION EXAMPLES:\n\n"
            f"Example A (PASS):\n"
            f"  User: 'What's our current Sharpe?'\n"
            f"  Response: 'Current best Sharpe is 1.1705 (DSR 0.9984) from run 52eb3ffe. "
            f"Sub-periods: A=0.89, B=0.92, C=1.88. All positive. 2× cost stress test: 0.91.'\n"
            f"  Scores: Accuracy=1.0, Completeness=0.9, Groundedness=1.0, Conciseness=1.0\n"
            f"  Verdict: PASS (avg 0.975)\n\n"
            f"Example B (FAIL):\n"
            f"  User: 'Why did momentum fail robustness?'\n"
            f"  Response: 'Momentum strategies can sometimes underperform in choppy markets. "
            f"It's important to consider various factors when evaluating signal robustness.'\n"
            f"  Scores: Accuracy=0.3, Completeness=0.2, Groundedness=0.1, Conciseness=0.8\n"
            f"  Verdict: FAIL (avg 0.35 — vague, no data, doesn't answer the question)\n\n"
            f"Example C (FAIL → IMPROVE):\n"
            f"  User: 'Compare last 3 evaluator runs'\n"
            f"  Response: 'The evaluator has been running well recently with good scores.'\n"
            f"  Scores: Accuracy=0.4, Completeness=0.1, Groundedness=0.1, Conciseness=0.9\n"
            f"  Verdict: FAIL (avg 0.375 — no specific data from actual runs)\n\n"
            f"NOW EVALUATE THE RESPONSE ABOVE.\n\n"
            f"Output format (EXACTLY this, nothing else):\n"
            f"ACCURACY: <score>\n"
            f"COMPLETENESS: <score>\n"
            f"GROUNDEDNESS: <score>\n"
            f"CONCISENESS: <score>\n"
            f"VERDICT: PASS|FAIL\n"
            f"If FAIL, add on the next line:\n"
            f"IMPROVED: <your complete replacement response>\n"
        )

        comms_config = AGENT_CONFIGS[AgentType.COMMUNICATION]
        gate_config = type(comms_config)(
            agent_type=comms_config.agent_type,
            name="Quality Gate",
            model=comms_config.model,
            system_prompt=(
                "You are a quality reviewer using a scoring rubric. Be skeptical — "
                "the agent that wrote this cannot judge its own work. "
                "Score each criterion independently BEFORE deciding the verdict. "
                "Never upgrade a score after seeing the overall picture. "
                "If uncertain between two scores, pick the LOWER one."
            ),
            max_tokens=2000,
        )

        loop = asyncio.get_event_loop()
        gate_response, usage = await loop.run_in_executor(
            None, self._call_agent, gate_config, gate_prompt,
        )

        # Parse scoring rubric response
        response_upper = gate_response.strip().upper()
        scores = {}
        for line in gate_response.strip().split('\n'):
            line_upper = line.strip().upper()
            for criterion in ('ACCURACY', 'COMPLETENESS', 'GROUNDEDNESS', 'CONCISENESS'):
                if line_upper.startswith(f"{criterion}:"):
                    try:
                        score_str = line_upper.split(':', 1)[1].strip()
                        scores[criterion.lower()] = float(score_str)
                    except (ValueError, IndexError):
                        pass

        # Log scores
        if scores:
            avg = sum(scores.values()) / len(scores)
            score_str = ', '.join(f"{k}={v:.1f}" for k, v in scores.items())
            logger.info(f"📋 Quality gate scores: {score_str} (avg={avg:.2f})")

            # Check thresholds: any < 0.6 or avg < 0.7 = FAIL
            any_below = any(v < 0.6 for v in scores.values())
            if any_below or avg < 0.7:
                # Extract improved response if present
                improved = None
                for marker in ('IMPROVED:', 'IMPROVED RESPONSE:'):
                    if marker in gate_response.upper():
                        idx = gate_response.upper().index(marker)
                        improved = gate_response[idx + len(marker):].strip()
                        break
                if improved:
                    logger.info(f"🔄 Quality gate: FAIL (avg={avg:.2f}) → improved")
                    return improved, usage
                else:
                    logger.info(f"🔄 Quality gate: FAIL (avg={avg:.2f}) but no improvement provided")
                    return None, usage
            else:
                logger.info(f"✅ Quality gate: PASS (avg={avg:.2f})")
                return None, usage

        # Fallback: old-style PASS/FAIL parsing
        if 'VERDICT: PASS' in response_upper or response_upper == 'PASS':
            logger.info("✅ Quality gate: PASS")
            return None, usage
        elif 'VERDICT: FAIL' in response_upper:
            improved = None
            for marker in ('IMPROVED:', 'IMPROVED RESPONSE:'):
                if marker in gate_response.upper():
                    idx = gate_response.upper().index(marker)
                    improved = gate_response[idx + len(marker):].strip()
                    break
            if improved:
                logger.info(f"🔄 Quality gate: FAIL → improved ({len(improved)} chars)")
                return improved, usage
            logger.info("🔄 Quality gate: FAIL but no improvement")
            return None, usage
        else:
            # If we can't parse, treat non-PASS as improvement
            logger.info(f"🔄 Quality gate: unparseable, treating as improvement ({len(gate_response)} chars)")
            return gate_response, usage

    # ═══════════════════════════════════════════════════════════════
    # MEMORY INTEGRATION
    # ═══════════════════════════════════════════════════════════════

    def _save_to_memory(self, entry: str):
        """Save to episodic memory (daily log)."""
        try:
            from backend.agents.harness_memory import HarnessMemory
            memory = HarnessMemory()
            memory.append_episodic(entry)
        except Exception as e:
            logger.debug(f"Memory save skipped: {e}")

    def _get_harness_context(self, agent_type: AgentType) -> str:
        """Inject harness state into agent prompts."""
        try:
            from backend.agents.harness_state_reader import get_harness_reader
            return get_harness_reader().build_agent_context(
                agent_type=agent_type.value, max_tokens=800,
            )
        except Exception as e:
            logger.debug(f"Harness context unavailable: {e}")
            return ""

    # ═══════════════════════════════════════════════════════════════
    # CONTEXT RESET (Anthropic pattern for long sessions)
    #
    # From the article: "Context resets — clearing the context window
    # entirely and starting a fresh agent, combined with a structured
    # handoff that carries the previous agent's state and the next
    # steps — addresses [context anxiety]."
    #
    # This differs from compaction. A reset gives a clean slate at
    # the cost of needing a good handoff artifact.
    # ═══════════════════════════════════════════════════════════════

    def build_context_handoff(self, messages: list[dict], query: str) -> str:
        """Build a structured handoff artifact from current conversation.

        Used when context is approaching limits and we need to reset.
        Captures: original query, key findings, decisions made, next steps.

        Args:
            messages: Current conversation messages
            query: The original user query

        Returns:
            Structured handoff string for the next fresh agent.
        """
        # Extract key content from messages
        findings = []
        for msg in messages:
            content = str(msg.get("content", ""))
            role = msg.get("role", "")
            if role == "assistant" and len(content) > 100:
                findings.append(content[:300])

        handoff = (
            f"=== CONTEXT HANDOFF (reset from previous session) ===\n"
            f"ORIGINAL QUERY: {query}\n\n"
            f"FINDINGS FROM PREVIOUS SESSION ({len(findings)} messages):\n"
        )
        for i, f in enumerate(findings[-5:], 1):  # Last 5 substantive messages
            handoff += f"  {i}. {f}\n"

        handoff += (
            f"\nINSTRUCTIONS:\n"
            f"- You are continuing work from a previous session that was reset\n"
            f"- The findings above summarize what was already discovered\n"
            f"- Build on these findings, do NOT repeat the same searches\n"
            f"- Focus on gaps and unanswered aspects\n"
            f"=== END HANDOFF ===\n"
        )
        return handoff

    def should_reset_context(self, messages: list[dict], model: str = "claude-opus-4-6") -> bool:
        """Check if context should be reset (approaching 80% of window)."""
        from backend.agents.harness_memory import approx_token_count, get_context_window
        total = sum(approx_token_count(str(m.get("content", ""))) for m in messages)
        window = get_context_window(model)
        usage_pct = total / window if window > 0 else 0
        return usage_pct >= 0.80

    # ═══════════════════════════════════════════════════════════════
    # LLM CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════

    async def _classify_via_llm(self, message):
        comms_config = AGENT_CONFIGS[AgentType.COMMUNICATION]
        loop = asyncio.get_event_loop()
        text, usage = await loop.run_in_executor(
            None, self._call_agent, comms_config, message,
        )
        logger.info(f"🔀 Classification ({usage.get('input',0)}+{usage.get('output',0)} tok): {text[:100]}")
        return parse_llm_classification(text)

    # ═══════════════════════════════════════════════════════════════
    # DIRECT (local, no API)
    # ═══════════════════════════════════════════════════════════════

    def _handle_direct(self, message):
        from backend.slack_bot.direct_responder import get_direct_response
        return get_direct_response(message) or "👋 I'm here. What do you need?"

    # ═══════════════════════════════════════════════════════════════
    # API CALL (simple — for routing, planning, quality gate)
    # ═══════════════════════════════════════════════════════════════

    def _call_agent(self, agent_config, task):
        """Simple one-shot API call (no tools). Used for classification, planning, synthesis, quality gate."""
        client = self._get_client()
        try:
            response = client.messages.create(
                model=agent_config.model,
                max_tokens=agent_config.max_tokens,
                system=agent_config.system_prompt,
                messages=[{"role": "user", "content": task}],
            )
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            usage = {
                "input": response.usage.input_tokens if response.usage else 0,
                "output": response.usage.output_tokens if response.usage else 0,
            }
            return text or "No response.", usage
        except Exception as e:
            logger.error(f"API call to {agent_config.name} failed: {type(e).__name__}: {e}")
            raise

    # ═══════════════════════════════════════════════════════════════
    # API CALL WITH TOOLS (subagent tool loop — GAP 1.14 FIX)
    #
    # From the diagram: each subagent runs:
    #   web_search → think(evaluate) → search again → complete_task
    #
    # Our agents get harness tools so they can iteratively read
    # experiment data, evaluator critiques, and research plans,
    # then refine their analysis based on what they find.
    # ═══════════════════════════════════════════════════════════════

    def _call_agent_with_tools(self, agent_config, task, max_turns=None):
        """
        Call agent with tool-use loop + interleaved thinking.

        From Anthropic's multi-agent research article:
          "Subagents use interleaved thinking after tool results to evaluate
           quality, identify gaps, and refine their next query. This makes
           subagents more effective in adapting to any task."

        Pattern: search → think(evaluate) → search again → complete_task
        """
        if max_turns is None:
            max_turns = MAX_TOOL_TURNS

        client = self._get_client()
        messages = [{"role": "user", "content": task}]
        total_usage = {"input": 0, "output": 0}

        for turn in range(max_turns):
            try:
                # Interleaved thinking (Anthropic GAP fix):
                # Extended thinking lets subagents plan after each tool result,
                # evaluate quality, identify gaps, and decide next action.
                # Budget of 2048 tokens per turn keeps costs reasonable.
                response = client.messages.create(
                    model=agent_config.model,
                    max_tokens=agent_config.max_tokens + 2048,
                    system=agent_config.system_prompt,
                    tools=AGENT_TOOLS,
                    messages=messages,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 2048,
                    },
                )
            except Exception as e:
                logger.error(f"Tool-loop call failed on turn {turn}: {e}")
                raise

            total_usage["input"] += response.usage.input_tokens if response.usage else 0
            total_usage["output"] += response.usage.output_tokens if response.usage else 0

            if response.stop_reason == "tool_use":
                # Agent wants to call tools — execute them in parallel
                # (Gap 6 fix: parallel tool execution within turns)
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                tool_names_called = [b.name for b in tool_blocks]

                if len(tool_blocks) > 1:
                    # Multiple tools requested — run in parallel via threads
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=len(tool_blocks)) as pool:
                        futures = {
                            b.id: pool.submit(self._execute_tool, b.name, b.input)
                            for b in tool_blocks
                        }
                    tool_results = [
                        {
                            "type": "tool_result",
                            "tool_use_id": b.id,
                            "content": futures[b.id].result(),
                        }
                        for b in tool_blocks
                    ]
                else:
                    # Single tool — no overhead
                    tool_results = [
                        {
                            "type": "tool_result",
                            "tool_use_id": b.id,
                            "content": self._execute_tool(b.name, b.input),
                        }
                        for b in tool_blocks
                    ]

                # Emit tool_call events for dashboard
                bus = get_event_bus()
                for tn in tool_names_called:
                    bus.emit(MASEvent(
                        event_type="tool_call", agent=agent_config.name,
                        data={"tool": tn, "turn": turn + 1},
                    ))

                logger.info(
                    f"  🔧 Turn {turn+1}: {agent_config.name} called "
                    f"{', '.join(tool_names_called)}"
                )

                # Add assistant response + tool results to conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                # Apply observation masking if context is growing large
                # (ACON-inspired: compress older observations at 60% window)
                masker = self._get_masker()
                if masker and masker.should_mask(messages):
                    messages, mask_report = masker.mask_observations(messages)
                    if mask_report.get('masked'):
                        logger.info(
                            f"  🗜️ Observation masking: saved {mask_report['tokens_saved']} tokens "
                            f"({mask_report['usage_pct_before']:.0%} → {mask_report['usage_pct_after']:.0%})"
                        )

            else:
                # Agent finished — stop_reason is "end_turn" or "max_tokens"
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                if turn > 0:
                    logger.info(
                        f"  ✅ {agent_config.name} completed after {turn+1} turns "
                        f"({total_usage['input']}+{total_usage['output']} tok)"
                    )
                return text or "No response.", total_usage

        # Max turns reached — extract whatever text we have
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        logger.warning(f"  ⚠️ {agent_config.name} hit max tool turns ({max_turns})")
        return text or "Max tool turns reached.", total_usage

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a harness_state_reader tool and return JSON result."""
        try:
            from backend.agents.harness_state_reader import get_harness_reader
            reader = get_harness_reader()

            if tool_name == "read_evaluator_critique":
                result = reader.get_evaluator_critique()
            elif tool_name == "read_experiment_results":
                result = reader.get_experiment_results()
            elif tool_name == "read_research_plan":
                result = reader.get_research_plan()
            elif tool_name == "read_experiment_log":
                last_n = tool_input.get("last_n", 20)
                result = reader.get_experiment_summary(last_n=last_n)
            elif tool_name == "read_best_params":
                result = reader.get_best_params()
            elif tool_name == "read_contract":
                result = reader.get_contract()
            elif tool_name == "read_harness_log":
                last_n = tool_input.get("last_n", 5)
                result = reader.get_harness_log(last_n=last_n)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            return json.dumps(result, default=str, indent=2)[:4000]  # Cap output

        except Exception as e:
            logger.warning(f"Tool {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})

    # ═══════════════════════════════════════════════════════════════
    # CITATION AGENT (GAP 1.13 FIX)
    #
    # From the diagram: after quality gate, CitationAgent processes
    # the research report to identify source locations for citations.
    # ═══════════════════════════════════════════════════════════════

    async def _add_citations(self, response, classification):
        """
        CitationAgent: process response to add structured source citations.

        Only runs for Q&A and Research agent responses (operational
        responses from Ford don't need citations).

        Uses Sonnet for cost efficiency — citation processing is
        mechanical, not creative.
        """
        # Skip citation for operational/trivial responses
        if classification.agent_type not in (AgentType.QA, AgentType.RESEARCH):
            return response, {"input": 0, "output": 0}

        # Skip if response is too short to benefit from citations
        if len(response) < 200:
            return response, {"input": 0, "output": 0}

        cite_prompt = (
            f"RESPONSE TO PROCESS:\n{response}\n\n"
            f"YOUR TASK: Add numbered source citations to this response.\n\n"
            f"RULES:\n"
            f"- Add [1], [2], etc. markers after claims that have identifiable sources\n"
            f"- At the end, add a '---\\n📚 *Sources:*' section listing each numbered source\n"
            f"- If the response mentions author/year (e.g. 'Bailey & López de Prado (2014)'),"
            f" convert to numbered format: claim [1] → [1] Bailey & López de Prado (2014)\n"
            f"- If a claim references harness data (Sharpe values, evaluator scores), cite as"
            f" '[N] pyfinAgent harness — evaluator_critique.md' or similar\n"
            f"- If a claim has NO identifiable source, leave it uncited — do NOT invent sources\n"
            f"- Preserve ALL original content — only add citation markers and the sources list\n"
            f"- Keep the same formatting (bullets, headers, etc.)\n"
        )

        citation_config = AgentConfig(
            agent_type=AgentType.COMMUNICATION,
            name="CitationAgent",
            model="claude-sonnet-4-6",
            system_prompt="You are a citation processor. Add numbered source citations to research responses. Be precise — only cite real sources mentioned in the text.",
            max_tokens=2000,
        )

        loop = asyncio.get_event_loop()
        cited_response, usage = await loop.run_in_executor(
            None, self._call_agent, citation_config, cite_prompt,
        )

        logger.info(
            f"📚 CitationAgent processed ({usage.get('input',0)}+{usage.get('output',0)} tok)"
        )
        return cited_response, usage

    # ═══════════════════════════════════════════════════════════════
    # PROMPT BUILDING (with harness context + 4-component delegation)
    # ═══════════════════════════════════════════════════════════════

    def _build_task_prompt(self, message, classification, context=None, plan=None):
        parts = [f"USER REQUEST:\n{message}"]

        if plan:
            parts.append(f"\nYOUR PLAN:\n{plan}")

        harness_ctx = self._get_harness_context(classification.agent_type)
        if harness_ctx:
            parts.append(f"\nHARNESS STATE:\n{harness_ctx}")

        if context:
            parts.append(f"\nCONTEXT:\n{self._fmt_ctx(context)}")

        parts.append(
            "\nINSTRUCTIONS:\n"
            "- Lead with the key answer\n"
            "- Reference harness state when relevant\n"
            "- Keep under 300 words for Slack\n"
            "- Add [DELEGATE:qa] or [DELEGATE:research] for sub-agent help\n"
            "- Add [TRIGGER_HARNESS] if new experiments are needed\n"
        )
        return "\n".join(parts)

    def _build_subtask_prompt(self, message, agent_type, classification,
                               context=None, plan=None, previous_findings=None):
        config = AGENT_CONFIGS[agent_type]

        # 4-component delegation (Anthropic's requirement)
        delegation = {
            AgentType.QA: {
                "objective": "Analyze the quantitative aspects using harness data and experiment results.",
                "output_format": "Key finding first, then supporting data. Specific numbers required.",
                "tool_guidance": "Use harness artifacts (evaluator_critique.md, experiment_results.md). Cross-reference sub-periods.",
                "boundaries": "Do NOT run experiments. Do NOT modify harness state. Analyze existing data only.",
            },
            AgentType.RESEARCH: {
                "objective": "Find relevant academic and practitioner evidence.",
                "output_format": "Finding → Evidence (author/year) → Application to pyfinAgent → Pitfalls.",
                "tool_guidance": "Start broad (arXiv, SSRN). Cite sources. Include formulas and thresholds.",
                "boundaries": "Do NOT run experiments. Do NOT modify RESEARCH.md. Find information only.",
            },
            AgentType.MAIN: {
                "objective": "Address operational aspects — system status, deployments, tasks.",
                "output_format": "Status indicators, specific numbers, actionable steps.",
                "tool_guidance": "Check system status, git, harness log, sprint contract.",
                "boundaries": "Do NOT do deep analysis. Do NOT trigger harness from sub-task.",
            },
        }.get(agent_type, {})

        parts = [
            f"USER REQUEST:\n{message}",
            f"\nDELEGATION TO {config.name}:",
            f"  OBJECTIVE: {delegation.get('objective', 'Provide your perspective.')}",
            f"  OUTPUT FORMAT: {delegation.get('output_format', 'Concise, under 250 words.')}",
            f"  TOOL GUIDANCE: {delegation.get('tool_guidance', 'Use available data.')}",
            f"  BOUNDARIES: {delegation.get('boundaries', 'Stay in your domain.')}",
        ]

        if plan:
            parts.append(f"\nLEAD AGENT PLAN:\n{plan[:500]}")

        if previous_findings:
            parts.append(f"\nPREVIOUS FINDINGS (build on these, don't repeat):")
            for i, f in enumerate(previous_findings[-3:], 1):
                parts.append(f"  {i}. {f[:200]}...")

        harness_ctx = self._get_harness_context(agent_type)
        if harness_ctx:
            parts.append(f"\nHARNESS STATE:\n{harness_ctx}")

        return "\n".join(parts)

    def _fmt_ctx(self, context):
        if not context:
            return ""
        lines = []
        for k, v in context.items():
            if isinstance(v, dict):
                lines.append(f"- {k}: {', '.join(f'{a}={b}' for a, b in v.items())}")
            elif isinstance(v, list):
                lines.append(f"- {k}: {', '.join(str(x) for x in v[:5])}")
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _build_result(self, response, classification, start_time, token_usage):
        return {
            "response": response,
            "agent_type": classification.agent_type.value,
            "classification": classification,
            "processing_time_ms": round((time.time() - start_time) * 1000, 1),
            "token_usage": token_usage,
            "triggers_harness": getattr(classification, "triggers_harness", False),
        }


# Singleton
_orchestrator: Optional[MultiAgentOrchestrator] = None

def get_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator
