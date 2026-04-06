# RESEARCH_HARNESS_OPTIMIZATION.md
> Phase 2.12 Research — Multi-Agent Harness Optimization  
> Compiled: 2026-04-01 | Researcher: Ford (subagent)  
> Scope: Resource scaling, context engineering, hierarchical memory, daily context management

---

## Executive Summary

State-of-the-art research on multi-agent harness optimization reveals four key findings that should reshape pyfinAgent's harness design:

1. **More agents ≠ better performance.** Google's 180-config study proves multi-agent coordination degrades performance 39–70% on sequential tasks but boosts parallelizable tasks 80.9%.
2. **Simple observation masking beats LLM summarization.** JetBrains Research (NeurIPS 2025) found masking halves cost vs. raw agent and **matches or exceeds** LLM-Summary effectiveness.
3. **Prefix stability via prompt caching** yields 90% cost reduction and 85% latency drop — the single highest-ROI optimization available today.
4. **CoALA 4-memory architecture** (working + episodic + semantic + procedural) is the canonical cognitive framework for production agents, validated by Princeton/LangChain/Anthropic.

---

## 1. Resource Scaling Heuristics

### 1.1 The Myth of "More Agents"

**Source:** Google Research + Google DeepMind, "Towards a Science of Scaling Agent Systems: When and Why Agent Systems Work" (arXiv:2512.08296, 2025)  
**URL:** https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/

**Study design:** 180 agent configurations, 5 architectures × 4 benchmarks (Finance-Agent, BrowseComp-Plus, PlanCraft, Workbench), 3 model families (GPT, Gemini, Claude).

**Key findings with thresholds:**

| Architecture | Parallelizable Task | Sequential Task |
|---|---|---|
| Single-Agent | baseline | baseline |
| Independent (parallel, no comms) | +some | +some |
| Centralized (orchestrator → workers) | **+80.9%** | **−39% to −70%** |
| Decentralized (peer mesh) | degraded | degraded |
| Hybrid (hierarchical + peer) | mixed | mixed |

**Critical thresholds:**
- Centralized orchestration is the **only safe default** — it contains error amplification to 4.4× vs. independent agents (17.2×).
- More tools = worse multi-agent performance. The "tool-coordination trade-off" is real: 16+ tools kills multi-agent gains.
- Sequential workflows → single agent wins. Parallelizable tasks → centralized multi-agent wins.

**Predictive model (R² = 0.513):** Task decomposability + tool count → optimal architecture. The model is correct 87% of the time on unseen tasks.

**Pitfalls:**
- Independent multi-agent (no orchestrator) amplifies errors 17.2×
- Decentralized peer mesh almost always underperforms
- Identical agents (homogeneous) hit diminishing returns near-immediately

### 1.2 Collaborative Scaling Law

**Source:** "Scaling Large Language Model-based Multi-Agent Collaboration" (arXiv:2406.07155, 2024)

**Finding:** Performance follows a **logistic growth curve** as agent count increases — not linear. Collaborative emergence occurs earlier than neural scaling emergence.

**Optimal ensemble size:** 10–40 agents before saturation, but this applies to homogeneous parallel voting agents. For structured harnesses (PLAN → GENERATE → EVALUATE), the optimal team size is **3–5 specialized agents**.

**Heterogeneity multiplier:** Diverse agents (different models, prompts, tools) produce substantially more gain per agent than homogeneous agents.

### 1.3 Token Budgeting Strategy

**Source:** Anthropic Engineering + Search synthesis

**Model tier heuristics:**
```
Task Complexity → Model Tier
─────────────────────────────────────────────────────
Monitoring, routing, simple reads     → Haiku 4.5   (~5× cheaper than Opus)
Production agents, pipelines          → Sonnet 4.6  (1M ctx in beta)
Complex multi-step research/planning  → Opus 4+     (best reasoning)
```

**Tool response budget:**
- Paginate tool results; never return unbounded output
- Use range selection, filtering, truncation at the tool layer
- Code execution instead of natural language tool calls: **85% token reduction** on tool routing

**Subagent delegation rule:**
> Delegate verbose tasks (logs, tests, large file processing) to subagents. Keep parent context clean.

---

## 2. Context Engineering

### 2.1 Anthropic's Framework (Primary Source)

**Source:** Anthropic Engineering Blog, "Effective Context Engineering for AI Agents" (2025)  
**URL:** https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

**Core principle:** *Find the smallest possible set of high-signal tokens that maximize the likelihood of the desired outcome.*

**Context rot:** As token count grows, recall accuracy degrades across ALL models. This is architectural — n² pairwise attention relationships stretch thin at long ranges. Models have less training experience with long sequences.

**System prompt best practices:**
- Goldilocks altitude: specific enough to guide, flexible enough to allow heuristics
- Organize into named sections: `<background_information>`, `<instructions>`, `## Tool guidance`, `## Output description`
- XML tags or Markdown headers for section delineation
- Minimal set of canonical few-shot examples (quality > quantity)
- Avoid brittle if-else hardcoded logic in prompts

**Tool design rules:**
- Self-contained, minimal overlap between tools
- Bloated tool sets are a primary failure mode
- If a human engineer can't definitively say which tool to use → the LLM can't either
- Curate a minimal viable tool set

**Just-in-time context (JIT):**
- Agents maintain lightweight identifiers (file paths, stored queries, links) and load data on demand
- Progressive disclosure: discover context through exploration rather than front-loading
- Claude Code CLAUDE.md pattern: some context front-loaded, rest discovered via grep/glob
- Trade-off: JIT is slower but prevents context bloat for large codebases

**Long-horizon task management:**
- Break into subgoals (HiAgent approach — see §3.2)
- Note-taking agents: structured persistent notes instead of holding everything in context
- External storage for heavy intermediate results

### 2.2 Prefix Stability via Prompt Caching

**Source:** Anthropic documentation + industry synthesis (2024)

**What it is:** Cache the stable "prefix" portion of prompts (system instructions, tool definitions, examples) so subsequent calls only pay for new tokens.

**Implementation:**
```python
# Mark stable content with cache_control
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": STABLE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    }
]
# Header: anthropic-beta: prompt-caching-2024-07-31
```

**Economics:**
- Cache write: 125% of base input price (one-time cost)
- Cache hit: 10% of base input price (**90% cheaper**)
- Latency: 85% reduction on cache hits
- Cache TTL: 5 minutes (refreshed on each use)
- Minimum tokens to qualify: 1,024 (Sonnet/Opus), 2,048 (Haiku)
- Maximum cache breakpoints: 4 per prompt

**What to cache:**
1. System prompt
2. Tool definitions
3. Stable few-shot examples
4. Large background documents

**What NOT to cache:**
1. Dynamic user messages
2. Tool call results
3. Session-specific state

### 2.3 Working Window Optimization

**Source:** Anthropic context engineering + JetBrains Research (2025)

**"Lost in the middle" problem:** Even 1M-token context windows suffer from poor recall for information buried in the middle. This is not solved by simply extending context.

**Window optimization strategies:**
1. **Recency bias:** Keep most recent tool observations; mask older ones
2. **Relevance filtering:** Load only context relevant to current subgoal
3. **Progressive disclosure:** Start narrow, expand only when needed
4. **Position encoding:** Critical information at start (system prompt) and end (current task); avoid critical data in the middle of long contexts

**TALE Framework (arXiv:2412.18547, 2024):**
- Dynamically estimates token budget for each problem based on complexity
- Guides reasoning process with budget-awareness
- Result: **68.64% reduction in output tokens** with only slight accuracy impact

---

## 3. Hierarchical Memory Systems

### 3.1 CoALA: Canonical Memory Architecture

**Source:** Sumers, Yao, Narasimhan, Griffiths — "Cognitive Architectures for Language Agents" (Princeton, 2023, Trans. Mach. Learn. Res.)  
**URL:** https://arxiv.org/html/2309.02427v3  
**arXiv:** 2309.02427

**Four memory types:**

| Memory Type | Human Analogy | Agent Implementation | Update Pattern |
|---|---|---|---|
| **Working** | Short-term scratchpad | Context window | Each turn |
| **Episodic** | Specific past events | Past action-observation logs | Post-session |
| **Semantic** | Factual world knowledge | Extracted facts, preferences, knowledge base | Background extraction |
| **Procedural** | How to do things | LLM weights + agent code + system prompt | Slow/manual |

**Insight:** Most existing systems only optimize episodic memory (cross-trial). Working memory (in-trial) is underexplored and offers high ROI.

**Action space:**
- Internal actions: memory read/write, planning, reasoning
- External actions: API calls, tool use, environment interaction

### 3.2 HiAgent: Hierarchical Working Memory

**Source:** Hu, Chen, Chen, Mu, Shao, Luo — "HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model" (arXiv:2408.09559, August 2024)  
**URL:** https://arxiv.org/abs/2408.09559  
**Project:** https://github.com/HiAgent2024/HiAgent

**Core idea:** Use **subgoals as memory chunks**. When a subgoal completes, compress its action-observation history into a summary observation. Only retain raw detail for the current active subgoal.

**Algorithm:**
```
1. LLM formulates subgoal before each action cluster
2. LLM generates actions to achieve subgoal
3. On subgoal completion → replace action-observation pairs with summary
4. Retain full detail only for current subgoal
5. Trajectory retrieval module for looking up past subgoal summaries when needed
```

**Measured outcomes (5 long-horizon tasks):**
- **2× success rate improvement**
- **3.8 fewer steps on average**
- **35% context length reduction**
- **19.4% runtime reduction**

**Why it works:** Mirrors human working memory chunking. Reduces redundancy in long trajectories. Prevents context bloat from irrelevant completed-subgoal detail.

**Implementation pattern for pyfinAgent:**
```
Harness step = subgoal chunk
On step completion → summarize to handoff artifact
Next step loads: summary of prior step + full context for current step
```

### 3.3 Multi-Tier Memory Architecture (LangChain/Production Pattern)

**Source:** LangChain Blog, "Memory for Agents" (2024)  
**URL:** https://blog.langchain.com/memory-for-agents/  
**Reference paper:** CoALA (above)

**LangGraph Memory Store pattern:**

```
Tier 1: Working Memory (context window)
  ↕ Hot path or background
Tier 2: Episodic Memory (event log, append-only)
  ↕ Background extraction
Tier 3: Semantic Memory (extracted facts, preferences)
  ↕ Manual/slow update
Tier 4: Procedural Memory (system prompt, agent code)
```

**Update strategies:**

| Strategy | Pattern | Trade-offs |
|---|---|---|
| **Hot path** | Agent explicitly saves memory before responding | +immediate availability; −added latency; −mixes logic |
| **Background** | Separate process runs post-session | +no latency; +separation of concerns; −not immediately available |
| **User feedback** | Mark good interactions, save as episodic | +high quality; −requires user engagement |

**Application-specific guidance:**
- Memory shape varies by application — don't use a generic memory schema
- Episodic memory → few-shot prompting for task-specific behaviors
- Semantic memory → user preferences, domain knowledge for personalization

### 3.4 MIRIX: 6-Type Modular Memory (2025)

**Source:** MIRIX multi-agent memory system (2025)

**Six memory types:**
1. Core memory (always-in-context facts)
2. Episodic memory (past events, timestamped)
3. Semantic memory (extracted knowledge)
4. Procedural memory (skills, how-tos)
5. Resource memory (file paths, tool references)
6. Knowledge graph (inter-concept relationships)

**Key distinction from simpler systems:** Multimodal support + explicit knowledge graph for relationship traversal.

---

## 4. Daily Context Management

### 4.1 Observation Masking (Preferred Strategy)

**Source:** Lindenbauer, Slinko, Felder, Bogomolov, Zharov — "The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management" (JetBrains Research / TU Munich, NeurIPS 2025 Workshop)  
**URL:** https://arxiv.org/html/2508.21433v3  
**Dataset/Code:** https://github.com/JetBrains-Research/the-complexity-trap

**Key insight:** Observation tokens make up **~84% of average SWE-agent turn context**. Targeting them directly is the highest-leverage optimization.

**Comparison of strategies:**

| Strategy | Cost vs. Raw | Solve Rate | Notes |
|---|---|---|---|
| Raw Agent (no management) | 1× (baseline) | baseline | Often 2× more expensive than needed |
| LLM-Summary | ~0.5× | ≈baseline | Complex, adds overhead, masks failure signals |
| **Observation Masking** | **~0.5×** | **≈ or slightly > LLM-Summary** | Simple, fast, no extra LLM calls |
| **Hybrid (Masking + LLM)** | **~0.43×** | **≈ best** | 7% cheaper than masking alone |

**Critical pitfall of LLM-Summary:**
> "Smoothed summaries can mask failure signals, leading agents to persist in unproductive loops."

LLM-based summarization hides signs of failure. The agent can't tell if prior steps went wrong.

**Observation masking algorithm:**
```python
MAX_OBSERVATIONS_IN_CONTEXT = 3  # or based on token budget

def mask_observations(trajectory):
    """Keep last N observations, replace older ones with placeholder."""
    for i, step in enumerate(trajectory[:-MAX_OBSERVATIONS_IN_CONTEXT]):
        step.observation = f"[observation from step {i+1} removed — completed]"
    return trajectory
```

**Trigger heuristics (from SWE-agent):**
- Fixed context window size threshold
- Fixed turn count threshold (e.g., mask after turn 5)
- Hybrid: mask when context > 60% of window capacity

### 4.2 ACON: Agent Context Optimization

**Source:** "Agent Context Optimization (ACON)" (arXiv:2510.00615, 2024)  
**URL:** https://arxiv.org/html/2510.00615v1

**Framework:** Unified context compression for both environment observations AND interaction histories.

**Mechanism:**
1. Feedback loop refines compression guidelines in natural language
2. Compression guidelines distilled from LLM feedback
3. Distilled compressors → small models (lower overhead)

**Results:**
- **26–54% memory usage reduction**
- Improved decision quality over uncompressed baselines
- Compressors can be distilled to smaller models for production use

### 4.3 Tokenomics in Multi-Agent Systems

**Source:** Capgemini analysis + arXiv tokenomics study (2024)

**Token distribution in typical multi-agent SE pipeline:**
- Input tokens: ~53.9% of total consumption
- Iterative review stages (code review, critique loops): 59.4% of total usage

**Implication:** The Evaluator/Critic stage is the biggest cost driver. Optimize there first.

**Strategies:**
1. Batch critique requests — don't make one LLM call per item
2. Use smaller models for critique (Haiku vs. Opus)
3. Apply observation masking to critique outputs before feeding next iteration
4. Cap iteration cycles with diminishing-returns detection

### 4.4 Cascading Compression Architecture

**Source:** Synthesis of Anthropic, JetBrains, LangChain, and CoALA patterns

**Recommended 4-stage compression cascade:**

```
Stage 1: Tool Output Truncation (at tool boundary)
  → Truncate/paginate all tool outputs before injection
  → Budget: max 2,000 tokens per tool response

Stage 2: Observation Masking (in-context)
  → Replace observations older than N turns with placeholders
  → Trigger: context > 60% of window capacity

Stage 3: Subgoal Compression (on subgoal completion)
  → Summarize completed subgoal's action-observation pairs
  → Store summary in handoff artifact for retrieval
  → HiAgent pattern (arXiv:2408.09559)

Stage 4: Session Compaction (cross-session)
  → Daily log (append-only raw events) → background extraction → MEMORY.md update
  → Long-term facts in MEMORY.md; raw events in memory/YYYY-MM-DD.md
  → LangChain background memory update pattern
```

### 4.5 Session Continuity Methods

**Source:** Synthesis of Anthropic, LangChain, and CoALA patterns

**Compaction survival checklist:**
1. ✅ Never hold critical state only in context — always externalize to files
2. ✅ HEARTBEAT.md = current position in plan (survives any restart)
3. ✅ Daily log = append-only raw events (never overwrite)
4. ✅ MEMORY.md = curated long-term facts (background extraction)
5. ✅ handoff/ artifacts = step-level state (contract, results, critique)
6. ✅ Cron jobs for monitoring (survive session death; poll loops don't)

**Anti-patterns:**
- Storing state only in conversation history (lost on compaction)
- Long poll loops in main session (blocks responsiveness)
- Monolithic MEMORY.md that conflates raw events with curated facts

---

## 5. Synthesis: Actionable Recommendations for pyfinAgent Harness

### Priority 1: Prompt Caching (Highest ROI, implement immediately)

```python
# In run_harness.py or any LLM client wrapper:
# Mark PLAN.md content, tool defs, and system prompts as cacheable
# Cost: 1.25× on first call, 0.10× on repeats
# With 84 commits of institutional knowledge in prompts → enormous savings
```

**Cache targets:**
- System prompt (harness role descriptions)
- Tool definitions list
- Background context (PLAN.md summary, AGENTS.md rules)
- Few-shot examples for GENERATE/EVALUATE agents

### Priority 2: Observation Masking (Replace LLM-Summary)

If the harness currently uses LLM-based summarization for context management, **replace it with observation masking**.

```python
MASK_AFTER_TURNS = 5       # Keep last 5 turn observations raw
TOKEN_BUDGET_TRIGGER = 0.6  # Mask when context > 60% of window
PLACEHOLDER = "[observation from step {n} cleared — archived in handoff/]"
```

**Why:** Same or better performance at half the cost, no failure-signal masking.

### Priority 3: HiAgent Subgoal Chunking

Map each harness step to a subgoal chunk:
- PLAN step starts: clear prior step's raw detail from context
- Load: prior step's summary (from handoff artifact) + current step full context
- Achieves: 35% context reduction, 2× task success rate improvement

### Priority 4: Architecture Selection Rule

Before spawning subagents, classify the task:
```
Is the task parallelizable?
  YES → Centralized orchestrator with worker subagents (→ +80% perf)
   NO → Single agent (→ multi-agent loses 39–70%)

How many tools involved?
  < 8 tools → multi-agent OK
  > 16 tools → strongly prefer single agent
```

### Priority 5: 4-Tier Memory Separation

```
Tier 1: Context window (working memory)
  → Current harness step detail only
  → HiAgent masking for prior steps

Tier 2: handoff/ artifacts (episodic, in-session)
  → contract.md, experiment_results.md, evaluator_critique.md
  → Per-step subgoal summaries

Tier 3: memory/YYYY-MM-DD.md (episodic, cross-session)
  → Append-only raw event log
  → Never compress or overwrite

Tier 4: MEMORY.md (semantic, long-term)
  → Background extraction of curated facts
  → Updated by main agent post-milestone only
  → Procedural patterns, learned rules, important decisions
```

### Priority 6: Model Tier Assignment

```
Harness component      → Model
────────────────────────────────────
PLAN (main agent)      → Opus 4+ (complex reasoning)
GENERATE (coding)      → Sonnet 4.6 (balance speed/quality)
EVALUATE (critique)    → Haiku 4.5 (batch evaluation, cost ~5× cheaper)
HEARTBEAT              → Haiku 4.5
Cron monitoring        → Haiku 4.5
Slack routing          → Haiku 4.5
```

---

## 6. Cost/Benefit Summary

| Optimization | Implementation Complexity | Expected Savings | Risk |
|---|---|---|---|
| Prompt caching | Low (API header + cache_control) | Up to 90% cost, 85% latency | Low |
| Observation masking | Low (replace older tools with placeholders) | ~50% context cost | Very low |
| Subgoal chunking (HiAgent) | Medium (restructure step transitions) | 35% context, 2× success rate | Low |
| Architecture selection (Google) | Low (decision rule) | 40–80% perf improvement | Low |
| 4-tier memory separation | Medium (refactor memory writes) | Better continuity, lower MEMORY.md bloat | Low |
| Model tier assignment | Low (routing config) | 3–5× cost reduction on batch tasks | Low |
| ACON compression | High (train/distill compressor model) | 26–54% memory reduction | Medium |

---

## 7. Full Citations

1. **Google Research/DeepMind** — "Towards a Science of Scaling Agent Systems: When and Why Agent Systems Work" (arXiv:2512.08296, 2025)  
   URL: https://arxiv.org/abs/2512.08296  
   Blog: https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/

2. **JetBrains Research / TU Munich** — "The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management" (NeurIPS 2025 Workshop, arXiv:2508.21433)  
   URL: https://arxiv.org/html/2508.21433v3  
   Code: https://github.com/JetBrains-Research/the-complexity-trap  
   Data: https://huggingface.co/datasets/JetBrains-Research/the-complexity-trap

3. **Hu, Chen, Chen, Mu, Shao, Luo (2024)** — "HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model" (arXiv:2408.09559)  
   URL: https://arxiv.org/abs/2408.09559  
   Code: https://github.com/HiAgent2024/HiAgent

4. **Sumers, Yao, Narasimhan, Griffiths (Princeton, 2023)** — "Cognitive Architectures for Language Agents (CoALA)" (Trans. Mach. Learn. Res.)  
   URL: https://arxiv.org/html/2309.02427v3  
   arXiv: 2309.02427

5. **Anthropic Engineering (2025)** — "Effective Context Engineering for AI Agents"  
   URL: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

6. **LangChain (2024)** — "Memory for Agents"  
   URL: https://blog.langchain.com/memory-for-agents/  
   Reference: CoALA paper (above)

7. **Han et al. (2024)** — "TALE: Token-Budget-Aware LLM Reasoning" (arXiv:2412.18547)  
   URL: https://arxiv.org/abs/2412.18547  
   Result: 68.64% output token reduction with minimal accuracy impact

8. **Qian et al. (2024)** — "Scaling Large Language Model-based Multi-Agent Collaboration" (arXiv:2406.07155)  
   URL: https://arxiv.org/abs/2406.07155  
   Result: Collaborative scaling law — logistic growth pattern

9. **arXiv:2510.00615 (2024)** — "ACON: Agent Context Optimization"  
   URL: https://arxiv.org/html/2510.00615v1  
   Result: 26–54% memory reduction via feedback-refined compression guidelines

10. **Anthropic API Documentation (2024)** — Prompt Caching  
    URL: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching  
    Key: 90% cost reduction, 85% latency reduction, 5-min TTL, 4 breakpoints max

---

## 8. Pitfalls to Avoid

1. **LLM-Summary as default context management** — masks failure signals, costs same as masking, gives no performance benefit (JetBrains 2025)
2. **Homogeneous agent swarms** — hit diminishing returns near-immediately (arXiv:2406.07155)
3. **Multi-agent for sequential tasks** — degrades performance 39–70% (Google 2025)
4. **Independent agents without orchestrator** — 17.2× error amplification (Google 2025)
5. **16+ tools in multi-agent system** — tool-coordination tax kills gains (Google 2025)
6. **Storing critical state only in context window** — dies on compaction/restart
7. **Front-loading all context** — "lost in the middle" problem; JIT retrieval is safer
8. **Long poll loops in main session** — blocks responsiveness, dies on compaction
9. **Monolithic memory file** — conflates raw events with curated knowledge; should be separated
10. **Exceeding 60% context window before masking** — leave headroom for tool outputs and reasoning

---

*This document is the research foundation for Phase 2.12 harness redesign. Prioritize items by ROI: start with prompt caching (30-min implementation, 90% savings) before tackling architectural changes.*
