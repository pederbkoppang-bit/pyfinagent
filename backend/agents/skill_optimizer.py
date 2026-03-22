"""
Skill Optimizer — autonomous experiment loop for improving agent skills.md files.

Mirrors Karpathy's autoresearch pattern: establish baseline → propose modification
→ measure metric → keep/discard/crash → LOOP FOREVER.

The optimizer modifies ONLY the ## Prompt Template section of each agent's skills.md.
The data tools, orchestrator pipeline, output schemas, and evaluation formula are FIXED.
"""

import csv
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.config.prompts import SKILLS_DIR, load_skill, reload_skills
from backend.config.settings import Settings
from backend.db.bigquery_client import BigQueryClient
from backend.services.outcome_tracker import OutcomeTracker

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = SKILLS_DIR / "experiments"
RESULTS_TSV = EXPERIMENTS_DIR / "skill_results.tsv"

# TSV columns
TSV_HEADER = [
    "timestamp", "commit", "agent", "metric_before", "metric_after",
    "delta", "status", "description",
]

# Agents eligible for optimization (only those actively used in the pipeline)
OPTIMIZABLE_AGENTS = [
    "rag_agent", "market_agent", "competitor_agent",
    "insider_agent", "options_agent", "social_sentiment_agent",
    "patent_agent", "earnings_tone_agent", "enhanced_macro_agent",
    "alt_data_agent", "sector_analysis_agent", "nlp_sentiment_agent",
    "anomaly_agent", "scenario_agent", "quant_model_agent",
    "bull_agent", "bear_agent", "devils_advocate_agent", "moderator_agent",
    "aggressive_analyst", "conservative_analyst", "neutral_analyst",
    "risk_judge", "synthesis_agent", "critic_agent", "deep_dive_agent",
]


def _git(*args: str, cwd: str | None = None) -> str:
    """Run a git command and return stdout. Raises on failure."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd or str(SKILLS_DIR.parent.parent.parent),
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _get_short_hash(cwd: str | None = None) -> str:
    """Get current git short commit hash."""
    try:
        return _git("rev-parse", "--short", "HEAD", cwd=cwd)
    except RuntimeError:
        return "no-git"


class SkillOptimizer:
    """
    Autonomous experiment loop for skill optimization.

    Usage:
        optimizer = SkillOptimizer(settings)
        optimizer.establish_baseline()
        optimizer.run_loop()  # runs until stopped
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.bq = BigQueryClient(settings)
        self.outcome_tracker = OutcomeTracker(settings)
        self._running = False
        self._consecutive_discards: dict[str, int] = {}

        # Ensure experiments directory and TSV exist
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        if not RESULTS_TSV.exists():
            with open(RESULTS_TSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(TSV_HEADER)

        # Create Gemini model for reflection/proposal generation
        self._model = None

    def _get_model(self):
        """Lazy-initialize the Gemini model."""
        if self._model is None:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(
                project=self.settings.gcp_project_id,
                location=self.settings.gcp_location,
            )
            model_name = self.settings.deep_think_model or self.settings.gemini_model
            self._model = GenerativeModel(model_name)
        return self._model

    # ── Metric Computation ───────────────────────────────────────

    def compute_metric(self) -> float:
        """
        Compute the single optimization metric via PerformanceSkill.

        Delegates to perf_metrics.get_scalar_metric_from_bq which computes:
            risk_adjusted_return × (1 − tx_cost_drag)

        Higher is better. Returns 0.0 if no outcome data is available.
        """
        from backend.services.perf_metrics import get_scalar_metric_from_bq
        return get_scalar_metric_from_bq(self.bq)

    # ── Baseline ─────────────────────────────────────────────────

    def establish_baseline(self) -> float:
        """
        BASELINE FIRST (autoresearch rule).

        Evaluate current skills without changes and record starting metric.
        """
        # Evaluate any pending outcomes first
        self.outcome_tracker.evaluate_all_pending()

        metric = self.compute_metric()
        self._log_experiment(
            agent="BASELINE",
            metric_before=metric,
            metric_after=metric,
            delta=0.0,
            status="keep",
            description="Baseline measurement — no modifications",
        )
        logger.info(f"Baseline metric: {metric}")
        return metric

    # ── Agent Performance Analysis ───────────────────────────────

    def analyze_agent_performance(self) -> list[dict]:
        """
        Query BQ to determine which agents are weakest contributors.

        Returns agents sorted by weakness (worst first), with performance data.
        """
        # Get recent reports with agent-level signal data
        query = f"""
            SELECT
                ticker, analysis_date, recommendation, final_score,
                debate_consensus, debate_confidence,
                bull_confidence, bear_confidence,
                insider_signal, options_signal, patent_signal,
                sector_signal, social_sentiment_score, nlp_sentiment_score,
                earnings_confidence, data_quality_score,
                risk_judge_decision, risk_adjusted_confidence,
                bias_count, conflict_count
            FROM `{self.bq.reports_table}`
            ORDER BY analysis_date DESC
            LIMIT 50
        """
        try:
            rows = [dict(r) for r in self.bq.client.query(query).result()]
        except Exception as e:
            logger.warning(f"Could not analyze agent performance: {e}")
            return []

        # Get outcome data for these reports
        outcomes = {}
        try:
            outcome_query = f"""
                SELECT ticker, analysis_date, return_pct, beat_benchmark
                FROM `{self.bq.outcomes_table}`
            """
            for row in self.bq.client.query(outcome_query).result():
                key = (row["ticker"], row["analysis_date"])
                outcomes[key] = dict(row)
        except Exception:
            pass

        # Per-agent signal quality heuristic:
        # For enrichment agents, check if their signal direction aligned with actual outcome
        agent_scores: dict[str, list[float]] = {a: [] for a in OPTIMIZABLE_AGENTS}

        for report in rows:
            key = (report["ticker"], report["analysis_date"])
            outcome = outcomes.get(key)
            if not outcome:
                continue

            actual_positive = (outcome.get("return_pct") or 0) > 0

            # Score debate agents by consensus accuracy
            consensus = (report.get("debate_consensus") or "").upper()
            if consensus in ("STRONG_BUY", "BUY") and actual_positive:
                agent_scores["bull_agent"].append(1.0)
                agent_scores["moderator_agent"].append(1.0)
            elif consensus in ("STRONG_SELL", "SELL") and not actual_positive:
                agent_scores["bear_agent"].append(1.0)
                agent_scores["moderator_agent"].append(1.0)
            else:
                # Wrong direction
                if consensus in ("STRONG_BUY", "BUY"):
                    agent_scores["bull_agent"].append(0.0)
                    agent_scores["moderator_agent"].append(0.0)
                elif consensus in ("STRONG_SELL", "SELL"):
                    agent_scores["bear_agent"].append(0.0)
                    agent_scores["moderator_agent"].append(0.0)

            # Score enrichment agents by signal direction alignment
            signal_map = {
                "insider_agent": report.get("insider_signal", ""),
                "options_agent": report.get("options_signal", ""),
                "patent_agent": report.get("patent_signal", ""),
                "sector_analysis_agent": report.get("sector_signal", ""),
            }
            for agent_name, signal in signal_map.items():
                sig = signal.upper()
                if "BULL" in sig or "RISING" in sig or "BREAKOUT" in sig or "TAILWIND" in sig:
                    agent_scores[agent_name].append(1.0 if actual_positive else 0.0)
                elif "BEAR" in sig or "DECLINING" in sig or "RISK" in sig or "LAGGING" in sig:
                    agent_scores[agent_name].append(0.0 if actual_positive else 1.0)

            # Score numeric signal agents
            for agent_name, field in [
                ("social_sentiment_agent", "social_sentiment_score"),
                ("nlp_sentiment_agent", "nlp_sentiment_score"),
            ]:
                val = report.get(field)
                if val is not None:
                    bullish_signal = val > 0.5
                    agent_scores[agent_name].append(
                        1.0 if bullish_signal == actual_positive else 0.0
                    )

        # Compute average accuracy per agent, sort weakest first
        results = []
        for agent_name in OPTIMIZABLE_AGENTS:
            scores = agent_scores.get(agent_name, [])
            if scores:
                avg = sum(scores) / len(scores)
                results.append({
                    "agent": agent_name,
                    "accuracy": round(avg, 3),
                    "sample_size": len(scores),
                })
            else:
                results.append({
                    "agent": agent_name,
                    "accuracy": 0.5,  # neutral if no data
                    "sample_size": 0,
                })

        results.sort(key=lambda x: (x["accuracy"], x["sample_size"]))
        return results

    # ── Read In-Scope Context ────────────────────────────────────

    def read_in_scope_files(self, agent_name: str) -> dict[str, Any]:
        """
        READ CONTEXT (autoresearch rule): gather everything relevant
        before proposing a modification.
        """
        context: dict[str, Any] = {"agent_name": agent_name}

        # 1. Current skill file
        try:
            skill_path = SKILLS_DIR / f"{agent_name}.md"
            context["current_skill"] = skill_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            context["current_skill"] = ""

        # 2. Past experiment history for this agent
        context["past_experiments"] = self._get_agent_experiments(agent_name)

        # 3. Performance data from BQ
        perf = self.analyze_agent_performance()
        agent_perf = next((p for p in perf if p["agent"] == agent_name), None)
        context["performance"] = agent_perf

        # 4. Recent outcome data
        outcomes = self.bq.get_performance_stats()
        context["overall_performance"] = outcomes

        return context

    # ── Propose Modification ─────────────────────────────────────

    def propose_skill_modification(
        self, agent_name: str, context: dict
    ) -> Optional[dict]:
        """
        Use LLM to propose a specific edit to the agent's Prompt Template section.

        Returns dict with 'old_text', 'new_text', 'description', or None if no change proposed.
        """
        current_skill = context.get("current_skill", "")
        performance = context.get("performance", {})
        past_experiments = context.get("past_experiments", [])
        overall_perf = context.get("overall_performance", {})

        # Format past experiments for context
        exp_summary = ""
        if past_experiments:
            recent = past_experiments[-10:]  # last 10
            lines = []
            for e in recent:
                lines.append(
                    f"  - [{e.get('status', '?')}] delta={e.get('delta', '?')}: "
                    f"{e.get('description', 'no description')}"
                )
            exp_summary = "\n".join(lines)

        prompt = (
            f"You are an expert prompt engineer optimizing a financial analysis agent.\n\n"
            f"## Agent: {agent_name}\n"
            f"## Current accuracy: {performance.get('accuracy', 'unknown')} "
            f"(sample size: {performance.get('sample_size', 0)})\n"
            f"## Overall system performance: avg_return={overall_perf.get('avg_return', 'N/A')}, "
            f"benchmark_beat_rate={overall_perf.get('benchmark_beat_rate', 'N/A')}\n\n"
            f"## Past Experiments on this Agent:\n{exp_summary or '(none yet)'}\n\n"
            f"## Current Skills File:\n```\n{current_skill[:6000]}\n```\n\n"
            "## RULES:\n"
            "1. You may ONLY modify text within the ## Prompt Template section.\n"
            "2. Do NOT change {{variable}} placeholder names.\n"
            "3. Do NOT change the output JSON schema or required fields.\n"
            "4. Focus on: reasoning strategies, analytical techniques, emphasis, thresholds, anti-patterns.\n"
            "5. Simpler is better — remove unhelpful instructions rather than always adding.\n"
            "6. If past experiments show a pattern (e.g., adding thresholds helped), build on it.\n"
            "7. If past experiments show a pattern of failures, try a different approach.\n"
            "8. Be specific and testable — vague changes like 'be more careful' are useless.\n\n"
            "## YOUR TASK:\n"
            "Propose ONE specific, testable modification to the ## Prompt Template section.\n\n"
            "Respond in this EXACT JSON format:\n"
            "```json\n"
            "{\n"
            '  "old_text": "the exact text to find and replace (2-10 lines from the current Prompt Template)",\n'
            '  "new_text": "the replacement text",\n'
            '  "description": "one-line description of what this change does and why",\n'
            '  "hypothesis": "what improvement this should cause and how to measure it"\n'
            "}\n"
            "```\n\n"
            "If you believe no useful modification can be made right now, respond with:\n"
            '```json\n{"skip": true, "reason": "explanation"}\n```'
        )

        try:
            model = self._get_model()
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.7, "max_output_tokens": 2048},
            )
            text = response.text.strip()

            # Extract JSON from response (may be wrapped in ```json ... ```)
            json_match = _extract_json(text)
            if not json_match:
                logger.warning(f"No JSON found in optimizer response for {agent_name}")
                return None

            proposal = json.loads(json_match)

            if proposal.get("skip"):
                logger.info(f"Optimizer skipped {agent_name}: {proposal.get('reason', 'no reason')}")
                return None

            if not proposal.get("old_text") or not proposal.get("new_text"):
                logger.warning(f"Invalid proposal for {agent_name}: missing old_text or new_text")
                return None

            return proposal

        except Exception as e:
            logger.error(f"Failed to generate proposal for {agent_name}: {e}")
            return None

    # ── Apply / Revert Modification ──────────────────────────────

    def apply_modification(self, agent_name: str, proposal: dict) -> bool:
        """
        Apply a proposed modification to the agent's skills.md file.

        Returns True if the modification was applied successfully.
        """
        skill_path = SKILLS_DIR / f"{agent_name}.md"
        old_text = proposal["old_text"]
        new_text = proposal["new_text"]

        content = skill_path.read_text(encoding="utf-8")
        if old_text not in content:
            logger.warning(
                f"Cannot apply modification to {agent_name}: old_text not found in skill file"
            )
            return False

        # Ensure old_text appears only once (safety check)
        if content.count(old_text) > 1:
            logger.warning(
                f"old_text appears {content.count(old_text)} times in {agent_name}.md — "
                "ambiguous replacement, skipping"
            )
            return False

        new_content = content.replace(old_text, new_text, 1)
        skill_path.write_text(new_content, encoding="utf-8")

        # Clear skill cache so the new prompt is picked up
        reload_skills()

        # Validate the modified skill can still be loaded
        try:
            load_skill(agent_name)
        except Exception as e:
            logger.error(f"Modified skill for {agent_name} failed to load: {e}. Reverting.")
            skill_path.write_text(content, encoding="utf-8")
            reload_skills()
            return False

        # Git commit the change
        try:
            _git("add", str(skill_path))
            _git("commit", "-m", f"skill-opt: {agent_name} — {proposal.get('description', 'modification')[:80]}")
        except RuntimeError as e:
            logger.warning(f"Git commit failed (non-fatal): {e}")

        return True

    def revert_modification(self, agent_name: str) -> bool:
        """Revert the last modification to an agent's skill file via git."""
        try:
            skill_path = SKILLS_DIR / f"{agent_name}.md"
            _git("checkout", "HEAD~1", "--", str(skill_path))
            _git("commit", "-m", f"skill-opt: revert {agent_name}")
            reload_skills()
            return True
        except RuntimeError as e:
            logger.error(f"Git revert failed for {agent_name}: {e}")
            return False

    # ── Crash Recovery ───────────────────────────────────────────

    def handle_crash(self, agent_name: str, error: str) -> None:
        """
        CRASH RECOVERY (autoresearch rule): revert the modification,
        log the crash, and move on.
        """
        logger.error(f"Crash detected for {agent_name}: {error}")
        self.revert_modification(agent_name)
        self._log_experiment(
            agent=agent_name,
            metric_before=0.0,
            metric_after=0.0,
            delta=0.0,
            status="crash",
            description=f"Crash: {error[:200]}",
        )

    # ── Think Harder ─────────────────────────────────────────────

    def think_harder(self, agent_name: str) -> Optional[dict]:
        """
        WHEN STUCK (autoresearch rule): re-read research, review near-misses,
        try radical approaches.

        Called after 5+ consecutive discards on the same agent.
        """
        past_experiments = self._get_agent_experiments(agent_name)

        # Find near-misses (discards with small negative delta)
        near_misses = [
            e for e in past_experiments
            if e.get("status") == "discard" and abs(float(e.get("delta", 0))) < 0.5
        ]

        # Read the AGENTS.md research foundations
        agents_md = Path(SKILLS_DIR.parent.parent.parent / "AGENTS.md")
        research_section = ""
        if agents_md.exists():
            content = agents_md.read_text(encoding="utf-8")
            # Extract research foundations section
            import re
            match = re.search(
                r"## Research Foundations\s*\n(.*?)(?=^## |\Z)",
                content,
                re.MULTILINE | re.DOTALL,
            )
            if match:
                research_section = match.group(1).strip()[:3000]

        near_miss_text = "\n".join(
            f"  - delta={e.get('delta')}: {e.get('description')}"
            for e in near_misses[-5:]
        ) if near_misses else "(no near-misses)"

        prompt = (
            f"You are an expert prompt engineer who has been stuck optimizing the "
            f"'{agent_name}' financial analysis agent. The last 5+ modifications were discarded.\n\n"
            f"## Near-miss experiments (almost worked):\n{near_miss_text}\n\n"
            f"## Research context from academic papers:\n{research_section[:2000]}\n\n"
            f"## Current skill file:\n"
            f"```\n{(SKILLS_DIR / f'{agent_name}.md').read_text(encoding='utf-8')[:4000]}\n```\n\n"
            "## STRATEGIES TO TRY:\n"
            "1. COMBINE multiple near-miss approaches that individually almost worked.\n"
            "2. Try a RADICAL restructure of the prompt template reasoning flow.\n"
            "3. Apply a research insight that hasn't been tried yet.\n"
            "4. SIMPLIFY — remove instructions that may be causing confusion.\n"
            "5. Reverse a previous assumption about what the agent should prioritize.\n\n"
            "Propose ONE bold modification using the same JSON format:\n"
            "```json\n"
            "{\n"
            '  "old_text": "exact text to replace",\n'
            '  "new_text": "replacement text",\n'
            '  "description": "what and why",\n'
            '  "hypothesis": "expected improvement"\n'
            "}\n```"
        )

        try:
            model = self._get_model()
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.9, "max_output_tokens": 2048},
            )
            json_str = _extract_json(response.text.strip())
            if json_str:
                return json.loads(json_str)
        except Exception as e:
            logger.error(f"think_harder failed for {agent_name}: {e}")
        return None

    # ── Simplicity Criterion ─────────────────────────────────────

    @staticmethod
    def passes_simplicity_criterion(proposal: dict, delta: float) -> bool:
        """
        Check if the complexity cost is justified by the improvement.

        Rule: require improvement >= 0.5% return delta per 10 lines added.
        Simplifications (fewer lines) always pass.
        """
        old_lines = proposal.get("old_text", "").count("\n") + 1
        new_lines = proposal.get("new_text", "").count("\n") + 1
        lines_added = new_lines - old_lines

        # Simplification = always good if metric didn't drop
        if lines_added <= 0 and delta >= 0:
            return True

        # Added complexity requires proportional improvement
        if lines_added > 0:
            required_delta = 0.005 * (lines_added / 10)  # 0.5% per 10 lines
            return delta >= required_delta

        return delta > 0

    # ── Experiment Tracking ──────────────────────────────────────

    def _log_experiment(
        self,
        agent: str,
        metric_before: float,
        metric_after: float,
        delta: float,
        status: str,
        description: str,
    ) -> None:
        """Append a row to skill_results.tsv."""
        commit = _get_short_hash()
        row = [
            datetime.utcnow().isoformat(),
            commit,
            agent,
            f"{metric_before:.4f}",
            f"{metric_after:.4f}",
            f"{delta:+.4f}",
            status,
            description,
        ]
        with open(RESULTS_TSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(row)

    def _get_agent_experiments(self, agent_name: str) -> list[dict]:
        """Read all past experiments for a specific agent from the TSV."""
        if not RESULTS_TSV.exists():
            return []
        experiments = []
        with open(RESULTS_TSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("agent") == agent_name:
                    experiments.append(row)
        return experiments

    def get_all_experiments(self) -> list[dict]:
        """Read all experiments from the TSV."""
        if not RESULTS_TSV.exists():
            return []
        with open(RESULTS_TSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            return list(reader)

    # ── Main Loop ────────────────────────────────────────────────

    def run_loop(self, max_iterations: int = 0, target_agents: Optional[list[str]] = None) -> None:
        """
        LOOP FOREVER (until stopped or max_iterations reached).

        max_iterations=0 means loop indefinitely.
        target_agents: if provided by MetaCoordinator, restricts optimization to these agents.
        """
        self._running = True
        iteration = 0

        logger.info("Skill optimization loop started")

        while self._running:
            if max_iterations > 0 and iteration >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations}). Stopping.")
                break

            iteration += 1
            logger.info(f"=== Optimization iteration {iteration} ===")

            try:
                self._run_one_iteration(target_agents=target_agents)
            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}", exc_info=True)
                time.sleep(10)

        self._running = False
        logger.info("Skill optimization loop stopped")

    def stop(self) -> None:
        """Signal the loop to stop gracefully."""
        self._running = False
        logger.info("Stop signal received")

    @property
    def is_running(self) -> bool:
        return self._running

    def _run_one_iteration(self, target_agents: Optional[list[str]] = None) -> None:
        """Execute one full optimization cycle.

        Args:
            target_agents: If provided (from MetaCoordinator MDA→Agent bridge),
                overrides the weakest-agent heuristic and cycles through these agents.
        """
        # 1. Evaluate pending outcomes
        logger.info("Step 1: Evaluating pending outcomes...")
        self.outcome_tracker.evaluate_all_pending()

        # 2. Compute current metric
        metric_before = self.compute_metric()
        logger.info(f"Current metric: {metric_before}")

        # 3. Analyze agent performance — find weakest agent
        perf = self.analyze_agent_performance()
        if not perf:
            logger.warning("No performance data available. Skipping iteration.")
            return

        # Pick target: MDA-targeted agents take priority over weakest-agent heuristic
        target = None
        if target_agents:
            # MetaCoordinator provided MDA-targeted agents — cycle through them
            valid_targets = [a for a in target_agents if a in OPTIMIZABLE_AGENTS]
            if valid_targets:
                pick = valid_targets[iteration_counter(len(valid_targets))]
                # Find this agent in perf data, or create a minimal entry
                for agent_data in perf:
                    if agent_data["agent"] == pick:
                        target = agent_data
                        break
                if not target:
                    target = {"agent": pick, "accuracy": 0.5, "sample_size": 0}
                logger.info(f"MDA-targeted agent: {target['agent']}")

        if not target:
            for agent_data in perf:
                agent_name = agent_data["agent"]
                if agent_data["sample_size"] >= 3 and agent_data["accuracy"] < 0.8:
                    target = agent_data
                    break

        if not target:
            # All agents performing well or insufficient data — pick round-robin
            target = perf[iteration_counter(len(perf))]
            logger.info(f"All agents performing well. Round-robin pick: {target['agent']}")
        else:
            logger.info(
                f"Weakest agent: {target['agent']} "
                f"(accuracy={target['accuracy']}, n={target['sample_size']})"
            )

        agent_name = target["agent"]

        # 4. Check if stuck
        consecutive = self._consecutive_discards.get(agent_name, 0)
        if consecutive >= 5:
            logger.info(f"{agent_name} stuck ({consecutive} consecutive discards). Thinking harder...")
            proposal = self.think_harder(agent_name)
            if not proposal:
                logger.info(f"think_harder produced nothing for {agent_name}. Moving on.")
                self._consecutive_discards[agent_name] = 0
                return
        else:
            # 5. Read context and propose modification
            context = self.read_in_scope_files(agent_name)
            proposal = self.propose_skill_modification(agent_name, context)
            if not proposal:
                logger.info(f"No modification proposed for {agent_name}. Skipping.")
                return

        # 6. Apply modification
        logger.info(f"Applying: {proposal.get('description', 'no description')}")
        if not self.apply_modification(agent_name, proposal):
            self._log_experiment(
                agent=agent_name,
                metric_before=metric_before,
                metric_after=metric_before,
                delta=0.0,
                status="crash",
                description=f"Failed to apply: {proposal.get('description', '')}",
            )
            return

        # 7. Measure new metric (requires waiting for new analyses to complete)
        # In practice, the metric won't change immediately — it requires new analyses
        # to be run with the modified skill. For the autonomous loop, we re-evaluate
        # after the modification is live.
        metric_after = self.compute_metric()
        delta = metric_after - metric_before

        # 8. Keep or discard
        if delta > 0 and self.passes_simplicity_criterion(proposal, delta):
            # KEEP
            self._log_experiment(
                agent=agent_name,
                metric_before=metric_before,
                metric_after=metric_after,
                delta=delta,
                status="keep",
                description=proposal.get("description", ""),
            )
            self._consecutive_discards[agent_name] = 0
            logger.info(f"KEEP: {agent_name} delta={delta:+.4f}")
        elif delta == 0.0:
            # No change yet — try proxy validation (1-window quant backtest)
            proxy_sharpe = self._run_proxy_validation()
            if proxy_sharpe is not None:
                logger.info(f"Proxy validation Sharpe: {proxy_sharpe:.4f}")
                self._log_experiment(
                    agent=agent_name,
                    metric_before=metric_before,
                    metric_after=metric_after,
                    delta=delta,
                    status="pending",
                    description=f"{proposal.get('description', '')} [proxy_sharpe={proxy_sharpe:.4f}]",
                )
            else:
                self._log_experiment(
                    agent=agent_name,
                    metric_before=metric_before,
                    metric_after=metric_after,
                    delta=delta,
                    status="pending",
                    description=proposal.get("description", ""),
                )
            logger.info(f"PENDING: {agent_name} delta=0 (awaiting new analyses)")
        else:
            # DISCARD
            self.revert_modification(agent_name)
            self._log_experiment(
                agent=agent_name,
                metric_before=metric_before,
                metric_after=metric_after,
                delta=delta,
                status="discard",
                description=proposal.get("description", ""),
            )
            self._consecutive_discards[agent_name] = consecutive + 1
            logger.info(f"DISCARD: {agent_name} delta={delta:+.4f}")

    # ── Proxy Validation ────────────────────────────────────────

    def _run_proxy_validation(self) -> Optional[float]:
        """
        Run a 1-window quant-only backtest as fast proxy feedback.

        Returns Sharpe ratio or None on failure. Delegates to
        MetaCoordinator.run_proxy_validation() — no LLM cost.
        """
        try:
            from backend.agents.meta_coordinator import MetaCoordinator
            return MetaCoordinator.run_proxy_validation(self.settings)
        except Exception as e:
            logger.debug(f"Proxy validation skipped: {e}")
            return None

    # ── Status ───────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return current optimizer status for API consumption."""
        experiments = self.get_all_experiments()
        kept = sum(1 for e in experiments if e.get("status") == "keep")
        discarded = sum(1 for e in experiments if e.get("status") == "discard")
        crashed = sum(1 for e in experiments if e.get("status") == "crash")
        pending = sum(1 for e in experiments if e.get("status") == "pending")

        return {
            "running": self._running,
            "total_experiments": len(experiments),
            "kept": kept,
            "discarded": discarded,
            "crashed": crashed,
            "pending": pending,
            "keep_rate": round(kept / (kept + discarded), 3) if (kept + discarded) > 0 else 0.0,
            "current_metric": self.compute_metric() if self._running else None,
        }


# ── Helpers ──────────────────────────────────────────────────────

_iteration_counter = 0


def iteration_counter(mod: int) -> int:
    """Simple round-robin counter."""
    global _iteration_counter
    idx = _iteration_counter % mod
    _iteration_counter += 1
    return idx


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON from LLM response, handling ```json ... ``` wrapping."""
    import re
    # Try ```json ... ``` blocks first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try raw JSON
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start >= 0:
            # Find matching end
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
    return None
