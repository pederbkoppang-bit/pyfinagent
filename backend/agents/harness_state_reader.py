"""
Harness State Reader — Read-only access to harness artifacts for Slack agents.

Research finding: "File-based artifacts are the correct communication channel
between agents operating in different context windows."

This module provides:
1. Read-only access to harness handoff artifacts
2. Memory injection from harness_memory.py (episodic + semantic)
3. Experiment log summaries from quant_results.tsv
4. Current best params from optimizer_best.json

CRITICAL: Slack agents MUST NEVER write to harness artifacts.
The harness owns its own state. This is READ-ONLY.

Reference: Anthropic harness design + multi-agent research system
"""

import csv
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Paths relative to project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_HANDOFF_DIR = _PROJECT_ROOT / "handoff"
_EXPERIMENTS_DIR = _PROJECT_ROOT / "backend" / "backtest" / "experiments"
_BEST_PARAMS = _EXPERIMENTS_DIR / "optimizer_best.json"
_TSV_PATH = _EXPERIMENTS_DIR / "quant_results.tsv"


class HarnessStateReader:
    """
    Read-only interface to harness state for Slack agents.

    Exposes:
    - Latest evaluator critique (scores, verdict, weak periods)
    - Current sprint contract (hypothesis, success criteria)
    - Experiment results (last cycle's output)
    - Research plan (planner's next direction)
    - Experiment log summary (trends from quant_results.tsv)
    - Best params snapshot
    - Memory layers (episodic + semantic from harness_memory.py)
    """

    def __init__(self, project_root: Path = None):
        self.root = project_root or _PROJECT_ROOT
        self.handoff = self.root / "handoff"
        self.experiments = self.root / "backend" / "backtest" / "experiments"

    # ── Harness Artifacts (read-only) ────────────────────────────

    def get_evaluator_critique(self) -> dict:
        """Read latest evaluator_critique.md — scores, verdict, weak periods."""
        path = self.handoff / "evaluator_critique.md"
        if not path.exists():
            return {"available": False, "content": None}

        content = path.read_text(encoding="utf-8")
        # Extract structured data from markdown
        result = {
            "available": True,
            "raw": content[:3000],  # Cap for context budget
            "verdict": self._extract_verdict(content),
            "scores": self._extract_scores(content),
            "modified": path.stat().st_mtime,
        }
        return result

    def get_contract(self) -> dict:
        """Read current contract.md — hypothesis, success criteria."""
        path = self.handoff / "contract.md"
        if not path.exists():
            return {"available": False, "content": None}

        content = path.read_text(encoding="utf-8")
        return {
            "available": True,
            "raw": content[:2000],
            "modified": path.stat().st_mtime,
        }

    def get_experiment_results(self) -> dict:
        """Read latest experiment_results.md — what was tried, what happened."""
        path = self.handoff / "experiment_results.md"
        if not path.exists():
            return {"available": False, "content": None}

        content = path.read_text(encoding="utf-8")
        return {
            "available": True,
            "raw": content[:2500],
            "modified": path.stat().st_mtime,
        }

    def get_research_plan(self) -> dict:
        """Read current research_plan.md — planner's next direction."""
        path = self.handoff / "research_plan.md"
        if not path.exists():
            return {"available": False, "content": None}

        content = path.read_text(encoding="utf-8")
        return {
            "available": True,
            "raw": content[:2000],
            "modified": path.stat().st_mtime,
        }

    def get_harness_log(self, last_n: int = 5) -> dict:
        """Read last N cycles from harness_log.md."""
        path = self.handoff / "harness_log.md"
        if not path.exists():
            return {"available": False, "cycles": []}

        content = path.read_text(encoding="utf-8")
        # Split by cycle markers
        cycles = content.split("## Cycle")
        recent = cycles[-last_n:] if len(cycles) > last_n else cycles[1:]  # skip header

        return {
            "available": True,
            "total_cycles": len(cycles) - 1,
            "recent": [f"## Cycle{c}" for c in recent],
        }

    # ── Best Params & Experiment Log ─────────────────────────────

    def get_best_params(self) -> dict:
        """Read current best params from optimizer_best.json."""
        path = self.experiments / "optimizer_best.json"
        if not path.exists():
            return {"available": False}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "available": True,
                "sharpe": data.get("sharpe", 0),
                "dsr": data.get("dsr", 0),
                "params": data.get("params", {}),
                "modified": path.stat().st_mtime,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def get_experiment_summary(self, last_n: int = 20) -> dict:
        """Summarize recent experiments from quant_results.tsv."""
        path = self.experiments / "quant_results.tsv"
        if not path.exists():
            return {"available": False, "total": 0}

        try:
            experiments = []
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    experiments.append(row)

            total = len(experiments)
            recent = experiments[-last_n:]

            # Compute summary stats
            kept = sum(1 for e in recent if e.get("status") == "keep")
            discarded = sum(1 for e in recent if e.get("status") in ("discard", "dsr_reject"))

            return {
                "available": True,
                "total_experiments": total,
                "recent_n": len(recent),
                "recent_kept": kept,
                "recent_discarded": discarded,
                "keep_rate": f"{kept / len(recent) * 100:.0f}%" if recent else "0%",
                "recent_experiments": recent[-5:],  # Last 5 for context
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    # ── Memory Integration ───────────────────────────────────────

    def get_memory_context(self, max_tokens: int = 800) -> str:
        """
        Load episodic + semantic memory for context injection.

        Research finding: "Inject 500-token episodic + 300-token semantic
        into each agent's context."
        """
        try:
            from backend.agents.harness_memory import HarnessMemory
            memory = HarnessMemory(workspace_root=self.root)
            return memory.format_for_context(max_tokens=max_tokens)
        except ImportError:
            logger.debug("harness_memory.py not available")
            return ""
        except Exception as e:
            logger.warning(f"Memory load failed: {e}")
            return ""

    # ── Composite Context Block ──────────────────────────────────

    def build_agent_context(self, agent_type: str = "qa", max_tokens: int = 1500) -> str:
        """
        Build a composite context block for agent prompt injection.

        Includes:
        - Current best Sharpe/DSR (always)
        - Latest evaluator verdict + scores (always)
        - Memory context (episodic + semantic)
        - Experiment summary (for Q&A)
        - Research plan (for Research agent)

        Returns a formatted string that fits within the token budget.
        """
        parts = []
        remaining = max_tokens

        # 1. Best params snapshot (always, ~50 tokens)
        best = self.get_best_params()
        if best.get("available"):
            parts.append(
                f"CURRENT BEST: Sharpe={best['sharpe']:.4f}, DSR={best['dsr']:.4f}"
            )
            remaining -= 50

        # 2. Evaluator state (always, ~150 tokens)
        critique = self.get_evaluator_critique()
        if critique.get("available"):
            scores = critique.get("scores", {})
            parts.append(
                f"LAST EVALUATOR: verdict={critique.get('verdict', '?')}, "
                f"scores=[stat={scores.get('statistical_validity', '?')}, "
                f"robust={scores.get('robustness', '?')}, "
                f"simple={scores.get('simplicity', '?')}, "
                f"reality={scores.get('reality_gap', '?')}]"
            )
            remaining -= 150

        # 3. Memory context (500 tokens)
        if remaining > 500:
            memory = self.get_memory_context(max_tokens=min(500, remaining))
            if memory:
                parts.append(f"\n{memory}")
                remaining -= min(500, len(memory) // 4)

        # 4. Agent-specific context
        if agent_type == "qa" and remaining > 200:
            exp = self.get_experiment_summary(last_n=10)
            if exp.get("available"):
                parts.append(
                    f"\nEXPERIMENT LOG: {exp['total_experiments']} total, "
                    f"recent {exp['recent_n']}: {exp['recent_kept']} kept, "
                    f"{exp['recent_discarded']} discarded ({exp['keep_rate']} rate)"
                )

        elif agent_type == "research" and remaining > 200:
            plan = self.get_research_plan()
            if plan.get("available"):
                # Truncate to fit
                raw = plan["raw"][:remaining * 4]
                parts.append(f"\nRESEARCH PLAN:\n{raw}")

        return "\n".join(parts) if parts else ""

    # ── Helpers ──────────────────────────────────────────────────

    def _extract_verdict(self, content: str) -> str:
        """Extract verdict (PASS/FAIL/CONDITIONAL) from critique markdown."""
        for line in content.split("\n"):
            if "Verdict:" in line or "verdict:" in line:
                upper = line.upper()
                if "PASS" in upper:
                    return "PASS"
                if "FAIL" in upper:
                    return "FAIL"
                if "CONDITIONAL" in upper:
                    return "CONDITIONAL"
        return "UNKNOWN"

    def _extract_scores(self, content: str) -> dict:
        """Extract criterion scores from critique markdown."""
        scores = {}
        for line in content.split("\n"):
            for criterion in ["Statistical Validity", "Robustness", "Simplicity", "Reality Gap"]:
                if criterion in line and "/10" in line:
                    try:
                        # Extract "X/10" pattern
                        parts = line.split("/10")[0].split()
                        score = int(parts[-1].replace(":", ""))
                        scores[criterion.lower().replace(" ", "_")] = score
                    except (ValueError, IndexError):
                        pass
        return scores


# ── Module singleton ─────────────────────────────────────────────

_reader: Optional[HarnessStateReader] = None

def get_harness_reader() -> HarnessStateReader:
    global _reader
    if _reader is None:
        _reader = HarnessStateReader()
    return _reader
