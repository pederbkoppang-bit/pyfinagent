"""
DEPRECATED — Phase 4 stub. Not part of the active MAS architecture.

The active harness is run_harness.py (Planner → Generator → Evaluator pattern).
The active MAS orchestrator is backend/agents/multi_agent_orchestrator.py.
This file is a skeleton for future autonomous cycling and should not be extended
until Phase 4 activates.

Original purpose:
Autonomous Harness — Self-driving backtest + optimization loop.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AutonomousHarness:
    """
    Self-driving harness that:
    1. Runs backtests on schedule
    2. Evaluates Sharpe + robustness
    3. Adjusts parameters (if improving)
    4. Logs all decisions
    5. Reports to Slack
    
    Designed for multi-day/week autonomous optimization.
    """
    
    def __init__(self, config_path: str = "backend/backtest/experiments/optimizer_best.json"):
        self.config_path = config_path
        self.running = False
        self.current_sharpe = 0.0
        self.current_params = {}
        
    async def start(self):
        """Start the autonomous loop."""
        self.running = True
        logger.info("[bot] AUTONOMOUS HARNESS STARTED")
        
        while self.running:
            try:
                # 1. Check if enough time has passed since last run
                if self._should_run():
                    # 2. Run backtest
                    result = await self._run_backtest()
                    
                    # 3. Evaluate result
                    decision = await self._evaluate_result(result)
                    
                    # 4. Log decision
                    await self._log_decision(decision)
                    
                    # 5. Report to Slack
                    await self._report_slack(decision)
                
                # Wait 60s before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in autonomous harness: {e}")
                await asyncio.sleep(60)
    
    def _should_run(self) -> bool:
        """Check if enough time has passed since last run."""
        # TODO: Check timestamp of last run
        # Default: run every 4 hours
        return True
    
    async def _run_backtest(self) -> Dict[str, Any]:
        """Execute backtest with current parameters."""
        logger.info("[stats] Running backtest...")
        # TODO: Import and run backtest_engine.py
        return {
            "sharpe": 0.0,
            "return": 0.0,
            "max_dd": 0.0,
            "trades": 0,
            "parameters": self.current_params
        }
    
    async def _evaluate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if result is:
        - Better than current best → ACCEPT + save
        - Worse → REJECT
        - Marginal → CONDITIONAL (save for comparison)
        """
        new_sharpe = result.get("sharpe", 0.0)
        
        if new_sharpe > self.current_sharpe * 1.01:  # 1% improvement threshold
            return {
                "decision": "ACCEPT",
                "reason": f"Sharpe improved: {self.current_sharpe:.4f} → {new_sharpe:.4f}",
                "new_best": True
            }
        elif new_sharpe > self.current_sharpe:
            return {
                "decision": "CONDITIONAL",
                "reason": f"Marginal improvement: {new_sharpe - self.current_sharpe:.4f}",
                "new_best": False
            }
        else:
            return {
                "decision": "REJECT",
                "reason": f"Degradation: {new_sharpe:.4f} < {self.current_sharpe:.4f}",
                "new_best": False
            }
    
    async def _log_decision(self, decision: Dict[str, Any]):
        """Log decision to harness_log.md"""
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(f"[Research] {decision['decision']}: {decision['reason']}")
        # TODO: Append to handoff/harness_log.md
    
    async def _report_slack(self, decision: Dict[str, Any]):
        """Send status to Slack #ford-approvals"""
        # TODO: Send message to Slack with decision + metrics
        pass
    
    def stop(self):
        """Stop the autonomous loop."""
        self.running = False
        logger.info("[bot] AUTONOMOUS HARNESS STOPPED")


# Global instance
_harness = None


async def start_autonomous_harness(config_path: str = "backend/backtest/experiments/optimizer_best.json"):
    """Start the autonomous harness."""
    global _harness
    _harness = AutonomousHarness(config_path)
    await _harness.start()


def get_autonomous_harness() -> AutonomousHarness:
    """Get the global harness instance."""
    global _harness
    return _harness


# ═══════════════════════════════════════════════════════════════════
# phase-4.9.8 — Autoresearch Gauntlet hook
# ═══════════════════════════════════════════════════════════════════
#
# `promote_strategy(name)` is the single entry point autoresearch
# (phase-8.5) uses before allocating capital to a new strategy. It
# enforces the phase-4.9 Gauntlet contract:
#
#   1. A Gauntlet report must exist at
#      `handoff/gauntlet/<name>/report.json`.
#   2. The report must PASS the phase-4.9.6 evaluator
#      (`overall_pass=True`).
#   3. The strategy must not be on the 30-day blocklist at
#      `handoff/gauntlet_blocklist.jsonl` -- blocked strategies
#      cannot re-apply for promotion for 30 calendar days after
#      a failed attempt.
#
# On ANY of those conditions, the function raises. The caller
# (autoresearch orchestrator) is expected to catch + log, NOT retry.
# Successful promotions append an "annotated" row to
# `handoff/harness_log.md` so the Harness tab surfaces them.

from pathlib import Path as _Path
import json as _json
from datetime import datetime as _dt, timedelta as _td, timezone as _tz

_REPO = _Path(__file__).resolve().parents[1]
_GAUNTLET_ROOT = _REPO / "handoff" / "gauntlet"
_BLOCKLIST_PATH = _REPO / "handoff" / "gauntlet_blocklist.jsonl"
_HARNESS_LOG = _REPO / "handoff" / "harness_log.md"
_BLOCKLIST_DAYS = 30


class PromotionBlocked(Exception):
    """Raised by promote_strategy when the Gauntlet contract is not met."""


def _read_blocklist() -> list[dict]:
    if not _BLOCKLIST_PATH.exists():
        return []
    rows: list[dict] = []
    with _BLOCKLIST_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_json.loads(line))
            except Exception:
                continue
    return rows


def _on_blocklist(strategy: str, now: _dt | None = None) -> dict | None:
    now = now or _dt.now(_tz.utc)
    for row in _read_blocklist():
        if row.get("strategy") != strategy:
            continue
        try:
            blocked_at = _dt.fromisoformat(row["blocked_at"])
        except Exception:
            continue
        if (now - blocked_at) < _td(days=_BLOCKLIST_DAYS):
            return row
    return None


def _append_blocklist(strategy: str, reason: str) -> None:
    _BLOCKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "strategy": strategy,
        "blocked_at": _dt.now(_tz.utc).isoformat(),
        "reason": reason[:400],
        "expires_after_days": _BLOCKLIST_DAYS,
    }
    with _BLOCKLIST_PATH.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(row) + "\n")


def _annotate_harness_log(strategy: str, verdict: dict) -> None:
    try:
        line = (
            "\n[phase-4.9.8] autoresearch promoted strategy="
            f"{strategy} overall_pass={verdict['overall_pass']} "
            f"at={_dt.now(_tz.utc).isoformat()}\n"
        )
        with _HARNESS_LOG.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        logger.debug("autoresearch: harness-log annotate failed")


def promote_strategy(strategy: str) -> dict:
    """Authorise `strategy` for capital allocation via the Gauntlet gate.

    Returns the evaluator verdict dict on success. Raises
    `PromotionBlocked` (subclass of `Exception`) on any failure --
    missing report, failed pass criteria, or active blocklist entry.
    """
    # 1. Blocklist check (30-day cooldown)
    block = _on_blocklist(strategy)
    if block:
        raise PromotionBlocked(
            f"strategy {strategy!r} on 30-day blocklist since "
            f"{block['blocked_at']}: {block.get('reason', '?')}"
        )

    # 2. Gauntlet report must exist
    report_path = _GAUNTLET_ROOT / strategy / "report.json"
    if not report_path.exists():
        _append_blocklist(strategy, f"missing gauntlet report at {report_path}")
        raise PromotionBlocked(
            f"no gauntlet report for strategy {strategy!r} at {report_path}; "
            "added to 30-day blocklist"
        )

    # 3. Gauntlet pass-criteria evaluator
    from backend.backtest.gauntlet.evaluator import evaluate  # lazy import
    report = _json.loads(report_path.read_text(encoding="utf-8"))
    verdict = evaluate(report)
    if not verdict["overall_pass"]:
        reason = "; ".join(verdict.get("reasons", [])) or "overall_pass=False"
        _append_blocklist(strategy, reason)
        raise PromotionBlocked(
            f"strategy {strategy!r} failed gauntlet: {reason}; "
            "added to 30-day blocklist"
        )

    _annotate_harness_log(strategy, verdict)
    return verdict
