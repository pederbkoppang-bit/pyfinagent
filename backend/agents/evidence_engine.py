"""
Evidence Engine — Phase 3.1

Maintains history of:
- Backtest results (Sharpe, Return, MaxDD, trades)
- Feature success/failure tracking
- Proposal acceptance rates
- Sharpe gains attributed to specific features
- Market regime classification

Feeds this evidence to the Planner agent for smarter proposals.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

EVIDENCE_FILE = Path("handoff/evidence_history.json")


class EvidenceEngine:
    """Tracks backtest evidence and feature performance."""

    def __init__(self):
        """Initialize evidence engine, load history if exists."""
        self.history = self._load_history()
        self.current_best_sharpe = 1.1705  # Known baseline from Phase 2.12
        self.accepted_proposals = 0
        self.rejected_proposals = 0

    def record_backtest_result(
        self,
        sharpe: float,
        ret_pct: float,
        max_dd: float,
        num_trades: int,
        features: List[str],
        params: Dict[str, Any],
        notes: Optional[str] = None
    ):
        """Record a new backtest result."""

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sharpe": sharpe,
            "return_pct": ret_pct,
            "max_dd": max_dd,
            "num_trades": num_trades,
            "features": features,
            "params": params,
            "notes": notes
        }

        self.history["backtest_results"].append(result)

        # Update running best
        if sharpe > self.current_best_sharpe:
            self.current_best_sharpe = sharpe
            logger.info(f"[target] NEW BEST: Sharpe {sharpe:.4f} (+{sharpe - 1.1705:.4f})")

        # Track feature success
        for feature in features:
            if feature not in self.history["feature_stats"]:
                self.history["feature_stats"][feature] = {
                    "count": 0,
                    "total_sharpe_delta": 0.0
                }
            self.history["feature_stats"][feature]["count"] += 1
            self.history["feature_stats"][feature]["total_sharpe_delta"] += (sharpe - 1.1705)

        self._save_history()

    def record_proposal_verdict(
        self,
        proposal: Dict[str, Any],
        verdict: str,  # ACCEPT | REJECT | REVISE
        feedback: str
    ):
        """Record evaluator verdict on proposal."""

        proposal_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "proposal": proposal,
            "verdict": verdict,
            "feedback": feedback
        }

        self.history["proposals"].append(proposal_record)

        if verdict == "ACCEPT":
            self.accepted_proposals += 1
        elif verdict == "REJECT":
            self.rejected_proposals += 1

        logger.info(f"[stats] Proposals: {self.accepted_proposals} accepted, {self.rejected_proposals} rejected")

        self._save_history()

    def get_recent_results(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent backtest results."""
        return self.history["backtest_results"][-limit:]

    def get_feature_success_rate(self, feature: str) -> float:
        """Get success rate of a feature (how often it appeared in improving backtests)."""
        if feature not in self.history["feature_stats"]:
            return 0.0

        stats = self.history["feature_stats"][feature]
        if stats["count"] == 0:
            return 0.0

        # Sharpe gains > +2% considered success
        return stats["total_sharpe_delta"] / stats["count"] / 0.02

    def get_weakness_summary(self) -> str:
        """Identify current strategy weaknesses for planner."""

        if len(self.history["backtest_results"]) < 2:
            return "Insufficient history for analysis."

        recent = self.get_recent_results(5)
        latest = recent[-1]

        weaknesses = []

        # Check trade frequency
        if latest["num_trades"] > 40:
            weaknesses.append(f"High trade frequency ({latest['num_trades']} trades/month)")

        # Check drawdown
        if latest["max_dd"] > 0.15:
            weaknesses.append(f"Drawdown too large ({latest['max_dd']:.2%})")

        # Check return consistency (volatility across recent runs)
        returns = [r["return_pct"] for r in recent]
        return_std = (sum((r - sum(returns) / len(returns)) ** 2 for r in returns) / len(returns)) ** 0.5
        if return_std > 15:
            weaknesses.append(f"Inconsistent returns (σ={return_std:.1f}%)")

        if not weaknesses:
            weaknesses.append("Strategy performing well; small improvements possible")

        return " | ".join(weaknesses)

    def acceptance_rate(self) -> float:
        """Get proposal acceptance rate."""
        total = self.accepted_proposals + self.rejected_proposals
        if total == 0:
            return 0.0
        return self.accepted_proposals / total

    def _load_history(self) -> Dict[str, Any]:
        """Load evidence history from file or create new."""
        if EVIDENCE_FILE.exists():
            try:
                with open(EVIDENCE_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load evidence history: {e}")

        return {
            "backtest_results": [],
            "proposals": [],
            "feature_stats": {}
        }

    def _save_history(self):
        """Save evidence history to file."""
        try:
            with open(EVIDENCE_FILE, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save evidence history: {e}")


def get_evidence_engine() -> EvidenceEngine:
    """Get or create global evidence engine."""
    global _engine
    if '_engine' not in globals():
        _engine = EvidenceEngine()
    return _engine
