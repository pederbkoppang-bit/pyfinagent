"""phase-8.5.10 Meta-search DSR calibration.

When running many trials in parallel (the autoresearch loop), the per-trial
DSR is NOT the right statistic -- you must recompute DSR at the cumulative
sample size across ALL trials, including abandoned ones. This is the
"multiple testing correction" Bailey & Lopez de Prado describe for DSR.

TrialLedger logs every trial, including abandoned. `meta_dsr(trials)`
recomputes DSR adjusted for cumulative N.

Pure functions. ASCII-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

STRICT_DSR_MIN = 0.99  # when cumulative N > 50
LOOSE_DSR_MIN = 0.95


@dataclass
class TrialLedger:
    """Log every trial, including abandoned. No mutation of logged trials."""

    trials: list[dict[str, Any]] = field(default_factory=list)

    def log(self, trial: dict[str, Any]) -> None:
        """Append a trial record. `abandoned: bool` may be set on abandoned trials."""
        self.trials.append(dict(trial))  # shallow copy to prevent external mutation

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def n_abandoned(self) -> int:
        return sum(1 for t in self.trials if t.get("abandoned"))

    @property
    def promoted(self) -> list[dict[str, Any]]:
        return [t for t in self.trials if t.get("promoted") and not t.get("abandoned")]


def meta_dsr(trials: list[dict[str, Any]], *, cumulative_n: int | None = None) -> dict[str, float]:
    """Recompute DSR adjusted for cumulative N across ALL trials (including abandoned).

    Formula (simplified multiple-testing correction):
        penalty = 0.1 * sqrt(log(max(2, N)))   # monotonically INCREASES with N
        adjusted_dsr = raw_dsr - penalty

    A full Bailey-Lopez de Prado DSR adjustment divides by an estimated
    skewness-kurtosis-corrected standard deviation and subtracts an explicit
    multiple-testing term; this scaffold uses the qualitatively-correct
    monotone penalty only, which is sufficient for gate wiring.
    """
    if not trials:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "n": 0, "penalty": 0.0}
    n = int(cumulative_n if cumulative_n is not None else len(trials))
    penalty = 0.1 * math.sqrt(math.log(max(2, n)))
    raws = [float(t.get("dsr", 0.0)) for t in trials]
    adj = [r - penalty for r in raws]
    return {
        "mean": sum(adj) / len(adj),
        "min": min(adj),
        "max": max(adj),
        "n": n,
        "penalty": float(penalty),
    }


def required_dsr(cumulative_n: int) -> float:
    """Step-up DSR threshold: 0.95 by default, 0.99 once cumulative_n > 50."""
    return STRICT_DSR_MIN if int(cumulative_n) > 50 else LOOSE_DSR_MIN


def cpcv_applied_on(trial: dict[str, Any]) -> bool:
    """Promoted trials must carry `cpcv_applied: True`; non-promoted need not."""
    if trial.get("promoted") and not trial.get("abandoned"):
        return bool(trial.get("cpcv_applied"))
    return True  # vacuously true for non-promoted


__all__ = ["TrialLedger", "meta_dsr", "required_dsr", "cpcv_applied_on", "STRICT_DSR_MIN", "LOOSE_DSR_MIN"]
