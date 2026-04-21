"""phase-8.5.6 Autoresearch -> paper-live promoter.

Requires >= 5 trading days of shadow-log history and DSR >= 0.95 before
promoting a trial to paper-live. Position size is tied to realized DSR.
Kill-switch callback fires on drawdown breach.

Pure functions. ASCII-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

SHADOW_MIN_DAYS = 5
DD_TRIGGER = 0.10
DSR_MIN_FOR_PROMOTION = 0.95


@dataclass(frozen=True)
class Promoter:
    shadow_min_days: int = SHADOW_MIN_DAYS
    dsr_min: float = DSR_MIN_FOR_PROMOTION
    dd_trigger: float = DD_TRIGGER

    def promote(self, trial: dict[str, Any]) -> dict[str, Any]:
        days = int(trial.get("shadow_trading_days", 0))
        dsr = float(trial.get("dsr", 0.0))
        if days < self.shadow_min_days:
            return {"promoted": False, "reason": f"shadow_days_below_min:{days}<{self.shadow_min_days}"}
        if dsr < self.dsr_min:
            return {"promoted": False, "reason": f"dsr_below_min:{dsr:.4f}<{self.dsr_min}"}
        return {"promoted": True, "reason": None, "trial_id": trial.get("trial_id")}

    def position_size(self, trial: dict[str, Any], capital: float) -> float:
        """Notional size tied to realized DSR. Below DSR=0.5 -> 0. At DSR=1.0 -> full capital."""
        dsr = float(trial.get("dsr", 0.0))
        fraction = max(0.0, min(1.0, (dsr - 0.5) * 2.0))
        return float(capital) * fraction

    def on_dd_breach(
        self,
        current_dd: float,
        kill_fn: Callable[[str], None],
    ) -> bool:
        """Fire kill_fn when abs(dd) exceeds dd_trigger. Returns True when fired."""
        if abs(float(current_dd)) > self.dd_trigger:
            kill_fn(f"dd_breach:{current_dd:.4f}>{self.dd_trigger}")
            return True
        return False


__all__ = ["Promoter", "SHADOW_MIN_DAYS", "DD_TRIGGER"]
