"""phase-8.5.2 Wall-clock + USD budget enforcer.

Deterministic termination of an autoresearch cycle when either wall-clock
elapsed time or cumulative USD spend exceeds its cap. Alerts via an
injectable callable (default: logger warning); tests pass a captive list.

Example:
    enforcer = BudgetEnforcer(wallclock_seconds=3600, usd_budget=5.00)
    while True:
        state = enforcer.tick(usd_spent_this_iter)
        if state["terminated"]:
            logger.warning("budget: terminated reason=%s", state["reason"])
            break
        ... do work ...

Fail-open. ASCII-only.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BudgetEnforcer:
    """Deterministic budget enforcement over wall-clock AND USD spend.

    Parameters
    ----------
    wallclock_seconds : float
        Budget over time.time() elapsed from first tick.
    usd_budget : float
        Budget over cumulative USD spend supplied via tick(usd_spent).
    alert_fn : Callable[[str, dict], None] | None
        Called once on first budget breach with (reason, state). Default
        None -> logger warning only.
    """

    def __init__(
        self,
        wallclock_seconds: float,
        usd_budget: float,
        alert_fn: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if wallclock_seconds < 0 or usd_budget < 0:
            raise ValueError("budgets must be non-negative")
        self.wallclock_seconds = float(wallclock_seconds)
        self.usd_budget = float(usd_budget)
        self.alert_fn = alert_fn
        self._start_ts: float | None = None
        self._spent_usd: float = 0.0
        self._terminated: bool = False
        self._reason: str | None = None
        self._alerted: bool = False

    @property
    def state(self) -> dict[str, Any]:
        elapsed = 0.0 if self._start_ts is None else (time.monotonic() - self._start_ts)
        return {
            "elapsed_s": elapsed,
            "spent_usd": self._spent_usd,
            "wallclock_seconds": self.wallclock_seconds,
            "usd_budget": self.usd_budget,
            "terminated": self._terminated,
            "reason": self._reason,
        }

    def tick(self, usd_spent: float = 0.0) -> dict[str, Any]:
        """Record spend, evaluate budget, return state dict.

        On first breach, calls `alert_fn(reason, state)` exactly once.
        Subsequent ticks are idempotent: they do not re-alert and keep
        the prior `reason`.
        """
        if self._start_ts is None:
            self._start_ts = time.monotonic()
        if usd_spent < 0:
            usd_spent = 0.0
        self._spent_usd += float(usd_spent)
        if not self._terminated:
            elapsed = time.monotonic() - self._start_ts
            if elapsed >= self.wallclock_seconds:
                self._terminated = True
                self._reason = "wallclock"
            elif self._spent_usd >= self.usd_budget:
                self._terminated = True
                self._reason = "usd"
        if self._terminated and not self._alerted:
            self._alerted = True
            if self.alert_fn is not None:
                try:
                    self.alert_fn(self._reason or "unknown", self.state)
                except Exception as exc:  # pragma: no cover
                    logger.warning("budget: alert_fn raised: %r", exc)
            else:
                logger.warning(
                    "budget: breached reason=%s elapsed=%.3fs spent_usd=%.4f",
                    self._reason, self.state["elapsed_s"], self._spent_usd,
                )
        return self.state


__all__ = ["BudgetEnforcer"]
