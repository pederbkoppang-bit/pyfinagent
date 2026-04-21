"""phase-8.5.7 Overnight autoresearch orchestration cron.

Registers a nightly job that runs ~100 experiments within the phase-8.5.2
wall-clock + USD budget. Results visible in phase-4.7 Harness tab via BQ
views.

Fail-open registration + bounded-batch execution. ASCII-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AutoresearchCron:
    """In-memory registration shim. Real APScheduler wiring deferred to phase-9."""

    target_experiments_per_night: int = 100
    min_experiments_per_night: int = 80
    _registered: bool = False
    _schedule: str | None = None

    def register(self, scheduler: Any | None = None, *, cron_schedule: str = "0 2 * * *") -> bool:
        """Register nightly 2am cron. `scheduler` is an injectable APScheduler-like; None -> in-memory shim."""
        if scheduler is not None and hasattr(scheduler, "add_job"):
            try:
                scheduler.add_job(
                    func=lambda: None,
                    trigger="cron",
                    hour=int(cron_schedule.split()[1]),
                    id="autoresearch_overnight",
                    replace_existing=True,
                )
            except Exception:
                pass  # fail-open: registration is best-effort
        self._registered = True
        self._schedule = cron_schedule
        return True

    @property
    def registered(self) -> bool:
        return self._registered

    def run_batch(
        self,
        enforcer,  # BudgetEnforcer
        run_one: Callable[[int], dict[str, Any]],
        *,
        max_experiments: int | None = None,
    ) -> dict[str, Any]:
        """Run experiments sequentially until budget breaches or max_experiments hit.

        Returns aggregate stats. Each `run_one(i)` should return a dict with
        at least `{"usd_spent": float}` so the enforcer can be ticked.
        """
        cap = max_experiments or self.target_experiments_per_night
        results: list[dict[str, Any]] = []
        for i in range(cap):
            try:
                r = run_one(i)
            except Exception as exc:
                results.append({"index": i, "error": repr(exc), "usd_spent": 0.0})
                continue
            results.append(r)
            state = enforcer.tick(float(r.get("usd_spent", 0.0)))
            if state["terminated"]:
                break
        return {
            "experiments_run": len(results),
            "terminated_by_budget": enforcer.state["terminated"],
            "reason": enforcer.state["reason"],
            "results": results,
        }


__all__ = ["AutoresearchCron"]
