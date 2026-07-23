"""phase-9.8 Cost-budget watcher with circuit breaker.

Monitors daily + monthly BigQuery spend. Trips circuit breaker when daily or
monthly cap is exceeded. Re-uses BudgetEnforcer from phase-8.5.2.

pyfinagent runs on a Claude Max subscription (flat-fee LLM access), so the
only variable cost worth watching is BigQuery bytes billed. phase-9.9.2
swapped the signal from the Anthropic Cost API to
`region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT * $6.25/TiB`.

APScheduler fires run() with zero args; `daily_spend_usd` and
`monthly_spend_usd` are optional, resolved via `fetch_fn or
_default_fetch_spend` when absent. Fail-open returns (0.0, 0.0).
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from backend.autoresearch.budget import BudgetEnforcer
from backend.slack_bot.job_runtime import IdempotencyKey, IdempotencyStore, heartbeat

logger = logging.getLogger(__name__)
JOB_NAME = "cost_budget_watcher"
_BQ_USD_PER_TIB = 6.25  # on-demand pricing, stable 2023-07-05 -> 2026


def run(
    *,
    daily_spend_usd: float | None = None,
    monthly_spend_usd: float | None = None,
    fetch_fn: Callable[[], tuple[float, float]] | None = None,
    daily_cap_usd: float = 5.0,
    monthly_cap_usd: float = 50.0,
    alert_fn: Callable[[str, dict], None] | None = None,
    store: IdempotencyStore | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    if daily_spend_usd is None or monthly_spend_usd is None:
        fetched_daily, fetched_monthly = (fetch_fn or _default_fetch_spend)()
        if daily_spend_usd is None:
            daily_spend_usd = fetched_daily
        if monthly_spend_usd is None:
            monthly_spend_usd = fetched_monthly

    key = IdempotencyKey.daily(JOB_NAME, day=day)
    result: dict[str, Any] = {
        "tripped": False,
        "reason": None,
        "daily": daily_spend_usd,
        "monthly": monthly_spend_usd,
        "key": key,
        "skipped": False,
    }
    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        fired: list[tuple[str, dict]] = []
        for scope, spend, cap in (
            ("daily", daily_spend_usd, daily_cap_usd),
            ("monthly", monthly_spend_usd, monthly_cap_usd),
        ):
            e = BudgetEnforcer(
                wallclock_seconds=10**9,
                usd_budget=cap,
                alert_fn=lambda reason, st, scope=scope: fired.append((scope, {"reason": reason, "state": st})),
            )
            e.tick(float(spend))
        if fired:
            result["tripped"] = True
            result["reason"] = fired[0][0]
            if alert_fn is not None:
                try:
                    alert_fn(fired[0][0], fired[0][1])
                except Exception as exc:
                    logger.warning("cost_budget_watcher: alert_fn fail-open: %r", exc)
    return result


def _default_fetch_spend() -> tuple[float, float]:
    """DEPRECATED back-compat alias -- delegates to the public
    `backend.services.observability.fetch_spend`.

    phase-75.5 (arch-04) promoted the implementation out of this private symbol
    because llm_client and /api/cost-budget both reached across the layer boundary
    into it. The alias is RETAINED deliberately, not for politeness: an existing test
    (tests/slack_bot/test_scheduler_wiring_phase991.py:150) monkeypatches this exact
    attribute, and that file lives OUTSIDE backend/tests/ -- i.e. outside this step's
    verification command -- so removing the name would break it SILENTLY.
    """
    from backend.services.observability import fetch_spend

    return fetch_spend()


__all__ = ["run", "JOB_NAME", "_default_fetch_spend"]
