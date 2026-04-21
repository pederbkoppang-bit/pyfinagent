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
import os
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
    """Fetch today + this-month BQ spend from INFORMATION_SCHEMA.JOBS_BY_PROJECT.

    Price: $6.25/TiB on-demand (stable 2023-07-05 -> 2026). Uses
    `total_bytes_billed` (reflects 10 MB minimum-billing floor; cache hits are
    already 0). Requires `roles/bigquery.resourceViewer` on the project.
    Fail-open to (0.0, 0.0) on any exception.
    """
    try:
        from google.cloud import bigquery
        project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
        client = bigquery.Client(project=project)
        sql = f"""
            SELECT
              SUM(IF(DATE(creation_time) = CURRENT_DATE(), total_bytes_billed, 0))
                AS daily_bytes,
              SUM(total_bytes_billed) AS monthly_bytes
            FROM `{project}.region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
            WHERE
              creation_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH)
              AND state = 'DONE'
        """
        rows = list(client.query(sql).result())
        if not rows:
            return 0.0, 0.0
        row = rows[0]
        daily_bytes = float(row["daily_bytes"] or 0)
        monthly_bytes = float(row["monthly_bytes"] or 0)
        daily = daily_bytes / 1e12 * _BQ_USD_PER_TIB
        monthly = monthly_bytes / 1e12 * _BQ_USD_PER_TIB
        return daily, monthly
    except Exception as exc:
        logger.warning("cost_budget_watcher: BQ spend fetch fail-open: %r", exc)
        return 0.0, 0.0


__all__ = ["run", "JOB_NAME"]
