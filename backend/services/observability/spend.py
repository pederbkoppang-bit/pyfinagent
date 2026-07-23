"""phase-75.5 (arch-04): the public home for the cloud-spend fetch.

Before this module, the fetch lived as `_default_fetch_spend` -- a PRIVATE-by-name
symbol inside `backend/slack_bot/jobs/cost_budget_watcher.py`, whose `__all__` exported
only `["run", "JOB_NAME"]`. Two other packages nevertheless reached across the layer
boundary into it: the llm_client hard-block (`llm_client.py:427`) and the
`/api/cost-budget` endpoint (`cost_budget_api.py:24`). A money guard that three
subsystems depend on should not be a private symbol in a Slack job.

SCOPE WARNING -- READ BEFORE USING THIS AS AN "LLM SPEND" NUMBER.
This function measures **BigQuery** spend: it prices
`INFORMATION_SCHEMA.JOBS_BY_PROJECT.total_bytes_billed` at the on-demand $6.25/TiB rate.
It does NOT measure LLM/Anthropic/Gemini spend. That matters because
`settings.cost_budget_daily_usd` is documented as the *"Daily LLM-spend cap"* and
CLAUDE.md describes the $25/day cap as the LLM circuit breaker -- so the guard does not
measure what its consumers' names imply. phase-75.5 deliberately promoted this
**unchanged** (behavior-preserving refactor); reconciling the metric with its name is
queued as its own masterplan step. Do not paper over it here.

Fail-open is intentional and preserved: a spend-fetch failure must never block a live
call. But phase-75.5 adds a degradation COUNTER + alert seam, because the pre-existing
behavior was to emit a single WARNING line and return (0.0, 0.0) -- which is
indistinguishable from "you have spent nothing", i.e. the guard silently opens.
"""
from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

# BigQuery on-demand analysis pricing, stable 2023-07-05 -> 2026.
_BQ_USD_PER_TIB = 6.25

# phase-75.5 (arch-04): degradation counter. A fail-open that returns (0.0, 0.0) looks
# exactly like "no spend" to every caller, so an outage silently DISABLES the budget
# guard. Counting it makes the degradation observable instead of inferable.
_DEGRADED_LOCK = threading.Lock()
_DEGRADED_COUNT = 0
_LAST_ERROR: str = ""

# Alert once per process on the healthy->degraded transition (alert-on-transition, the
# same discipline the phase-66.1 rail breaker uses), not on every call.
_ALERTED = False


def spend_guard_status() -> dict:
    """Observable state of the spend guard. Read by tests and diagnostics."""
    with _DEGRADED_LOCK:
        return {
            "degraded_count": _DEGRADED_COUNT,
            "last_error": _LAST_ERROR,
            "alerted": _ALERTED,
        }


def reset_spend_guard_status() -> None:
    """Test seam -- reset the counter between cases."""
    global _DEGRADED_COUNT, _LAST_ERROR, _ALERTED
    with _DEGRADED_LOCK:
        _DEGRADED_COUNT = 0
        _LAST_ERROR = ""
        _ALERTED = False


def _record_degradation(exc: BaseException) -> None:
    """Count a fail-open, and page exactly once on the closed->open transition."""
    global _DEGRADED_COUNT, _LAST_ERROR, _ALERTED
    should_alert = False
    with _DEGRADED_LOCK:
        _DEGRADED_COUNT += 1
        _LAST_ERROR = repr(exc)[:400]
        if not _ALERTED:
            _ALERTED = True
            should_alert = True
        count = _DEGRADED_COUNT
    logger.warning(
        "observability.spend: BQ spend fetch fail-open (#%d): %r -- the cost-budget "
        "guard is returning (0.0, 0.0), which is INDISTINGUISHABLE from zero spend, "
        "so the budget hard-block is effectively disabled until this recovers.",
        count, exc,
    )
    if should_alert:
        try:  # fail-open: alerting must never break the money path
            from backend.services.observability.alerting import raise_cron_alert_sync

            raise_cron_alert_sync(
                source="cost_budget_guard",
                error_type="spend_fetch_degraded",
                severity="P2",
                title="Cost-budget spend fetch degraded -- guard is fail-open",
                detail=(
                    f"fetch_spend() fail-open: {exc!r}. Callers receive (0.0, 0.0), so "
                    f"the daily/monthly budget hard-block cannot trip while this "
                    f"persists."
                ),
            )
        except Exception as alert_exc:  # pragma: no cover -- fail-open
            logger.debug("observability.spend: alert skipped: %r", alert_exc)


def fetch_spend() -> tuple[float, float]:
    """Return (daily_usd, monthly_usd) of BigQuery spend for the current project.

    Prices `total_bytes_billed` at $6.25/TiB (reflects the 10 MB minimum-billing floor;
    cache hits are already 0). Requires `roles/bigquery.resourceViewer`.

    Fail-open to (0.0, 0.0) on ANY exception -- preserved verbatim from the
    pre-phase-75.5 behavior -- but the failure is now counted and alerted once.

    See the module docstring: this is BIGQUERY spend, not LLM spend.
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
        return (
            daily_bytes / 1e12 * _BQ_USD_PER_TIB,
            monthly_bytes / 1e12 * _BQ_USD_PER_TIB,
        )
    except Exception as exc:
        _record_degradation(exc)
        return 0.0, 0.0


__all__ = ["fetch_spend", "spend_guard_status", "reset_spend_guard_status"]
