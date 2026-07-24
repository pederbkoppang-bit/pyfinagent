"""phase-75.5 (arch-04): the public home for the cloud-spend fetch.

Before this module, the fetch lived as `_default_fetch_spend` -- a PRIVATE-by-name
symbol inside `backend/slack_bot/jobs/cost_budget_watcher.py`, whose `__all__` exported
only `["run", "JOB_NAME"]`. Two other packages nevertheless reached across the layer
boundary into it: the llm_client hard-block (`llm_client.py:427`) and the
`/api/cost-budget` endpoint (`cost_budget_api.py:24`). A money guard that three
subsystems depend on should not be a private symbol in a Slack job.

SCOPE WARNING -- READ BEFORE USING `fetch_spend` AS AN "LLM SPEND" NUMBER.
`fetch_spend` measures **BigQuery** spend: it prices
`INFORMATION_SCHEMA.JOBS_BY_PROJECT.total_bytes_billed` at the on-demand $6.25/TiB rate.
It does NOT measure LLM/Anthropic/Gemini spend. That matters because
`settings.cost_budget_daily_usd` is documented as the *"Daily LLM-spend cap"* and
CLAUDE.md describes the $25/day cap as the LLM circuit breaker.

phase-75.5.1 resolves the mismatch with `fetch_llm_spend`: metered LLM tokens from
`llm_call_log` priced against the live `cost_tracker.MODEL_PRICING` table with the
cache-aware formula. The budget gate selects it via
`settings.cost_budget_use_llm_spend_enabled` (default OFF = the BQ metric, byte-identical
to pre-75.5.1). Three invariants the LLM metric MUST keep:

  1. METERED-ONLY. Flat-fee CC-rail rows (provider='claude-code', or
     provider='anthropic' with agent LIKE 'cc_rail:%') record tokens whose real cost is
     ~$0 (Claude Code Max rail). Pricing them at API rates would trip the $25 breaker on
     FREE tokens and falsely halt trading -- the same phantom class as the 2026-06
     session_cost_usd staircase.
  2. RAW TOKENS x PRICING, never stored dollars. llm_call_log has no per-call cost
     column, and session_cost_usd is a per-cycle cumulative GAUGE (phase-66.3: never
     sum it). Token counts are also invariant across the 75.5 cache-cost fix, so a
     day window spanning that boundary still prices correctly.
  3. CACHE-AWARE pricing ported from cost_tracker (read 0.1x input rate, write 2.0x) --
     the sovereign_api variant that ignores cache columns under-counts and would let
     the breaker trip late.

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


def _price_llm_tokens(
    model: str,
    input_tok: float,
    output_tok: float,
    cache_creation_tok: float,
    cache_read_tok: float,
) -> float:
    """Cache-aware pricing, ported from cost_tracker.CostTracker (the accurate
    formula -- read 0.1x input rate, write 2.0x for the 1h-TTL tier) against the
    LIVE MODEL_PRICING table. Single source of truth: import, never copy."""
    from backend.agents.cost_tracker import _DEFAULT_PRICING, MODEL_PRICING

    pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
    if cache_read_tok > 0 or cache_creation_tok > 0:
        return (
            cache_read_tok * pricing[0] * 0.1
            + cache_creation_tok * pricing[0] * 2.0
            + input_tok * pricing[0]
            + output_tok * pricing[1]
        ) / 1_000_000
    return (input_tok * pricing[0] + output_tok * pricing[1]) / 1_000_000


def fetch_llm_spend() -> tuple[float, float]:
    """Return (daily_usd, monthly_usd) of METERED LLM spend.

    Sums raw token columns from `llm_call_log` per model (month-to-date, with a
    same-day split) and prices them in Python via `_price_llm_tokens`. Excludes
    the flat-fee CC-rail rows and failed calls (see the module docstring's three
    invariants). Fail-open to (0.0, 0.0) through the SAME arch-04 degradation
    seam as `fetch_spend` -- a spend-fetch failure must never block a live call,
    but it must be counted and alerted once.
    """
    try:
        from google.cloud import bigquery

        from backend.config.settings import get_settings

        settings = get_settings()
        project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
        dataset = getattr(settings, "bq_dataset_observability", "pyfinagent_data")
        client = bigquery.Client(project=project)
        sql = f"""
            SELECT
              model,
              SUM(IF(DATE(ts) = CURRENT_DATE(), input_tok, 0)) AS d_in,
              SUM(IF(DATE(ts) = CURRENT_DATE(), output_tok, 0)) AS d_out,
              SUM(IF(DATE(ts) = CURRENT_DATE(), cache_creation_tok, 0)) AS d_cw,
              SUM(IF(DATE(ts) = CURRENT_DATE(), cache_read_tok, 0)) AS d_cr,
              SUM(input_tok) AS m_in,
              SUM(output_tok) AS m_out,
              SUM(cache_creation_tok) AS m_cw,
              SUM(cache_read_tok) AS m_cr
            FROM `{project}.{dataset}.llm_call_log`
            WHERE
              ts >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH)
              AND ok
              AND provider != 'claude-code'
              AND (agent IS NULL OR agent NOT LIKE 'cc_rail:%')
            GROUP BY model
        """
        daily = 0.0
        monthly = 0.0
        for row in client.query(sql, timeout=30).result():
            model = str(row["model"] or "")
            daily += _price_llm_tokens(
                model,
                float(row["d_in"] or 0), float(row["d_out"] or 0),
                float(row["d_cw"] or 0), float(row["d_cr"] or 0),
            )
            monthly += _price_llm_tokens(
                model,
                float(row["m_in"] or 0), float(row["m_out"] or 0),
                float(row["m_cw"] or 0), float(row["m_cr"] or 0),
            )
        return daily, monthly
    except Exception as exc:
        _record_degradation(exc)
        return 0.0, 0.0


__all__ = [
    "fetch_llm_spend",
    "fetch_spend",
    "reset_spend_guard_status",
    "spend_guard_status",
]
