"""phase-15.1 Cost-budget watcher endpoint.

Exposes today's daily + month-to-date BigQuery spend (from
`region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT * $6.25/TiB`) alongside the
$5/day and $50/month caps that the phase-9.9.2 cost-budget watcher enforces.
The harness-tab tile renders this for at-a-glance visibility.

Reuses `backend.slack_bot.jobs.cost_budget_watcher._default_fetch_spend` --
the canonical BQ fetcher. Do not duplicate the SQL here.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.api_cache import get_api_cache
# phase-75.5 (arch-04): public observability home, not the private slack_bot symbol.
from backend.services.observability import fetch_spend as _default_fetch_spend

logger = logging.getLogger(__name__)


def structured_log(endpoint: str, duration_ms: float, status: str, **extra) -> None:
    """phase-15.10: emit one structured JSON log line per endpoint call.

    Fields are a stable contract for the observability tile; extras flow
    through as-is for per-endpoint enrichment (e.g. tripped, reason).
    """
    try:
        logger.info(
            _json.dumps(
                {
                    "endpoint": endpoint,
                    "duration_ms": round(duration_ms, 1),
                    "status": status,
                    "ts": time.time(),
                    **extra,
                }
            )
        )
    except Exception as exc:
        logger.warning("structured_log fail-open: %r", exc)

router = APIRouter(prefix="/api/cost-budget", tags=["cost-budget"])

_DAILY_CAP_USD = 5.0
_MONTHLY_CAP_USD = 50.0
_CACHE_KEY = "cost_budget:today"
_CACHE_TTL = 60.0


class CostBudgetToday(BaseModel):
    daily_usd: float
    monthly_usd: float
    daily_cap: float
    monthly_cap: float
    tripped: bool
    reason: Optional[str] = None
    # phase-15.10 cost-per-call rollup (optional; best-effort from BQ).
    llm_tokens_today: Optional[int] = None
    cost_per_llm_call_usd: Optional[float] = None
    # phase-82.54: the COMPONENTS behind llm_tokens_today, actually exposed.
    #
    # An earlier revision of this step claimed the breakdown was exposed while
    # the endpoint still returned one conflated number -- caught by the 82.54
    # Q/A. That claim was load-bearing, not decorative: the stated reason for
    # summing all four token columns is that a single conflated number is
    # exactly what let a 26x undercount hide (measured 2026-08-05: input+output
    # 353,896 vs all four 9,159,745). Shipping the same conflation while saying
    # otherwise would have reproduced the defect and the excuse together.
    llm_input_tokens_today: Optional[int] = None
    llm_output_tokens_today: Optional[int] = None
    llm_cache_creation_tokens_today: Optional[int] = None
    llm_cache_read_tokens_today: Optional[int] = None


#: phase-82.54. A PLAIN string literal, deliberately -- NOT an f-string.
#:
#: The pre-82.54 query interpolated the project id, and
#: `schema_oracle.extract_sql_literals` reassembles an f-string from its CONSTANT
#: parts only: `FROM \`{project}.pyfinagent_data.llm_call_log\`` collapses to
#: `FROM \`.pyfinagent_data.llm_call_log\``, whose empty project group cannot
#: match `_FQ_TABLE_RE`. So this file resolved ZERO SQL literals and the
#: unknown-column sweep was structurally blind to it -- which is exactly how a
#: live phantom-column defect survived here while 82.39's twin was caught.
#:
#: THE COLUMN NAMES WERE WRONG: it selected `input_tokens` / `output_tokens`,
#: which do not exist. BigQuery answered "Unrecognized name: input_tokens; Did
#: you mean input_tok?" on every call, and the fail-open `except` below returned
#: (None, None).
#:
#: AND THE FIX IS NOT A RENAME. `llm_call_log` carries FOUR token columns, and
#: the cache pair dominates: measured 2026-08-05, input+output = 353,896 while
#: all four = 9,159,745, a 25.9x difference. Renaming the two would have shipped
#: a number that under-reports by an order of magnitude and looks plausible.
#: This is a TOKEN count, not a billed-cost figure -- cost weighting (cache reads
#: are ~10x cheaper) lives in `observability/spend.py::fetch_llm_spend`, which is
#: correct and untouched. The components are exposed so a future consumer is not
#: forced to guess which definition a single number used.
#:
#: `ts` is TIMESTAMP and the table is day-partitioned on it, so
#: `DATE(ts) = CURRENT_DATE()` prunes to 0 bytes. Unlike `paper_trades.created_at`
#: (82.39) and `historical_fundamentals.report_date` (82.21), there is no
#: STRING-date trap here -- do not "fix" this predicate.
LLM_TOKENS_TODAY_SQL = """
          SELECT
            COALESCE(SUM(input_tok), 0) AS input_tokens,
            COALESCE(SUM(output_tok), 0) AS output_tokens,
            COALESCE(SUM(cache_creation_tok), 0) AS cache_creation_tokens,
            COALESCE(SUM(cache_read_tok), 0) AS cache_read_tokens,
            COALESCE(SUM(input_tok) + SUM(output_tok)
                     + SUM(cache_creation_tok) + SUM(cache_read_tok), 0) AS tokens,
            COUNT(*) AS calls
          FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log`
          WHERE DATE(ts) = CURRENT_DATE()
        """


def _alert_llm_tokens_failed(exc: Exception) -> None:
    """phase-82.54: make the swallowed BQ failure operator-visible.

    P1, never P2: only `_CRITICAL_SEVERITIES` reach `_bot_token_fallback` while
    `slack_webhook_url` is empty, so a P2 would be logged and dropped -- the same
    invisibility this step removes. Function-local import, so a test MUST patch
    `backend.services.observability.alerting.raise_cron_alert_sync`.
    """
    try:
        from backend.services.observability.alerting import raise_cron_alert_sync

        raise_cron_alert_sync(
            source="cost_budget_api",
            error_type="llm_tokens_fetch_failed",
            severity="P1",
            title="cost_budget_api: llm_call_log token fetch failed",
            details={
                "error": f"{type(exc).__name__}: {exc}"[:600],
                "consequence": "llm_tokens_today reports null rather than a count",
            },
        )
    except Exception as alert_exc:  # noqa: BLE001 -- never break the endpoint
        logger.warning("cost_budget_api: alert dispatch fail-open: %r", alert_exc)


def _fetch_llm_tokens_today() -> tuple[Optional[int], Optional[int], Optional[dict]]:
    """Return (tokens_today, calls_today) from pyfinagent_data.llm_call_log.

    Fail-open to (None, None) -- the column rolls up into CostBudgetToday
    optionally so the tile stays truthful when the log is empty or BQ is
    unreachable.
    """
    try:
        from google.cloud import bigquery

        client = bigquery.Client(
            project=os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
        )
        rows = list(client.query(LLM_TOKENS_TODAY_SQL, timeout=30).result())
        # NOTE: this is an aggregate with no GROUP BY, so it ALWAYS returns
        # exactly one row and COALESCE makes it non-NULL -- `if not rows` here
        # would be dead code, and "the total is non-null" is not evidence the
        # query works. Measured: a day with ZERO calls returns tokens=0/calls=0.
        r = rows[0]
        return (
            int(r["tokens"] or 0),
            int(r["calls"] or 0),
            {
                "input": int(r["input_tokens"] or 0),
                "output": int(r["output_tokens"] or 0),
                "cache_creation": int(r["cache_creation_tokens"] or 0),
                "cache_read": int(r["cache_read_tokens"] or 0),
            },
        )
    except Exception as exc:
        logger.warning("cost_budget_api: llm_tokens fetch fail-open: %r", exc)
        _alert_llm_tokens_failed(exc)
        return None, None, None


@router.get("/status", response_model=CostBudgetToday)
async def get_cost_budget_status() -> CostBudgetToday:
    """phase-16.22 alias: masterplan verification command hits
    /api/cost-budget/status; the canonical implementation is /today.
    Same response model + same fetch path.
    """
    return await get_cost_budget_today()


@router.get("/today", response_model=CostBudgetToday)
async def get_cost_budget_today() -> CostBudgetToday:
    """Return today's + month-to-date BQ spend vs the $5/$50 caps.

    Fail-open to zeros if the BQ query fails (permission, network, quota).
    Cached 60s to avoid re-scanning INFORMATION_SCHEMA.JOBS on every render.
    """
    start = time.perf_counter()
    cache = get_api_cache()
    cached = cache.get(_CACHE_KEY)
    if cached is not None:
        structured_log(
            "/api/cost-budget/today",
            (time.perf_counter() - start) * 1000,
            "cache_hit",
        )
        return cached

    try:
        daily_usd, monthly_usd = await asyncio.to_thread(_default_fetch_spend)
    except Exception as exc:
        logger.warning("cost_budget_api: fetch fail-open: %r", exc)
        daily_usd, monthly_usd = 0.0, 0.0

    daily = float(daily_usd or 0.0)
    monthly = float(monthly_usd or 0.0)
    tripped = daily >= _DAILY_CAP_USD or monthly >= _MONTHLY_CAP_USD
    if daily >= _DAILY_CAP_USD:
        reason: Optional[str] = "daily"
    elif monthly >= _MONTHLY_CAP_USD:
        reason = "monthly"
    else:
        reason = None

    tokens, calls, parts = await asyncio.to_thread(_fetch_llm_tokens_today)
    cost_per_call = (
        round(daily / calls, 6) if (calls and calls > 0 and daily > 0) else None
    )

    result = CostBudgetToday(
        daily_usd=round(daily, 4),
        monthly_usd=round(monthly, 4),
        daily_cap=_DAILY_CAP_USD,
        monthly_cap=_MONTHLY_CAP_USD,
        tripped=tripped,
        reason=reason,
        llm_tokens_today=tokens,
        llm_input_tokens_today=(parts or {}).get("input"),
        llm_output_tokens_today=(parts or {}).get("output"),
        llm_cache_creation_tokens_today=(parts or {}).get("cache_creation"),
        llm_cache_read_tokens_today=(parts or {}).get("cache_read"),
        cost_per_llm_call_usd=cost_per_call,
    )
    cache.set(_CACHE_KEY, result, _CACHE_TTL)

    structured_log(
        "/api/cost-budget/today",
        (time.perf_counter() - start) * 1000,
        "tripped" if tripped else "ok",
        daily_usd=daily,
        monthly_usd=monthly,
        tokens=tokens,
    )
    return result


__all__ = ["router", "CostBudgetToday"]
