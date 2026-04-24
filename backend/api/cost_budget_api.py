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
from backend.slack_bot.jobs.cost_budget_watcher import _default_fetch_spend

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


def _fetch_llm_tokens_today() -> tuple[Optional[int], Optional[int]]:
    """Return (tokens_today, calls_today) from pyfinagent_data.llm_call_log.

    Fail-open to (None, None) -- the column rolls up into CostBudgetToday
    optionally so the tile stays truthful when the log is empty or BQ is
    unreachable.
    """
    try:
        from google.cloud import bigquery
        project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
        client = bigquery.Client(project=project)
        sql = f"""
          SELECT
            COALESCE(SUM(input_tokens) + SUM(output_tokens), 0) AS tokens,
            COUNT(*) AS calls
          FROM `{project}.pyfinagent_data.llm_call_log`
          WHERE DATE(ts) = CURRENT_DATE()
        """
        rows = list(client.query(sql, timeout=30).result())
        if not rows:
            return None, None
        r = rows[0]
        return int(r["tokens"] or 0), int(r["calls"] or 0)
    except Exception as exc:
        logger.warning("cost_budget_api: llm_tokens fetch fail-open: %r", exc)
        return None, None


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

    tokens, calls = await asyncio.to_thread(_fetch_llm_tokens_today)
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
