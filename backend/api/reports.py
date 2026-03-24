"""
Reports API routes — list past reports, get single report, performance stats.
"""

import logging
import traceback
import asyncio

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from google.api_core.exceptions import GoogleAPIError

from backend.api.models import PerformanceStats, ReportSummary
from backend.config.settings import Settings, get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.services.api_cache import ENDPOINT_TTLS, get_api_cache
from backend.services.outcome_tracker import OutcomeTracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reports", tags=["reports"])


def _get_bq(settings: Settings = Depends(get_settings)) -> BigQueryClient:
    return BigQueryClient(settings)


@router.get("/", response_model=list[ReportSummary])
async def list_reports(limit: int = 20, bq: BigQueryClient = Depends(_get_bq)):
    """List recent analysis reports."""
    try:
        cache = get_api_cache()
        cache_key = f"reports:list:{limit}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        rows = await asyncio.to_thread(bq.get_recent_reports, limit=limit)
        result = [ReportSummary(**r) for r in rows]
        cache.set(cache_key, result, ENDPOINT_TTLS["reports:list"])
        return result
    except GoogleAPIError as exc:
        logger.error("BigQuery error listing reports: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"BigQuery error: {exc}",
        )
    except Exception as exc:
        logger.error("Unexpected error listing reports: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list reports: {type(exc).__name__}: {exc}",
        )


@router.get("/performance", response_model=PerformanceStats)
async def get_performance(settings: Settings = Depends(get_settings)):
    """Get aggregated recommendation performance statistics."""
    try:
        tracker = OutcomeTracker(settings)
        return await asyncio.to_thread(tracker.get_performance_summary)
    except GoogleAPIError as exc:
        logger.error("BigQuery error fetching performance: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"BigQuery error: {exc}")
    except Exception as exc:
        logger.error("Unexpected error fetching performance: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch performance: {type(exc).__name__}: {exc}",
        )


@router.post("/evaluate")
async def evaluate_outcomes(settings: Settings = Depends(get_settings)):
    """Trigger evaluation of all pending recommendation outcomes."""
    try:
        tracker = OutcomeTracker(settings)
        results = await asyncio.to_thread(tracker.evaluate_all_pending)
        return {"evaluated": len(results), "outcomes": results}
    except Exception as exc:
        logger.error("Error evaluating outcomes: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate outcomes: {type(exc).__name__}: {exc}",
        )


@router.get("/cost-history")
async def get_cost_history(limit: int = 50, bq: BigQueryClient = Depends(_get_bq)):
    """Get cost/token usage history for past analyses."""
    try:
        cache = get_api_cache()
        cache_key = f"reports:cost-history:{limit}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        result = await asyncio.to_thread(bq.get_cost_history, limit=limit)
        cache.set(cache_key, result, ENDPOINT_TTLS["reports:cost-history"])
        return result
    except GoogleAPIError as exc:
        logger.error("BigQuery error fetching cost history: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"BigQuery error: {exc}")
    except Exception as exc:
        logger.error("Error fetching cost history: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch cost history: {type(exc).__name__}: {exc}",
        )


@router.get("/latest-cost-summary")
async def get_latest_cost_summary(bq: BigQueryClient = Depends(_get_bq)):
    """Get the per-agent cost breakdown from the most recent analysis.

    This extracts the cost_summary object from full_report_json of the latest
    report, providing real token counts per agent for the cost estimator.
    """
    try:
        cache = get_api_cache()
        cache_key = "reports:cost-summary"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        # Single BQ query instead of previous two-call pattern
        row = await asyncio.to_thread(bq.get_latest_report_json)
        if not row:
            return {"agents": [], "total_cost_usd": 0, "total_tokens": 0}
        full_json = row.get("full_report_json")
        if isinstance(full_json, dict):
            cost_summary = full_json.get("cost_summary") or full_json.get("final_synthesis", {}).get("cost_summary")
            if cost_summary:
                cost_summary["ticker"] = row["ticker"]
                cost_summary["analysis_date"] = row.get("analysis_date", "")
                cache.set(cache_key, cost_summary, ENDPOINT_TTLS["reports:cost-summary"])
                return cost_summary
        result = {"agents": [], "total_cost_usd": 0, "total_tokens": 0, "ticker": row["ticker"]}
        cache.set(cache_key, result, ENDPOINT_TTLS["reports:cost-summary"])
        return result
    except GoogleAPIError as exc:
        logger.error("BigQuery error fetching latest cost summary: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"BigQuery error: {exc}")
    except Exception as exc:
        logger.error("Error fetching latest cost summary: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch latest cost summary: {type(exc).__name__}: {exc}",
        )


@router.get("/{ticker}")
async def get_report(
    ticker: str,
    analysis_date: Optional[str] = Query(None, description="Fetch a specific report by analysis_date"),
    bq: BigQueryClient = Depends(_get_bq),
):
    """Get a report for a specific ticker, optionally by analysis_date."""
    try:
        cache = get_api_cache()
        cache_key = f"reports:ticker:{ticker.upper()}:{analysis_date or 'latest'}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        report = await asyncio.to_thread(bq.get_report, ticker.upper(), analysis_date=analysis_date)
        if not report:
            raise HTTPException(status_code=404, detail=f"No report found for {ticker}")
        cache.set(cache_key, report, ENDPOINT_TTLS["reports:ticker"])
        return report
    except HTTPException:
        raise
    except GoogleAPIError as exc:
        logger.error("BigQuery error fetching report for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"BigQuery error: {exc}")
    except Exception as exc:
        logger.error("Unexpected error fetching report for %s: %s", ticker, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch report: {type(exc).__name__}: {exc}",
        )
