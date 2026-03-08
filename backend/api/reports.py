"""
Reports API routes — list past reports, get single report, performance stats.
"""

import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException
from google.api_core.exceptions import GoogleAPIError

from backend.api.models import PerformanceStats, ReportSummary
from backend.config.settings import Settings, get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.services.outcome_tracker import OutcomeTracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reports", tags=["reports"])


def _get_bq(settings: Settings = Depends(get_settings)) -> BigQueryClient:
    return BigQueryClient(settings)


@router.get("/", response_model=list[ReportSummary])
async def list_reports(limit: int = 20, bq: BigQueryClient = Depends(_get_bq)):
    """List recent analysis reports."""
    try:
        rows = bq.get_recent_reports(limit=limit)
        return [ReportSummary(**r) for r in rows]
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
        return tracker.get_performance_summary()
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
        results = tracker.evaluate_all_pending()
        return {"evaluated": len(results), "outcomes": results}
    except Exception as exc:
        logger.error("Error evaluating outcomes: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate outcomes: {type(exc).__name__}: {exc}",
        )


@router.get("/{ticker}")
async def get_report(ticker: str, bq: BigQueryClient = Depends(_get_bq)):
    """Get the latest report for a specific ticker."""
    try:
        report = bq.get_report(ticker.upper())
        if not report:
            raise HTTPException(status_code=404, detail=f"No report found for {ticker}")
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
