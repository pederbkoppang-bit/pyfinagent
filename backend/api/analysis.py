"""
Analysis API routes — start analysis, check status, get results.

Supports two execution modes:
- Celery (USE_CELERY=true): dispatches to a Celery worker via Redis.
- Sync  (USE_CELERY=false): runs in a background asyncio task in-process.
"""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.api.models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    AnalysisStatusResponse,
    SynthesisReport,
)
from backend.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# ── In-memory task store (used when USE_CELERY=false) ────────────
_tasks: dict[str, dict[str, Any]] = {}


async def _run_sync_analysis(task_id: str, ticker: str, settings: Settings):
    """Run the full analysis pipeline in-process and update _tasks."""
    from backend.agents.orchestrator import AnalysisOrchestrator
    from backend.db.bigquery_client import BigQueryClient
    from backend.tools.slack import send_notification

    steps_completed: list[str] = []

    def on_step(step_name: str, status: str, message: str = ""):
        if status == "completed":
            steps_completed.append(step_name)
        _tasks[task_id].update(
            status=AnalysisStatus.RUNNING,
            current_step=step_name,
            step_status=status,
            message=message,
            steps_completed=list(steps_completed),
        )

    try:
        orchestrator = AnalysisOrchestrator(settings)
        report = await orchestrator.run_full_analysis(ticker, on_step=on_step)

        synthesis = report.get("final_synthesis", {})
        quant = report.get("quant", {})

        # Save to BigQuery (best-effort)
        try:
            bq = BigQueryClient(settings)
            bq.save_report(
                ticker=ticker,
                company_name=quant.get("company_name", "N/A"),
                final_score=synthesis.get("final_weighted_score", 0),
                recommendation=synthesis.get("recommendation", {}).get("action", "N/A"),
                summary=synthesis.get("final_summary", ""),
                full_report=report,
            )
        except Exception as e:
            logger.error(f"Failed to save report to BigQuery: {e}")

        # Slack notification (best-effort)
        try:
            if settings.slack_webhook_url:
                await send_notification(
                    settings.slack_webhook_url,
                    f"Analysis Complete: {ticker}",
                    {
                        "Score": f"{synthesis.get('final_weighted_score', 'N/A')}/10",
                        "Verdict": synthesis.get("recommendation", {}).get("action", "N/A"),
                    },
                    "success",
                )
        except Exception:
            pass

        _tasks[task_id].update(
            status=AnalysisStatus.COMPLETED,
            steps_completed=list(steps_completed),
            result={"ticker": ticker, "final_synthesis": synthesis, "steps_completed": steps_completed},
        )

    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}", exc_info=True)
        _tasks[task_id].update(
            status=AnalysisStatus.FAILED,
            error=str(e),
        )


# ── Endpoints ────────────────────────────────────────────────────

@router.post("/", response_model=AnalysisResponse)
async def start_analysis(req: AnalysisRequest, settings: Settings = Depends(get_settings)):
    """Start a new analysis for the given ticker. Returns a task ID for polling."""
    ticker = req.ticker.upper()

    if settings.use_celery:
        from backend.tasks.analysis import run_analysis_task
        task = run_analysis_task.delay(ticker)
        task_id = task.id
    else:
        task_id = str(uuid.uuid4())
        _tasks[task_id] = {
            "ticker": ticker,
            "status": AnalysisStatus.PENDING,
            "current_step": "Queued",
            "steps_completed": [],
        }
        # Launch as a fire-and-forget asyncio task
        asyncio.create_task(_run_sync_analysis(task_id, ticker, settings))

    logger.info(f"Analysis started for {ticker}, task_id={task_id}")
    return AnalysisResponse(analysis_id=task_id, ticker=ticker, status=AnalysisStatus.PENDING)


@router.get("/{analysis_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: str, settings: Settings = Depends(get_settings)):
    """Poll for analysis status and progress. Returns final report when complete."""

    if settings.use_celery:
        return _poll_celery(analysis_id)

    # ── Sync / in-memory mode ──
    task = _tasks.get(analysis_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    status = task["status"]
    report_model = None

    if status == AnalysisStatus.COMPLETED:
        synthesis = task.get("result", {}).get("final_synthesis", {})
        try:
            report_model = SynthesisReport(**synthesis)
        except Exception:
            logger.warning("Could not parse synthesis into SynthesisReport model")

    return AnalysisStatusResponse(
        analysis_id=analysis_id,
        ticker=task.get("ticker", ""),
        status=status,
        current_step=task.get("current_step", ""),
        steps_completed=task.get("steps_completed", []),
        report=report_model,
        error=task.get("error"),
    )


def _poll_celery(analysis_id: str) -> AnalysisStatusResponse:
    """Poll Celery AsyncResult for task status."""
    from celery.result import AsyncResult

    result = AsyncResult(analysis_id)

    if result.state == "PENDING":
        return AnalysisStatusResponse(
            analysis_id=analysis_id, ticker="", status=AnalysisStatus.PENDING,
            current_step="Queued", steps_completed=[],
        )
    if result.state == "PROGRESS":
        meta = result.info or {}
        return AnalysisStatusResponse(
            analysis_id=analysis_id, ticker="", status=AnalysisStatus.RUNNING,
            current_step=meta.get("current_step", ""),
            steps_completed=meta.get("steps_completed", []),
        )
    if result.state == "SUCCESS":
        data = result.result or {}
        synthesis = data.get("final_synthesis", {})
        report = None
        try:
            report = SynthesisReport(**synthesis)
        except Exception:
            logger.warning("Could not parse synthesis into SynthesisReport model")
        return AnalysisStatusResponse(
            analysis_id=analysis_id, ticker=data.get("ticker", ""),
            status=AnalysisStatus.COMPLETED,
            steps_completed=data.get("steps_completed", []),
            report=report,
        )
    # FAILURE or REVOKED
    return AnalysisStatusResponse(
        analysis_id=analysis_id, ticker="", status=AnalysisStatus.FAILED,
        error=str(result.info) if result.info else "Unknown error",
    )
