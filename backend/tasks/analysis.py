"""
Celery app configuration and async analysis task.

Settings are loaded lazily so the module can be imported without requiring
all environment variables to be present (e.g. during FastAPI startup checks).
"""

import asyncio
import json
import logging
import os

from celery import Celery

logger = logging.getLogger(__name__)

# Create Celery app with env-based defaults; the worker will pick up real
# values from the environment at runtime.
_broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery("pyfinagent", broker=_broker, backend=_backend)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="run_analysis")
def run_analysis_task(self, ticker: str):
    """
    Celery task that runs the full analysis pipeline.
    Updates task state with step-by-step progress.
    """
    from backend.agents.orchestrator import AnalysisOrchestrator
    from backend.config.settings import get_settings
    from backend.db.bigquery_client import BigQueryClient
    from backend.tools.slack import send_notification

    settings = get_settings()
    steps_completed = []

    def on_step(step_name, status, message):
        if status == "completed":
            steps_completed.append(step_name)
        self.update_state(
            state="PROGRESS",
            meta={
                "current_step": step_name,
                "step_status": status,
                "message": message,
                "steps_completed": steps_completed,
            },
        )

    try:
        orchestrator = AnalysisOrchestrator(settings)

        # Run the async orchestrator in a sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            report = loop.run_until_complete(orchestrator.run_full_analysis(ticker, on_step=on_step))
        finally:
            loop.close()

        # Save to BigQuery
        synthesis = report.get("final_synthesis", {})
        quant = report.get("quant", {})
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

        # Slack notification
        try:
            loop2 = asyncio.new_event_loop()
            loop2.run_until_complete(send_notification(
                settings.slack_webhook_url,
                f"✅ Analysis Complete: {ticker}",
                {
                    "Score": f"{synthesis.get('final_weighted_score', 'N/A')}/10",
                    "Verdict": synthesis.get("recommendation", {}).get("action", "N/A"),
                },
                "success",
            ))
            loop2.close()
        except Exception:
            pass

        return {
            "ticker": ticker,
            "status": "completed",
            "final_synthesis": synthesis,
            "steps_completed": steps_completed,
        }

    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}", exc_info=True)
        # Try to notify on failure
        try:
            loop3 = asyncio.new_event_loop()
            loop3.run_until_complete(send_notification(
                settings.slack_webhook_url,
                f"❌ Analysis Failed: {ticker}",
                {"Error": str(e)[:200]},
                "error",
            ))
            loop3.close()
        except Exception:
            pass
        raise
