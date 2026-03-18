"""
Scheduled jobs: morning digest + proactive anomaly alerts.
Uses APScheduler to run tasks within the Slack bot process.
"""

import logging
from datetime import datetime

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from slack_bolt.async_app import AsyncApp

from backend.config.settings import get_settings
from backend.slack_bot.formatters import format_morning_digest

logger = logging.getLogger(__name__)

_BACKEND_URL = "http://backend:8000"
_scheduler: AsyncIOScheduler | None = None


def start_scheduler(app: AsyncApp):
    """Start the APScheduler with daily digest and alert jobs."""
    global _scheduler
    settings = get_settings()

    if not settings.slack_channel_id:
        logger.warning("SLACK_CHANNEL_ID not set — scheduled jobs disabled")
        return

    _scheduler = AsyncIOScheduler()

    # Morning digest — daily at configured hour
    _scheduler.add_job(
        _send_morning_digest,
        "cron",
        hour=settings.morning_digest_hour,
        minute=0,
        args=[app],
        id="morning_digest",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(f"Scheduler started: morning digest at {settings.morning_digest_hour}:00")


async def _send_morning_digest(app: AsyncApp):
    """Fetch portfolio performance and post morning digest."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Portfolio performance
            portfolio_res = await client.get(f"{_BACKEND_URL}/api/portfolio/performance")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            # Recent reports
            reports_res = await client.get(f"{_BACKEND_URL}/api/reports/?limit=5")
            reports_data = reports_res.json() if reports_res.status_code == 200 else []

        blocks = format_morning_digest(portfolio_data, reports_data)

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"PyFinAgent Morning Digest — {datetime.now().strftime('%B %d, %Y')}",
        )
        logger.info("Morning digest sent")

    except Exception:
        logger.exception("Failed to send morning digest")


async def send_analysis_alert(app: AsyncApp, ticker: str, report: dict):
    """Post a proactive alert after analysis completes (called from orchestrator)."""
    settings = get_settings()
    if not settings.slack_channel_id:
        return

    try:
        score = report.get("final_weighted_score", 0)
        rec = report.get("recommendation", {})
        action = rec.get("action", "N/A") if isinstance(rec, dict) else str(rec)

        emoji = ":chart_with_upwards_trend:" if score >= 7 else ":chart_with_downwards_trend:" if score < 4 else ":bar_chart:"
        color = "#22c55e" if score >= 7 else "#ef4444" if score < 4 else "#f59e0b"

        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"{emoji} *Analysis Complete: {ticker}*"}},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Score:* {score:.1f}/10"},
                    {"type": "mrkdwn", "text": f"*Recommendation:* {action}"},
                ],
            },
        ]

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"Analysis complete: {ticker} — {action} ({score:.1f}/10)",
            attachments=[{"color": color, "blocks": []}],
        )
    except Exception:
        logger.exception(f"Failed to send alert for {ticker}")
