"""
Scheduled jobs: morning digest, evening digest, and watchdog health check.
Uses APScheduler to run tasks within the Slack bot process.
"""

import logging
from datetime import datetime

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from slack_bolt.async_app import AsyncApp

from backend.config.settings import get_settings
from backend.slack_bot.formatters import format_morning_digest, format_evening_digest

logger = logging.getLogger(__name__)

_BACKEND_URL = "http://backend:8000"
_scheduler: AsyncIOScheduler | None = None


def start_scheduler(app: AsyncApp):
    """Start the APScheduler with daily digests and watchdog jobs."""
    global _scheduler
    settings = get_settings()

    if not settings.slack_channel_id:
        logger.warning("SLACK_CHANNEL_ID not set -- scheduled jobs disabled")
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

    # Evening digest — daily at configured hour
    _scheduler.add_job(
        _send_evening_digest,
        "cron",
        hour=settings.evening_digest_hour,
        minute=0,
        args=[app],
        id="evening_digest",
        replace_existing=True,
    )

    # Watchdog health check — interval-based, alerts on failure only
    _scheduler.add_job(
        _watchdog_health_check,
        "interval",
        minutes=settings.watchdog_interval_minutes,
        args=[app],
        id="watchdog_health_check",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "Scheduler started: morning digest at %d:00, evening digest at %d:00, "
        "watchdog every %d min",
        settings.morning_digest_hour,
        settings.evening_digest_hour,
        settings.watchdog_interval_minutes,
    )


async def _send_morning_digest(app: AsyncApp):
    """Fetch portfolio performance and post morning digest."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            portfolio_res = await client.get(f"{_BACKEND_URL}/api/portfolio/performance")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            reports_res = await client.get(f"{_BACKEND_URL}/api/reports/?limit=5")
            reports_data = reports_res.json() if reports_res.status_code == 200 else []

        blocks = format_morning_digest(portfolio_data, reports_data)

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"PyFinAgent Morning Digest -- {datetime.now().strftime('%B %d, %Y')}",
        )
        logger.info("Morning digest sent")

    except Exception:
        logger.exception("Failed to send morning digest")


async def _send_evening_digest(app: AsyncApp):
    """Fetch end-of-day portfolio summary and post evening digest."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            portfolio_res = await client.get(f"{_BACKEND_URL}/api/portfolio/performance")
            portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

            trades_res = await client.get(f"{_BACKEND_URL}/api/paper-trading/trades?limit=10")
            trades_data = trades_res.json() if trades_res.status_code == 200 else []

        blocks = format_evening_digest(portfolio_data, trades_data)

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=blocks,
            text=f"PyFinAgent Evening Digest -- {datetime.now().strftime('%B %d, %Y')}",
        )
        logger.info("Evening digest sent")

    except Exception:
        logger.exception("Failed to send evening digest")


async def _watchdog_health_check(app: AsyncApp):
    """Probe backend health endpoint; post to Slack only on failure."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{_BACKEND_URL}/api/health")
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                logger.debug("Watchdog health check passed")
                return

        await app.client.chat_postMessage(
            channel=settings.slack_channel_id,
            blocks=[{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        ":warning: *Watchdog Alert* -- Backend health check failed\n"
                        f"Status: {resp.status_code} at {datetime.now().strftime('%H:%M:%S')}"
                    ),
                },
            }],
            text="Watchdog Alert: backend health check failed",
        )
        logger.warning("Watchdog health check failed -- status %d", resp.status_code)

    except Exception:
        try:
            await app.client.chat_postMessage(
                channel=settings.slack_channel_id,
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            ":rotating_light: *Watchdog Alert* -- Backend unreachable\n"
                            f"Time: {datetime.now().strftime('%H:%M:%S')}"
                        ),
                    },
                }],
                text="Watchdog Alert: backend unreachable",
            )
        except Exception:
            pass
        logger.exception("Watchdog health check -- backend unreachable")


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
            text=f"Analysis complete: {ticker} -- {action} ({score:.1f}/10)",
            attachments=[{"color": color, "blocks": []}],
        )
    except Exception:
        logger.exception(f"Failed to send alert for {ticker}")
