"""
Slack Bot entry point — Bolt async app with Socket Mode.

Runs as a standalone service: python -m backend.slack_bot.app
Uses Socket Mode (outbound WebSocket) — no public URL needed.
Starts ticket queue processor and SLA monitor as background tasks.
"""

import asyncio
import logging

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from backend.config.settings import get_settings
from backend.slack_bot.commands import register_commands
from backend.slack_bot.scheduler import start_scheduler
from backend.slack_bot.assistant_lifecycle import register_assistant_lifecycle
from backend.services.ticket_queue_processor import start_queue_processor
from backend.services.sla_monitor import start_sla_monitoring
from backend.services.stuck_task_reaper import start_stuck_task_reaper

logger = logging.getLogger(__name__)


def create_app() -> AsyncApp:
    settings = get_settings()
    app = AsyncApp(token=settings.slack_bot_token)
    
    # Register event handlers
    register_commands(app)
    register_assistant_lifecycle(app)  # ← NEW: Slack AI Agent lifecycle
    
    return app


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    if not settings.slack_bot_token or not settings.slack_app_token:
        logger.error("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required. Set them in .env")
        return

    app = create_app()

    # Start scheduled jobs (morning digest, proactive alerts)
    start_scheduler(app)

    # Start ticket queue processor in background (processes tickets every 30s to avoid rate limits)
    # Anthropic API is heavily rate limited; we need significant spacing between requests
    asyncio.create_task(start_queue_processor(batch_interval=30.0))
    logger.info("🎫 Ticket queue processor started as background task")

    # Start SLA monitoring in background (checks every 5 minutes)
    asyncio.create_task(start_sla_monitoring(check_interval=300))
    logger.info("🔍 SLA monitor started as background task")
    
    # Start stuck-task reaper (checks every 60s for >15min hung tickets)
    asyncio.create_task(start_stuck_task_reaper(check_interval=60))
    logger.info("🔪 Stuck-Task Reaper started as background task")

    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    logger.info("Slack bot starting in Socket Mode...")
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
