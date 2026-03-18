"""
Slack Bot entry point — Bolt async app with Socket Mode.

Runs as a standalone service: python -m backend.slack_bot.app
Uses Socket Mode (outbound WebSocket) — no public URL needed.
"""

import asyncio
import logging

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from backend.config.settings import get_settings
from backend.slack_bot.commands import register_commands
from backend.slack_bot.scheduler import start_scheduler

logger = logging.getLogger(__name__)


def create_app() -> AsyncApp:
    settings = get_settings()
    app = AsyncApp(token=settings.slack_bot_token)
    register_commands(app)
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

    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    logger.info("Slack bot starting in Socket Mode...")
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
