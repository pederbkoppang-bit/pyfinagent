"""
Slack Bot entry point — SAFE startup with fallback mode.

SAFETY DESIGN:
  If slack_bolt/slack_sdk are new enough → Full AI Agent mode
    (Assistant panel, streaming, task plan, feedback, governance)
  If packages are old → Fallback basic mode
    (Channel messages still work, deploy commands still work, nothing breaks)

This ensures the bot ALWAYS starts, even if packages are outdated.
The Mac Mini lifeline is NEVER broken by a bad deploy.

Requires minimum: slack_bolt >= 1.18.0 (basic), >= 1.22.0 (full AI features)
"""

import logging
import sys
import time

from slack_bolt.app import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Safe feature detection ───────────────────────────────────────

ASSISTANT_AVAILABLE = False
STREAMING_AVAILABLE = False

try:
    from slack_bolt.assistant import Assistant
    ASSISTANT_AVAILABLE = True
except ImportError:
    logger.warning(
        "⚠️ slack_bolt.assistant not available — "
        "upgrade with: pip install 'slack_bolt>=1.22.0'"
    )

try:
    from slack_sdk.models.messages.chunk import MarkdownTextChunk
    STREAMING_AVAILABLE = True
except ImportError:
    logger.warning(
        "⚠️ slack_sdk streaming chunks not available — "
        "upgrade with: pip install 'slack_sdk>=3.35.0'"
    )

# Reconnection settings
MAX_RECONNECT_ATTEMPTS = 50
INITIAL_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 60


def create_app() -> App:
    """
    Create the Bolt app with the best available features.

    Full mode (packages up to date):
      Assistant side-panel + streaming + task plan + governance

    Fallback mode (old packages):
      Channel messages + deploy commands + direct responses
      Everything still works, just without the fancy AI Agent UI
    """
    settings = get_settings()
    app = App(token=settings.slack_bot_token)

    # ── Mode 1: Full AI Agent (if packages support it) ──────────
    if ASSISTANT_AVAILABLE and STREAMING_AVAILABLE:
        try:
            from backend.slack_bot.assistant_handler import (
                handle_thread_started,
                handle_context_changed,
                handle_user_message,
                handle_feedback_positive,
                handle_feedback_negative,
            )

            assistant = Assistant()

            @assistant.thread_started
            def on_thread_started(say, set_suggested_prompts, get_thread_context, logger):
                handle_thread_started(say=say, set_suggested_prompts=set_suggested_prompts,
                                      get_thread_context=get_thread_context, logger=logger)

            @assistant.thread_context_changed
            def on_context_changed(get_thread_context, logger):
                handle_context_changed(get_thread_context=get_thread_context, logger=logger)

            @assistant.user_message
            def on_user_message(client, context, get_thread_context, logger, payload, say, set_status):
                handle_user_message(client=client, context=context,
                                    get_thread_context=get_thread_context, logger=logger,
                                    payload=payload, say=say, set_status=set_status)

            app.use(assistant)

            @app.action("agent_feedback_positive")
            def action_positive(ack, body, logger):
                handle_feedback_positive(ack=ack, body=body, logger=logger)

            @app.action("agent_feedback_negative")
            def action_negative(ack, body, logger):
                handle_feedback_negative(ack=ack, body=body, logger=logger)

            logger.info("✅ Full AI Agent mode enabled (Assistant + streaming + tasks)")

        except Exception as e:
            logger.error(f"⚠️ Failed to init Assistant, falling back to basic mode: {e}")
            ASSISTANT_AVAILABLE = False
    else:
        logger.info("ℹ️ Running in basic mode (upgrade slack_bolt/sdk for AI Agent features)")

    # ── Register commands (works in ALL modes) ──────────────────
    from backend.slack_bot.commands import register_commands
    register_commands(app)

    # ── Register governance (App Home + /agent) — guarded ───────
    try:
        from backend.slack_bot.app_home import register_governance
        register_governance(app)
        logger.info("✅ Governance features registered (App Home + /agent)")
    except Exception as e:
        logger.warning(f"⚠️ Governance features unavailable: {e}")

    # ── Suppress noisy events ───────────────────────────────────
    @app.event("message_changed")
    def handle_message_changed(event, logger):
        pass

    @app.event("message_deleted")
    def handle_message_deleted(event, logger):
        pass

    @app.event("channel_join")
    def handle_channel_join(event, logger):
        pass

    @app.event("member_joined_channel")
    def handle_member_joined(event, logger):
        pass

    return app


def _start_background_tasks():
    """Start background tasks in daemon threads."""
    import threading

    def queue_loop():
        import asyncio
        try:
            from backend.services.ticket_queue_processor import get_queue_processor
            processor = get_queue_processor()
            logger.info("🎫 Ticket queue processor started")
            while True:
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(processor.process_queue_batch(batch_size=5))
                    loop.close()
                except Exception as e:
                    logger.error(f"Queue processor error: {e}")
                time.sleep(30.0)
        except Exception as e:
            logger.warning(f"Queue processor not available: {e}")

    def sla_loop():
        import asyncio
        try:
            from backend.services.sla_monitor import get_sla_monitor
            monitor = get_sla_monitor()
            logger.info("🔍 SLA monitor started")
            while True:
                try:
                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(monitor.monitor_sla_compliance())
                    loop.close()
                    if result.get("active_breaches", 0) > 0:
                        logger.warning(f"⚠️ SLA: {result['active_breaches']} breaches")
                except Exception as e:
                    logger.error(f"SLA monitor error: {e}")
                time.sleep(300.0)
        except Exception as e:
            logger.warning(f"SLA monitor not available: {e}")

    threading.Thread(target=queue_loop, daemon=True).start()
    threading.Thread(target=sla_loop, daemon=True).start()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()

    if not settings.slack_bot_token or not settings.slack_app_token:
        logger.error("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required.")
        return

    app = create_app()

    # Start scheduled jobs (morning digest, proactive alerts)
    try:
        from backend.slack_bot.scheduler import start_scheduler
        start_scheduler(app)
        logger.info("✅ Scheduler started")
    except Exception as e:
        logger.warning(f"⚠️ Scheduler not available: {e}")

    # Start background tasks
    _start_background_tasks()

    # ── Socket Mode with reconnection loop ──────────────────────
    reconnect_count = 0
    backoff = INITIAL_BACKOFF_SECONDS

    mode = "AI Agent" if (ASSISTANT_AVAILABLE and STREAMING_AVAILABLE) else "Basic"

    while reconnect_count < MAX_RECONNECT_ATTEMPTS:
        try:
            handler = SocketModeHandler(app, settings.slack_app_token)

            if reconnect_count == 0:
                logger.info(f"🚀 Slack bot starting in Socket Mode ({mode} mode)...")
                if mode == "AI Agent":
                    logger.info("   Features: Assistant panel, streaming, tasks, governance, deploy")
                else:
                    logger.info("   Features: Channel messages, deploy commands, direct responses")
                    logger.info("   Upgrade packages for full AI Agent features")
            else:
                logger.info(f"🔄 Reconnecting (attempt {reconnect_count})...")

            handler.start()

            logger.warning("Socket Mode exited, reconnecting...")
            backoff = INITIAL_BACKOFF_SECONDS
            reconnect_count += 1

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            try:
                handler.close()
            except Exception:
                pass
            return

        except ConnectionResetError as e:
            reconnect_count += 1
            logger.error(f"Connection reset (attempt {reconnect_count}): {e}")
            time.sleep(backoff)
            backoff = min(backoff * 1.5, MAX_BACKOFF_SECONDS)

        except Exception as e:
            reconnect_count += 1
            logger.error(
                f"Socket Mode error (attempt {reconnect_count}): "
                f"[{type(e).__name__}] {str(e)[:200]}"
            )
            if "invalid_auth" in str(e).lower():
                backoff = min(backoff * 3, MAX_BACKOFF_SECONDS * 2)
            time.sleep(backoff)
            backoff = min(backoff * 1.5, MAX_BACKOFF_SECONDS)

    logger.error(f"🔴 Failed after {MAX_RECONNECT_ATTEMPTS} attempts.")
    sys.exit(1)


if __name__ == "__main__":
    main()
