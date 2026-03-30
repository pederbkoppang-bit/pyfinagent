"""
Slack Bot entry point -- Bolt async app with Socket Mode.

Runs as a standalone service: python -m backend.slack_bot.app
Uses Socket Mode (outbound WebSocket) -- no public URL needed.

Connection stability features:
- Reconnection loop with exponential backoff (survives transient network failures)
- Configurable ping interval for faster disconnect detection
- Graceful SIGTERM/SIGINT shutdown (clean close for OpenClaw/container restarts)
- Retry handlers on the Slack web client for API call resilience
"""

import asyncio
import logging
import signal

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.http_retry.builtin_async_handlers import AsyncConnectionErrorRetryHandler
from slack_sdk.http_retry.async_handler import AsyncRetryHandler
from slack_sdk.http_retry.interval_calculator import BackoffRetryIntervalCalculator
from slack_sdk.http_retry.state import RetryState
from slack_sdk.http_retry.request import HttpRequest
from slack_sdk.http_retry.response import HttpResponse

from backend.config.settings import get_settings
from backend.slack_bot.commands import register_commands
from backend.slack_bot.scheduler import start_scheduler

logger = logging.getLogger(__name__)

# Sentinel used to signal a clean shutdown from signal handlers
_shutdown_event: asyncio.Event | None = None

# Backoff parameters for the outer reconnection loop
_INITIAL_BACKOFF_S = 2
_MAX_BACKOFF_S = 120


class _AsyncRateLimitRetryHandler(AsyncRetryHandler):
    """Retry on HTTP 429 (rate-limited) responses from Slack Web API."""

    def __init__(self):
        super().__init__(
            max_retry_count=3,
            interval_calculator=BackoffRetryIntervalCalculator(backoff_factor=1.0),
        )

    async def _can_retry_async(
        self,
        *,
        state: RetryState,
        request: HttpRequest,
        response: HttpResponse | None = None,
        error: Exception | None = None,
    ) -> bool:
        return response is not None and response.status_code == 429


def create_app() -> AsyncApp:
    settings = get_settings()
    app = AsyncApp(
        token=settings.slack_bot_token,
    )
    # Attach retry handlers so transient Slack API errors are retried
    # automatically (connection resets, 429 rate limits).
    app.client.retry_handlers = [
        AsyncConnectionErrorRetryHandler(max_retry_count=3),
        _AsyncRateLimitRetryHandler(),
    ]
    register_commands(app)
    return app


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register SIGTERM / SIGINT so the bot shuts down cleanly."""
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    def _handle_signal(sig: signal.Signals) -> None:
        logger.info("Received %s -- initiating graceful shutdown", sig.name)
        if _shutdown_event is not None:
            _shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except NotImplementedError:
            # Windows does not support add_signal_handler; fall back to
            # signal.signal which is less reliable but better than nothing.
            signal.signal(sig, lambda s, _f: _handle_signal(signal.Signals(s)))


async def _run_with_reconnect(app: AsyncApp) -> None:
    """Connect to Slack Socket Mode with automatic reconnection on failure.

    The SDK's built-in auto-reconnect handles transient WebSocket drops.
    This outer loop catches cases the SDK cannot recover from (e.g. token
    rotation, prolonged outage) and retries with exponential backoff.

    ``attempt`` resets after every successful connection so the retry
    limit applies per connection cycle, not globally.
    """
    settings = get_settings()
    max_retries = settings.slack_reconnect_max_retries  # 0 = infinite
    attempt = 0
    backoff = _INITIAL_BACKOFF_S

    # How often (seconds) to check if the WebSocket is still alive.
    _HEALTH_CHECK_INTERVAL = 30

    while True:
        attempt += 1
        handler: AsyncSocketModeHandler | None = None
        try:
            # Lower ping_interval = faster detection of stale connections
            handler = AsyncSocketModeHandler(
                app,
                settings.slack_app_token,
            )
            # The underlying SocketModeClient exposes ping_interval; set it
            # via the client attribute that the handler wraps.
            if hasattr(handler, "client") and hasattr(handler.client, "ping_interval"):
                handler.client.ping_interval = settings.slack_ping_interval

            logger.info(
                "Socket Mode connecting (attempt %d, ping_interval=%ds)...",
                attempt,
                settings.slack_ping_interval,
            )
            await handler.connect_async()
            logger.info("Socket Mode connected successfully")

            # Reset backoff/attempt on successful connection (per-cycle limit)
            backoff = _INITIAL_BACKOFF_S
            attempt = 0

            # Periodically verify the connection is alive.  If the SDK's
            # internal reconnect fails silently, we detect it here and
            # trigger the outer reconnection loop.
            while True:
                if _shutdown_event is not None and _shutdown_event.is_set():
                    logger.info("Shutdown event received -- closing Socket Mode handler")
                    await handler.close_async()
                    return

                # Check the underlying client connection state
                client = getattr(handler, "client", None)
                if client is not None and hasattr(client, "is_connected"):
                    connected = await client.is_connected()
                    if not connected:
                        logger.warning(
                            "Socket Mode health check: connection lost -- triggering reconnect"
                        )
                        break  # falls through to the reconnection path

                # Interruptible sleep for the health check interval
                if _shutdown_event is not None:
                    try:
                        await asyncio.wait_for(
                            _shutdown_event.wait(),
                            timeout=_HEALTH_CHECK_INTERVAL,
                        )
                        # shutdown_event was set during sleep
                        logger.info("Shutdown event received -- closing Socket Mode handler")
                        await handler.close_async()
                        return
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(_HEALTH_CHECK_INTERVAL)

        except asyncio.CancelledError:
            logger.info("Socket Mode task cancelled -- shutting down")
            if handler is not None:
                try:
                    await handler.close_async()
                except Exception:
                    pass
            return

        except Exception:
            logger.exception(
                "Socket Mode connection lost (attempt %d)", attempt
            )

        # Check retry limit (attempt was already incremented at loop top)
        if 0 < max_retries < attempt:
            logger.error(
                "Reached max reconnect attempts (%d) -- giving up",
                max_retries,
            )
            return

        logger.info("Reconnecting in %ds...", backoff)
        # Interruptible sleep so shutdown signals are honoured during backoff
        if _shutdown_event is not None:
            try:
                await asyncio.wait_for(
                    _shutdown_event.wait(), timeout=backoff
                )
                logger.info("Shutdown during backoff -- exiting")
                return
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(backoff)

        backoff = min(backoff * 2, _MAX_BACKOFF_S)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()

    if not settings.slack_bot_token or not settings.slack_app_token:
        logger.error(
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required. Set them in .env"
        )
        return

    app = create_app()

    # Start scheduled jobs (morning digest, proactive alerts)
    start_scheduler(app)

    # Install OS signal handlers for graceful shutdown (OpenClaw / containers)
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)

    logger.info("Slack bot starting in Socket Mode...")
    await _run_with_reconnect(app)
    logger.info("Slack bot stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run can surface KeyboardInterrupt on Windows
        pass
