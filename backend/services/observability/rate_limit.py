"""phase-6.7 per-source async rate limiter (leaky bucket, aiolimiter).

Research (brief 2026-04-19):
- leaky-bucket semantics enforce steady-state rate respecting external
  limits (Finnhub 30 req/sec); token-bucket allows bursts which we do
  NOT want against an external server
- aiolimiter: ~7 KB, zero runtime deps, asyncio-native, Python 3.9+
- module-level singleton per source prevents thundering-herd on startup

Fall-through: if aiolimiter is not installed (not in requirements.txt on
some dev boxes), `get_rate_limiter()` returns a no-op context manager so
callers do not break.

Async and sync adaptors are provided. Sync callers wrapping `requests.get`
can still benefit from rate limiting via `acquire_sync()` which blocks on
a thread-local asyncio loop.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from aiolimiter import AsyncLimiter  # type: ignore[import-not-found]

    _AIOLIMITER_ERR: Exception | None = None
except Exception as exc:  # pragma: no cover
    AsyncLimiter = None  # type: ignore[assignment]
    _AIOLIMITER_ERR = exc


# Default rates per source (requests per second) -- can be overridden by
# settings.<source>_rate_limit_rps. Conservative vs upstream server limits
# to leave headroom.
_DEFAULT_RATES: dict[str, int] = {
    "finnhub": 25,
    "benzinga": 2,
    "alpaca": 30,
    "fred": 5,
    "alphavantage": 1,
    "anthropic": 40,
    "gemini": 10,
}


class _NoOpLimiter:
    """Fallback when aiolimiter is absent. Always permits immediately."""

    async def __aenter__(self) -> "_NoOpLimiter":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def acquire_sync(self) -> None:
        return None


_registry: dict[str, Any] = {}


def _resolve_rate(source: str) -> int:
    try:
        from backend.config.settings import get_settings

        s = get_settings()
        override = getattr(s, f"{source}_rate_limit_rps", None)
        if override:
            return int(override)
    except Exception:  # pragma: no cover
        pass
    return _DEFAULT_RATES.get(source, 5)


def get_rate_limiter(source: str) -> Any:
    """Return a module-level singleton rate limiter for `source`.

    Usage (async):
        async with get_rate_limiter("finnhub"):
            resp = await httpx_client.get(url)

    Usage (sync helper, in a thread):
        limiter = get_rate_limiter("finnhub")
        limiter.acquire_sync()   # blocks until the rate budget allows
        resp = requests.get(url)

    Both adaptors are no-ops when aiolimiter is unavailable.
    """
    if source in _registry:
        return _registry[source]
    if AsyncLimiter is None:
        logger.debug(
            "aiolimiter absent (%s); rate_limit is no-op for %s",
            _AIOLIMITER_ERR,
            source,
        )
        lim = _NoOpLimiter()
    else:
        rate = _resolve_rate(source)
        # aiolimiter: AsyncLimiter(max_rate, time_period_seconds). 25 req/sec
        # means max_rate=25, time_period=1.
        lim = AsyncLimiter(max_rate=rate, time_period=1)
        lim = _SyncAdaptor(lim, rate)
    _registry[source] = lim
    return lim


class _SyncAdaptor:
    """Wraps AsyncLimiter with a sync `acquire_sync()` helper.

    `acquire_sync` uses asyncio.run in a fresh loop per call so it is safe to
    call from threads not running a loop (e.g., sync `requests.get` callers).
    This is intentionally simple, not high-throughput -- sync callers should
    migrate to httpx/asyncio for real throughput gains.
    """

    def __init__(self, limiter: Any, rate: int) -> None:
        self._limiter = limiter
        self._rate = rate

    async def __aenter__(self) -> Any:
        return await self._limiter.__aenter__()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return await self._limiter.__aexit__(exc_type, exc, tb)

    def acquire_sync(self) -> None:
        import asyncio

        async def _once() -> None:
            async with self._limiter:
                return None

        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_once())
            finally:
                loop.close()
        except Exception as exc:  # pragma: no cover
            logger.debug("acquire_sync failure (rate=%d): %r", self._rate, exc)


def reset_registry() -> None:
    """Test helper: drop singletons so fresh limiters are created on next get()."""
    _registry.clear()


__all__ = ["get_rate_limiter", "reset_registry"]
