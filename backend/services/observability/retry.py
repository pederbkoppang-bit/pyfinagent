"""phase-6.7 retry with capped exponential backoff + full jitter.

Research (brief 2026-04-19):
- AWS builders-library: full-jitter delay = uniform(0, cap) spreads retries
- AWS prescriptive guidance: 3 retries, base, multiplier, retry on 429+5xx
- Honour `Retry-After` header when present (Anthropic sends it on 429)

Both sync and async variants. Sync variant is the one used by news/calendar
source adapters (which use `requests.get`); async variant for httpx callers.

Caller responsibility: ensure the wrapped fn is idempotent. GET requests are
idempotent by HTTP spec; POST/PUT require explicit caller opt-in.
"""
from __future__ import annotations

import logging
import random
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RetryExhausted(Exception):
    """Raised after all attempts fail. Wraps the last underlying exception."""

    def __init__(self, attempts: int, last_exc: BaseException) -> None:
        super().__init__(f"retry exhausted after {attempts} attempts: {last_exc!r}")
        self.attempts = attempts
        self.last_exc = last_exc


def _parse_retry_after(value: str | None) -> float | None:
    """Accepts integer seconds OR HTTP-date; returns seconds-from-now or None."""
    if not value:
        return None
    v = value.strip()
    # integer-seconds form
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    # HTTP-date form
    try:
        dt = parsedate_to_datetime(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


def _delay_for(
    attempt: int, base: float, multiplier: float, cap: float, jitter: str
) -> float:
    raw = min(cap, base * (multiplier ** attempt))
    if jitter == "full":
        return random.uniform(0.0, raw)
    if jitter == "equal":
        return raw / 2.0 + random.uniform(0.0, raw / 2.0)
    # "none"
    return raw


def _should_retry(exc: BaseException | None, status: int | None, retry_on: tuple[int, ...]) -> bool:
    if status is not None and status in retry_on:
        return True
    if exc is None:
        return False
    # Network errors: connection/read/timeout. We retry on broad classes
    # because the underlying lib (requests, httpx, anthropic sdk) maps these
    # inconsistently.
    msg = type(exc).__name__.lower()
    for tok in ("timeout", "connection", "network", "transport", "apiconnection"):
        if tok in msg:
            return True
    return False


def _extract_status_and_retry_after(exc_or_response: Any) -> tuple[int | None, float | None]:
    """Best-effort extraction of HTTP status + Retry-After from an object.

    Handles:
    - `requests.Response` (status_code + headers.get("Retry-After"))
    - `httpx.HTTPStatusError.response` (same shape)
    - `anthropic.APIStatusError` (status_code)
    - plain exceptions: (None, None)
    """
    status: int | None = None
    retry_after: float | None = None
    resp = getattr(exc_or_response, "response", exc_or_response)
    sc = getattr(resp, "status_code", None)
    if isinstance(sc, int):
        status = sc
    headers = getattr(resp, "headers", None)
    if headers:
        ra_header: str | None = None
        try:
            ra_header = headers.get("Retry-After") or headers.get("retry-after")
        except Exception:  # pragma: no cover
            pass
        retry_after = _parse_retry_after(ra_header)
    return status, retry_after


def retry_with_backoff(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 3,
    base: float = 1.0,
    multiplier: float = 2.0,
    cap: float = 30.0,
    jitter: str = "full",
    retry_on: tuple[int, ...] = (429, 502, 503, 504),
    honor_retry_after: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call `fn()` up to `max_attempts` times with backoff.

    `fn` is a zero-arg callable. The caller binds positional/keyword args via
    `functools.partial` or a closure. This keeps the retry helper itself
    call-shape-agnostic.

    Raises `RetryExhausted` after the last attempt (wraps the last exception)
    or returns the first successful result. If `fn` returns a response-like
    object with a `status_code` in `retry_on`, the response is treated as a
    retryable failure, and the final attempt's response is returned to the
    caller unchanged (so HTTP-level success/failure handling stays in the
    caller).
    """
    last_exc: BaseException | None = None
    last_response: Any = None

    for attempt in range(max_attempts):
        status: int | None = None
        retry_after: float | None = None
        try:
            result = fn()
        except BaseException as exc:
            last_exc = exc
            status, retry_after = _extract_status_and_retry_after(exc)
            if attempt == max_attempts - 1 or not _should_retry(exc, status, retry_on):
                raise
        else:
            status_val = getattr(result, "status_code", None)
            if isinstance(status_val, int) and status_val in retry_on:
                last_response = result
                status = status_val
                _, retry_after = _extract_status_and_retry_after(result)
                if attempt == max_attempts - 1:
                    return result
            else:
                return result

        delay = _delay_for(attempt, base, multiplier, cap, jitter)
        if honor_retry_after and retry_after is not None:
            delay = max(delay, retry_after)
        logger.debug(
            "retry attempt=%d sleep=%.2fs status=%s exc=%r",
            attempt + 1,
            delay,
            status,
            last_exc,
        )
        sleep(delay)

    # Should not reach here; defensive
    if last_exc is not None:  # pragma: no cover
        raise RetryExhausted(max_attempts, last_exc)
    return last_response


async def retry_with_backoff_async(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 3,
    base: float = 1.0,
    multiplier: float = 2.0,
    cap: float = 30.0,
    jitter: str = "full",
    retry_on: tuple[int, ...] = (429, 502, 503, 504),
    honor_retry_after: bool = True,
) -> Any:
    """Async variant. `fn` is an async zero-arg callable (returns a coroutine)."""
    import asyncio

    last_exc: BaseException | None = None

    for attempt in range(max_attempts):
        status: int | None = None
        retry_after: float | None = None
        try:
            result = await fn()
        except BaseException as exc:
            last_exc = exc
            status, retry_after = _extract_status_and_retry_after(exc)
            if attempt == max_attempts - 1 or not _should_retry(exc, status, retry_on):
                raise
        else:
            status_val = getattr(result, "status_code", None)
            if isinstance(status_val, int) and status_val in retry_on:
                status = status_val
                _, retry_after = _extract_status_and_retry_after(result)
                if attempt == max_attempts - 1:
                    return result
            else:
                return result

        delay = _delay_for(attempt, base, multiplier, cap, jitter)
        if honor_retry_after and retry_after is not None:
            delay = max(delay, retry_after)
        logger.debug(
            "retry async attempt=%d sleep=%.2fs status=%s exc=%r",
            attempt + 1,
            delay,
            status,
            last_exc,
        )
        await asyncio.sleep(delay)

    if last_exc is not None:  # pragma: no cover
        raise RetryExhausted(max_attempts, last_exc)


__all__ = [
    "retry_with_backoff",
    "retry_with_backoff_async",
    "RetryExhausted",
    "_parse_retry_after",
    "_delay_for",
]
