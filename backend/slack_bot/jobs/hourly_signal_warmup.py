"""phase-9.5 Hourly signal cache warmup for the watchlist."""
from __future__ import annotations

import logging
from typing import Any, Callable

from backend.slack_bot.job_runtime import IdempotencyKey, IdempotencyStore, heartbeat

logger = logging.getLogger(__name__)
JOB_NAME = "hourly_signal_warmup"


def run(
    *,
    watchlist: list[str] | None = None,
    compute_signal_fn: Callable[[str], Any] | None = None,
    cache_backend: dict | None = None,
    store: IdempotencyStore | None = None,
    iso_hour: str | None = None,
) -> dict[str, Any]:
    key = IdempotencyKey.hourly(JOB_NAME, iso_hour=iso_hour)
    cache = cache_backend if cache_backend is not None else {}
    result: dict[str, Any] = {"warmed": 0, "key": key, "skipped": False, "cache_size": 0}
    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        wl = watchlist or _load_watchlist()
        fn = compute_signal_fn or (lambda t: {"score": 0.0})
        for ticker in wl:
            cache[ticker] = fn(ticker)
        result["warmed"] = len(wl)
        result["cache_size"] = len(cache)
    return result


def _load_watchlist() -> list[str]:
    """Injectable default: production reads `settings.watchlist`."""
    try:
        from backend.config.settings import get_settings
        s = get_settings()
        wl = getattr(s, "watchlist", None) or []
        return list(wl)
    except Exception:  # pragma: no cover
        return ["AAPL", "MSFT", "SPY"]


__all__ = ["run", "JOB_NAME"]
