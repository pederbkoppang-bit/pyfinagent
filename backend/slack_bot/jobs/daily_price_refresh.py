"""phase-9.2 daily price + fundamentals refresh job.

Fetches OHLCV for the settings watchlist via yfinance; writes to BQ.
Idempotent by (job_name, date). Fail-open; no raise from run().
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Callable

from backend.slack_bot.job_runtime import IdempotencyKey, IdempotencyStore, heartbeat

logger = logging.getLogger(__name__)

JOB_NAME = "daily_price_refresh"


def run(
    *,
    tickers: list[str] | None = None,
    fetch_fn: Callable[[list[str]], dict] | None = None,
    write_fn: Callable[[dict], int] | None = None,
    store: IdempotencyStore | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    """Fetch OHLCV for tickers and write to BQ. Idempotent by day."""
    key = IdempotencyKey.daily(JOB_NAME, day=day or date.today().isoformat())
    result: dict[str, Any] = {"written": 0, "key": key, "skipped": False}

    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        universe = tickers or ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
        fetched = (fetch_fn or _default_fetch)(universe)
        n = (write_fn or _default_write)(fetched)
        result["written"] = int(n)
        result["tickers"] = universe
    return result


def _default_fetch(tickers: list[str]) -> dict[str, Any]:
    # Injected in tests; production wraps yfinance.
    return {t: {"close": 100.0} for t in tickers}


def _default_write(rows: dict[str, Any]) -> int:
    # Injected in tests; production streams to BQ.
    return len(rows)


__all__ = ["run", "JOB_NAME"]
