"""
live_prices -- Cached + rate-limited intraday price fetcher for the
paper-trading dashboard (4.5.6).

yfinance rate-limits aggressively; this module uses:

  - In-process per-ticker TTL cache (default 60s, matches the frontend poll
    cadence).
  - Per-call upper bound on the number of unique tickers that can bypass the
    cache in a given window.
  - Thread-safe via a module-level lock (same pattern as api_cache.py).

Design note: we do NOT spawn background workers. The endpoint pulls-on-demand,
uses the cache when fresh, and only round-trips yfinance for genuinely stale
tickers. This keeps the invariant "never exceed yfinance rate limits" simple
to reason about, matching the api_cache/perf_tracker module pattern.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_TTL_SEC = 60.0
MAX_UNIQUE_REFRESH_PER_MIN = 30  # hard cap across the entire process


class LivePriceCache:
    """Thread-safe TTL cache with a yfinance rate gate."""

    def __init__(self, ttl_sec: float = DEFAULT_TTL_SEC, max_refresh_per_min: int = MAX_UNIQUE_REFRESH_PER_MIN):
        self._ttl = float(ttl_sec)
        self._max_refresh = int(max_refresh_per_min)
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, Optional[float]]] = {}
        self._refresh_log: list[float] = []  # unix timestamps of recent refreshes

    def _rate_ok(self, now: float) -> bool:
        # Drop log entries older than 60 seconds
        cutoff = now - 60.0
        self._refresh_log = [t for t in self._refresh_log if t >= cutoff]
        return len(self._refresh_log) < self._max_refresh

    def get_many(self, tickers: list[str]) -> dict[str, dict]:
        """
        Return {ticker: {price, age_sec, cached}} for each ticker. `age_sec` is
        the seconds since the price was fetched from yfinance (not since the
        cache entry was served) -- required so the UI can render a freshness
        indicator (Coinpaprika 2024 anti-pattern: polling without staleness).
        """
        cleaned = []
        for t in tickers or []:
            t = str(t).strip().upper()
            if t and all(c.isalnum() or c in ".-" for c in t):
                cleaned.append(t)
        result: dict[str, dict] = {}
        now = time.time()

        with self._lock:
            stale: list[str] = []
            for t in cleaned:
                entry = self._cache.get(t)
                if entry and (now - entry[0]) < self._ttl:
                    ts, price = entry
                    result[t] = {
                        "price": price,
                        "age_sec": round(now - ts, 1),
                        "cached": True,
                    }
                else:
                    stale.append(t)

            for t in stale:
                if not self._rate_ok(now):
                    logger.warning("live_prices: rate gate tripped, serving stale/None")
                    entry = self._cache.get(t)
                    if entry:
                        ts, price = entry
                        result[t] = {
                            "price": price,
                            "age_sec": round(now - ts, 1),
                            "cached": True,
                            "rate_gated": True,
                        }
                    else:
                        result[t] = {"price": None, "age_sec": None, "cached": False, "rate_gated": True}
                    continue
                try:
                    price = _fetch_price(t)
                except Exception as e:
                    logger.debug(f"live_prices fetch failed for {t}: {e}")
                    price = None
                self._cache[t] = (now, price)
                self._refresh_log.append(now)
                result[t] = {
                    "price": price,
                    "age_sec": 0.0,
                    "cached": False,
                }

            return result


def _fetch_price(ticker: str) -> Optional[float]:
    """One synchronous yfinance call. Caller holds the rate gate."""
    hist = yf.Ticker(ticker).history(period="1d", interval="1m")
    if hist.empty:
        return None
    return float(hist["Close"].iloc[-1])


_live_cache = LivePriceCache()


def get_live_cache() -> LivePriceCache:
    return _live_cache
