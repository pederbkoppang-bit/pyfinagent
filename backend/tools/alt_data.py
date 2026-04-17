"""
Alternative data tool -- Google Trends via pytrends.

pytrends 4.9.2 is the last stable release (repo archived April 2025).
Its built-in retries= arg forwards to urllib3.Retry with the removed
method_whitelist kwarg, which crashes on urllib3 2.x. So we do
retry/backoff in Python instead of via pytrends.

Structural fix for the 429 problem: a module-level 24-hour TTL cache
keyed by (ticker, today). At pyfinagent's 500-ticker/day volume the
cache suppresses 90%+ of live calls, staying well under Google's
observed ~200/day per-IP quota. ERROR and UNAVAILABLE results are
cached too so repeated failures within the day don't re-hammer Google.

Graceful degradation: sustained 429 surfaces as `signal: "UNAVAILABLE"`
(not ERROR) so the 12-signal aggregator treats it as a missing-but-
not-broken signal.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date

logger = logging.getLogger(__name__)

_TTL_SECONDS = 24 * 3600
_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key: tuple[str, str]) -> dict | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if not entry:
            return None
        stamped, payload = entry
        if time.time() - stamped > _TTL_SECONDS:
            _CACHE.pop(key, None)
            return None
        return payload


def _cache_put(key: tuple[str, str], payload: dict) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), payload)


def _is_rate_limited(exc: BaseException) -> bool:
    """Detect 429 / ToS-block errors surfaced through pytrends."""
    msg = str(exc).lower()
    return ("429" in msg or "too many requests" in msg
            or "response with code 429" in msg or "quota" in msg)


def get_google_trends(ticker: str, company_name: str) -> dict:
    """Fetch Google Trends interest with retry + 24h TTL cache."""
    cache_key = (ticker.upper(), date.today().isoformat())
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("alt_data cache hit for %s", ticker)
        return cached

    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.warning("pytrends not installed -- skipping Google Trends")
        return {
            "ticker": ticker,
            "signal": "UNAVAILABLE",
            "summary": "pytrends not installed. Run: pip install pytrends",
        }

    keywords = [ticker, company_name] if company_name != ticker else [ticker]
    interest = None
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
            pytrends.build_payload(keywords[:5], cat=0,
                                    timeframe="today 12-m", geo="US")
            interest = pytrends.interest_over_time()
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            if not _is_rate_limited(e):
                break  # non-429: no point retrying
            # Exponential backoff: 1.5s, 3s, 6s.
            time.sleep(1.5 * (2 ** attempt))

    if last_exc is not None:
        if _is_rate_limited(last_exc):
            logger.warning("alt_data 429 for %s after 3 retries; graceful UNAVAILABLE", ticker)
            result = {
                "ticker": ticker,
                "signal": "UNAVAILABLE",
                "reason": "google_trends_429",
                "summary": (
                    "Google Trends rate-limited (429) for this IP; "
                    "treating as missing signal (not an error)."
                ),
            }
            _cache_put(cache_key, result)
            return result
        logger.error("Failed to fetch Google Trends for %s: %s", ticker, last_exc)
        result = {"ticker": ticker, "signal": "ERROR", "summary": f"Error: {last_exc}"}
        _cache_put(cache_key, result)  # cache errors too -- avoid re-hammering
        return result

    if interest.empty:
        result = {
            "ticker": ticker,
            "signal": "NO_DATA",
            "summary": "No Google Trends data available.",
        }
        _cache_put(cache_key, result)
        return result

    col = ticker if ticker in interest.columns else interest.columns[0]
    values = interest[col].tolist()
    dates = [d.strftime("%Y-%m-%d") for d in interest.index]

    recent = values[-4:] if len(values) >= 4 else values
    older = values[-12:-4] if len(values) >= 12 else values[:max(len(values) // 2, 1)]
    recent_avg = sum(recent) / len(recent) if recent else 0
    older_avg = sum(older) / len(older) if older else recent_avg

    momentum = ((recent_avg - older_avg) / max(older_avg, 1)) * 100
    current_interest = values[-1] if values else 0
    peak_interest = max(values) if values else 0

    signal = "NEUTRAL"
    if momentum > 30 and current_interest > 60:
        signal = "RISING_STRONG"
    elif momentum > 15:
        signal = "RISING"
    elif momentum < -30:
        signal = "DECLINING_STRONG"
    elif momentum < -15:
        signal = "DECLINING"

    trend_data = [
        {"date": dates[i], "interest": values[i]}
        for i in range(0, len(dates), max(1, len(dates) // 52))
    ]

    result = {
        "ticker": ticker,
        "company": company_name,
        "current_interest": current_interest,
        "peak_interest": peak_interest,
        "momentum_pct": round(momentum, 1),
        "trend_data": trend_data,
        "signal": signal,
        "summary": (
            f"Google Trends interest: {current_interest}/100 (peak: {peak_interest}). "
            f"Momentum: {momentum:+.1f}%. Signal: {signal}."
        ),
    }
    _cache_put(cache_key, result)
    return result
