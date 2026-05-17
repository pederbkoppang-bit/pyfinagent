"""
Short-interest exclusion data — FINRA bimonthly CSV primary, yfinance fallback.

Returns dict[ticker, shortPercentOfFloat] for use by the screener exclusion filter
(phase-28.5). Boehmer-Jones-Zhang (2008): top-decile shorted stocks underperform
by 1.16%/month; Oxford RAPS (2022): cross-sectional confirmation in 32 countries.

Data path priority:
1. FINRA bimonthly equity short-interest CSV (https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files)
   — free, bulk, no per-ticker HTTP. Cached for 14 days (matches FINRA publication cadence).
2. yfinance Ticker.info["shortPercentOfFloat"] per ticker — fallback when FINRA
   path fails or ticker is missing. Throttled to 0.5s/call to respect Yahoo rate limits.

Returns empty dict on any unrecoverable error (default-OFF safety pattern, mirrors
news_screen / pead_signal).
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "_cache" / "short_interest"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FINRA_CSV_BASE = "https://cdn.finra.org/equity/regsho/monthly/shrt{date}.csv"
_USER_AGENT = "PyFinAgent/2.0 ShortInterest (peder.bkoppang@hotmail.no)"
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,10}$")


def _cache_path() -> Path:
    return _CACHE_DIR / "finra_short_interest_latest.csv"


def _cache_meta_path() -> Path:
    return _CACHE_DIR / "finra_short_interest_meta.txt"


def _cache_is_fresh(cache_days: int) -> bool:
    p = _cache_path()
    if not p.exists():
        return False
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    return age < timedelta(days=cache_days)


async def _try_download_finra_csv(client: httpx.AsyncClient) -> Optional[str]:
    """Try a few recent FINRA bimonthly settlement dates until one returns HTTP 200.

    FINRA publishes on the 15th and end-of-month for each cycle; the exact filename
    is `shrt<YYYYMMDD>.csv`. We probe a small window of recent settlement dates.
    """
    today = datetime.now(timezone.utc).date()
    candidates: list[str] = []
    for delta in range(0, 35):
        d = today - timedelta(days=delta)
        if d.day == 15 or (d.day in (28, 29, 30, 31) and d.month != (d + timedelta(days=1)).month):
            candidates.append(d.strftime("%Y%m%d"))
    seen: set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for date_str in candidates[:6]:
        url = _FINRA_CSV_BASE.format(date=date_str)
        try:
            resp = await client.get(url, timeout=30, headers={"User-Agent": _USER_AGENT})
            if resp.status_code == 200 and resp.text and "Symbol" in resp.text[:500]:
                logger.info("FINRA short-interest CSV downloaded: %s (%d bytes)", date_str, len(resp.text))
                return resp.text
        except Exception as e:
            logger.debug("FINRA probe %s failed: %s", date_str, e)
            continue
    logger.warning("FINRA short-interest CSV: no recent settlement date returned 200 (last 35 days tried)")
    return None


def _parse_finra_csv(csv_text: str) -> dict[str, float]:
    """Parse the FINRA pipe-delimited CSV and return {ticker: shortPercentOfFloat}.

    FINRA columns (canonical): Settlement Date | Symbol | Market Class Code |
    Current Short Position | Previous Short Position | Change | Percent Change |
    Average Daily Volume Quantity | Days to Cover | Revised
    """
    lookup: dict[str, float] = {}
    lines = csv_text.splitlines()
    if not lines:
        return lookup
    header = lines[0]
    sep = "|" if "|" in header else ","
    cols = [c.strip().lower() for c in header.split(sep)]
    try:
        sym_idx = next(i for i, c in enumerate(cols) if "symbol" in c)
        short_pos_idx = next(i for i, c in enumerate(cols) if "current" in c and "short" in c)
    except StopIteration:
        logger.warning("FINRA CSV header missing expected Symbol / Current Short Position columns: %s", cols)
        return lookup

    pof_idx: Optional[int] = None
    for i, c in enumerate(cols):
        if "float" in c or "percent" in c:
            pof_idx = i
            break

    parsed = 0
    for line in lines[1:]:
        parts = line.split(sep)
        if len(parts) <= max(sym_idx, short_pos_idx):
            continue
        ticker = parts[sym_idx].strip().upper()
        if not _TICKER_RE.match(ticker):
            continue
        try:
            if pof_idx is not None and pof_idx < len(parts):
                raw = parts[pof_idx].replace("%", "").strip()
                if raw:
                    val = float(raw)
                    lookup[ticker] = val / 100.0 if val > 1.0 else val
                    parsed += 1
                    continue
            short_pos = float(parts[short_pos_idx].replace(",", "").strip() or 0)
            if short_pos > 0:
                lookup[ticker] = -1.0
        except (ValueError, IndexError):
            continue
    logger.info("Parsed FINRA short-interest CSV: %d tickers with shortPercentOfFloat", parsed)
    return lookup


async def _yfinance_fallback(tickers: list[str], throttle_s: float = 0.5) -> dict[str, float]:
    """Fallback: per-ticker yfinance Ticker.info call. Slow; only use for missing tickers."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; short-interest fallback unavailable")
        return {}

    out: dict[str, float] = {}
    for t in tickers:
        try:
            info = await asyncio.to_thread(lambda x=t: yf.Ticker(x).info)
            pct = info.get("shortPercentOfFloat") if isinstance(info, dict) else None
            if pct is not None:
                out[t.upper()] = float(pct)
        except Exception as e:
            logger.debug("yfinance shortPercentOfFloat fetch failed for %s: %s", t, e)
        await asyncio.sleep(throttle_s)
    logger.info("yfinance fallback: fetched shortPercentOfFloat for %d/%d tickers", len(out), len(tickers))
    return out


async def fetch_short_interest_lookup(
    fallback_tickers: Optional[list[str]] = None,
    use_cache: bool = True,
) -> dict[str, float]:
    """Return {ticker: shortPercentOfFloat} for use by the screener exclusion filter.

    Args:
        fallback_tickers: If FINRA path returns nothing for these tickers, try
            yfinance per-ticker. Pass the screener's candidate list to bound cost.
        use_cache: When True (default), reuse the local FINRA CSV cache if fresh
            (< short_interest_cache_days old per settings).

    Returns:
        dict[ticker, float] -- keys are uppercase tickers, values in [0, 1].
        Empty dict if all paths fail.
    """
    settings = get_settings()
    cache_days = getattr(settings, "short_interest_cache_days", 14)

    csv_text: Optional[str] = None
    if use_cache and _cache_is_fresh(cache_days):
        try:
            csv_text = _cache_path().read_text(encoding="utf-8")
            logger.info("FINRA short-interest cache hit (%d bytes)", len(csv_text))
        except Exception as e:
            logger.warning("FINRA cache unreadable: %s", e)

    if csv_text is None:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            csv_text = await _try_download_finra_csv(client)
        if csv_text:
            try:
                _cache_path().write_text(csv_text, encoding="utf-8")
                _cache_meta_path().write_text(
                    datetime.now(timezone.utc).isoformat(), encoding="utf-8"
                )
            except Exception as e:
                logger.warning("FINRA cache write failed: %s", e)

    lookup: dict[str, float] = {}
    if csv_text:
        try:
            lookup = _parse_finra_csv(csv_text)
        except Exception as e:
            logger.warning("FINRA CSV parse failed: %s", e)

    if fallback_tickers:
        missing = [t for t in fallback_tickers if t.upper() not in lookup]
        if missing and not lookup:
            yf_lookup = await _yfinance_fallback(missing[:50])
            lookup.update(yf_lookup)

    logger.info("Short-interest lookup: %d tickers total", len(lookup))
    return lookup
