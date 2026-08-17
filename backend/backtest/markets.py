"""
Multi-market abstractions for Phase 2.9 expansion.

Supports: US (default), NO, CA, EU, KR with proper calendars, currencies, and ticker namespacing.
Ticker namespacing: {market}:{ticker}  (e.g., "US:AAPL", "NO:EQNR", "KR:005930")
"""

import logging
from typing import Optional

try:
    import exchange_calendars as xcals
except ImportError:
    xcals = None

logger = logging.getLogger(__name__)

DEFAULT_MARKET = "US"

# Market configuration: exchange calendar, currency, timezone, benchmark.
# phase-50.5: `benchmark` is the per-market index ticker used by the backtest
# baseline/alpha computation (analytics.compute_baseline_strategies). US stays
# "SPY" so the US backtest is byte-identical; intl benchmarks are FX-converted
# to the base currency. yfinance index symbols: ^GDAXI (DAX), ^KS11 (KOSPI),
# ^OSEAX (Oslo All-Share), ^GSPTSE (S&P/TSX Composite).
MARKET_CONFIG = {
    "US": {
        "exchange": "XNYS",
        "currency": "USD",
        "timezone": "America/New_York",
        "description": "NYSE/NASDAQ (United States)",
        "benchmark": "SPY",
    },
    "NO": {
        "exchange": "XOSL",
        "currency": "NOK",
        "timezone": "Europe/Oslo",
        "description": "Oslo Børs (Norway)",
        "benchmark": "^OSEAX",
    },
    "CA": {
        "exchange": "XTSE",
        "currency": "CAD",
        "timezone": "America/Toronto",
        "description": "Toronto Stock Exchange (Canada)",
        "benchmark": "^GSPTSE",
    },
    "EU": {
        "exchange": "XETR",  # XETRA (Germany) as primary
        "currency": "EUR",
        "timezone": "Europe/Berlin",
        "description": "XETRA/Euronext (European equities)",
        "benchmark": "^GDAXI",
    },
    "KR": {
        "exchange": "XKRX",
        "currency": "KRW",
        "timezone": "Asia/Seoul",
        "description": "KRX - KOSPI/KOSDAQ (South Korea)",
        "benchmark": "^KS11",
    },
    # goal-multimarket-ux: Nordic markets (additive metadata; trading only happens
    # for codes listed in PAPER_MARKETS, so adding these does NOT change the live
    # US/EU/KR loop). Currencies: SE=SEK, DK=DKK, FI=EUR (euro-zone), IS=ISK.
    "SE": {
        "exchange": "XSTO",
        "currency": "SEK",
        "timezone": "Europe/Stockholm",
        "description": "Nasdaq Stockholm (Sweden)",
        "benchmark": "^OMX",
    },
    "DK": {
        "exchange": "XCSE",
        "currency": "DKK",
        "timezone": "Europe/Copenhagen",
        "description": "Nasdaq Copenhagen (Denmark)",
        "benchmark": "^OMXC25",
    },
    "FI": {
        "exchange": "XHEL",
        "currency": "EUR",
        "timezone": "Europe/Helsinki",
        "description": "Nasdaq Helsinki (Finland)",
        "benchmark": "^OMXH25",
    },
    "IS": {
        "exchange": "XICE",
        "currency": "ISK",
        "timezone": "Atlantic/Reykjavik",
        "description": "Nasdaq Iceland",
        "benchmark": "^OMXIPI",
    },
}


def parse_namespaced_ticker(ticker: str) -> tuple[str, str]:
    """
    Parse {market}:{ticker} format.

    Args:
        ticker: e.g., "US:AAPL" or just "AAPL" (defaults to US)

    Returns:
        (market, ticker) tuple
    """
    if ":" in ticker:
        market, t = ticker.split(":", 1)
        market = market.upper()
        if market not in MARKET_CONFIG:
            logger.warning(f"Unknown market: {market}, defaulting to US")
            market = DEFAULT_MARKET
        return market, t
    return DEFAULT_MARKET, ticker


def get_market_config(market: str = DEFAULT_MARKET) -> dict:
    """Get configuration for a market."""
    market = market.upper()
    return MARKET_CONFIG.get(market, MARKET_CONFIG[DEFAULT_MARKET])


# phase-50.3: yfinance ticker suffix per market (US is bare).
# goal-multimarket-ux: Nordic suffixes added (Stockholm .ST, Copenhagen .CO,
# Helsinki .HE, Iceland .IC).
YF_SUFFIX = {
    "US": "", "NO": ".OL", "CA": ".TO", "EU": ".DE", "KR": ".KS",
    "SE": ".ST", "DK": ".CO", "FI": ".HE", "IS": ".IC",
}


def to_yfinance_symbol(namespaced: str) -> str:
    """phase-50.3: {market}:{ticker} -> yfinance symbol. US:AAPL->AAPL,
    EU:SAP->SAP.DE, KR:005930->005930.KS. A bare or already-suffixed symbol is
    returned unchanged (the curated universe lists already store suffixed forms)."""
    market, t = parse_namespaced_ticker(namespaced)
    suffix = YF_SUFFIX.get(market, "")
    if not suffix or t.endswith(suffix):
        return t
    return f"{t}{suffix}"


def market_for_symbol(symbol: str) -> str:
    """phase-50.3: derive the market code from a yfinance symbol's suffix
    (the suffix IS the source of truth -- AIR.PA is EU despite market='EU' not
    yielding .PA). Bare ticker -> US. .KS/.KQ -> KR; .DE/.PA/.AS/.F -> EU;
    .OL -> NO; .TO -> CA."""
    s = (symbol or "").upper()
    if s.endswith((".KS", ".KQ")):
        return "KR"
    if s.endswith((".DE", ".PA", ".AS", ".F")):
        return "EU"
    if s.endswith(".OL"):
        return "NO"
    # goal-multimarket-ux: Nordic suffixes (Stockholm/Copenhagen/Helsinki/Iceland).
    if s.endswith(".ST"):
        return "SE"
    if s.endswith(".CO"):
        return "DK"
    if s.endswith(".HE"):
        return "FI"
    if s.endswith(".IC"):
        return "IS"
    if s.endswith(".TO"):
        return "CA"
    return DEFAULT_MARKET


def get_trading_calendar(market: str = DEFAULT_MARKET):
    """
    Get exchange trading calendar for market.

    Args:
        market: Market code (US, NO, CA, EU, KR)

    Returns:
        exchange_calendars calendar object, or None if xcals not installed
    """
    if not xcals:
        logger.warning("exchange_calendars not installed, returning None")
        return None

    config = get_market_config(market)
    exchange = config["exchange"]

    try:
        return xcals.get_calendar(exchange)
    except Exception as e:
        logger.error(f"Failed to load calendar for {exchange}: {e}")
        return None


def is_trading_day(date, market: str = DEFAULT_MARKET) -> bool:
    """Check if `date` is a trading session for the market's exchange.

    phase-50.4: rewritten to exchange_calendars 4.x `cal.is_session()`. The
    previous body used `date in cal.days`, but `.days` was removed in
    exchange_calendars 4.0 -> the bare except swallowed the AttributeError and
    this ALWAYS returned True (a latent no-op gate; it had zero live callers).
    `is_session` requires a tz-NAIVE session label (midnight); it rejects
    tz-aware Timestamps. Fail-open (return True) when the calendar lib is
    unavailable -- never block a trade because exchange_calendars is missing."""
    cal = get_trading_calendar(market)
    if cal is None:
        return True  # fail-open: calendar unavailable
    try:
        import pandas as pd
        ts = pd.Timestamp(date)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)  # is_session rejects tz-aware labels
        return bool(cal.is_session(ts.normalize()))
    except Exception as e:
        logger.warning("Calendar check failed for %s/%s: %s; assuming trading day", market, date, e)
        return True


def is_us_trading_day_now(market: str = DEFAULT_MARKET) -> bool:
    """phase-86.109: True iff TODAY, in the exchange's own timezone, is a
    trading session.

    **This is the ONE definition.** It was extracted from
    `backend/slack_bot/scheduler.py::_is_us_trading_day_now` (phase-51.3, which
    gates the morning/evening digests) so that a second consumer -- the
    data-freshness notifier -- could reuse it rather than grow a parallel copy.
    Step 86.109's criterion 2 names that risk explicitly, and it is not
    hypothetical: `backend/services/cycle_health.py` already carries three
    separate calendar notions, one of which (`is_weekday_et`) is holiday-BLIND,
    so a fourth would have drifted from the day it was written.

    Timezone: America/New_York, because the sessions being gated are US
    exchange sessions and the digest cron already runs on ET. A UTC "today"
    would be the wrong day for five hours of every evening.

    **Fail-open, deliberately.** `is_trading_day` returns True when
    `exchange_calendars` is unavailable, so a calendar-library failure can
    never SUPPRESS a page or a digest. For a notification gate the safe
    direction is to notify: a spurious page is noise, a suppressed one is a
    silent failure.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et_today = datetime.now(ZoneInfo("America/New_York")).date()
    return is_trading_day(et_today, market)
