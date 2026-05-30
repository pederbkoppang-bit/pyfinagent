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

# Market configuration: exchange calendar, currency, timezone
MARKET_CONFIG = {
    "US": {
        "exchange": "XNYS",
        "currency": "USD",
        "timezone": "America/New_York",
        "description": "NYSE/NASDAQ (United States)",
    },
    "NO": {
        "exchange": "XOSL",
        "currency": "NOK",
        "timezone": "Europe/Oslo",
        "description": "Oslo Børs (Norway)",
    },
    "CA": {
        "exchange": "XTSE",
        "currency": "CAD",
        "timezone": "America/Toronto",
        "description": "Toronto Stock Exchange (Canada)",
    },
    "EU": {
        "exchange": "XETR",  # XETRA (Germany) as primary
        "currency": "EUR",
        "timezone": "Europe/Berlin",
        "description": "XETRA/Euronext (European equities)",
    },
    "KR": {
        "exchange": "XKRX",
        "currency": "KRW",
        "timezone": "Asia/Seoul",
        "description": "KRX - KOSPI/KOSDAQ (South Korea)",
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
YF_SUFFIX = {"US": "", "NO": ".OL", "CA": ".TO", "EU": ".DE", "KR": ".KS"}


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
    """Check if date is a trading day in the specified market."""
    cal = get_trading_calendar(market)
    if cal is None:
        return True  # Safe default if calendar unavailable

    from datetime import datetime

    if isinstance(date, str):
        date = datetime.fromisoformat(date).date()

    try:
        return date in cal.days
    except Exception as e:
        logger.warning(f"Calendar check failed: {e}, assuming trading day")
        return True
