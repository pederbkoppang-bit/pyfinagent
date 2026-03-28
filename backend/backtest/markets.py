"""
Multi-market abstractions (Phase 2.9).

Lightweight layer that prepares the codebase for future market expansion
(NO, CA, EU, KR) without implementing actual multi-market support yet.
All defaults point to US — zero behavioral change for existing code.

Ticker namespacing: {MARKET}:{TICKER}  e.g. US:AAPL, NO:EQNR, KR:005930
"""

from dataclasses import dataclass

# ── Market Definitions ───────────────────────────────────────────────

@dataclass(frozen=True)
class MarketDef:
    """Definition of a supported market."""
    code: str               # Short code (US, NO, CA, DE, KR)
    name: str               # Human-readable name
    currency: str           # ISO 4217 currency code
    exchange_cal: str       # exchange_calendars calendar name
    default_tx_cost_bps: int  # Default transaction cost in basis points (round-trip)
    timezone: str           # Market timezone

MARKETS: dict[str, MarketDef] = {
    "US": MarketDef(
        code="US", name="United States (NYSE/NASDAQ)", currency="USD",
        exchange_cal="XNYS", default_tx_cost_bps=10, timezone="America/New_York",
    ),
    "NO": MarketDef(
        code="NO", name="Norway (Oslo Børs)", currency="NOK",
        exchange_cal="XOSL", default_tx_cost_bps=15, timezone="Europe/Oslo",
    ),
    "CA": MarketDef(
        code="CA", name="Canada (TSX)", currency="CAD",
        exchange_cal="XTSE", default_tx_cost_bps=12, timezone="America/Toronto",
    ),
    "DE": MarketDef(
        code="DE", name="Germany (XETRA)", currency="EUR",
        exchange_cal="XETR", default_tx_cost_bps=12, timezone="Europe/Berlin",
    ),
    "KR": MarketDef(
        code="KR", name="South Korea (KRX)", currency="KRW",
        exchange_cal="XKRX", default_tx_cost_bps=15, timezone="Asia/Seoul",
    ),
}

DEFAULT_MARKET = "US"


# ── Ticker Namespacing ───────────────────────────────────────────────

def namespace_ticker(ticker: str, market: str = DEFAULT_MARKET) -> str:
    """Add market namespace to a ticker if not already namespaced.
    
    >>> namespace_ticker("AAPL")
    'US:AAPL'
    >>> namespace_ticker("US:AAPL")
    'US:AAPL'
    >>> namespace_ticker("EQNR", "NO")
    'NO:EQNR'
    """
    if ":" in ticker:
        return ticker
    return f"{market}:{ticker}"


def parse_ticker(namespaced: str) -> tuple[str, str]:
    """Parse a namespaced ticker into (market, ticker).
    
    >>> parse_ticker("US:AAPL")
    ('US', 'AAPL')
    >>> parse_ticker("AAPL")
    ('US', 'AAPL')
    """
    if ":" in namespaced:
        market, ticker = namespaced.split(":", 1)
        return market.upper(), ticker
    return DEFAULT_MARKET, namespaced


def strip_namespace(namespaced: str) -> str:
    """Strip market namespace from a ticker.
    
    >>> strip_namespace("US:AAPL")
    'AAPL'
    >>> strip_namespace("AAPL")
    'AAPL'
    """
    _, ticker = parse_ticker(namespaced)
    return ticker


# ── Trading Calendar ─────────────────────────────────────────────────

def get_trading_days(start: str, end: str, market: str = DEFAULT_MARKET):
    """Get valid trading days for a market.
    
    Uses exchange_calendars if available, falls back to pandas bdate_range.
    Returns a DatetimeIndex.
    """
    import pandas as pd
    
    try:
        import exchange_calendars as xcals
        market_def = MARKETS.get(market, MARKETS[DEFAULT_MARKET])
        cal = xcals.get_calendar(market_def.exchange_cal)
        sessions = cal.sessions_in_range(start, end)
        return sessions
    except ImportError:
        # Fallback: pandas business days (assumes NYSE-like schedule)
        return pd.bdate_range(start, end)
    except Exception:
        # Any calendar error: fallback to pandas
        import logging
        logging.getLogger(__name__).warning(
            "exchange_calendars failed for %s, falling back to bdate_range", market
        )
        return pd.bdate_range(start, end)


def get_market_def(market: str = DEFAULT_MARKET) -> MarketDef:
    """Get market definition, defaulting to US if unknown."""
    return MARKETS.get(market.upper(), MARKETS[DEFAULT_MARKET])


# ── Currency (passthrough for now) ───────────────────────────────────

def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    date: str | None = None,
) -> float:
    """Convert between currencies.
    
    PHASE 2.9: Passthrough — returns amount unchanged.
    PHASE 5+: Will use FX rate feed.
    """
    if from_currency == to_currency:
        return amount
    # TODO Phase 5: implement real FX conversion
    # For now, log a warning and return the amount unchanged
    import logging
    logging.getLogger(__name__).warning(
        "Currency conversion %s→%s not implemented yet (Phase 5). Returning unconverted amount.",
        from_currency, to_currency,
    )
    return amount
