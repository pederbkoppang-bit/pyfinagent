"""
Market definitions and configurations for multi-market support.

Phase 2.9: Lightweight preparation for future market expansion (CA, EU, NO, KR).
Currently defaults to US; other markets are data structures only.
"""

from enum import Enum
from typing import NamedTuple


class MarketConfig(NamedTuple):
    """Market configuration tuple."""
    code: str  # Market code (US, CA, EU, NO, KR)
    currency: str  # Primary currency (USD, CAD, EUR, NOK, KRW)
    exchange: str  # Primary exchange code (NYSE, TSX, XETRA, OSE, KRX)
    timezone: str  # IANA timezone (e.g., America/New_York)
    calendar: str  # exchange_calendars name (e.g., 'NYSE', 'TSX', 'XETRA', 'OSE', 'XKRX')


class Market(Enum):
    """Supported markets and their configurations."""
    
    US = MarketConfig(
        code="US",
        currency="USD",
        exchange="NYSE",
        timezone="America/New_York",
        calendar="NYSE"
    )
    
    CA = MarketConfig(
        code="CA",
        currency="CAD",
        exchange="TSX",
        timezone="America/Toronto",
        calendar="TSX"
    )
    
    EU = MarketConfig(
        code="EU",
        currency="EUR",
        exchange="XETRA",
        timezone="Europe/Berlin",
        calendar="XETRA"
    )
    
    NO = MarketConfig(
        code="NO",
        currency="NOK",
        exchange="OSE",
        timezone="Europe/Oslo",
        calendar="OSE"
    )
    
    KR = MarketConfig(
        code="KR",
        currency="KRW",
        exchange="KRX",
        timezone="Asia/Seoul",
        calendar="XKRX"
    )


# Default market (US only for now)
DEFAULT_MARKET = "US"
DEFAULT_MARKET_CONFIG = Market.US.value


def get_market_config(market: str) -> MarketConfig:
    """Get configuration for a market code.
    
    Args:
        market: Market code (e.g., 'US', 'CA')
        
    Returns:
        MarketConfig with all settings
        
    Raises:
        ValueError: If market is not supported
    """
    try:
        return Market[market].value
    except KeyError:
        supported = ", ".join(m.name for m in Market)
        raise ValueError(f"Unsupported market '{market}'. Supported: {supported}")


def get_timezone(market: str) -> str:
    """Get timezone for a market."""
    return get_market_config(market).timezone


def get_calendar(market: str) -> str:
    """Get exchange_calendars name for a market."""
    return get_market_config(market).calendar


def get_currency(market: str) -> str:
    """Get base currency for a market."""
    return get_market_config(market).currency
