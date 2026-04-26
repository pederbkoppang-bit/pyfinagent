"""phase-5.1 backend.markets -- multi-asset broker abstraction package.

Exposes:
- BrokerClient (abstract base)
- AlpacaBroker (concrete; US equity)
- get_broker(market, asset_class) -- factory

Future steps add OandaBroker (5.7), IBKRBroker (5.8), etc. by appending
to `_REGISTRY`. No module-level network calls or env reads -- safe to
import in any environment.
"""
from __future__ import annotations

from backend.markets.alpaca_broker import AlpacaBroker
from backend.markets.broker_base import (
    AccountInfo,
    BrokerClient,
    FillResult,
    OrderInfo,
    PositionInfo,
    QuoteInfo,
)

__all__ = [
    "AccountInfo",
    "AlpacaBroker",
    "BrokerClient",
    "FillResult",
    "OrderInfo",
    "PositionInfo",
    "QuoteInfo",
    "get_broker",
]


_REGISTRY: dict[tuple[str, str], type[BrokerClient]] = {
    ("US", "equity"): AlpacaBroker,
}


def get_broker(market: str, asset_class: str) -> BrokerClient:
    """Return a broker instance for (market, asset_class).

    `market` is normalized to upper-case ISO country code; `asset_class`
    to lower-case ('equity' / 'option' / 'fx' / 'future').

    Raises ValueError if the (market, asset_class) tuple is not
    registered. Adding a new broker is a one-line addition to _REGISTRY.
    """
    key = (market.upper(), asset_class.lower())
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"no broker registered for (market={market!r}, "
            f"asset_class={asset_class!r}); known: {sorted(_REGISTRY.keys())}"
        )
    return cls()
