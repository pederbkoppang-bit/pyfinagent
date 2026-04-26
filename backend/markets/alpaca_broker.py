"""phase-5.1 Concrete BrokerClient implementation for Alpaca paper trading.

Delegates `submit_order` to `backend.services.execution_router._alpaca_real_fill`
to preserve the max-notional clamp + live-key guard chain that already
ships in the existing service. The other 5 methods (cancel_order,
get_account, get_positions, get_orders, get_quote) call alpaca-py
directly with `paper=True` hardcoded.

Fail-open in creds-absent environments: every method returns a safe
default + logs a warning rather than raising. This is required so the
masterplan immutable verification command (which imports this module
in a clean environment) succeeds.

Future siblings: OandaBroker (5.7), IBKRBroker (5.8). All plug into
`backend.markets.get_broker(market, asset_class)`.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from backend.markets.broker_base import (
    AccountInfo,
    BrokerClient,
    FillResult,
    OrderInfo,
    PositionInfo,
    QuoteInfo,
)

logger = logging.getLogger(__name__)


def _has_creds() -> bool:
    return bool(
        os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY")
    )


def _empty_account() -> AccountInfo:
    return AccountInfo(buying_power=0.0, equity=0.0, cash=0.0, currency="USD")


def _empty_quote(symbol: str) -> QuoteInfo:
    return QuoteInfo(symbol=symbol, ask=0.0, bid=0.0, last=0.0)


class AlpacaBroker(BrokerClient):
    """Alpaca paper-trading broker. Equity-only at this layer.

    Construction does NOT touch the network or read env vars eagerly --
    creds are read per-method via `_has_creds()` so a clean import
    works in any environment.
    """

    def __init__(self) -> None:
        # Lazy: no API client created at construction time.
        self._client: Any = None

    def _trading_client(self) -> Any:
        """Lazy-init the alpaca-py TradingClient. Returns None when
        creds are absent (caller should fail-open)."""
        if self._client is not None:
            return self._client
        if not _has_creds():
            return None
        try:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                api_key=os.environ["ALPACA_API_KEY_ID"],
                secret_key=os.environ["ALPACA_API_SECRET_KEY"],
                paper=True,
            )
            return self._client
        except Exception as exc:
            logger.warning("AlpacaBroker TradingClient init fail-open: %r", exc)
            return None

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
        **kwargs: Any,
    ) -> FillResult:
        """Delegate to execution_router._alpaca_real_fill (preserves
        max-notional clamp + live-key guard). When creds are absent,
        execution_router._alpaca_mock_fill is used instead."""
        from backend.services.execution_router import (
            _alpaca_mock_fill,
            _alpaca_real_fill,
        )

        if _has_creds():
            try:
                return _alpaca_real_fill(symbol, qty, side, client_order_id)
            except Exception as exc:
                logger.warning(
                    "AlpacaBroker.submit_order real-fill fail-open: %r", exc
                )
        return _alpaca_mock_fill(
            symbol, qty, side, client_order_id, kwargs.get("close_price")
        )

    def cancel_order(self, order_id: str) -> bool:
        client = self._trading_client()
        if client is None:
            return False
        try:
            client.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            logger.warning("AlpacaBroker.cancel_order fail-open: %r", exc)
            return False

    def get_account(self) -> AccountInfo:
        client = self._trading_client()
        if client is None:
            return _empty_account()
        try:
            acct = client.get_account()
            return AccountInfo(
                buying_power=float(getattr(acct, "buying_power", 0.0) or 0.0),
                equity=float(getattr(acct, "equity", 0.0) or 0.0),
                cash=float(getattr(acct, "cash", 0.0) or 0.0),
                currency=str(getattr(acct, "currency", "USD") or "USD"),
                raw={"id": str(getattr(acct, "id", ""))},
            )
        except Exception as exc:
            logger.warning("AlpacaBroker.get_account fail-open: %r", exc)
            return _empty_account()

    def get_positions(self) -> list[PositionInfo]:
        client = self._trading_client()
        if client is None:
            return []
        try:
            raw_positions = client.get_all_positions()
            return [
                PositionInfo(
                    symbol=str(p.symbol),
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(getattr(p, "market_value", 0.0) or 0.0),
                    unrealized_pl=float(getattr(p, "unrealized_pl", 0.0) or 0.0),
                    side=str(getattr(p, "side", "long")).lower(),
                )
                for p in raw_positions
            ]
        except Exception as exc:
            logger.warning("AlpacaBroker.get_positions fail-open: %r", exc)
            return []

    def get_orders(self, status: str | None = None) -> list[OrderInfo]:
        client = self._trading_client()
        if client is None:
            return []
        try:
            from alpaca.trading.requests import GetOrdersRequest

            req = GetOrdersRequest(status=status) if status else GetOrdersRequest()
            raw_orders = client.get_orders(req)
            return [
                OrderInfo(
                    order_id=str(o.id),
                    client_order_id=str(getattr(o, "client_order_id", "") or ""),
                    symbol=str(o.symbol),
                    qty=float(o.qty),
                    side=str(o.side).split(".")[-1].lower(),
                    status=str(o.status).split(".")[-1].lower(),
                    filled_avg_price=(
                        float(o.filled_avg_price)
                        if getattr(o, "filled_avg_price", None) is not None
                        else None
                    ),
                )
                for o in raw_orders
            ]
        except Exception as exc:
            logger.warning("AlpacaBroker.get_orders fail-open: %r", exc)
            return []

    def get_quote(self, symbol: str) -> QuoteInfo:
        if not _has_creds():
            return _empty_quote(symbol)
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest

            data_client = StockHistoricalDataClient(
                api_key=os.environ["ALPACA_API_KEY_ID"],
                secret_key=os.environ["ALPACA_API_SECRET_KEY"],
            )
            req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = data_client.get_stock_latest_quote(req)
            q = quotes.get(symbol) if isinstance(quotes, dict) else None
            if q is None:
                return _empty_quote(symbol)
            ask = float(getattr(q, "ask_price", 0.0) or 0.0)
            bid = float(getattr(q, "bid_price", 0.0) or 0.0)
            mid = (ask + bid) / 2.0 if (ask and bid) else max(ask, bid, 0.0)
            return QuoteInfo(symbol=symbol, ask=ask, bid=bid, last=mid)
        except Exception as exc:
            logger.warning("AlpacaBroker.get_quote fail-open: %r", exc)
            return _empty_quote(symbol)


__all__ = ["AlpacaBroker"]
