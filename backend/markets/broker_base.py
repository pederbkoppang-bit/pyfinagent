"""phase-5.1 Broker abstraction base class + shared dataclasses.

This is the foundational interface that all multi-asset broker
implementations (AlpacaBroker today; OandaBroker / IBKRBroker in 5.7+)
plug into. The abstract `BrokerClient` ABC enforces the 6 methods every
broker must provide; subclasses missing any of them fail at
instantiation time with `TypeError`.

Design choice: `abc.ABC` (not `typing.Protocol`) because pyfinagent
owns all subclasses and we want runtime enforcement of the contract.
See `handoff/archive/phase-5.1/research-brief.md` for the full
ABC-vs-Protocol justification (7 sources read in full, 2026 best
practice consensus on internal class hierarchies).

`FillResult` is re-exported from `backend.services.execution_router`
to avoid duplicate definitions and keep the max-notional clamp /
live-key guard chain intact when AlpacaBroker delegates submit_order.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from backend.services.execution_router import FillResult

__all__ = [
    "AccountInfo",
    "BrokerClient",
    "FillResult",
    "OrderInfo",
    "PositionInfo",
    "QuoteInfo",
]


@dataclass(frozen=True)
class AccountInfo:
    """Snapshot of broker account state. Currency is ISO-4217 (default USD)."""

    buying_power: float
    equity: float
    cash: float
    currency: str = "USD"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionInfo:
    """One open position. `side` is 'long' or 'short'."""

    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    side: str = "long"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderInfo:
    """One order record. `status` follows broker conventions
    ('new' / 'filled' / 'partially_filled' / 'canceled' / 'rejected')."""

    order_id: str
    client_order_id: str
    symbol: str
    qty: float
    side: str
    status: str
    filled_avg_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuoteInfo:
    """Latest NBBO snapshot for one symbol."""

    symbol: str
    ask: float
    bid: float
    last: float
    raw: dict[str, Any] = field(default_factory=dict)


class BrokerClient(abc.ABC):
    """Abstract base for every concrete broker implementation.

    Subclasses MUST implement all 6 abstract methods or instantiation
    raises TypeError. Methods that interact with live broker APIs MUST
    fail-open in test environments where credentials are absent --
    return safe defaults / log warnings, do NOT raise. This keeps
    test suites and CI pipelines functional without env injection.

    Concrete implementations (AlpacaBroker, future OandaBroker /
    IBKRBroker) should keep heavy I/O lazy: do NOT call broker APIs
    in `__init__`. Module-level network calls or env reads are also
    forbidden (the masterplan immutable verification command imports
    these modules in a clean environment).
    """

    @abc.abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
        **kwargs: Any,
    ) -> FillResult:
        """Place a market order. Returns a `FillResult` (shape-compatible
        with `backend.services.execution_router.FillResult`)."""

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success, False otherwise.
        Must NOT raise when creds are absent or the order does not exist."""

    @abc.abstractmethod
    def get_account(self) -> AccountInfo:
        """Return the current account snapshot. In creds-absent
        environments, return a zero-value AccountInfo (do NOT raise)."""

    @abc.abstractmethod
    def get_positions(self) -> list[PositionInfo]:
        """Return all open positions. Empty list when none / no creds."""

    @abc.abstractmethod
    def get_orders(self, status: str | None = None) -> list[OrderInfo]:
        """Return orders, optionally filtered by status. Empty list
        when none / no creds."""

    @abc.abstractmethod
    def get_quote(self, symbol: str) -> QuoteInfo:
        """Return the latest NBBO snapshot. In creds-absent
        environments, return a zero-value QuoteInfo (do NOT raise)."""
