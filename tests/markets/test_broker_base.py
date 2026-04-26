"""phase-5.1 unit tests for the broker abstraction layer.

12 tests covering ABC enforcement, AlpacaBroker subclass + factory
dispatch, fail-open behavior in creds-absent environments, and
no-regression on existing FillResult shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.markets import (  # noqa: E402
    AccountInfo,
    AlpacaBroker,
    BrokerClient,
    FillResult,
    OrderInfo,
    PositionInfo,
    QuoteInfo,
    get_broker,
)
from backend.services.execution_router import FillResult as RouterFillResult  # noqa: E402


# ----------------------
# ABC enforcement
# ----------------------

def test_brokerbase_cannot_instantiate():
    """BrokerClient is abstract -- direct instantiation raises TypeError."""
    with pytest.raises(TypeError):
        BrokerClient()  # type: ignore[abstract]


def test_alpacabroker_is_subclass():
    assert issubclass(AlpacaBroker, BrokerClient)


def test_alpacabroker_instantiation():
    """AlpacaBroker construction is lazy -- no network or env reads."""
    broker = AlpacaBroker()
    assert isinstance(broker, BrokerClient)


def test_incomplete_subclass_raises():
    """A subclass missing abstract methods cannot be instantiated."""

    class _BrokenBroker(BrokerClient):
        # Missing 5 of the 6 abstract methods.
        def submit_order(self, symbol, qty, side, client_order_id, **kwargs):
            return FillResult(
                client_order_id=client_order_id,
                symbol=symbol,
                qty=qty,
                side=side,
                fill_price=0.0,
                status="rejected",
                source="test",
            )

    with pytest.raises(TypeError):
        _BrokenBroker()  # type: ignore[abstract]


# ----------------------
# Factory dispatch
# ----------------------

def test_get_broker_us_equity():
    broker = get_broker("US", "equity")
    assert isinstance(broker, AlpacaBroker)
    assert isinstance(broker, BrokerClient)


def test_get_broker_unknown_raises():
    with pytest.raises(ValueError, match="no broker registered"):
        get_broker("UK", "equity")
    with pytest.raises(ValueError):
        get_broker("US", "crypto")


def test_get_broker_case_insensitive():
    """Market normalises to upper, asset_class to lower."""
    a = get_broker("us", "EQUITY")
    b = get_broker("US", "Equity")
    assert isinstance(a, AlpacaBroker)
    assert isinstance(b, AlpacaBroker)


# ----------------------
# FillResult shape (no duplication)
# ----------------------

def test_fillresult_is_not_duplicated():
    """The package re-exports execution_router.FillResult -- must be the SAME class."""
    assert FillResult is RouterFillResult


# ----------------------
# Fail-open behavior in creds-absent environment
# ----------------------

def test_get_account_no_creds_returns_empty(monkeypatch):
    """No creds -> AlpacaBroker.get_account() returns zero-value AccountInfo, no raise."""
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    broker = AlpacaBroker()
    out = broker.get_account()
    assert isinstance(out, AccountInfo)
    assert out.buying_power == 0.0
    assert out.equity == 0.0
    assert out.currency == "USD"


def test_get_positions_no_creds_returns_empty(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    assert AlpacaBroker().get_positions() == []


def test_get_orders_no_creds_returns_empty(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    assert AlpacaBroker().get_orders() == []


def test_get_quote_no_creds_returns_empty(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    out = AlpacaBroker().get_quote("AAPL")
    assert isinstance(out, QuoteInfo)
    assert out.symbol == "AAPL"
    assert out.ask == 0.0


def test_cancel_order_no_creds_returns_false(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    assert AlpacaBroker().cancel_order("any-id") is False


# ----------------------
# Submit order via mock-fill path (no creds)
# ----------------------

def test_submit_order_no_creds_returns_mock_fill(monkeypatch):
    """No creds -> AlpacaBroker.submit_order returns a mock FillResult, no raise."""
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    broker = AlpacaBroker()
    out = broker.submit_order("AAPL", 1.0, "buy", "test-coid-1", close_price=150.0)
    assert isinstance(out, FillResult)
    assert out.symbol == "AAPL"
    assert out.qty == 1.0
    assert out.side == "buy"
    # mock fill source label per execution_router._alpaca_mock_fill
    assert out.source.startswith("mock") or out.source == "alpaca_mock"


# ----------------------
# Existing-service regression
# ----------------------

def test_paper_trader_import_regression():
    """paper_trader.py must still import after backend.markets is added."""
    from backend.services import paper_trader  # noqa: F401
    from backend.services.execution_router import ExecutionRouter  # noqa: F401
