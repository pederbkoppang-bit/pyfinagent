"""phase-23.1.15: trade idempotency guard + paper_positions MERGE upsert."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.services.paper_trader import PaperTrader


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        paper_max_positions=15,
        paper_starting_capital=15000.0,
        paper_min_cash_reserve_pct=0.0,
        paper_default_stop_loss_pct=8.0,
        paper_transaction_cost_pct=0.1,
        paper_alpaca_paper_enabled=False,
    )


def _bq_mock(recent_buys: list[dict] | None = None) -> MagicMock:
    bq = MagicMock()
    bq.get_or_create_paper_portfolio.return_value = {
        "portfolio_id": "default",
        "current_cash": 10_000.0,
    }
    bq.get_paper_positions.return_value = []
    bq.get_paper_trades_for_ticker_since.return_value = recent_buys or []
    return bq


def test_idempotency_guard_blocks_duplicate_buy_within_window(monkeypatch):
    """A BUY of the same ticker at near-identical qty within 30 min must be skipped."""
    # Existing recent BUY: 2.351361 shares of WDC at $404 = $949.95 ten minutes ago.
    recent_qty = 2.351361
    recent_iso = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    bq = _bq_mock(recent_buys=[{
        "trade_id": "prior-uuid",
        "ticker": "WDC",
        "action": "BUY",
        "quantity": recent_qty,
        "created_at": recent_iso,
    }])

    # Patch portfolio cash readback path.
    pt = PaperTrader(_settings(), bq)
    monkeypatch.setattr(pt, "get_or_create_portfolio",
                        lambda: {"current_cash": 10_000.0, "portfolio_id": "default"})

    # Propose an identical BUY: $949.95 / $404 = 2.3514...
    result = pt.execute_buy(ticker="WDC", amount_usd=949.95, price=404.0)

    assert result is None, "idempotency guard must short-circuit duplicate BUY"
    bq.save_paper_trade.assert_not_called()
    bq.save_paper_position.assert_not_called()


def test_idempotency_guard_allows_different_qty(monkeypatch):
    """A BUY with quantity >1% different from the recent one is allowed through."""
    # Prior BUY at qty 1.0; proposed BUY at qty 5.0 — clearly different size.
    recent_iso = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    bq = _bq_mock(recent_buys=[{
        "trade_id": "prior-uuid",
        "ticker": "WDC",
        "action": "BUY",
        "quantity": 1.0,
        "created_at": recent_iso,
    }])
    pt = PaperTrader(_settings(), bq)
    monkeypatch.setattr(pt, "get_or_create_portfolio",
                        lambda: {"current_cash": 10_000.0, "portfolio_id": "default"})
    monkeypatch.setattr(pt, "_update_portfolio_cash", lambda _c: None)
    # Force the deterministic price path (skip ExecutionRouter).
    monkeypatch.setattr(
        "backend.services.paper_trader.ExecutionRouter",
        lambda: SimpleNamespace(submit_order=lambda **kw: SimpleNamespace(
            fill_price=kw["close_price"], source="bq_sim",
        )),
    )

    result = pt.execute_buy(ticker="WDC", amount_usd=2020.0, price=404.0)  # qty=5.0
    assert result is not None, "5x-larger BUY must NOT be flagged as duplicate"
    bq.save_paper_trade.assert_called_once()


def test_save_paper_position_uses_merge():
    """Smoke test: save_paper_position issues a MERGE statement, not plain INSERT."""
    from backend.db.bigquery_client import BigQueryClient

    settings = SimpleNamespace(
        gcp_project_id="test-proj",
        bq_dataset_reports="test_ds",
    )
    bq = BigQueryClient.__new__(BigQueryClient)
    bq.settings = settings
    bq.client = MagicMock()
    bq.client.query.return_value.result.return_value = None

    bq.save_paper_position({
        "position_id": "p1",
        "ticker": "AAPL",
        "quantity": 1.0,
        "avg_entry_price": 100.0,
        "cost_basis": 100.0,
        "current_price": 100.0,
        "market_value": 100.0,
        "entry_date": "2026-04-29T00:00:00+00:00",
    })

    assert bq.client.query.called
    sent_sql = bq.client.query.call_args[0][0]
    assert "MERGE" in sent_sql, f"expected MERGE in SQL, got: {sent_sql[:200]}"
    assert "ON T.ticker = S.ticker" in sent_sql
    assert "WHEN MATCHED" in sent_sql and "WHEN NOT MATCHED" in sent_sql


def test_save_paper_position_rejects_missing_ticker():
    """MERGE requires the ticker key for the merge predicate."""
    from backend.db.bigquery_client import BigQueryClient
    settings = SimpleNamespace(gcp_project_id="p", bq_dataset_reports="d")
    bq = BigQueryClient.__new__(BigQueryClient)
    bq.settings = settings
    bq.client = MagicMock()
    with pytest.raises(ValueError, match="requires 'ticker' field"):
        bq.save_paper_position({"position_id": "p1", "quantity": 1.0})
