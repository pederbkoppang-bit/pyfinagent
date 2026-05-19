"""phase-30.6 tests for paper_trader.execute_buy price-tolerance gate.

Audit basis: handoff/archive/phase-30.0/experiment_results.md cross-val
6.1 + P2-4: FIA WP July 2024 Sec 1.3 names "Price Tolerance" as a
canonical pre-trade gate; pyfinagent has none. Default 5% per SEC LULD
Tier 1 band for S&P 500 + Russell 1000 > $3 (the pyfinagent universe).

Test plan (5 cases):
  1. Pass: 1% deviation -> BUY succeeds.
  2. Reject up: live +10% over analysis -> BUY rejected.
  3. Reject down: live -10% under analysis -> BUY rejected.
  4. Disable: tolerance = 0 -> gate is a no-op.
  5. None analysis price -> gate is skipped (fail-open for lite-Claude
     path).
  6. Grep-equivalent: `paper_price_tolerance_pct` in settings.py +
     `price_tolerance` in paper_trader.py (masterplan verification
     command sanity).

The trade-execution side-effects (BQ writes, portfolio mutations) are
mocked; these tests cover the gate's reject/pass logic only.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.services.paper_trader import PaperTrader


def _mock_settings(
    price_tolerance_pct: float = 5.0,
    default_stop_loss_pct: float = 8.0,
    max_positions: int = 10,
    transaction_cost_pct: float = 0.05,
) -> SimpleNamespace:
    return SimpleNamespace(
        paper_price_tolerance_pct=price_tolerance_pct,
        paper_default_stop_loss_pct=default_stop_loss_pct,
        paper_max_positions=max_positions,
        paper_transaction_cost_pct=transaction_cost_pct,
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
    )


def _trader_with_mocks(settings: SimpleNamespace) -> PaperTrader:
    """Build a PaperTrader with a heavily-mocked BQ client so we can
    exercise execute_buy without touching BigQuery."""
    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "portfolio_id": "default",
        "current_cash": 5000.0,
        "starting_capital": 10000.0,
        "inception_date": "2026-05-01T00:00:00+00:00",
    }
    bq.get_paper_positions.return_value = []
    bq.get_paper_trades_for_ticker_since.return_value = []
    bq.save_paper_trade.return_value = None
    bq.save_paper_position.return_value = None
    bq.save_paper_portfolio.return_value = None
    return PaperTrader(settings=settings, bq_client=bq)


# ---------------------------------------------------------------------
# 1. Pass branch: 1% deviation
# ---------------------------------------------------------------------
def test_price_tolerance_pass_1pct_deviation():
    """phase-30.6 Test A: 1% live-vs-analysis deviation passes
    (tolerance = 5%)."""
    settings = _mock_settings(price_tolerance_pct=5.0)
    trader = _trader_with_mocks(settings)

    with patch(
        "backend.services.paper_trader.ExecutionRouter"
    ) as mock_router_cls:
        mock_router = mock_router_cls.return_value
        mock_router.submit_order.return_value = SimpleNamespace(
            fill_price=101.0,
            source="bq_sim",
        )
        trade = trader.execute_buy(
            ticker="WDC",
            amount_usd=500.0,
            price=101.0,            # live
            price_at_analysis=100.0,  # analysis-time; 1% deviation
        )

    assert trade is not None, (
        "phase-30.6 Test A: 1% deviation must pass the 5% gate; got reject"
    )
    assert trade["ticker"] == "WDC"
    assert trade["action"] == "BUY"


# ---------------------------------------------------------------------
# 2. Reject up: live +10% over analysis
# ---------------------------------------------------------------------
def test_price_tolerance_reject_live_10pct_above_analysis():
    """phase-30.6 Test B: live price 10% above analysis-time -> reject."""
    settings = _mock_settings(price_tolerance_pct=5.0)
    trader = _trader_with_mocks(settings)

    trade = trader.execute_buy(
        ticker="WDC",
        amount_usd=500.0,
        price=110.0,             # live (+10%)
        price_at_analysis=100.0,   # analysis-time
    )

    assert trade is None, (
        "phase-30.6 Test B: live +10% over analysis must be rejected "
        f"by the 5% gate; got trade={trade}"
    )


# ---------------------------------------------------------------------
# 3. Reject down: live -10% under analysis
# ---------------------------------------------------------------------
def test_price_tolerance_reject_live_10pct_below_analysis():
    """phase-30.6 Test C: live price 10% below analysis-time -> reject
    (gate is symmetric; protects both up-spikes and crash entries)."""
    settings = _mock_settings(price_tolerance_pct=5.0)
    trader = _trader_with_mocks(settings)

    trade = trader.execute_buy(
        ticker="WDC",
        amount_usd=500.0,
        price=90.0,              # live (-10%)
        price_at_analysis=100.0,   # analysis-time
    )

    assert trade is None, (
        "phase-30.6 Test C: live -10% under analysis must be rejected "
        f"by the 5% gate (symmetric); got trade={trade}"
    )


# ---------------------------------------------------------------------
# 4. Disable: tolerance = 0 -> gate is no-op
# ---------------------------------------------------------------------
def test_price_tolerance_zero_disables_gate():
    """phase-30.6 Test D: tolerance=0 disables the gate (legacy
    behavior preserved -- mirrors the 0-disables convention used
    elsewhere in settings)."""
    settings = _mock_settings(price_tolerance_pct=0.0)
    trader = _trader_with_mocks(settings)

    with patch(
        "backend.services.paper_trader.ExecutionRouter"
    ) as mock_router_cls:
        mock_router = mock_router_cls.return_value
        mock_router.submit_order.return_value = SimpleNamespace(
            fill_price=200.0,
            source="bq_sim",
        )
        trade = trader.execute_buy(
            ticker="WDC",
            amount_usd=500.0,
            price=200.0,             # +100% would be rejected if gate enabled
            price_at_analysis=100.0,
        )

    assert trade is not None, (
        "phase-30.6 Test D: tolerance=0 must disable gate even on +100% "
        f"deviation; got reject"
    )


# ---------------------------------------------------------------------
# 5. None analysis price -> gate skipped (fail-open)
# ---------------------------------------------------------------------
def test_price_tolerance_skipped_when_analysis_price_missing():
    """phase-30.6 Test E: when price_at_analysis is None (lite-Claude
    path that lacks a written analysis price), the gate is skipped --
    fail-open per the contract."""
    settings = _mock_settings(price_tolerance_pct=5.0)
    trader = _trader_with_mocks(settings)

    with patch(
        "backend.services.paper_trader.ExecutionRouter"
    ) as mock_router_cls:
        mock_router = mock_router_cls.return_value
        mock_router.submit_order.return_value = SimpleNamespace(
            fill_price=500.0,
            source="bq_sim",
        )
        trade = trader.execute_buy(
            ticker="WDC",
            amount_usd=500.0,
            price=500.0,             # no analysis reference -> gate cannot fire
            price_at_analysis=None,
        )

    assert trade is not None, (
        "phase-30.6 Test E: None price_at_analysis must fail-open; got reject"
    )


# ---------------------------------------------------------------------
# 6. Grep-equivalent: symbols present per masterplan verification
# ---------------------------------------------------------------------
def test_price_tolerance_symbols_present_in_source():
    """phase-30.6: mirrors the masterplan verification grep predicate
    so a future refactor that removes the wiring breaks pytest."""
    from pathlib import Path

    settings_src = (
        Path(__file__).resolve().parents[1] / "config" / "settings.py"
    ).read_text(encoding="utf-8")
    trader_src = (
        Path(__file__).resolve().parents[1] / "services" / "paper_trader.py"
    ).read_text(encoding="utf-8")

    assert "paper_price_tolerance_pct" in settings_src, (
        "phase-30.6: settings.py must contain `paper_price_tolerance_pct`"
    )
    assert "price_tolerance" in trader_src, (
        "phase-30.6: paper_trader.py must contain `price_tolerance` symbol"
    )
