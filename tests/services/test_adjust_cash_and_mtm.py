"""phase-23.2.17: PaperTrader.adjust_cash_and_mtm helper test.

The helper exists to prevent the bug class where raw cash mutations
(cleanup scripts, manual refunds, deposits via DML) silently leave
total_nav stale. The pattern caught phase-23.1.15 and phase-23.2.2 by
surprise; helper centralizes the cash + MtM + snapshot flow so future
operators don't have to remember 3 separate calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.services.paper_trader import PaperTrader


def _stub_settings() -> SimpleNamespace:
    return SimpleNamespace(
        paper_max_positions=15,
        paper_starting_capital=15000.0,
        paper_min_cash_reserve_pct=0.0,
        paper_default_stop_loss_pct=8.0,
        paper_transaction_cost_pct=0.1,
        paper_alpaca_paper_enabled=False,
    )


def test_adjust_cash_and_mtm_calls_each_step():
    """The helper must call _update_portfolio_cash, mark_to_market,
    save_daily_snapshot exactly once each, in that order."""
    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "portfolio_id": "default",
        "current_cash": 1000.0,
        "starting_capital": 15000.0,
    }
    pt = PaperTrader(_stub_settings(), bq)

    # Patch the trader's own methods we care about.
    call_order: list[str] = []
    pt._update_portfolio_cash = lambda c: call_order.append(f"cash:{c}")
    pt.mark_to_market = lambda: (call_order.append("mtm"), {"nav": 1500.0})[1]
    pt.save_daily_snapshot = (
        lambda trades_today=0, analysis_cost_today=0.0:
        (call_order.append("snapshot"), None)[1]
    )

    result = pt.adjust_cash_and_mtm(delta=950.43, reason="phase-23.2.2")

    assert call_order == ["cash:1950.43", "mtm", "snapshot"], \
        f"expected cash -> mtm -> snapshot, got {call_order}"
    assert result["old_cash"] == 1000.0
    assert result["new_cash"] == 1950.43
    assert result["delta"] == 950.43
    assert result["reason"] == "phase-23.2.2"
    assert result["post_nav"] == 1500.0


def test_adjust_cash_and_mtm_negative_delta():
    """Refund-reversal (negative delta) is supported."""
    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "current_cash": 5000.0, "starting_capital": 15000.0,
    }
    pt = PaperTrader(_stub_settings(), bq)
    pt._update_portfolio_cash = lambda c: None
    pt.mark_to_market = lambda: {"nav": 14500.0}
    pt.save_daily_snapshot = lambda **_: None

    result = pt.adjust_cash_and_mtm(delta=-500.0, reason="reversal-test")
    assert result["new_cash"] == 4500.0
    assert result["delta"] == -500.0
