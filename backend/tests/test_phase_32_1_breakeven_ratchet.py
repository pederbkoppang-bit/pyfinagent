"""phase-32.1 tests: breakeven-stop ratchet at +1R.

Audit basis: handoff/archive/phase-31.0/experiment_results.md section 4 P1.1.
Spec source: .claude/masterplan.json::phase-32.1.implementation_plan.test_specs.

Test plan (7 cases):
  1. test_no_advance_below_1R: MFE=+5% (< threshold 8%) -> (None, None).
  2. test_advance_exactly_at_1R: MFE=+8.0 -> (entry, ISO).
  3. test_advance_above_1R: MFE=+20% -> (entry, ISO). Stop does NOT advance
     past entry (that's phase-32.2's HWM trailing job).
  4. test_idempotent: position with stop_advanced_at_R already populated ->
     (None, None) even at MFE=+50%.
  5. test_monotonic_never_moves_down: position with stop_loss_price already
     >= entry_price (e.g. from a future trail) -> (None, None) regardless
     of MFE.
  6. test_mark_to_market_persists: integration via PaperTrader.mark_to_market.
     Feed a position at entry $100, current $110 (+10% pnl). Assert
     _safe_save_position is called with stop_loss_price = 100.0 and
     stop_advanced_at_R populated as an ISO string.
  7. test_position_below_threshold_in_mark_to_market: pos at entry $100,
     current $105 (+5% pnl, below 8% threshold) -> stop unchanged, no
     stop_advanced_at_R written.

Adversarial note (audit-confirmed): Kaminski-Lo Proposition 2 governs
trailing-stop EXIT thresholds under mean-reverting return processes; it
does NOT apply to one-shot breakeven mutations on already-profitable
positions. No strategy-conditional guard needed for the +1R ratchet.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.services.paper_trader import PaperTrader


# ── Helpers ───────────────────────────────────────────────────────


def _mock_settings(default_stop_loss_pct: float = 8.0) -> SimpleNamespace:
    return SimpleNamespace(
        paper_price_tolerance_pct=5.0,
        paper_default_stop_loss_pct=default_stop_loss_pct,
        paper_max_positions=10,
        paper_transaction_cost_pct=0.05,
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        paper_trailing_dd_limit_pct=10.0,
        paper_daily_loss_limit_pct=4.0,
    )


def _trader_with_mocks(settings: SimpleNamespace) -> PaperTrader:
    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "portfolio_id": "default",
        "current_cash": 5000.0,
        "starting_capital": 10000.0,
        "inception_date": "2026-05-01T00:00:00+00:00",
        "total_nav": 10000.0,
        "total_pnl_pct": 0.0,
        "benchmark_return_pct": 0.0,
        "updated_at": "2026-05-01T00:00:00+00:00",
    }
    bq.get_paper_positions.return_value = []
    bq.save_paper_position.return_value = None
    bq.delete_paper_position.return_value = None
    bq.upsert_paper_portfolio.return_value = None
    return PaperTrader(settings=settings, bq_client=bq)


# ── Pure-helper unit tests (5) ────────────────────────────────────


def test_no_advance_below_1R():
    """MFE=+5% with 8% threshold -> no ratchet."""
    trader = _trader_with_mocks(_mock_settings(default_stop_loss_pct=8.0))
    pos = {
        "ticker": "MU",
        "avg_entry_price": 500.0,
        "stop_loss_price": 460.0,
        "stop_advanced_at_R": None,
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=5.0)
    assert new_stop is None
    assert advance_iso is None


def test_advance_exactly_at_1R():
    """MFE = threshold -> ratchet fires; stop -> entry; ISO string returned."""
    trader = _trader_with_mocks(_mock_settings(default_stop_loss_pct=8.0))
    pos = {
        "ticker": "MU",
        "avg_entry_price": 500.0,
        "stop_loss_price": 460.0,
        "stop_advanced_at_R": None,
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=8.0)
    assert new_stop == 500.0
    assert isinstance(advance_iso, str) and advance_iso.endswith("+00:00")


def test_advance_above_1R():
    """MFE=+20% (well above threshold) -> ratchet fires; stop pinned to entry,
    NOT further (that's phase-32.2's HWM-trailing job)."""
    trader = _trader_with_mocks(_mock_settings(default_stop_loss_pct=8.0))
    pos = {
        "ticker": "SNDK",
        "avg_entry_price": 989.9,
        "stop_loss_price": None,  # legacy NO_STOP position
        "stop_advanced_at_R": None,
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=20.0)
    assert new_stop == 989.9
    # Sanity: the helper must NOT compute a trailing distance above entry.
    assert new_stop == pos["avg_entry_price"]
    assert advance_iso is not None


def test_idempotent_when_stop_advanced_at_R_already_populated():
    """Once stop_advanced_at_R is set, the helper short-circuits regardless
    of MFE -- this is the idempotency guarantee."""
    trader = _trader_with_mocks(_mock_settings(default_stop_loss_pct=8.0))
    pos = {
        "ticker": "MU",
        "avg_entry_price": 500.0,
        "stop_loss_price": 500.0,
        "stop_advanced_at_R": "2026-05-20T12:00:00+00:00",
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=50.0)
    assert new_stop is None
    assert advance_iso is None


def test_monotonic_never_moves_down():
    """If stop_loss_price >= entry_price already (e.g. from a hypothetical
    future trail), the helper does NOT pull it back down to entry."""
    trader = _trader_with_mocks(_mock_settings(default_stop_loss_pct=8.0))
    pos = {
        "ticker": "MU",
        "avg_entry_price": 500.0,
        "stop_loss_price": 550.0,  # already above entry from a future trail
        "stop_advanced_at_R": None,
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=30.0)
    assert new_stop is None
    assert advance_iso is None


# ── Integration tests via mark_to_market (2) ──────────────────────


def test_mark_to_market_persists_ratchet():
    """Full path: a position at entry $100 with current $110 (+10% pnl)
    must trigger the ratchet during mark_to_market. _safe_save_position
    is called with stop_loss_price = 100.0 and stop_advanced_at_R set."""
    settings = _mock_settings(default_stop_loss_pct=8.0)
    trader = _trader_with_mocks(settings)
    trader.bq.get_paper_positions.return_value = [
        {
            "ticker": "TEST",
            "quantity": 10.0,
            "avg_entry_price": 100.0,
            "cost_basis": 1000.0,
            "stop_loss_price": 92.0,
            "stop_advanced_at_R": None,
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "current_price": 100.0,
        }
    ]
    saved: list[dict] = []
    trader.bq.save_paper_position.side_effect = lambda row: saved.append(dict(row))

    with patch(
        "backend.services.paper_trader._get_live_price", return_value=110.0
    ), patch(
        "backend.services.paper_trader._get_benchmark_return", return_value=None
    ):
        trader.mark_to_market()

    assert len(saved) == 1
    persisted = saved[0]
    assert persisted["stop_loss_price"] == 100.0, (
        f"breakeven ratchet should pin stop to entry; got {persisted.get('stop_loss_price')}"
    )
    assert isinstance(persisted.get("stop_advanced_at_R"), str), (
        "stop_advanced_at_R must be populated as an ISO string"
    )
    assert persisted["mfe_pct"] == 10.0


def test_mark_to_market_below_threshold_no_ratchet():
    """Position with +5% pnl (below 8% threshold) -> stop unchanged, no
    stop_advanced_at_R written."""
    settings = _mock_settings(default_stop_loss_pct=8.0)
    trader = _trader_with_mocks(settings)
    trader.bq.get_paper_positions.return_value = [
        {
            "ticker": "TEST",
            "quantity": 10.0,
            "avg_entry_price": 100.0,
            "cost_basis": 1000.0,
            "stop_loss_price": 92.0,
            "stop_advanced_at_R": None,
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "current_price": 100.0,
        }
    ]
    saved: list[dict] = []
    trader.bq.save_paper_position.side_effect = lambda row: saved.append(dict(row))

    with patch(
        "backend.services.paper_trader._get_live_price", return_value=105.0
    ), patch(
        "backend.services.paper_trader._get_benchmark_return", return_value=None
    ):
        trader.mark_to_market()

    assert len(saved) == 1
    persisted = saved[0]
    assert persisted["stop_loss_price"] == 92.0, (
        f"stop must remain at entry-anchored level when MFE < threshold; "
        f"got {persisted.get('stop_loss_price')}"
    )
    assert persisted.get("stop_advanced_at_R") in (None, ""), (
        "stop_advanced_at_R must remain unset below threshold"
    )
    assert persisted["mfe_pct"] == 5.0
