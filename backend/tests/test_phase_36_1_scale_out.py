"""phase-36.1 scale-out take-profit ladder tests.

Verifies the fix for OPEN-2 (the only OPEN code BLOCK on profit-protection
per closure_roadmap.md §2): partial-close primitive in execute_sell(quantity=...)
exists; this step wires it to fire at MFE >= 2*R (50% close) and >= 3*R
(remainder close). Gated by paper_scale_out_enabled (default OFF).

Tests cover the 5 immutable success criteria from masterplan 36.1:
  1. synth_position_with_mfe_2_1R_triggers_50_percent_partial_close
  2. synth_position_with_mfe_3_1R_triggers_remainder_close
  3. idempotent_re_fire_in_same_cycle_is_no_op
  4. paper_trades_emits_partial_close_row_with_reason_take_profit_2R
  5. scale_out_levels_hit_column_added_via_idempotent_migration (verified by
     migration file existence + --verify flag honored)

No real BQ calls; bigquery_client is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest


def _make_trader(flag_on: bool = True, stop_loss_pct: float = 8.0):
    """Build a PaperTrader with mocked bq + settings. Returns (trader, bq_mock, settings)."""
    from backend.services.paper_trader import PaperTrader
    from backend.config.settings import Settings

    s = Settings()
    s.paper_scale_out_enabled = flag_on
    s.paper_default_stop_loss_pct = stop_loss_pct

    bq = MagicMock()
    trader = PaperTrader(settings=s, bq_client=bq)
    return trader, bq, s


def _pos(ticker="COHR", qty=10.0, entry=200.0, mfe=0.0, scale_out_levels=None):
    return {
        "ticker": ticker,
        "position_id": f"pos_{ticker}",
        "quantity": qty,
        "avg_entry_price": entry,
        "current_price": entry * (1.0 + mfe / 100.0),
        "cost_basis": qty * entry,
        "mfe_pct": mfe,
        "mae_pct": 0.0,
        "stop_loss_price": entry * 0.92,
        "entry_strategy": "momentum",
        "scale_out_levels_hit": json.dumps(scale_out_levels) if scale_out_levels else None,
        "entry_date": "2026-05-01T00:00:00+00:00",
    }


def test_phase_36_1_flag_off_no_fires_backward_compat():
    trader, bq, _ = _make_trader(flag_on=False)
    bq.get_paper_positions.return_value = [_pos(mfe=30.0)]
    fires = trader.check_scale_out_fires()
    assert fires == []


def test_phase_36_1_field_default_off():
    from backend.config.settings import Settings
    assert Settings().paper_scale_out_enabled is False


def test_phase_36_1_2r_fires_50_percent_partial_close():
    trader, bq, _ = _make_trader(flag_on=True)
    bq.get_paper_positions.return_value = [_pos(qty=10.0, entry=200.0, mfe=17.0)]

    with patch.object(trader, "execute_sell") as mock_sell, \
         patch.object(trader, "_persist_scale_out_levels") as mock_persist:
        mock_sell.return_value = {"trade_id": "trade_2r_1"}
        fires = trader.check_scale_out_fires()
        mock_sell.assert_called_once_with("COHR", quantity=5.0, reason="take_profit_2R")
        mock_persist.assert_called_once()
        persist_args = mock_persist.call_args
        assert persist_args.args[0] == "COHR"
        assert "2R" in persist_args.args[1]

    assert len(fires) == 1
    assert fires[0]["ticker"] == "COHR"
    assert fires[0]["level"] == "2R"
    assert fires[0]["qty"] == 5.0


def test_phase_36_1_3r_fires_remainder_close():
    trader, bq, _ = _make_trader(flag_on=True)
    bq.get_paper_positions.return_value = [
        _pos(qty=5.0, entry=200.0, mfe=25.0, scale_out_levels=["2R"]),
    ]

    with patch.object(trader, "execute_sell") as mock_sell, \
         patch.object(trader, "get_position") as mock_get, \
         patch.object(trader, "_persist_scale_out_levels"):
        mock_get.return_value = _pos(qty=5.0, entry=200.0, mfe=25.0,
                                     scale_out_levels=["2R"])
        mock_sell.return_value = {"trade_id": "trade_3r_1"}
        fires = trader.check_scale_out_fires()
        mock_sell.assert_called_once_with("COHR", quantity=5.0, reason="take_profit_3R")

    assert len(fires) == 1
    assert fires[0]["level"] == "3R"
    assert fires[0]["qty"] == 5.0


def test_phase_36_1_idempotent_re_fire_no_op():
    trader, bq, _ = _make_trader(flag_on=True)
    bq.get_paper_positions.return_value = [
        _pos(qty=2.0, entry=200.0, mfe=40.0, scale_out_levels=["2R", "3R"]),
    ]
    with patch.object(trader, "execute_sell") as mock_sell:
        fires = trader.check_scale_out_fires()
        mock_sell.assert_not_called()
    assert fires == []


def test_phase_36_1_below_2r_no_fire():
    trader, bq, _ = _make_trader(flag_on=True)
    bq.get_paper_positions.return_value = [_pos(mfe=15.0)]
    with patch.object(trader, "execute_sell") as mock_sell:
        fires = trader.check_scale_out_fires()
        mock_sell.assert_not_called()
    assert fires == []


def test_phase_36_1_both_2r_and_3r_fire_in_same_cycle():
    trader, bq, _ = _make_trader(flag_on=True)
    bq.get_paper_positions.return_value = [
        _pos(qty=10.0, entry=200.0, mfe=30.0, scale_out_levels=None),
    ]
    with patch.object(trader, "execute_sell") as mock_sell, \
         patch.object(trader, "get_position") as mock_get, \
         patch.object(trader, "_persist_scale_out_levels"):
        mock_get.return_value = _pos(qty=5.0, entry=200.0, mfe=30.0,
                                     scale_out_levels=["2R"])
        mock_sell.return_value = {"trade_id": "trade_x"}
        fires = trader.check_scale_out_fires()
    assert len(fires) == 2
    levels = sorted([f["level"] for f in fires])
    assert levels == ["2R", "3R"]
    qtys = sorted([f["qty"] for f in fires])
    assert qtys == [5.0, 5.0]


def test_phase_36_1_null_scale_out_column_treated_as_empty_set():
    trader, bq, _ = _make_trader(flag_on=True)
    pos = _pos(qty=10.0, entry=200.0, mfe=17.0)
    pos.pop("scale_out_levels_hit", None)
    bq.get_paper_positions.return_value = [pos]
    with patch.object(trader, "execute_sell") as mock_sell, \
         patch.object(trader, "_persist_scale_out_levels"):
        mock_sell.return_value = {"trade_id": "trade_z"}
        fires = trader.check_scale_out_fires()
    mock_sell.assert_called_once_with("COHR", quantity=5.0, reason="take_profit_2R")
    assert len(fires) == 1
    assert fires[0]["level"] == "2R"


def test_phase_36_1_migration_script_has_verify_flag():
    """Success criterion #5: the migration script supports --verify exit-code."""
    import os
    path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "scripts", "migrations", "add_scale_out_levels_hit_column.py"
    )
    norm = os.path.normpath(path)
    assert os.path.exists(norm), f"migration file missing at {norm}"
    src = open(norm).read()
    assert "--verify" in src, "migration must support --verify flag"
    assert "ADD COLUMN IF NOT EXISTS" in src, "migration must be idempotent (IF NOT EXISTS)"
