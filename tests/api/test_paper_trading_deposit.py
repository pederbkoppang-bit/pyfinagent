"""phase-23.1.9: DepositRequest validation + FullSettings/SettingsUpdate
exposes the 10 paper-trading settings fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.api.paper_trading import DepositRequest
from backend.api.settings_api import FullSettings, SettingsUpdate


PAPER_FIELDS = {
    "paper_max_positions",
    "paper_max_daily_cost_usd",
    "paper_default_stop_loss_pct",
    "paper_screen_top_n",
    "paper_analyze_top_n",
    "paper_transaction_cost_pct",
    "paper_daily_loss_limit_pct",
    "paper_trailing_dd_limit_pct",
    "paper_min_cash_reserve_pct",
    "paper_starting_capital",
}


def test_full_settings_exposes_10_paper_fields():
    missing = PAPER_FIELDS - set(FullSettings.model_fields.keys())
    assert not missing, f"FullSettings missing: {missing}"


def test_settings_update_exposes_9_writable_paper_fields():
    """paper_starting_capital is read-only; other 9 are writable."""
    writable = PAPER_FIELDS - {"paper_starting_capital"}
    missing = writable - set(SettingsUpdate.model_fields.keys())
    assert not missing, f"SettingsUpdate missing: {missing}"


def test_paper_starting_capital_is_NOT_writable():
    """starting_capital must not be in SettingsUpdate (operator can change via deposit only)."""
    assert "paper_starting_capital" not in SettingsUpdate.model_fields


def test_deposit_request_accepts_normal_amount():
    req = DepositRequest(amount=500.0)
    assert req.amount == 500.0


def test_deposit_request_rejects_zero():
    with pytest.raises(ValidationError):
        DepositRequest(amount=0.0)


def test_deposit_request_rejects_negative():
    with pytest.raises(ValidationError):
        DepositRequest(amount=-100.0)


def test_deposit_request_rejects_over_one_million():
    with pytest.raises(ValidationError):
        DepositRequest(amount=2_000_000.0)


def test_deposit_request_accepts_one_million_exactly():
    req = DepositRequest(amount=1_000_000.0)
    assert req.amount == 1_000_000.0


def test_deposit_request_accepts_small_amount():
    req = DepositRequest(amount=1.0)
    assert req.amount == 1.0


def test_settings_update_paper_fields_have_validators():
    """Numeric paper fields should have ge/le constraints."""
    paper_max_positions_field = SettingsUpdate.model_fields["paper_max_positions"]
    # Just confirm constraints are present (Pydantic v2 stores them in metadata)
    metadata_str = str(paper_max_positions_field.metadata)
    assert "Ge" in metadata_str or "ge" in metadata_str.lower()


def test_full_settings_paper_starting_capital_default_10000():
    # Construct with required fields; paper fields should default
    f = FullSettings(
        gemini_model="gemini-2.5-flash",
        deep_think_model="gemini-2.5-pro",
        max_debate_rounds=2,
        max_risk_debate_rounds=1,
        weight_corporate=0.2, weight_industry=0.2, weight_valuation=0.2,
        weight_sentiment=0.2, weight_governance=0.2,
        data_quality_min=0.5, lite_mode=False,
        max_analysis_cost_usd=2.0, max_synthesis_iterations=2,
    )
    assert f.paper_starting_capital == 10000.0
    assert f.paper_max_positions == 10
    assert f.paper_default_stop_loss_pct == 8.0


def test_settings_update_validators_reject_out_of_range():
    with pytest.raises(ValidationError):
        SettingsUpdate(paper_max_positions=0)  # ge=1
    with pytest.raises(ValidationError):
        SettingsUpdate(paper_max_positions=100)  # le=50
    with pytest.raises(ValidationError):
        SettingsUpdate(paper_default_stop_loss_pct=0.5)  # ge=1
    with pytest.raises(ValidationError):
        SettingsUpdate(paper_default_stop_loss_pct=60.0)  # le=50
