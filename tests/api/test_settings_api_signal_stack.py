"""phase-23.1.6: Settings API exposes the 13 signal-stack fields with correct
defaults, optional update semantics, and ge/le validators on numeric fields."""

from __future__ import annotations

import pytest

from backend.api.settings_api import FullSettings, SettingsUpdate, _FIELD_TO_ENV


SIGNAL_STACK_FIELDS = {
    "macro_regime_filter_enabled",
    "macro_regime_model",
    "pead_signal_enabled",
    "pead_signal_model",
    "pead_signal_lookback_quarters",
    "news_screen_enabled",
    "news_screen_model",
    "news_screen_max_headlines",
    "sector_calendars_enabled",
    "sector_calendars_lookahead_days",
    "meta_scorer_enabled",
    "meta_scorer_model",
    "meta_scorer_max_batch",
}


def test_full_settings_exposes_all_13_fields():
    missing = SIGNAL_STACK_FIELDS - set(FullSettings.model_fields.keys())
    assert not missing, f"FullSettings missing: {missing}"


def test_settings_update_exposes_all_13_fields():
    missing = SIGNAL_STACK_FIELDS - set(SettingsUpdate.model_fields.keys())
    assert not missing, f"SettingsUpdate missing: {missing}"


def test_field_to_env_dict_covers_signal_stack():
    missing = SIGNAL_STACK_FIELDS - set(_FIELD_TO_ENV.keys())
    assert not missing, f"_FIELD_TO_ENV missing: {missing}"
    # Check each entry has uppercase env var
    for f in SIGNAL_STACK_FIELDS:
        assert _FIELD_TO_ENV[f] == f.upper(), f"{f} -> {_FIELD_TO_ENV[f]}"


def test_full_settings_signal_stack_defaults_off():
    """All 5 enable flags default to False (default-OFF discipline)."""
    full = _make_full_settings()
    assert full.macro_regime_filter_enabled is False
    assert full.pead_signal_enabled is False
    assert full.news_screen_enabled is False
    assert full.sector_calendars_enabled is False
    assert full.meta_scorer_enabled is False


def test_full_settings_signal_stack_model_defaults():
    full = _make_full_settings()
    assert full.macro_regime_model == "claude-haiku-4-5"
    assert full.pead_signal_model == "claude-haiku-4-5"
    assert full.news_screen_model == "claude-haiku-4-5"
    assert full.meta_scorer_model == "claude-haiku-4-5"


def test_full_settings_signal_stack_numeric_defaults():
    full = _make_full_settings()
    assert full.pead_signal_lookback_quarters == 8
    assert full.news_screen_max_headlines == 100
    assert full.sector_calendars_lookahead_days == 7
    assert full.meta_scorer_max_batch == 30


def test_settings_update_accepts_individual_fields():
    upd = SettingsUpdate(macro_regime_filter_enabled=True)
    assert upd.macro_regime_filter_enabled is True
    assert upd.pead_signal_enabled is None  # other fields untouched


def test_settings_update_news_max_headlines_above_ceiling_rejected():
    with pytest.raises(Exception):
        SettingsUpdate(news_screen_max_headlines=600)  # ceiling = 500


def test_settings_update_news_max_headlines_below_floor_rejected():
    with pytest.raises(Exception):
        SettingsUpdate(news_screen_max_headlines=5)  # floor = 10


def test_settings_update_meta_scorer_batch_below_floor_rejected():
    with pytest.raises(Exception):
        SettingsUpdate(meta_scorer_max_batch=4)  # floor = 5


def test_settings_update_meta_scorer_batch_above_ceiling_rejected():
    with pytest.raises(Exception):
        SettingsUpdate(meta_scorer_max_batch=200)  # ceiling = 100


def test_settings_update_pead_lookback_below_floor_rejected():
    with pytest.raises(Exception):
        SettingsUpdate(pead_signal_lookback_quarters=0)  # floor = 1


def test_settings_update_sector_lookahead_above_ceiling_rejected():
    with pytest.raises(Exception):
        SettingsUpdate(sector_calendars_lookahead_days=60)  # ceiling = 30


def test_settings_update_partial_payload_only_touched_fields():
    upd = SettingsUpdate(news_screen_enabled=True, meta_scorer_max_batch=20)
    dumped = upd.model_dump(exclude_none=True)
    assert dumped == {"news_screen_enabled": True, "meta_scorer_max_batch": 20}


def _make_full_settings() -> FullSettings:
    """Helper: construct FullSettings with required existing fields stubbed."""
    return FullSettings(
        gemini_model="gemini-2.0-flash",
        deep_think_model="gemini-2.5-pro",
        max_debate_rounds=2,
        max_risk_debate_rounds=1,
        weight_corporate=0.2,
        weight_industry=0.2,
        weight_valuation=0.2,
        weight_sentiment=0.2,
        weight_governance=0.2,
        data_quality_min=0.5,
        lite_mode=False,
        max_analysis_cost_usd=2.0,
        max_synthesis_iterations=2,
    )
