"""Unit tests for macro_regime — schema validation, score application, cache logic."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import pytest

from backend.services.macro_regime import (
    MacroRegimeOutput,
    SectorWeights,
    apply_regime_to_score,
    _DEFAULT_MULTIPLIERS,
    _load_cache,
    _save_cache,
    _CACHE_PATH,
)


def _mk_regime(regime="risk_on", mult=1.15, ow=("XLK",), uw=()) -> MacroRegimeOutput:
    return MacroRegimeOutput(
        rationale="test",
        regime=regime,
        conviction=0.8,
        conviction_multiplier=mult,
        sector_hints=SectorWeights(overweight=list(ow), underweight=list(uw)),
        series_used=["T10Y2Y", "VIXCLS", "BAMLH0A0HYM2"],
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def test_schema_enforces_regime_enum():
    with pytest.raises(Exception):
        MacroRegimeOutput(
            rationale="x", regime="bull_market", conviction=0.5,
            conviction_multiplier=1.0, sector_hints=SectorWeights(),
            series_used=[], computed_at="2026-04-26T00:00:00+00:00",
        )


def test_schema_enforces_conviction_range():
    with pytest.raises(Exception):
        MacroRegimeOutput(
            rationale="x", regime="risk_on", conviction=1.5,
            conviction_multiplier=1.0, sector_hints=SectorWeights(),
            series_used=[], computed_at="2026-04-26T00:00:00+00:00",
        )


def test_schema_enforces_multiplier_range():
    with pytest.raises(Exception):
        MacroRegimeOutput(
            rationale="x", regime="risk_on", conviction=0.5,
            conviction_multiplier=2.0, sector_hints=SectorWeights(),
            series_used=[], computed_at="2026-04-26T00:00:00+00:00",
        )


def test_default_multipliers_are_in_valid_range():
    for tag, m in _DEFAULT_MULTIPLIERS.items():
        assert 0.5 <= m <= 1.5, f"{tag} multiplier {m} out of range"


def test_apply_regime_no_regime_passes_through():
    assert apply_regime_to_score(10.0, "Technology", {"Technology": "XLK"}, None) == 10.0


def test_apply_regime_multiplier_only():
    regime = _mk_regime(regime="risk_off", mult=0.7, ow=(), uw=())
    out = apply_regime_to_score(10.0, "Technology", {"Technology": "XLK"}, regime)
    assert out == pytest.approx(7.0)


def test_apply_regime_overweight_sector_boost():
    regime = _mk_regime(regime="risk_on", mult=1.15, ow=("XLK",), uw=())
    out = apply_regime_to_score(10.0, "Technology", {"Technology": "XLK"}, regime)
    assert out == pytest.approx(10.0 * 1.15 * 1.05)


def test_apply_regime_underweight_sector_penalty():
    regime = _mk_regime(regime="risk_off", mult=0.7, ow=(), uw=("XLK",))
    out = apply_regime_to_score(10.0, "Technology", {"Technology": "XLK"}, regime)
    assert out == pytest.approx(10.0 * 0.7 * 0.95)


def test_apply_regime_unknown_sector_no_tilt():
    regime = _mk_regime(regime="risk_on", mult=1.15, ow=("XLU",), uw=())
    out = apply_regime_to_score(10.0, None, {"Technology": "XLK"}, regime)
    assert out == pytest.approx(10.0 * 1.15)


def test_cache_roundtrip(tmp_path, monkeypatch):
    fake = tmp_path / "macro_regime.json"
    monkeypatch.setattr("backend.services.macro_regime._CACHE_PATH", fake)
    r = _mk_regime()
    _save_cache(r)
    loaded = _load_cache()
    assert loaded is not None
    assert loaded.regime == r.regime
    assert loaded.conviction_multiplier == r.conviction_multiplier


def test_cache_expired(tmp_path, monkeypatch):
    fake = tmp_path / "macro_regime.json"
    monkeypatch.setattr("backend.services.macro_regime._CACHE_PATH", fake)
    r = _mk_regime()
    payload = json.loads(r.model_dump_json())
    payload["computed_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    fake.write_text(json.dumps(payload), encoding="utf-8")
    assert _load_cache() is None


def test_cache_unreadable_returns_none(tmp_path, monkeypatch):
    fake = tmp_path / "macro_regime.json"
    fake.write_text("not json", encoding="utf-8")
    monkeypatch.setattr("backend.services.macro_regime._CACHE_PATH", fake)
    assert _load_cache() is None
