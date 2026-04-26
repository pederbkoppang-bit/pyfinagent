"""Unit tests for pead_signal — schema, score application, cache, trailing mean."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from backend.services.pead_signal import (
    PeadSignalOutput,
    apply_pead_to_score,
    _trailing_mean_from_cache,
    _save_pead_cache,
    _load_pead_cache,
    _ticker_cache_path,
    _CACHE_DIR,
    _LOOKBACK_QUARTERS,
    _VALID_HOLDING_WINDOWS,
)


def _mk(tag="positive_surprise", sentiment=0.7, surprise=0.2, hold=28) -> PeadSignalOutput:
    return PeadSignalOutput(
        rationale="test",
        sentiment_score=sentiment,
        surprise_score=surprise,
        sentiment_tag=tag,
        holding_window_days=hold,
        skip_reason="",
    )


def test_schema_enforces_tag_enum():
    with pytest.raises(Exception):
        PeadSignalOutput(
            rationale="x", sentiment_score=0.5, surprise_score=0.0,
            sentiment_tag="bullish", holding_window_days=28,
        )


def test_schema_enforces_sentiment_range():
    with pytest.raises(Exception):
        PeadSignalOutput(
            rationale="x", sentiment_score=1.5, surprise_score=0.0,
            sentiment_tag="neutral", holding_window_days=28,
        )


def test_apply_pead_no_signals_passes_through():
    assert apply_pead_to_score(10.0, "AAPL", None) == 10.0
    assert apply_pead_to_score(10.0, "AAPL", {}) == 10.0


def test_apply_pead_unmatched_ticker_passes_through():
    assert apply_pead_to_score(10.0, "MSFT", {"AAPL": _mk()}) == 10.0


def test_apply_pead_positive_surprise_boosts():
    sig = _mk(tag="positive_surprise", surprise=0.3)
    out = apply_pead_to_score(10.0, "AAPL", {"AAPL": sig})
    assert out == pytest.approx(10.0 * (1.0 + 0.15))  # 0.3 * 0.5 = 0.15


def test_apply_pead_positive_surprise_capped_at_30pct():
    sig = _mk(tag="positive_surprise", surprise=2.0)  # would be +100% uncapped
    out = apply_pead_to_score(10.0, "AAPL", {"AAPL": sig})
    assert out == pytest.approx(10.0 * 1.3)  # capped at +30%


def test_apply_pead_strong_negative_filters_out():
    sig = _mk(tag="negative_surprise", sentiment=0.2, surprise=-0.4)
    out = apply_pead_to_score(10.0, "AAPL", {"AAPL": sig})
    assert out is None


def test_apply_pead_mild_negative_penalizes():
    sig = _mk(tag="negative_surprise", sentiment=0.4, surprise=-0.2)
    out = apply_pead_to_score(10.0, "AAPL", {"AAPL": sig})
    assert out == pytest.approx(10.0 * 0.9)  # 1 + (-0.2 * 0.5) = 0.9


def test_apply_pead_negative_floor_at_60pct():
    sig = _mk(tag="negative_surprise", sentiment=0.3, surprise=-0.29)  # not strong-negative
    out = apply_pead_to_score(10.0, "AAPL", {"AAPL": sig})
    assert out is not None
    assert out >= 10.0 * 0.6


def test_apply_pead_neutral_passes_through():
    sig = _mk(tag="neutral", sentiment=0.5, surprise=0.05)
    out = apply_pead_to_score(10.0, "AAPL", {"AAPL": sig})
    assert out == pytest.approx(10.0)


def test_apply_pead_insufficient_history_passes_through():
    sig = _mk(tag="insufficient_history", sentiment=0.6, surprise=0.0)
    out = apply_pead_to_score(10.0, "AAPL", {"AAPL": sig})
    assert out == pytest.approx(10.0)


def test_holding_window_set_only_accepts_valid_values():
    assert _VALID_HOLDING_WINDOWS == {14, 28, 42, 60}
    for v in _VALID_HOLDING_WINDOWS:
        # Construct without raising
        PeadSignalOutput(
            rationale="x", sentiment_score=0.5, surprise_score=0.0,
            sentiment_tag="neutral", holding_window_days=v,
        )


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.pead_signal._CACHE_DIR", tmp_path)
    sig = _mk()
    _save_pead_cache("AAPL", "2026-03-31", sig)
    loaded = _load_pead_cache("AAPL", "2026-03-31")
    assert loaded is not None
    assert loaded.sentiment_tag == sig.sentiment_tag
    assert loaded.sentiment_score == pytest.approx(sig.sentiment_score)


def test_cache_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.pead_signal._CACHE_DIR", tmp_path)
    assert _load_pead_cache("AAPL", "2026-03-31") is None


def test_cache_unreadable_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.pead_signal._CACHE_DIR", tmp_path)
    path = tmp_path / "pead_AAPL_2026-03-31.json"
    path.write_text("not json", encoding="utf-8")
    assert _load_pead_cache("AAPL", "2026-03-31") is None


def test_trailing_mean_from_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.pead_signal._CACHE_DIR", tmp_path)
    mean, n = _trailing_mean_from_cache("AAPL", "2026-03-31")
    assert mean is None
    assert n == 0


def test_trailing_mean_excludes_current_quarter(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.pead_signal._CACHE_DIR", tmp_path)
    for q, score in [("2025-09-30", 0.5), ("2025-12-31", 0.6), ("2026-03-31", 0.9)]:
        sig = _mk(sentiment=score)
        _save_pead_cache("AAPL", q, sig)
    mean, n = _trailing_mean_from_cache("AAPL", "2026-03-31")
    assert n == 2
    assert mean == pytest.approx((0.5 + 0.6) / 2)


def test_trailing_mean_caps_at_lookback(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.pead_signal._CACHE_DIR", tmp_path)
    # 10 prior quarters, scores 0.1..1.0
    for i in range(10):
        q = f"2024-{i+1:02d}-01"
        _save_pead_cache("AAPL", q, _mk(sentiment=0.1 * (i + 1)))
    mean, n = _trailing_mean_from_cache("AAPL", "9999-99-99")
    assert n == _LOOKBACK_QUARTERS  # capped at 8
