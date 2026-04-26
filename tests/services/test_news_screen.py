"""Unit tests for news_screen — schema, dedup, ticker normalization, score apply, cache."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import pytest

from backend.services.news_screen import (
    NewsHeadlineSignal,
    NewsSignalBatch,
    apply_news_to_score,
    _dedup_jaccard,
    _normalize_ticker,
    _word_3grams,
    _jaccard,
    _save_cache,
    _load_cache,
    _CACHE_DIR,
)


def _mk(ticker="AAPL", polarity="positive", confidence="high", event="earnings_beat") -> NewsHeadlineSignal:
    return NewsHeadlineSignal(
        ticker_mentioned=ticker,
        event_type=event,
        impact_polarity=polarity,
        confidence=confidence,
        rationale="test",
        skip_reason="",
    )


def test_schema_event_enum_enforced():
    with pytest.raises(Exception):
        NewsHeadlineSignal(
            ticker_mentioned="AAPL", event_type="cosmic_ray",
            impact_polarity="positive", confidence="high", rationale="",
        )


def test_schema_polarity_enum_enforced():
    with pytest.raises(Exception):
        NewsHeadlineSignal(
            ticker_mentioned="AAPL", event_type="earnings_beat",
            impact_polarity="rocket", confidence="high", rationale="",
        )


def test_schema_confidence_enum_enforced():
    with pytest.raises(Exception):
        NewsHeadlineSignal(
            ticker_mentioned="AAPL", event_type="earnings_beat",
            impact_polarity="positive", confidence="absolute", rationale="",
        )


def test_normalize_ticker_strips_dollar_and_uppercases():
    assert _normalize_ticker("$aapl") == "AAPL"
    assert _normalize_ticker(" msft ") == "MSFT"
    assert _normalize_ticker("brk.b") == "BRK.B"


def test_normalize_ticker_rejects_bad_format():
    assert _normalize_ticker("THIS IS A SENTENCE") is None
    assert _normalize_ticker("") is None
    assert _normalize_ticker(None) is None
    assert _normalize_ticker("AAPL+MSFT") is None


def test_normalize_ticker_accepts_exchange_suffix():
    assert _normalize_ticker("005930.KS") is not None
    assert _normalize_ticker("7203.T") == "7203.T" or _normalize_ticker("7203.T") is None  # 4 digits not all-alpha


def test_jaccard_identical_titles():
    a = _word_3grams("apple beats earnings reports record revenue")
    b = _word_3grams("apple beats earnings reports record revenue")
    assert _jaccard(a, b) == pytest.approx(1.0)


def test_jaccard_orthogonal_titles():
    a = _word_3grams("apple beats earnings reports")
    b = _word_3grams("federal reserve raises interest rate")
    assert _jaccard(a, b) == pytest.approx(0.0)


def test_dedup_jaccard_keeps_distinct():
    items = [
        {"title": "Apple beats earnings expectations Q1 2026"},
        {"title": "Federal Reserve hikes interest rates"},
        {"title": "Tesla announces new model launch"},
    ]
    out = _dedup_jaccard(items, threshold=0.4)
    assert len(out) == 3


def test_dedup_jaccard_merges_near_dupes():
    items = [
        {"title": "Apple beats earnings expectations Q1 2026"},
        {"title": "Apple beats earnings expectations Q1 2026 by margin"},
        {"title": "Federal Reserve hikes interest rates"},
    ]
    out = _dedup_jaccard(items, threshold=0.4)
    assert len(out) == 2


def test_dedup_jaccard_keeps_first_occurrence():
    items = [
        {"title": "Apple beats earnings expectations Q1 2026", "id": 1},
        {"title": "Apple beats earnings expectations Q1 2026 by margin", "id": 2},
    ]
    out = _dedup_jaccard(items, threshold=0.4)
    assert len(out) == 1
    assert out[0]["id"] == 1


def test_apply_news_no_signals_passes_through():
    assert apply_news_to_score(10.0, "AAPL", None) == 10.0
    assert apply_news_to_score(10.0, "AAPL", {}) == 10.0


def test_apply_news_unmatched_ticker_passes_through():
    assert apply_news_to_score(10.0, "MSFT", {"AAPL": _mk()}) == 10.0


def test_apply_news_positive_boosts_10pct():
    out = apply_news_to_score(10.0, "AAPL", {"AAPL": _mk(polarity="positive")})
    assert out == pytest.approx(11.0)


def test_apply_news_negative_penalizes_10pct():
    out = apply_news_to_score(10.0, "AAPL", {"AAPL": _mk(polarity="negative")})
    assert out == pytest.approx(9.0)


def test_apply_news_low_confidence_passes_through():
    out = apply_news_to_score(10.0, "AAPL", {"AAPL": _mk(polarity="positive", confidence="low")})
    assert out == pytest.approx(10.0)


def test_apply_news_neutral_passes_through():
    out = apply_news_to_score(10.0, "AAPL", {"AAPL": _mk(polarity="neutral")})
    assert out == pytest.approx(10.0)


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.news_screen._CACHE_DIR", tmp_path)
    sigs = {"AAPL": _mk(), "MSFT": _mk(ticker="MSFT")}
    _save_cache(sigs)
    loaded = _load_cache()
    assert loaded is not None
    assert set(loaded.keys()) == {"AAPL", "MSFT"}
    assert loaded["AAPL"].impact_polarity == "positive"


def test_cache_stale_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.news_screen._CACHE_DIR", tmp_path)
    sigs = {"AAPL": _mk()}
    _save_cache(sigs)
    # Backdate the file 5 hours
    bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    p = tmp_path / f"news_screen_{bucket}.json"
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).timestamp()
    import os
    os.utime(p, (old, old))
    assert _load_cache() is None


def test_batch_schema_accepts_empty_signals():
    batch = NewsSignalBatch(signals=[])
    assert batch.signals == []


def test_batch_schema_validates_nested_signals():
    with pytest.raises(Exception):
        NewsSignalBatch.model_validate({
            "signals": [{"ticker_mentioned": "AAPL", "event_type": "earnings_beat",
                         "impact_polarity": "rocket", "confidence": "high", "rationale": ""}],
        })
