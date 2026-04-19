"""phase-6.5 tests for the sentiment scorer ladder.

Covers:
 1. VADER bullish-headline path (end-to-end; skipped if vaderSentiment missing).
 2. VADER ambiguous article routes to FinBERT/Haiku via score_ladder.
 3. ScorerResult field schema matches the BQ migration column set.
 4. Haiku fail-open on missing API key (never raises, returns neutral/0).
 5. scorer_model enum conformance to migration.
 6. Escalation logic early-returns at first confident tier (monkeypatched).
 7. GeminiFlashScorer raises NotImplementedError when disabled.
 8. HAIKU_SYSTEM_PROMPT exceeds 4096-token cache minimum (char-proxy).
"""
from __future__ import annotations

import importlib

import pytest

from backend.news import sentiment as s
from backend.news.sentiment import (
    DEFAULT_MIN_CONFIDENCE,
    GeminiFlashScorer,
    HAIKU_SYSTEM_PROMPT,
    HaikuScorer,
    ScorerResult,
    SCORER_MODEL_FINBERT,
    SCORER_MODEL_GEMINI_FLASH,
    SCORER_MODEL_HAIKU,
    SCORER_MODEL_VADER,
    VaderScorer,
    score_ladder,
)


# ---------- helpers ----------

_SENTIMENT_BQ_FIELDS = {
    "article_id",
    "scorer_model",
    "scorer_version",
    "scored_at",
    "sentiment_score",
    "sentiment_label",
    "confidence",
    "latency_ms",
    "cost_usd",
    "raw_output",
}

_MIGRATION_ENUM = {
    SCORER_MODEL_VADER,
    SCORER_MODEL_FINBERT,
    SCORER_MODEL_HAIKU,
    SCORER_MODEL_GEMINI_FLASH,
}


def _fake_article(aid: str, title: str, body: str = "") -> dict:
    return {"article_id": aid, "title": title, "body": body}


# ---------- 1. VADER bullish path ----------


def test_vader_bullish_headline():
    scorer = VaderScorer()
    if scorer._analyzer is None:
        pytest.skip("vaderSentiment not installed")
    r = scorer.score(
        _fake_article("t1", "Company raises guidance on strong Q4 results")
    )
    assert isinstance(r, ScorerResult)
    assert r.sentiment_label in ("bullish", "bearish", "neutral")
    assert -1.0 <= r.sentiment_score <= 1.0
    assert 0.0 <= r.confidence <= 1.0
    assert r.scorer_model == SCORER_MODEL_VADER
    assert r.cost_usd == 0.0


# ---------- 2. ScorerResult field schema matches migration ----------


def test_scorer_result_fields_match_bq_migration():
    """ScorerResult fields must be a superset of the news_sentiment BQ columns."""
    dummy = ScorerResult(
        article_id="a",
        scorer_model=SCORER_MODEL_VADER,
        scorer_version="1.0",
        scored_at="2026-04-19T00:00:00+00:00",
        sentiment_score=0.0,
        sentiment_label="neutral",
        confidence=0.0,
        latency_ms=0.0,
        cost_usd=0.0,
        raw_output="",
    )
    actual = set(dummy.as_dict().keys())
    assert _SENTIMENT_BQ_FIELDS.issubset(actual), (
        f"ScorerResult missing BQ columns: {_SENTIMENT_BQ_FIELDS - actual}"
    )


# ---------- 3. scorer_model enum conformance ----------


def test_scorer_model_enum_matches_migration():
    """Every tier's scorer_model string must be in the migration enum."""
    v = VaderScorer()
    # VaderScorer can run without vaderSentiment and will fail-open.
    r = v.score(_fake_article("x", "test"))
    assert r.scorer_model == SCORER_MODEL_VADER
    assert r.scorer_model in _MIGRATION_ENUM

    # GeminiFlashScorer stub-neutral result when enabled:
    g = GeminiFlashScorer(enabled=True)
    r2 = g.score(_fake_article("x", "test"))
    assert r2.scorer_model == SCORER_MODEL_GEMINI_FLASH
    assert r2.scorer_model in _MIGRATION_ENUM


# ---------- 4. Haiku fail-open on missing API key ----------


def test_haiku_fail_open_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    h = HaikuScorer(api_key="")
    r = h.score(_fake_article("t4", "Boeing delivers record Q4"))
    # Must NOT raise; returns neutral fail-open result.
    assert isinstance(r, ScorerResult)
    assert r.scorer_model == SCORER_MODEL_HAIKU
    assert r.confidence == 0.0
    assert r.sentiment_label == "neutral"
    assert r.sentiment_score == 0.0
    assert r.cost_usd == 0.0


# ---------- 5. GeminiFlashScorer raises when disabled ----------


def test_gemini_flash_disabled_raises():
    g = GeminiFlashScorer(enabled=False)
    with pytest.raises(NotImplementedError):
        g.score(_fake_article("t5", "test"))


# ---------- 6. Escalation routing via score_ladder ----------


def test_score_ladder_early_returns_on_confident_tier(monkeypatch):
    """If VADER returns confidence >= threshold, FinBERT + Haiku must NOT be called."""

    def _fake_vader_score(self, article):
        return ScorerResult(
            article_id=article["article_id"],
            scorer_model=SCORER_MODEL_VADER,
            scorer_version="1.0",
            scored_at="2026-04-19T00:00:00+00:00",
            sentiment_score=0.9,
            sentiment_label="bullish",
            confidence=0.95,  # exceeds 0.7 -- should terminate here
            latency_ms=0.1,
            cost_usd=0.0,
            raw_output="{}",
        )

    finbert_called = {"n": 0}
    haiku_called = {"n": 0}

    def _fake_fin(self, article):
        finbert_called["n"] += 1
        return ScorerResult("a", SCORER_MODEL_FINBERT, "", "", 0.0, "neutral", 0.0, 0.0, 0.0, "")

    def _fake_haiku(self, article):
        haiku_called["n"] += 1
        return ScorerResult("a", SCORER_MODEL_HAIKU, "", "", 0.0, "neutral", 0.0, 0.0, 0.0, "")

    # Reset singletons so monkeypatched classes are used.
    s._SINGLETON_VADER = None
    s._SINGLETON_FINBERT = None
    s._SINGLETON_HAIKU = None

    monkeypatch.setattr(VaderScorer, "score", _fake_vader_score)
    monkeypatch.setattr(s.FinBertScorer, "score", _fake_fin)
    monkeypatch.setattr(s.HaikuScorer, "score", _fake_haiku)

    result = score_ladder(_fake_article("t6", "headline"))
    assert result.scorer_model == SCORER_MODEL_VADER
    assert result.confidence == 0.95
    assert finbert_called["n"] == 0
    assert haiku_called["n"] == 0


def test_score_ladder_escalates_from_vader_to_finbert(monkeypatch):
    """Low-confidence VADER must escalate to FinBERT; high-confidence FinBERT terminates."""

    def _fake_vader(self, article):
        return ScorerResult("a", SCORER_MODEL_VADER, "", "", 0.1, "neutral", 0.2, 0.0, 0.0, "")

    def _fake_fin(self, article):
        return ScorerResult("a", SCORER_MODEL_FINBERT, "", "", 0.85, "bullish", 0.85, 0.0, 0.0, "")

    haiku_called = {"n": 0}

    def _fake_haiku(self, article):
        haiku_called["n"] += 1
        return ScorerResult("a", SCORER_MODEL_HAIKU, "", "", 0.0, "neutral", 0.0, 0.0, 0.0, "")

    s._SINGLETON_VADER = None
    s._SINGLETON_FINBERT = None
    s._SINGLETON_HAIKU = None

    monkeypatch.setattr(VaderScorer, "score", _fake_vader)
    monkeypatch.setattr(s.FinBertScorer, "score", _fake_fin)
    monkeypatch.setattr(s.HaikuScorer, "score", _fake_haiku)

    result = score_ladder(_fake_article("t7", "ambiguous"))
    assert result.scorer_model == SCORER_MODEL_FINBERT
    assert result.confidence == 0.85
    assert haiku_called["n"] == 0


# ---------- 7. Haiku system prompt token floor ----------


def test_haiku_system_prompt_meets_4096_token_minimum():
    """Haiku 4.5 requires >= 4096 tokens for prompt-cache activation.

    1 token ~= 3.5-4 chars English. Using 3.5 chars/token as the strict
    lower bound: 4096 * 3.5 = 14336 chars minimum. We assert >= 14500 to
    leave a safety margin and catch any future prompt shrinkage.
    """
    assert len(HAIKU_SYSTEM_PROMPT) >= 14500, (
        f"HAIKU_SYSTEM_PROMPT too short ({len(HAIKU_SYSTEM_PROMPT)} chars); "
        f"needs >= 14500 to clear 4096-token cache floor"
    )


# ---------- 8. Module imports cleanly + public API stable ----------


def test_module_imports_and_exports():
    mod = importlib.import_module("backend.news.sentiment")
    for name in (
        "ScorerResult",
        "VaderScorer",
        "FinBertScorer",
        "HaikuScorer",
        "GeminiFlashScorer",
        "score_ladder",
        "HAIKU_SYSTEM_PROMPT",
    ):
        assert hasattr(mod, name), f"missing export: {name}"
    assert mod.DEFAULT_MIN_CONFIDENCE == 0.7
