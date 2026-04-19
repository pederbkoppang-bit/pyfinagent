"""phase-6.8 tests for the BQ writer module.

Coverage:
 1. Writers fail-open when google-cloud-bigquery absent or auth fails.
 2. `_serialize_article` produces the expected news_articles row shape.
 3. `_serialize_sentiment` produces the expected news_sentiment row shape.
 4. `_serialize_calendar_event` produces the expected calendar_events row shape.
 5. `_resolve_target` reads settings.bq_dataset_observability.
 6. Empty input returns 0 rows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from backend.news.bq_writer import (
    _resolve_target,
    _serialize_article,
    _serialize_calendar_event,
    _serialize_sentiment,
    write_calendar_events,
    write_news_articles,
    write_news_sentiment,
)


_NEWS_ARTICLES_FIELDS = {
    "article_id", "published_at", "fetched_at", "source", "ticker",
    "title", "body", "url", "canonical_url", "body_hash", "language",
    "authors", "categories", "raw_payload",
}
_NEWS_SENTIMENT_FIELDS = {
    "article_id", "scorer_model", "scorer_version", "scored_at",
    "sentiment_score", "sentiment_label", "confidence",
    "latency_ms", "cost_usd", "raw_output",
}
_CALENDAR_EVENTS_FIELDS = {
    "event_id", "event_type", "ticker", "scheduled_at", "window",
    "fiscal_period_end", "source", "confidence", "blackout_start",
    "blackout_end", "eps_estimate", "revenue_estimate", "fetched_at", "metadata",
}


# ---- 1. Fail-open paths ----


def test_write_news_articles_fail_open_empty_input():
    assert write_news_articles([]) == 0


def test_write_news_sentiment_fail_open_empty_input():
    assert write_news_sentiment([]) == 0


def test_write_calendar_events_fail_open_empty_input():
    assert write_calendar_events([]) == 0


def test_write_news_articles_fail_open_no_bq_auth():
    """Non-empty input with no BQ auth / project mismatch still returns 0; never raises."""
    articles = [
        {
            "article_id": "a1",
            "published_at": "2026-04-19T00:00:00+00:00",
            "fetched_at": "2026-04-19T00:00:01+00:00",
            "source": "stub",
            "ticker": "AAPL",
            "title": "Test",
            "body": "Test body",
            "url": "https://example.com/a1",
            "canonical_url": "https://example.com/a1",
            "body_hash": "abc",
            "language": "en",
            "authors": [],
            "categories": [],
            "raw_payload": {"x": 1},
        }
    ]
    # Force a definitely-bad project to trigger client init failure or insert failure
    result = write_news_articles(articles, project="nonexistent-fail-open-test", dataset="nx")
    assert result == 0  # fail-open: 0 on failure, never raises


# ---- 2. Article serialization ----


def test_serialize_article_produces_expected_fields():
    article = {
        "article_id": "a1",
        "published_at": "2026-04-19T00:00:00+00:00",
        "fetched_at": "2026-04-19T00:00:01+00:00",
        "source": "finnhub",
        "ticker": "AAPL",
        "title": "Test headline",
        "body": "Test body text",
        "url": "https://example.com/a",
        "canonical_url": "https://example.com/a",
        "body_hash": "sha256hex",
        "language": "en",
        "authors": ["alice"],
        "categories": ["earnings"],
        "raw_payload": {"id": 42, "nested": {"ok": True}},
    }
    row = _serialize_article(article)
    assert set(row.keys()) == _NEWS_ARTICLES_FIELDS
    assert row["article_id"] == "a1"
    # raw_payload gets JSON-stringified for the BQ JSON column
    assert isinstance(row["raw_payload"], str)
    assert json.loads(row["raw_payload"]) == {"id": 42, "nested": {"ok": True}}


def test_serialize_article_handles_missing_fields():
    row = _serialize_article({"article_id": "x"})
    # All fields must be present (populated with defaults)
    assert set(row.keys()) == _NEWS_ARTICLES_FIELDS
    assert row["article_id"] == "x"
    assert row["title"] == ""
    assert row["authors"] == []
    # raw_payload default is empty JSON
    assert row["raw_payload"] == "{}"


# ---- 3. Sentiment serialization ----


@dataclass
class _FakeScorer:
    article_id: str
    scorer_model: str
    scorer_version: str
    scored_at: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    latency_ms: float
    cost_usd: float
    raw_output: str


def test_serialize_sentiment_from_dataclass():
    r = _FakeScorer(
        article_id="a1",
        scorer_model="vader",
        scorer_version="1.0",
        scored_at="2026-04-19T00:00:00+00:00",
        sentiment_score=0.75,
        sentiment_label="bullish",
        confidence=0.75,
        latency_ms=0.4,
        cost_usd=0.0,
        raw_output="{}",
    )
    row = _serialize_sentiment(r)
    assert set(row.keys()) == _NEWS_SENTIMENT_FIELDS
    assert row["sentiment_label"] == "bullish"
    assert row["scorer_model"] == "vader"


def test_serialize_sentiment_from_mapping():
    row = _serialize_sentiment(
        {
            "article_id": "a1",
            "scorer_model": "finbert",
            "scored_at": "2026-04-19T00:00:00+00:00",
            "sentiment_score": -0.5,
            "sentiment_label": "bearish",
            "confidence": 0.7,
        }
    )
    assert row["scorer_model"] == "finbert"
    assert row["sentiment_label"] == "bearish"
    assert row["sentiment_score"] == -0.5


# ---- 4. Calendar serialization ----


def test_serialize_calendar_event_produces_expected_fields():
    event = {
        "event_id": "abc",
        "event_type": "earnings",
        "ticker": "NVDA",
        "scheduled_at": "2026-02-20T21:00:00+00:00",
        "window": "post_close",
        "fiscal_period_end": "2026-01-31",
        "source": "finnhub",
        "confidence": "confirmed",
        "blackout_start": None,
        "blackout_end": None,
        "eps_estimate": 0.88,
        "revenue_estimate": 38.0e9,
        "fetched_at": "2026-04-19T10:00:00+00:00",
        "metadata": {"year": 2026, "quarter": 4},
    }
    row = _serialize_calendar_event(event)
    assert set(row.keys()) == _CALENDAR_EVENTS_FIELDS
    assert row["event_id"] == "abc"
    assert row["event_type"] == "earnings"
    # metadata gets JSON-stringified
    assert isinstance(row["metadata"], str)
    assert json.loads(row["metadata"])["year"] == 2026


# ---- 5. Settings resolution ----


def test_resolve_target_reads_settings():
    proj, ds = _resolve_target(None, None)
    # project comes from settings.gcp_project_id (required field)
    assert isinstance(proj, str)
    assert isinstance(ds, str)
    assert ds  # falls back to pyfinagent_data at minimum


def test_resolve_target_explicit_overrides_settings():
    proj, ds = _resolve_target("my-proj", "my-ds")
    assert proj == "my-proj"
    assert ds == "my-ds"
