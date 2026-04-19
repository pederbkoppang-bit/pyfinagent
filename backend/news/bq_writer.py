"""phase-6.8 BigQuery writer for all phase-6 tables.

Three writer functions, one per table, using `client.insert_rows_json`
(the proven pattern from `backend/services/observability/api_call_log.py`).

Design (see `handoff/current/phase-6.8-research-brief.md`):
- `insert_rows_json` over Storage Write API for <200 rows per batch --
  BQ docs confirm no complexity/cost justification at this scale.
- At-least-once semantics acceptable: intra-batch dedup (phase-6.4)
  runs BEFORE insert; event_id is deterministic sha256; downstream
  consumers SELECT DISTINCT where needed.
- Fail-open at every boundary: missing `google-cloud-bigquery`, auth
  failure, DDL drift all return 0 and log at WARNING. Never raise.
- Dataset resolved from `settings.bq_dataset_observability` (phase-6.8
  made this an explicit field; fallback to `pyfinagent_data`).

Row shapes map 1:1 to the migrations:
- news_articles: `scripts/migrations/add_news_sentiment_schema.py:64-86`
- news_sentiment: `scripts/migrations/add_news_sentiment_schema.py:89-107`
- calendar_events: `scripts/migrations/add_calendar_events_schema.py:36-57`

NormalizedArticle / ScorerResult / CalendarEvent are TypedDicts or
dataclasses; we convert via `dict(..)` / `asdict(..)` and let BQ
ignore unknown keys (BQ streaming inserts warn on unknown columns in
strict mode; tolerated on permissive).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

_NEWS_ARTICLES_TABLE = "news_articles"
_NEWS_SENTIMENT_TABLE = "news_sentiment"
_CALENDAR_EVENTS_TABLE = "calendar_events"


def _resolve_target(project: str | None, dataset: str | None) -> tuple[str, str]:
    """Resolve (project, dataset) from args + settings. Never raises."""
    proj = project
    ds = dataset
    if proj is None or ds is None:
        try:
            from backend.config.settings import get_settings

            s = get_settings()
            if proj is None:
                proj = s.gcp_project_id
            if ds is None:
                ds = getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        except Exception as exc:  # pragma: no cover
            logger.warning("bq_writer: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_client(project: str) -> Any:
    """Return a BQ Client or None (fail-open)."""
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("bq_writer: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("bq_writer: bigquery.Client() init failed (%r)", exc)
        return None


def _insert_rows(
    project: str, dataset: str, table: str, rows: Sequence[dict[str, Any]]
) -> int:
    """Streaming insert wrapper. Returns rows inserted (0 on any failure).

    Never raises.
    """
    if not rows:
        return 0
    client = _get_client(project)
    if client is None:
        return 0
    try:
        table_ref = f"{project}.{dataset}.{table}" if project else f"{dataset}.{table}"
        errors = client.insert_rows_json(table_ref, list(rows))
        if errors:
            logger.warning("bq_writer %s insert errors: %s", table, errors)
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("bq_writer %s insert fail-open: %r", table, exc)
        return 0


def _serialize_article(article: Mapping[str, Any]) -> dict[str, Any]:
    """Map a NormalizedArticle dict to the news_articles row shape.

    `raw_payload` is a JSON column in BQ; it must be serialized as a
    JSON string when passed through `insert_rows_json`.
    """
    raw_payload = article.get("raw_payload") or {}
    try:
        raw_payload_json = (
            raw_payload
            if isinstance(raw_payload, str)
            else json.dumps(raw_payload, default=str)
        )
    except Exception:
        raw_payload_json = "{}"
    return {
        "article_id": article.get("article_id") or "",
        "published_at": article.get("published_at"),
        "fetched_at": article.get("fetched_at"),
        "source": article.get("source") or "",
        "ticker": article.get("ticker"),
        "title": article.get("title") or "",
        "body": article.get("body") or "",
        "url": article.get("url") or "",
        "canonical_url": article.get("canonical_url") or "",
        "body_hash": article.get("body_hash") or "",
        "language": article.get("language"),
        "authors": list(article.get("authors") or []),
        "categories": list(article.get("categories") or []),
        "raw_payload": raw_payload_json,
    }


def write_news_articles(
    articles: Sequence[Mapping[str, Any]],
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    """Insert NormalizedArticle rows into `news_articles`. Returns rows inserted."""
    proj, ds = _resolve_target(project, dataset)
    rows = [_serialize_article(a) for a in articles]
    return _insert_rows(proj, ds, _NEWS_ARTICLES_TABLE, rows)


def _serialize_sentiment(result: Any) -> dict[str, Any]:
    """ScorerResult dataclass -> news_sentiment row shape."""
    if is_dataclass(result):
        d = asdict(result)
    elif isinstance(result, Mapping):
        d = dict(result)
    else:  # pragma: no cover -- defensive
        d = {}
    return {
        "article_id": d.get("article_id") or "",
        "scorer_model": d.get("scorer_model") or "",
        "scorer_version": d.get("scorer_version"),
        "scored_at": d.get("scored_at"),
        "sentiment_score": d.get("sentiment_score"),
        "sentiment_label": d.get("sentiment_label"),
        "confidence": d.get("confidence"),
        "latency_ms": d.get("latency_ms"),
        "cost_usd": d.get("cost_usd"),
        "raw_output": d.get("raw_output"),
    }


def write_news_sentiment(
    results: Sequence[Any],
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    """Insert ScorerResult rows into `news_sentiment`. Returns rows inserted."""
    proj, ds = _resolve_target(project, dataset)
    rows = [_serialize_sentiment(r) for r in results]
    return _insert_rows(proj, ds, _NEWS_SENTIMENT_TABLE, rows)


def _serialize_calendar_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """CalendarEvent TypedDict -> calendar_events row shape."""
    metadata = event.get("metadata") or {}
    try:
        metadata_json = (
            metadata if isinstance(metadata, str) else json.dumps(metadata, default=str)
        )
    except Exception:
        metadata_json = "{}"
    return {
        "event_id": event.get("event_id") or "",
        "event_type": event.get("event_type") or "",
        "ticker": event.get("ticker"),
        "scheduled_at": event.get("scheduled_at"),
        "window": event.get("window"),
        "fiscal_period_end": event.get("fiscal_period_end"),
        "source": event.get("source") or "",
        "confidence": event.get("confidence") or "estimated",
        "blackout_start": event.get("blackout_start"),
        "blackout_end": event.get("blackout_end"),
        "eps_estimate": event.get("eps_estimate"),
        "revenue_estimate": event.get("revenue_estimate"),
        "fetched_at": event.get("fetched_at"),
        "metadata": metadata_json,
    }


def write_calendar_events(
    events: Sequence[Mapping[str, Any]],
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    """Insert CalendarEvent rows into `calendar_events`. Returns rows inserted."""
    proj, ds = _resolve_target(project, dataset)
    rows = [_serialize_calendar_event(e) for e in events]
    return _insert_rows(proj, ds, _CALENDAR_EVENTS_TABLE, rows)


__all__ = [
    "write_news_articles",
    "write_news_sentiment",
    "write_calendar_events",
]
