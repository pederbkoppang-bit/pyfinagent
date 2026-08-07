"""phase-83.0: news corpus persistence -- tables + observable writer + source-agnostic.

Verification file named by the step's immutable command. Criteria map:
 C1  news_articles schema carries published_at AND a distinct ingested_at (both TIMESTAMP)
 C2  REQUIRED provenance column; live path emits 'live', backfill path 'backfill',
     never null, never the same value from both paths
 C3  injected insert failure increments the counter AND emits a log record while the
     writer returns normally -- both halves in ONE test
 C4  counter guard is mutation-resistant: STRICT numeric increase asserted (the
     mutation matrix in experiment_results_83.0.md exercises the deletions)
 C5  source-agnostic: rows from two distinct registered adapters accepted; neither the
     migration module nor bq_writer pulls in any module requiring ALPHAVANTAGE_API_KEY
 C6  finnhub.py / benzinga.py byte-unchanged, asserted over exactly those two paths

Schema oracle: live BigQuery when reachable, else the checked-in snapshot captured
verbatim from `bq show --schema` on 2026-08-07 immediately after the migration ran.
The oracle is asserted NON-EMPTY before use (a silently-empty oracle validates any
shape -- feedback_a_green_suite_can_be_blind).
"""
from __future__ import annotations

import importlib
import logging
import sys
from unittest.mock import patch

import pytest

from backend.news import bq_writer
from backend.news.fetcher import run_once
from backend.news.registry import clear_registry, register

# ── schema oracle ────────────────────────────────────────────────────
# Verbatim from `bq show --schema sunny-might-477607-p8:pyfinagent_data.<t>`
# captured 2026-08-07 post-migration (see handoff live_check_83.0.md).
_SNAPSHOT_NEWS_ARTICLES = {
    "article_id": ("STRING", "REQUIRED"),
    # phase-83.0.1: relaxed REQUIRED -> NULLABLE (quarantined rows store NULL);
    # effective_trade_date added. Snapshot re-captured from the live schema
    # after the 83.0.1 ALTER.
    "published_at": ("TIMESTAMP", "NULLABLE"),
    "ingested_at": ("TIMESTAMP", "REQUIRED"),
    "provenance": ("STRING", "REQUIRED"),
    "effective_trade_date": ("DATE", "NULLABLE"),
    "source": ("STRING", "REQUIRED"),
    "ticker": ("STRING", "NULLABLE"),
    "title": ("STRING", "NULLABLE"),
    "body": ("STRING", "NULLABLE"),
    "url": ("STRING", "NULLABLE"),
    "canonical_url": ("STRING", "NULLABLE"),
    "body_hash": ("STRING", "NULLABLE"),
    "language": ("STRING", "NULLABLE"),
    "authors": ("STRING", "REPEATED"),
    "categories": ("STRING", "REPEATED"),
    "raw_payload": ("JSON", "NULLABLE"),
}
_SNAPSHOT_NEWS_SENTIMENT = {
    "article_id": ("STRING", "REQUIRED"),
    "scorer_model": ("STRING", "REQUIRED"),
    "scorer_version": ("STRING", "NULLABLE"),
    "scored_at": ("TIMESTAMP", "REQUIRED"),
    "ingested_at": ("TIMESTAMP", "REQUIRED"),
    "provenance": ("STRING", "REQUIRED"),
    "sentiment_score": ("FLOAT", "NULLABLE"),
    "sentiment_label": ("STRING", "NULLABLE"),
    "confidence": ("FLOAT", "NULLABLE"),
    "latency_ms": ("FLOAT", "NULLABLE"),
    "cost_usd": ("FLOAT", "NULLABLE"),
    "raw_output": ("STRING", "NULLABLE"),
}


def _resolve_schema(table: str, snapshot: dict) -> dict:
    """Live schema map name -> (type, mode); snapshot fallback offline."""
    try:  # pragma: no cover -- network-dependent branch
        from google.cloud import bigquery

        client = bigquery.Client(project="sunny-might-477607-p8")
        t = client.get_table(f"sunny-might-477607-p8.pyfinagent_data.{table}")
        live = {f.name: (f.field_type, f.mode) for f in t.schema}
        if live:
            return live
    except Exception:
        pass
    return dict(snapshot)


# ── C1: published_at + distinct ingested_at ──────────────────────────


def test_c1_articles_schema_has_published_at_and_distinct_ingested_at():
    schema = _resolve_schema("news_articles", _SNAPSHOT_NEWS_ARTICLES)
    assert schema, "schema oracle resolved EMPTY -- refusing a vacuous pass"
    assert "published_at" in schema, "published_at column absent"
    assert "ingested_at" in schema, "ingested_at column absent"
    assert "published_at" != "ingested_at"  # distinct names by construction
    assert schema["published_at"][0] == "TIMESTAMP"
    assert schema["ingested_at"][0] == "TIMESTAMP"
    # the rename really happened -- the old column is gone
    assert "fetched_at" not in schema, "fetched_at survived the rename"


# ── C2: REQUIRED provenance; live vs backfill discrimination ─────────


def test_c2_provenance_required_in_both_tables():
    for table, snap in (
        ("news_articles", _SNAPSHOT_NEWS_ARTICLES),
        ("news_sentiment", _SNAPSHOT_NEWS_SENTIMENT),
    ):
        schema = _resolve_schema(table, snap)
        assert schema, f"{table}: schema oracle resolved EMPTY"
        assert "provenance" in schema, f"{table}: provenance column absent"
        assert schema["provenance"] == ("STRING", "REQUIRED"), (
            f"{table}: provenance must be REQUIRED STRING, got {schema['provenance']}"
        )


def test_c2_live_and_backfill_paths_emit_distinct_provenance():
    live_report = run_once(["stub"], dry_run=True)
    backfill_report = run_once(["stub"], dry_run=True, provenance="backfill")
    assert live_report.articles and backfill_report.articles
    live_vals = {a.get("provenance") for a in live_report.articles}
    backfill_vals = {a.get("provenance") for a in backfill_report.articles}
    assert live_vals == {"live"}, f"live path emitted {live_vals}"
    assert backfill_vals == {"backfill"}, f"backfill path emitted {backfill_vals}"
    assert None not in live_vals and None not in backfill_vals
    assert live_vals != backfill_vals, "both paths emitted the same provenance"
    # the serialized BQ rows carry it too (REQUIRED => omission fails the batch)
    row_live = bq_writer._serialize_article(live_report.articles[0])
    row_bf = bq_writer._serialize_article(backfill_report.articles[0])
    assert row_live["provenance"] == "live"
    assert row_bf["provenance"] == "backfill"


# ── C3 + C4: swallowed failure -> counter strictly increases + log ───


class _FakeClient:
    def __init__(self, errors):
        self.errors = errors
        self.calls = []

    def insert_rows_json(self, table_ref, rows):
        self.calls.append((table_ref, rows))
        return self.errors  # [] == success; non-empty == BQ rejected the batch


_ROW = {
    "article_id": "t1",
    "published_at": "2026-08-07T00:00:00+00:00",
    "ingested_at": "2026-08-07T00:00:01+00:00",
    "provenance": "live",
    "source": "stub",
}


def test_c3_swallowed_write_increments_counter_and_logs(caplog):
    bq_writer.reset_write_failures_for_test()
    fake = _FakeClient(errors=[{"index": 0, "errors": [{"reason": "invalid"}]}])
    before = bq_writer.write_failure_count("news_articles")
    with patch.object(bq_writer, "_get_client", return_value=fake), caplog.at_level(
        logging.WARNING, logger="backend.news.bq_writer"
    ):
        n = bq_writer.write_news_articles([_ROW])  # must NOT raise
    after = bq_writer.write_failure_count("news_articles")
    assert n == 0, "fail-open contract broken: non-zero return on failed write"
    # C4: STRICT numeric increase -- attribute-existence alone does not count
    assert after == before + 1, f"counter did not strictly increase ({before} -> {after})"
    assert any(
        "write FAILED" in rec.getMessage() and "news_articles" in rec.getMessage()
        for rec in caplog.records
    ), "no WARNING log record emitted for the swallowed failure"


def test_c3_negative_control_success_does_not_count():
    bq_writer.reset_write_failures_for_test()
    fake = _FakeClient(errors=[])  # successful insert
    with patch.object(bq_writer, "_get_client", return_value=fake):
        n = bq_writer.write_news_articles([_ROW])
    assert n == 1
    assert bq_writer.write_failure_count("news_articles") == 0, (
        "counter moved on a SUCCESSFUL write -- it no longer measures failure"
    )


def test_c3_negative_control_empty_input_does_not_count():
    bq_writer.reset_write_failures_for_test()
    assert bq_writer.write_news_articles([]) == 0
    assert bq_writer.write_failure_count() == 0, (
        "empty input counted as a failure -- 'nothing to write' is not a failure"
    )


# ── C2 (cycle 3): the sentiment write seam stamps provenance ─────────
# ScorerResult carries no provenance field, so write_news_sentiment's
# `provenance` kwarg is the ONLY channel a backfill scoring run has.
# Cycle-2 Q/A proved the column was a constant 'live' before this seam
# existed; these tests make the stamp mutation-killable.


def test_c2_sentiment_writer_stamps_default_provenance():
    bq_writer.reset_write_failures_for_test()
    fake = _FakeClient(errors=[])
    row_in = {
        "article_id": "s1",
        "scorer_model": "finbert",
        "scored_at": "2026-08-07T00:00:00+00:00",
        "sentiment_score": 0.1,
    }
    with patch.object(bq_writer, "_get_client", return_value=fake):
        n = bq_writer.write_news_sentiment([row_in], provenance="backfill")
    assert n == 1
    (_, rows), = fake.calls
    assert rows[0]["provenance"] == "backfill", (
        "sentiment write seam did not stamp default provenance"
    )
    # explicit row value wins over the default
    fake2 = _FakeClient(errors=[])
    with patch.object(bq_writer, "_get_client", return_value=fake2):
        bq_writer.write_news_sentiment(
            [dict(row_in, provenance="live")], provenance="backfill"
        )
    (_, rows2), = fake2.calls
    assert rows2[0]["provenance"] == "live"
    # no kwarg, no row value -> 'live' (and never null: the column is REQUIRED)
    fake3 = _FakeClient(errors=[])
    with patch.object(bq_writer, "_get_client", return_value=fake3):
        bq_writer.write_news_sentiment([row_in])
    (_, rows3), = fake3.calls
    assert rows3[0]["provenance"] == "live"


def test_smoke_pipeline_threads_provenance_at_both_seams(monkeypatch):
    """The phase6_e2e remediation is guarded: deleting either provenance
    threading (fetch seam or sentiment-write seam) turns this red."""
    import backend.econ_calendar.watcher as watcher_mod
    import backend.news.fetcher as fetcher_mod
    import backend.services.observability as obs_mod
    import scripts.smoketest.phase6_e2e as e2e

    from backend.news.fetcher import FetchReport

    fetch_calls = []
    sent_calls = []

    def fake_run_once(source_names=None, dry_run=False, dedup=True, provenance="live"):
        fetch_calls.append({"provenance": provenance, "dry_run": dry_run})
        return FetchReport(n_sources=0, n_articles=0, dry_run=dry_run)

    def fake_write_sentiment(results, **kwargs):
        sent_calls.append(kwargs)
        return 0

    class _EmptyCal:
        events = []
        n_events = 0
        by_type = {}
        by_source = {}
        errors = []

    monkeypatch.setattr(fetcher_mod, "run_once", fake_run_once)
    monkeypatch.setattr(bq_writer, "write_news_articles", lambda *a, **k: 0)
    monkeypatch.setattr(bq_writer, "write_news_sentiment", fake_write_sentiment)
    monkeypatch.setattr(bq_writer, "write_calendar_events", lambda *a, **k: 0)
    monkeypatch.setattr(watcher_mod, "run_once", lambda **k: _EmptyCal())
    monkeypatch.setattr(obs_mod, "flush", lambda: 0)
    monkeypatch.setattr(obs_mod, "flush_llm", lambda: 0)
    monkeypatch.setattr(e2e, "_slack_heartbeat", lambda payload: False)

    e2e._run_pipeline(sources=["stub"], dry_run=True, backfill=True, days_forward=0)
    e2e._run_pipeline(sources=["stub"], dry_run=True, backfill=False, days_forward=0)

    assert [c["provenance"] for c in fetch_calls] == ["backfill", "live"], (
        f"fetch seam threading broken: {fetch_calls}"
    )
    assert [c.get("provenance") for c in sent_calls] == ["backfill", "live"], (
        f"sentiment-write seam threading broken: {sent_calls}"
    )


# ── C5: source-agnostic writer + no Alpha Vantage import coupling ────


@pytest.fixture
def _isolated_registry():
    """Wipe the registry for the test, then restore the real adapters."""
    clear_registry()
    yield
    clear_registry()
    for mod in ("backend.news.fetcher", "backend.news.sources"):
        sys.modules.pop(mod, None)
    importlib.import_module("backend.news.fetcher")
    importlib.import_module("backend.news.sources")


def test_c5_two_distinct_adapters_flow_through_writer(_isolated_registry):
    @register("srcA")
    class _SrcA:
        name = "srcA"

        def fetch(self):
            yield {
                "source": "srcA",
                "title": "A headline",
                "body": "a body",
                "url": "https://a.example.com/1",
                "published_at": "2026-08-01T00:00:00+00:00",
            }

    @register("srcB")
    class _SrcB:
        name = "srcB"

        def fetch(self):
            yield {
                "source": "srcB",
                "title": "B headline",
                "body": "b body",
                "url": "https://b.example.com/1",
                "published_at": "2026-08-02T00:00:00+00:00",
            }

    report = run_once(["srcA", "srcB"], dry_run=True)
    assert report.n_sources == 2
    sources = {a["source"] for a in report.articles}
    assert sources == {"srcA", "srcB"}
    rows = [bq_writer._serialize_article(a) for a in report.articles]
    assert {r["source"] for r in rows} == {"srcA", "srcB"}, (
        "writer did not accept rows from two distinct registered adapters"
    )


def test_c5_no_alphavantage_import_chain(monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    for name in list(sys.modules):
        if name.startswith("scripts.migrations.add_news_sentiment_schema"):
            sys.modules.pop(name, None)
    importlib.import_module("backend.news.bq_writer")
    importlib.import_module("scripts.migrations.add_news_sentiment_schema")
    reached = {m for m in sys.modules if "alphavantage" in m.lower() or "alpha_vantage" in m.lower()}
    assert not reached, f"alphavantage module pulled in by migration/writer: {reached}"


# ── C6: finnhub.py / benzinga.py byte-unchanged ──────────────────────
# TOMBSTONE (phase-83.0.1, contract D5): test_c6_finnhub_benzinga_byte_unchanged
# is RETIRED. Its criterion belonged to step 83.0 ("byte-unchanged ... asserted
# by a committed diff over exactly those two paths") and was DISCHARGED at the
# 83.0 close (commit 06911cb5 carries the empty diff). As a living test it
# asserted `git diff HEAD` emptiness forever, which would forbid every future
# legitimate edit to those files -- including 83.0.1's authorized removal of
# the adapters' wall-clock fabrication sites (the step 83.0's research had
# already flagged as upstream dead-code coupling). Retired, not weakened: no
# replacement assertion pretends to cover what it covered.
