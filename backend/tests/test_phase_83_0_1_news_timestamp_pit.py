"""phase-83.0.1: news timestamp point-in-time integrity.

Criteria map:
 C1  fixture RawArticle with no published_at -> persisted row published_at is
     NULL (asserted on the VALUE) AND the quarantine counter strictly
     increments -- both halves in one test
 C2  mutation: restoring the `or _now_iso()` fallback in fetcher._normalize
     makes the C1 test FAIL (matrix in experiment_results; the strict
     value-is-None + strict-increase asserts are what kill it)
 C3  backfill rows: ingested_at is never earlier than published_at for a
     2022-dated article -- ingested_at is the backfill-RUN moment, not the
     article's era
 C4  effective_trade_date derived + persisted; over the committed corpus the
     minimum of (effective_trade_date - DATE(published_at)) is >= 1 trading
     day; simulated entry is the session OPEN, not the publication-day close
 C5  mutation: _EMBARGO_DAYS = 0 makes the C4 minimum assertion FAIL (killable
     via the intraday-weekday fixture; a weekend-only corpus would survive)
 C6  the quarantine count over the corpus is RECORDED, not thresholded

Upstream scope (contract D2): the three adapters no longer substitute
wall-clock for missing vendor timestamps; per-adapter seam tests feed raw
vendor payloads with the timestamp field missing and assert NULL + quarantine.
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime

import pytest

from backend.news import bq_writer, fetcher
from backend.news.fetcher import (
    _derive_effective_trade_date,
    _normalize,
    _parse_published_at,
    quarantine_count,
    reset_quarantine_for_test,
)

_CORPUS_PATH = pathlib.Path(__file__).parent / "fixtures" / "news_pit_corpus.json"


def _corpus() -> list[dict]:
    data = json.loads(_CORPUS_PATH.read_text())
    articles = data["articles"]
    assert len(articles) >= 10, "fixture corpus degraded"
    return articles


@pytest.fixture(autouse=True)
def _clean_quarantine():
    reset_quarantine_for_test()
    yield
    reset_quarantine_for_test()


# ── C1 (+C2 kill surface): NULL + strict quarantine increment ────────


def test_c1_missing_published_at_yields_null_and_quarantines():
    before = quarantine_count("missing_published_at")
    row = _normalize({"title": "no ts", "url": "https://x.example.com/1"}, "fixture")
    after = quarantine_count("missing_published_at")
    assert row["published_at"] is None, (
        f"published_at was fabricated: {row['published_at']!r}"
    )
    assert after == before + 1, (
        f"quarantine counter did not strictly increase ({before} -> {after})"
    )
    # a quarantined row cannot carry an embargoed entry session either
    assert row["effective_trade_date"] is None
    # and the serialized BQ row keeps the NULL (NULLABLE since the 83.0.1 ALTER)
    assert bq_writer._serialize_article(row)["published_at"] is None


@pytest.mark.parametrize("bad", ["", None, "n/a", "yesterday", "13:45", "2022-99-99T00:00:00"])
def test_c1_parse_rejects_all_missing_and_malformed_shapes(bad):
    assert _parse_published_at(bad) is None


def test_c1_negative_control_valid_timestamp_not_quarantined():
    before = quarantine_count()
    row = _normalize(
        {"title": "ok", "url": "https://x.example.com/2",
         "published_at": "2022-06-08T15:45:00+00:00", "ticker": "MSFT"},
        "fixture",
    )
    assert quarantine_count() == before, "valid article moved the quarantine counter"
    assert row["published_at"] == "2022-06-08T15:45:00+00:00"
    assert row["effective_trade_date"] == "2022-06-09"  # next session after Wed


# ── C3: backfill ingest timestamp is the run moment, not the era ─────


def test_c3_backfill_ingested_at_never_earlier_than_published_at():
    row = _normalize(
        {"title": "2022 article", "url": "https://x.example.com/3",
         "published_at": "2022-03-15T14:30:00+00:00", "ticker": "AAPL"},
        "fixture", provenance="backfill",
    )
    assert row["provenance"] == "backfill"
    pub = datetime.fromisoformat(row["published_at"])
    ing = datetime.fromisoformat(row["ingested_at"])
    assert ing >= pub, f"ingested_at {ing} earlier than published_at {pub}"
    # the sharp half: the ingest stamp is the RUN era (now), not the article's
    assert ing.year >= 2026, (
        f"ingested_at written into the article's own era: {ing.isoformat()}"
    )


# ── C4 + C6: effective_trade_date over the committed corpus ──────────


def _sessions_between(cal, pub_date, eff_date) -> int:
    import pandas as pd
    sessions = cal.sessions_in_range(
        pd.Timestamp(pub_date) + pd.Timedelta(days=1), pd.Timestamp(eff_date)
    )
    return len(sessions)


def test_c4_c6_corpus_minimum_embargo_and_recorded_quarantine():
    from backend.backtest.markets import get_trading_calendar

    cal = get_trading_calendar("US")
    assert cal is not None, "exchange calendar unavailable -- cannot verify embargo"

    reset_quarantine_for_test()
    deltas_days: list[int] = []
    quarantined = 0
    for art in _corpus():
        row = _normalize({k: v for k, v in art.items() if k != "case"}, "fixture")
        if row["published_at"] is None:
            assert row["effective_trade_date"] is None
            quarantined += 1
            continue
        assert row["effective_trade_date"] is not None, art["case"]
        pub_date = datetime.fromisoformat(row["published_at"]).date()
        eff_date = datetime.fromisoformat(row["effective_trade_date"]).date()
        # C4: at least one trading day strictly after the publication date
        n_sessions = _sessions_between(cal, pub_date, eff_date)
        assert n_sessions >= 1, (
            f"{art['case']}: only {n_sessions} sessions in ({pub_date}, {eff_date}]"
        )
        # RuleA exactness: eff is the FIRST session strictly after pub_date
        assert n_sessions == 1, (
            f"{art['case']}: over-embargoed -- {n_sessions} sessions skipped"
        )
        deltas_days.append((eff_date - pub_date).days)

    # C6: RECORD the quarantine count -- no threshold assertion by design
    print(f"\nQUARANTINE RECORDED: {quarantine_count('missing_published_at')} of "
          f"{len(_corpus())} corpus articles (matches loop count {quarantined})")
    assert quarantine_count("missing_published_at") == quarantined
    # the corpus exercises weekend/holiday spans, so deltas must span >1 shape
    assert min(deltas_days) == 1 and max(deltas_days) >= 2, deltas_days


def test_c4_entry_price_is_next_session_open_not_publication_close():
    """The simulated entry anchor over the fixture: session OPEN of the
    embargoed date, which strictly follows publication (a same-day close
    entry would violate the one-session embargo)."""
    from backend.backtest.markets import get_trading_calendar

    cal = get_trading_calendar("US")
    assert cal is not None
    row = _normalize(
        {"title": "intraday", "url": "https://x.example.com/4",
         "published_at": "2022-06-08T15:45:00+00:00", "ticker": "MSFT"},
        "fixture",
    )
    import pandas as pd
    eff = pd.Timestamp(row["effective_trade_date"])
    entry_dt = cal.session_open(eff)
    pub_dt = pd.Timestamp(row["published_at"])
    assert entry_dt > pub_dt, "entry does not strictly follow publication"
    assert entry_dt.date() > pub_dt.date(), "entry landed on the publication day"
    assert (entry_dt.hour, entry_dt.minute) in {(13, 30), (14, 30)}, (
        f"not a session OPEN anchor: {entry_dt}"
    )


# ── D2: per-adapter seam -- missing vendor timestamps reach NULL ─────


class _FakeResp:
    status_code = 200
    text = ""
    content = b"{}"

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSecret:
    def __init__(self, v):
        self._v = v

    def get_secret_value(self):
        return self._v


@pytest.mark.parametrize(
    "mod_name,cls_name,settings_attrs,payload",
    [
        ("backend.news.sources.finnhub", "FinnhubSource",
         {"finnhub_api_key": "test-key"},
         [{"headline": "h", "url": "https://v.example.com/f", "summary": "s"}]),
        ("backend.news.sources.benzinga", "BenzingaSource",
         {"benzinga_api_key": "test-key"},
         [{"title": "h", "url": "https://v.example.com/b", "body": "s"}]),
        ("backend.news.sources.alpaca", "AlpacaSource",
         {"alpaca_api_key_id": _FakeSecret("id"),
          "alpaca_api_secret_key": _FakeSecret("sec")},
         {"news": [{"headline": "h", "url": "https://v.example.com/a", "content": "s"}]}),
    ],
)
def test_adapters_no_longer_fabricate_wall_clock(
    monkeypatch, mod_name, cls_name, settings_attrs, payload
):
    """Drive each REAL adapter fetch() (HTTP + key gates patched at the
    module seams) with a vendor payload whose timestamp field is MISSING and
    assert the yielded RawArticle carries no substituted wall clock and the
    chokepoint quarantines it. Kills the restore-now() adapter mutants (the
    LIVE fabrication site was alpaca -- its keys are set in this env)."""
    import importlib

    mod = importlib.import_module(mod_name)
    real_settings = fetcher and __import__(
        "backend.config.settings", fromlist=["get_settings"]
    ).get_settings()

    class _Settings:
        def __getattr__(self, item):
            if item in settings_attrs:
                return settings_attrs[item]
            return getattr(real_settings, item)

    monkeypatch.setattr(mod, "get_settings", lambda: _Settings())
    monkeypatch.setattr(mod, "retry_with_backoff", lambda fn, **kw: _FakeResp(payload))

    raws = list(getattr(mod, cls_name)().fetch())
    assert raws, f"{mod_name}: adapter yielded nothing through the real fetch()"
    raw = raws[0]
    assert not raw.get("published_at"), (
        f"{mod_name}: adapter substituted a timestamp: {raw.get('published_at')!r}"
    )
    before = quarantine_count("missing_published_at")
    row = _normalize(raw, raw.get("source") or cls_name)
    assert row["published_at"] is None, f"{mod_name}: fabrication survived"
    assert quarantine_count("missing_published_at") == before + 1


# ── fail-CLOSED derivation ───────────────────────────────────────────


def test_derivation_fails_closed_without_calendar(monkeypatch):
    import backend.backtest.markets as markets

    monkeypatch.setattr(markets, "get_trading_calendar", lambda market="US": None)
    assert _derive_effective_trade_date("2022-06-08T15:45:00+00:00", "MSFT") is None
    before = quarantine_count("calendar_unresolvable")
    row = _normalize(
        {"title": "t", "url": "https://x.example.com/5",
         "published_at": "2022-06-08T15:45:00+00:00", "ticker": "MSFT"},
        "fixture",
    )
    assert row["effective_trade_date"] is None
    assert quarantine_count("calendar_unresolvable") == before + 1
