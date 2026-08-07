"""phase-6.2 news fetcher core.

`run_once(source_names=None, dry_run=False) -> FetchReport` is the
single orchestration entry point.

1. Iterate registered sources (filtered by `source_names` if given).
2. Call `.fetch()` on each.
3. Normalize each raw article -> NormalizedArticle dict matching the
   phase-6.1 BigQuery `news_articles` schema.
4. Append to the batch (dedup is phase-6.4).
5. If `dry_run`, skip BQ write; return the FetchReport.
6. If NOT dry_run, call `_write_batch_to_bq(batch)` (the caller must
   pass BQ auth; live writes wired in phase-6.8 smoketest).

Also defines the built-in `StubSource` used by the contract's inline
smoke test to prove the pipeline end-to-end without a real network
call.
"""
from __future__ import annotations

import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TypedDict

# Allow `python backend/news/fetcher.py` (direct invocation) by
# prepending the repo root to sys.path so the absolute `backend.*`
# imports below resolve. The `-m backend.news.fetcher` form works
# without this because `python -m` adds cwd to sys.path automatically.
if __package__ in (None, ""):
    _REPO = Path(__file__).resolve().parents[2]
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

from backend.news.normalize import body_hash, canonical_url
from backend.news.registry import get_sources, register

logger = logging.getLogger(__name__)


class RawArticle(TypedDict, total=False):
    """Shape each source's .fetch() is expected to yield."""
    source: str
    title: str
    body: str
    url: str
    published_at: str     # ISO 8601
    ticker: str
    language: str
    authors: list[str]
    categories: list[str]
    raw_payload: dict[str, Any]


class NormalizedArticle(TypedDict, total=False):
    """Row shape that maps 1:1 to the news_articles schema (phase-83.0/.1)."""
    article_id: str
    published_at: str | None
    ingested_at: str
    provenance: str
    effective_trade_date: str | None
    source: str
    ticker: str | None
    title: str
    body: str
    url: str
    canonical_url: str
    body_hash: str
    language: str | None
    authors: list[str]
    categories: list[str]
    raw_payload: dict[str, Any]


@dataclass
class FetchReport:
    n_sources: int
    n_articles: int
    per_source_counts: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    articles: list[NormalizedArticle] = field(default_factory=list)
    dry_run: bool = False
    # phase-6.4 additions
    n_deduped: int = 0
    dedup_dropped_url: int = 0
    dedup_dropped_hash: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── phase-83.0.1: point-in-time integrity ────────────────────────────
# A missing/unparseable publication timestamp is stored as NULL and the row
# quarantined -- NEVER substituted with the wall clock (a backfill would
# stamp today's date onto a years-old article with no way to detect it).
# The predicate is PARSE-based at this single chokepoint because the three
# vendor adapters historically substituted now() upstream and malformed
# non-empty strings pass every presence check.

import threading

_EMBARGO_DAYS = 1  # one-session embargo: entry at the NEXT session's open

_QUARANTINE: dict[str, int] = {}  # reason -> count
_quarantine_lock = threading.Lock()


def quarantine_count(reason: str | None = None) -> int:
    """Quarantined-article count for `reason`, or the total when None."""
    with _quarantine_lock:
        if reason is None:
            return sum(_QUARANTINE.values())
        return _QUARANTINE.get(reason, 0)


def reset_quarantine_for_test() -> None:
    """Test-only helper: zero all quarantine counters."""
    with _quarantine_lock:
        _QUARANTINE.clear()


def _quarantine(reason: str, source: str, detail: str) -> None:
    """Count + log one quarantined article. Never raises."""
    with _quarantine_lock:
        _QUARANTINE[reason] = _QUARANTINE.get(reason, 0) + 1
        n = _QUARANTINE[reason]
    logger.warning(
        "news quarantine reason=%s source=%s count=%d detail=%s",
        reason, source, n, detail[:200],
    )


def _parse_published_at(raw_value: Any) -> str | None:
    """Strict ISO-8601 parse. None on missing/empty/malformed -- no fallback."""
    if raw_value is None:
        return None
    s = str(raw_value).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        # Naive vendor timestamps are treated as UTC (documented assumption);
        # this changes no ordering and avoids rejecting entire vendor feeds.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _derive_effective_trade_date(published_at_iso: str | None, ticker: str | None) -> str | None:
    """First trading session STRICTLY AFTER the publication UTC date.

    RuleA: cal.date_to_session(pub_date + _EMBARGO_DAYS, direction="next") --
    measured over 2022-01-01..2026-06-30: zero violations, min sessions in
    (pub, eff] exactly 1. FAIL-CLOSED: no parseable timestamp, no calendar,
    or any derivation error -> None (caller quarantines). Deliberately does
    NOT use markets.is_trading_day, which fails OPEN when the calendar lib
    is absent -- correct for "never block a trade", wrong for an embargo.
    """
    if published_at_iso is None:
        return None
    try:
        import pandas as pd

        from backend.backtest.markets import get_trading_calendar, market_for_symbol

        market = market_for_symbol(ticker) if ticker else "US"
        cal = get_trading_calendar(market)
        if cal is None:
            return None
        pub_date = datetime.fromisoformat(published_at_iso).date()
        target = pd.Timestamp(pub_date) + pd.Timedelta(days=_EMBARGO_DAYS)
        eff = cal.date_to_session(target, direction="next")
        return eff.date().isoformat()
    except Exception as exc:
        logger.warning("effective_trade_date derivation failed: %r", exc)
        return None


def _normalize(
    raw: RawArticle, source_name: str, provenance: str = "live"
) -> NormalizedArticle:
    url = str(raw.get("url") or "")
    body = str(raw.get("body") or "")
    # phase-83.0.1: strict parse -> NULL + quarantine, never wall-clock.
    published_at = _parse_published_at(raw.get("published_at"))
    if published_at is None:
        _quarantine(
            "missing_published_at", source_name,
            f"raw value {raw.get('published_at')!r} url={url}",
        )
        effective_trade_date = None
    else:
        effective_trade_date = _derive_effective_trade_date(
            published_at, raw.get("ticker")
        )
        if effective_trade_date is None:
            # parseable timestamp but no derivable session: fail-CLOSED.
            _quarantine(
                "calendar_unresolvable", source_name,
                f"published_at={published_at} ticker={raw.get('ticker')!r}",
            )
    return NormalizedArticle(
        article_id=str(uuid.uuid4()),
        published_at=published_at,
        # phase-83.0: renamed from fetched_at. phase-83.0.1: for backfill runs
        # this is the backfill-RUN moment -- a real ingest event, >= published_at
        # for any historical article -- never the article's own era.
        ingested_at=_now_iso(),
        provenance=provenance,
        effective_trade_date=effective_trade_date,
        source=source_name,
        ticker=raw.get("ticker"),
        title=str(raw.get("title") or "")[:2000],
        body=body,
        url=url,
        canonical_url=canonical_url(url),
        body_hash=body_hash(body),
        language=raw.get("language"),
        authors=list(raw.get("authors") or []),
        categories=list(raw.get("categories") or []),
        raw_payload=dict(raw.get("raw_payload") or {}),
    )


def _write_batch_to_bq(batch: list[NormalizedArticle]) -> int:
    """phase-6.8: live BQ writer.

    Delegates to `backend.news.bq_writer.write_news_articles` which
    uses `client.insert_rows_json`, fails-open on missing deps / auth,
    and returns rows inserted. Import is function-scoped so
    `fetcher.run_once(dry_run=True)` unit tests do NOT require the
    google-cloud-bigquery package.
    """
    try:
        from backend.news.bq_writer import write_news_articles
        return write_news_articles(batch)
    except Exception as exc:  # pragma: no cover -- fail-open
        logger.warning("fetcher: _write_batch_to_bq fail-open err=%r", exc)
        return 0


def run_once(
    source_names: list[str] | None = None,
    dry_run: bool = False,
    dedup: bool = True,
    provenance: str = "live",
) -> FetchReport:
    """Run one fetcher pass across registered sources.

    Args:
        source_names: optional subset of registered source names.
        dry_run: if True, skip the BQ write.
        dedup: if True (default), apply phase-6.4 intra-batch dedup
          on `canonical_url` / `body_hash` before the BQ-write guard.
        provenance: "live" (default) or "backfill" -- stamped onto every
          row (phase-83.0) so a backfilled row is never indistinguishable
          from a live-captured one.
    """
    sources = get_sources(source_names)
    report = FetchReport(
        n_sources=len(sources),
        n_articles=0,
        dry_run=bool(dry_run),
    )
    for name, src in sources.items():
        count = 0
        try:
            for raw in src.fetch():
                report.articles.append(
                    _normalize(raw, source_name=name, provenance=provenance)
                )
                count += 1
        except Exception as e:
            report.errors.append({"source": name, "error": f"{type(e).__name__}: {e}"})
            logger.warning("news fetch failed for %s: %s", name, e)
        report.per_source_counts[name] = count

    if dedup and report.articles:
        from backend.news.dedup import dedup_intra_batch
        kept, dedup_report = dedup_intra_batch(report.articles)
        report.n_deduped = dedup_report.n_in - dedup_report.n_kept
        report.dedup_dropped_url = dedup_report.n_dropped_url
        report.dedup_dropped_hash = dedup_report.n_dropped_hash
        report.articles = kept

    report.n_articles = len(report.articles)

    if not dry_run and report.articles:
        _write_batch_to_bq(report.articles)

    return report


# ═══════════════════════════════════════════════════════════════════
# Built-in StubSource (registered unconditionally so the smoke test
# runs without a real adapter). phase-6.3 will register Finnhub /
# Benzinga / Alpaca alongside it.
# ═══════════════════════════════════════════════════════════════════


@register("stub")
class StubSource:
    name = "stub"

    def fetch(self) -> Iterable[RawArticle]:
        yield RawArticle(
            source="stub",
            title="AAPL beats expectations on services revenue",
            body="<p>Apple reported Q1 services revenue of $X, beating estimates.</p>",
            url="https://example.com/aapl-earnings?utm_source=rss&id=1",
            published_at="2026-04-19T14:00:00+00:00",
            ticker="AAPL",
            authors=["Test Author"],
            categories=["earnings", "tech"],
            raw_payload={"origin": "stub", "id": 1},
        )
        yield RawArticle(
            source="stub",
            title="Fed signals rate path caution",
            body="Fed officials reiterated a data-dependent stance on rates.",
            url="https://example.com/fed-update/?fbclid=abc&ref=twitter",
            published_at="2026-04-19T13:30:00+00:00",
            ticker=None,
            authors=[],
            categories=["macro"],
            raw_payload={"origin": "stub", "id": 2},
        )
        yield RawArticle(
            source="stub",
            title="MSFT announces AI partnership",
            body="Microsoft announced a new AI partnership spanning Azure and Copilot.",
            url="https://example.com/msft-ai",
            published_at="2026-04-19T12:00:00+00:00",
            ticker="MSFT",
            authors=["Stub Reporter"],
            categories=["tech", "ai"],
            raw_payload={"origin": "stub", "id": 3},
        )


# ═══════════════════════════════════════════════════════════════════
# Inline smoke-test (runs when the module is executed directly).
# No pytest dependency so contract verification is self-contained.
# ═══════════════════════════════════════════════════════════════════


def _smoke() -> int:
    # Canonical URL strips trackers + sorts remaining params.
    u = canonical_url("https://Example.com/a?utm_source=foo&id=1")
    assert u == "http://example.com/a?id=1" or u == "https://example.com/a?id=1", u
    # The stub uses https; the canonical_url function preserves scheme.
    u2 = canonical_url("https://X.com/path/?utm_source=foo&z=2&a=1")
    assert u2 == "https://x.com/path?a=1&z=2", u2

    # body_hash: same input -> same; different input -> different.
    h1 = body_hash("<p>Hello World</p>")
    h2 = body_hash("hello    world")
    h3 = body_hash("Goodbye world")
    assert h1 == h2, (h1, h2)
    assert h1 != h3

    # Registry + fetcher end-to-end (stub source).
    report = run_once(["stub"], dry_run=True)
    assert report.n_sources == 1
    assert report.n_articles == 3
    assert report.per_source_counts == {"stub": 3}
    for a in report.articles:
        assert a["article_id"]
        assert a["canonical_url"]
        assert a["body_hash"]
        assert a["ingested_at"]
        assert a["provenance"] == "live"
        assert a["source"] == "stub"
        # phase-83.0.1: stub articles carry valid timestamps -> parsed, not
        # quarantined, and an embargoed session is derived.
        assert a["published_at"] is not None
        assert a["effective_trade_date"] is not None
        assert a["effective_trade_date"] > a["published_at"][:10]
    # UTM stripped from first article URL.
    assert "utm_source" not in report.articles[0]["canonical_url"]
    # fbclid + ref stripped from second.
    assert "fbclid" not in report.articles[1]["canonical_url"]
    assert "ref=" not in report.articles[1]["canonical_url"]

    print("phase-6.2 smoke: OK")
    print(f"  n_articles={report.n_articles}")
    print(f"  per_source_counts={report.per_source_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
