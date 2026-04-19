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
from backend.news.registry import NewsSource, clear_registry, get_sources, register

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
    """Row shape that maps 1:1 to the phase-6.1 news_articles schema."""
    article_id: str
    published_at: str
    fetched_at: str
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


def _normalize(raw: RawArticle, source_name: str) -> NormalizedArticle:
    url = str(raw.get("url") or "")
    body = str(raw.get("body") or "")
    return NormalizedArticle(
        article_id=str(uuid.uuid4()),
        published_at=str(raw.get("published_at") or _now_iso()),
        fetched_at=_now_iso(),
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
) -> FetchReport:
    """Run one fetcher pass across registered sources.

    Args:
        source_names: optional subset of registered source names.
        dry_run: if True, skip the BQ write.
        dedup: if True (default), apply phase-6.4 intra-batch dedup
          on `canonical_url` / `body_hash` before the BQ-write guard.
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
                report.articles.append(_normalize(raw, source_name=name))
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
        assert a["fetched_at"]
        assert a["source"] == "stub"
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
