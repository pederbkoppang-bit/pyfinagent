"""phase-6.4 news dedup layer.

Two-phase dedup pattern:
1. `dedup_intra_batch(articles)` -- in-memory seen-set filter. Keeps
   first occurrence, drops any later article whose `canonical_url`
   OR `body_hash` matches a prior anchor.
2. `dedup_against_bq(articles, bq_client=None, ...)` -- cross-batch
   filter against the `news_articles` BQ table. When `bq_client is
   None`, returns the input unchanged (safe in dry-run / unit tests).

Drops on EITHER anchor match, not both -- an article with a seen
URL but unseen hash is still a duplicate.

Empty `canonical_url` / `body_hash` are NOT treated as anchors
(all empty strings would otherwise collide on the first one seen).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DedupReport:
    n_in: int = 0
    n_kept: int = 0
    n_dropped_url: int = 0
    n_dropped_hash: int = 0
    reasons: list[str] = field(default_factory=list)


def dedup_intra_batch(articles: list[dict]) -> tuple[list[dict], DedupReport]:
    """Drop duplicates within a single batch on canonical_url OR body_hash.

    Returns `(kept, report)`. `kept` preserves the input order for
    first-occurrence entries.
    """
    report = DedupReport(n_in=len(articles))
    if not articles:
        return [], report

    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    kept: list[dict] = []
    for a in articles:
        url = (a.get("canonical_url") or "").strip()
        h = (a.get("body_hash") or "").strip()
        if url and url in seen_urls:
            report.n_dropped_url += 1
            report.reasons.append(f"url_dup:{url[:80]}")
            continue
        if h and h in seen_hashes:
            report.n_dropped_hash += 1
            report.reasons.append(f"hash_dup:{h[:16]}")
            continue
        if url:
            seen_urls.add(url)
        if h:
            seen_hashes.add(h)
        kept.append(a)
    report.n_kept = len(kept)
    return kept, report


def dedup_against_bq(
    articles: list[dict],
    bq_client: Any = None,
    dataset: str = "pyfinagent_data",
    table: str = "news_articles",
    lookback_days: int = 7,
) -> list[dict]:
    """Filter articles against recent rows in `news_articles`.

    When `bq_client is None`, this is a no-op and the full input is
    returned -- this keeps phase-6.4 safe in dry-run contexts. Real
    BQ integration is wired in phase-6.8 smoketest when the caller
    passes an authenticated `bigquery.Client`.

    Drops on EITHER `canonical_url` OR `body_hash` match against the
    lookback window. BQ streaming buffer is a known caveat -- rows
    written in the same run may not be visible; acceptable at this
    phase, documented in the research brief.
    """
    if not articles or bq_client is None:
        return list(articles)

    urls = tuple({(a.get("canonical_url") or "").strip() for a in articles if a.get("canonical_url")})
    hashes = tuple({(a.get("body_hash") or "").strip() for a in articles if a.get("body_hash")})
    if not urls and not hashes:
        return list(articles)

    sql = (
        f"SELECT canonical_url, body_hash "
        f"FROM `{bq_client.project}.{dataset}.{table}` "
        f"WHERE published_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY) "
        f"AND (canonical_url IN UNNEST(@urls) OR body_hash IN UNNEST(@hashes))"
    )
    try:
        from google.cloud import bigquery  # type: ignore
        params = [
            bigquery.ScalarQueryParameter("days", "INT64", int(lookback_days)),
            bigquery.ArrayQueryParameter("urls", "STRING", list(urls)),
            bigquery.ArrayQueryParameter("hashes", "STRING", list(hashes)),
        ]
        job = bq_client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
        rows = list(job.result(timeout=30))
    except Exception as e:
        logger.warning("dedup_against_bq lookup failed: %s: %s", type(e).__name__, e)
        return list(articles)

    seen_urls = {r.get("canonical_url") for r in rows if r.get("canonical_url")}
    seen_hashes = {r.get("body_hash") for r in rows if r.get("body_hash")}

    kept = []
    for a in articles:
        u = (a.get("canonical_url") or "").strip()
        h = (a.get("body_hash") or "").strip()
        if u and u in seen_urls:
            continue
        if h and h in seen_hashes:
            continue
        kept.append(a)
    return kept


# ═══════════════════════════════════════════════════════════════════
# Inline smoke (runs when this module is invoked directly).
# ═══════════════════════════════════════════════════════════════════


def _smoke() -> int:
    sample = [
        {"canonical_url": "https://x.com/a", "body_hash": "AAA"},
        {"canonical_url": "https://x.com/b", "body_hash": "BBB"},
        {"canonical_url": "https://x.com/c", "body_hash": "CCC"},
        # URL duplicate of #1
        {"canonical_url": "https://x.com/a", "body_hash": "DDD"},
        # Hash duplicate of #2
        {"canonical_url": "https://x.com/e", "body_hash": "BBB"},
    ]
    kept, report = dedup_intra_batch(sample)
    assert len(kept) == 3, f"expected 3 kept, got {len(kept)}"
    assert report.n_dropped_url == 1
    assert report.n_dropped_hash == 1
    assert report.n_in == 5

    # Empty anchors must not collide
    empty = [
        {"canonical_url": "", "body_hash": ""},
        {"canonical_url": "", "body_hash": ""},
    ]
    kept2, _ = dedup_intra_batch(empty)
    assert len(kept2) == 2, "empty anchors should NOT be treated as duplicates"

    # bq_client=None -> no-op
    out = dedup_against_bq(sample, bq_client=None)
    assert len(out) == len(sample)

    print("phase-6.4 dedup smoke: OK")
    print(f"  intra_batch: n_in={report.n_in} n_kept={report.n_kept} "
          f"dropped_url={report.n_dropped_url} dropped_hash={report.n_dropped_hash}")
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(_smoke())
