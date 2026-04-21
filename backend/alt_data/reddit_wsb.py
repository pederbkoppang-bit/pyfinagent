"""phase-7.5 Reddit WSB cashtag sentiment -- scaffold.

Persists `(subreddit, cashtag, sentiment_score, sentiment_label, score,
upvote_ratio, author_hash, ...)` rows to `pyfinagent_data.alt_reddit_sentiment`.
This cycle ships the scaffold; live PRAW 7.8.x calls + FinBERT scoring are
deferred to phase-7.12.

Compliance:
- `docs/compliance/reddit-license.md` (this cycle's companion doc)
- `docs/compliance/alt-data.md` row 7.5 + Sec. 2.2 + Sec. 5 + Sec. 6.2
- Advisory `adv_70_oauth_tos` active: OAuth script-app click-through IS contract
  formation. Registration + RBP submission gated to phase-7.12.

Reddit-specific deltas vs twitter.py:
- User-Agent MUST be Reddit format: `python:pyfinagent:1.0 (by /u/pederbkoppang)`
- Cashtag regex floor is 2 chars (`\\$[A-Z]{2,5}\\b`), not 1, to avoid $A/$I noise
- Rate cap: 100 QPM per OAuth client_id -> _RATE_INTERVAL_S = 0.6
- App type is script (not installed) for server pipelines; both client_id +
  client_secret required at phase-7.12 runtime.

CLI:
    python -m backend.alt_data.reddit_wsb [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

# Reddit mandates this exact User-Agent format; deviation triggers aggressive
# throttling per the API wiki. Do NOT reuse backend.alt_data.twitter._USER_AGENT.
_USER_AGENT = "python:pyfinagent:1.0 (by /u/pederbkoppang)"
_TABLE = "alt_reddit_sentiment"
_RATE_INTERVAL_S = 0.6  # 100 QPM free-tier ceiling

_STARTER_SUBS: tuple[str, ...] = ("wallstreetbets", "stocks", "investing")

# Floor is 2 chars (not 1 like twitter.py) to avoid false positives on $A, $I.
_CASHTAG_RE = re.compile(r"\$[A-Z]{2,5}\b")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  post_id STRING NOT NULL,
  as_of_date DATE NOT NULL,
  subreddit STRING,
  author_hash STRING,
  cashtag STRING,
  title STRING,
  text STRING,
  sentiment_score FLOAT64,
  sentiment_label STRING,
  score INT64,
  upvote_ratio FLOAT64,
  created_at TIMESTAMP,
  source STRING,
  raw_payload JSON
)
PARTITION BY as_of_date
CLUSTER BY subreddit, cashtag
OPTIONS (
  description = "phase-7.5 Reddit WSB + stocks + investing sentiment; fetch + model deferred to phase-7.12"
)
""".strip()


def extract_cashtags(text: str) -> list[str]:
    """Return uppercase cashtags (2-5 chars, e.g. '$AAPL') found in text."""
    if not text:
        return []
    return [m.group(0) for m in _CASHTAG_RE.finditer(text)]


def _hash_author(author_name: str | None) -> str | None:
    """PII discipline: sha256 the Reddit username; never persist raw."""
    if author_name is None or author_name == "[deleted]":
        return None
    return hashlib.sha256(str(author_name).encode("utf-8")).hexdigest()


def fetch_wsb_posts(
    subreddit: str,
    *,
    limit: int = 100,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Scaffold -- deferred to phase-7.12.

    Live impl will use PRAW 7.8.x:
        import praw
        reddit = praw.Reddit(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=_USER_AGENT,
        )
        for post in reddit.subreddit(subreddit).new(limit=limit):
            yield {
                "post_id": post.id,
                "author": str(post.author) if post.author else None,
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "created_utc": post.created_utc,
            }

    NOTE: Script-app OAuth; NOT installed-app. client_secret required.
    NOTE: env vars must be read inside this function, NOT at import time.
    Returns [] until implemented.
    """
    logger.debug("reddit_wsb: fetch_wsb_posts scaffold subreddit=%s", subreddit)
    return []


def score_sentiment(text: str) -> tuple[float, str]:
    """Scaffold -- returns (0.0, 'neutral') until phase-7.12 loads FinBERT."""
    return 0.0, "neutral"


def _resolve_target(project: str | None, dataset: str | None) -> tuple[str, str]:
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
            logger.warning("reddit_wsb: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_bq_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("reddit_wsb: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("reddit_wsb: bigquery.Client() init failed (%r)", exc)
        return None


def ensure_table(*, project: str | None = None, dataset: str | None = None) -> bool:
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return False
    sql = _CREATE_TABLE_SQL.format(project=proj, dataset=ds, table=_TABLE)
    try:
        client.query(sql).result(timeout=60)
        return True
    except Exception as exc:
        logger.warning("reddit_wsb: ensure_table fail-open: %r", exc)
        return False


def upsert(
    rows: list[dict[str, Any]],
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    if not rows:
        return 0
    proj, ds = _resolve_target(project, dataset)
    client = _get_bq_client(proj)
    if client is None:
        return 0
    table_ref = f"{proj}.{ds}.{_TABLE}" if proj else f"{ds}.{_TABLE}"
    try:
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            logger.warning("reddit_wsb: insert errors: %s", errors[:3])
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("reddit_wsb: upsert fail-open: %r", exc)
        return 0


def ingest_subreddit(
    sub: str,
    *,
    project: str | None = None,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    """Scaffold orchestrator. Returns 0 until phase-7.12 wires live PRAW + model."""
    today_iso = date.today().isoformat()
    rows: list[dict[str, Any]] = []
    for post in fetch_wsb_posts(sub):
        title = post.get("title") or ""
        body = post.get("selftext") or ""
        tags = extract_cashtags(title) + extract_cashtags(body)
        # Emit one row per cashtag mention (zero rows if no cashtags).
        for tag in tags or [None]:
            if tag is None:
                continue
            score, label = score_sentiment(f"{title}\n\n{body}")
            rows.append(
                {
                    "post_id": post.get("post_id"),
                    "as_of_date": today_iso,
                    "subreddit": sub,
                    "author_hash": _hash_author(post.get("author")),
                    "cashtag": tag,
                    "title": title[:1000],
                    "text": body[:2000],
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "score": post.get("score"),
                    "upvote_ratio": post.get("upvote_ratio"),
                    "created_at": post.get("created_utc"),
                    "source": "reddit.com/api",
                    "raw_payload": json.dumps(post, default=str, ensure_ascii=True),
                }
            )
    if dry_run:
        return len(rows)
    return upsert(rows, project=project, dataset=dataset)


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="phase-7.5 Reddit WSB sentiment ingester (scaffold)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    total = 0
    for sub in _STARTER_SUBS:
        total += ingest_subreddit(sub, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "dry_run": args.dry_run,
                "ingested": total,
                "scaffold_only": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
