"""phase-7.6 Twitter/X cashtag sentiment -- scaffold.

Plans to persist `(cashtag, sentiment_score, sentiment_label, ...)` rows to
`pyfinagent_data.alt_twitter_sentiment`. This cycle ships the scaffold; live
X API v2 `/2/tweets/search/recent` calls + FinBERT scoring are deferred to
phase-7.12.

Compliance: `docs/compliance/alt-data.md` row 7.6 -- X API v2 with OAuth app
key (paid tier for volume). Per advisory `adv_70_oauth_tos`, the developer
app registration happens in phase-7.12 (click-through = contract formation;
not done at scaffold time). PII: author_id is sha256'd before persistence
per compliance Section 5.5.

Pricing flag (research brief): Basic ~$100-200/mo may BLOCK the cashtag
operator; Pro ($5K/mo) confirmed safe. TODO phase-7.12 verify tier.

CLI:
    python -m backend.alt_data.twitter [--dry-run]
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

_USER_AGENT = "pyfinagent/1.0 peder.bkoppang@hotmail.no"
_TABLE = "alt_twitter_sentiment"

_STARTER_CASHTAGS: tuple[str, ...] = ("$SPY", "$QQQ", "$AAPL", "$TSLA", "$NVDA")
_CASHTAG_RE = re.compile(r"\$[A-Z]{1,5}\b")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  tweet_id STRING NOT NULL,
  as_of_date DATE NOT NULL,
  cashtag STRING,
  author_id_hash STRING,
  text STRING,
  sentiment_score FLOAT64,
  sentiment_label STRING,
  created_at TIMESTAMP,
  source STRING,
  raw_payload JSON
)
PARTITION BY as_of_date
CLUSTER BY cashtag, author_id_hash
OPTIONS (
  description = "phase-7.6 X/Twitter cashtag sentiment; fetch + model deferred to phase-7.12"
)
""".strip()


def extract_cashtags(text: str) -> list[str]:
    """Return all uppercase cashtags (e.g. '$AAPL') found in text."""
    if not text:
        return []
    return [m.group(0) for m in _CASHTAG_RE.finditer(text)]


def _hash_author(author_id: str | int | None) -> str | None:
    """PII discipline: never persist raw author_id. sha256 hash."""
    if author_id is None:
        return None
    return hashlib.sha256(str(author_id).encode("utf-8")).hexdigest()


def fetch_cashtag_tweets(
    cashtag: str,
    *,
    since: str | None = None,
    max_results: int = 100,
) -> list[dict[str, Any]]:
    """Scaffold -- deferred to phase-7.12.

    Live impl will call `GET /2/tweets/search/recent?query={cashtag}&max_results=...`
    with an app-only Bearer token (env var X_BEARER_TOKEN). Returns [] until
    implemented.
    """
    logger.debug("twitter: fetch_cashtag_tweets scaffold cashtag=%s", cashtag)
    return []


def score_sentiment(text: str) -> tuple[float, str]:
    """Scaffold -- deferred to phase-7.12.

    Live impl will load `ProsusAI/finbert` and return a softmax score + label
    over {positive, neutral, negative}. For now returns the neutral prior.
    """
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
            logger.warning("twitter: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_bq_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("twitter: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("twitter: bigquery.Client() init failed (%r)", exc)
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
        logger.warning("twitter: ensure_table fail-open: %r", exc)
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
            logger.warning("twitter: insert errors: %s", errors[:3])
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("twitter: upsert fail-open: %r", exc)
        return 0


def ingest_cashtags(
    cashtags: Iterable[str] = _STARTER_CASHTAGS,
    *,
    project: str | None = None,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    """Scaffold orchestrator. Returns 0 until phase-7.12 wires live fetch + model."""
    today_iso = date.today().isoformat()
    rows: list[dict[str, Any]] = []
    for tag in cashtags:
        tweets = fetch_cashtag_tweets(tag)
        for t in tweets:
            text = t.get("text") or ""
            score, label = score_sentiment(text)
            rows.append(
                {
                    "tweet_id": t.get("id"),
                    "as_of_date": today_iso,
                    "cashtag": tag,
                    "author_id_hash": _hash_author(t.get("author_id")),
                    "text": text,
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "created_at": t.get("created_at"),
                    "source": "x.com/api/2",
                    "raw_payload": json.dumps(t, default=str, ensure_ascii=True),
                }
            )
    if dry_run:
        return len(rows)
    return upsert(rows, project=project, dataset=dataset)


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="phase-7.6 Twitter/X sentiment ingester (scaffold)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    count = ingest_cashtags(dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "dry_run": args.dry_run,
                "ingested": count,
                "scaffold_only": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
