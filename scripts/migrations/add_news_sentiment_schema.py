"""phase-6.1 migration: create `news_articles` + `news_sentiment` BQ tables.

Two tables in `pyfinagent_data` (configurable via
`settings.bq_dataset_observability` fallback):

`news_articles` -- append-only raw fact table:
    article_id STRING NOT NULL       -- uuid4 surrogate
    published_at TIMESTAMP NOT NULL  -- source-asserted timestamp
    fetched_at TIMESTAMP NOT NULL    -- our ingestion time
    source STRING NOT NULL           -- finnhub | benzinga | alpaca | manual
    ticker STRING                    -- optional primary ticker
    title STRING
    body STRING                      -- up to 1MB
    url STRING
    canonical_url STRING             -- dedup anchor
    body_hash STRING                 -- sha256 dedup anchor
    language STRING
    authors ARRAY<STRING>
    categories ARRAY<STRING>
    raw_payload JSON                 -- original API row for audit
    PARTITION BY DATE(published_at)
    CLUSTER BY source, ticker

`news_sentiment` -- re-scorable enrichment table, joined on article_id:
    article_id STRING NOT NULL       -- FK to news_articles.article_id
    scorer_model STRING NOT NULL     -- gemini-2.0-flash | claude-haiku-4-5 | finbert | vader
    scorer_version STRING
    scored_at TIMESTAMP NOT NULL
    sentiment_score FLOAT64          -- normalised to [-1, +1]
    sentiment_label STRING           -- bullish | bearish | neutral | mixed
    confidence FLOAT64               -- [0, 1]
    latency_ms FLOAT64
    cost_usd FLOAT64
    raw_output STRING                -- truncated verbatim scorer output
    PARTITION BY DATE(scored_at)
    CLUSTER BY article_id, scorer_model

Design choices (see `handoff/current/phase-6.1-research-brief.md`):
- Daily (not hourly) partitions: news volume is < 6 mo historical window
  on the hot path; hourly adds cost without scan-reduction benefit.
- Two tables (not inlined sentiment): re-scoring is a first-class op;
  body columns are expensive to scan in a wide fact table; FNSPID /
  FinBERT split article storage from scoring.
- `raw_payload JSON` (not STRING): BQ Standard SQL JSON type enables
  JSON_EXTRACT_SCALAR downstream without re-parsing.
- Dedup logic is NOT in this migration. Lives in the ingestion cron
  (phase-6.2+) using `canonical_url` + `body_hash` as anchors.

Run:
    python scripts/migrations/add_news_sentiment_schema.py           # execute
    python scripts/migrations/add_news_sentiment_schema.py --dry-run # print DDL only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config.settings import get_settings


DDL_NEWS_ARTICLES = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.news_articles` (
  article_id STRING NOT NULL,
  published_at TIMESTAMP NOT NULL,
  fetched_at TIMESTAMP NOT NULL,
  source STRING NOT NULL,
  ticker STRING,
  title STRING,
  body STRING,
  url STRING,
  canonical_url STRING,
  body_hash STRING,
  language STRING,
  authors ARRAY<STRING>,
  categories ARRAY<STRING>,
  raw_payload JSON
)
PARTITION BY DATE(published_at)
CLUSTER BY source, ticker
OPTIONS (
  description = "phase-6.1 news ingestion fact table (append-only)"
)
"""


DDL_NEWS_SENTIMENT = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.news_sentiment` (
  article_id STRING NOT NULL,
  scorer_model STRING NOT NULL,
  scorer_version STRING,
  scored_at TIMESTAMP NOT NULL,
  sentiment_score FLOAT64,
  sentiment_label STRING,
  confidence FLOAT64,
  latency_ms FLOAT64,
  cost_usd FLOAT64,
  raw_output STRING
)
PARTITION BY DATE(scored_at)
CLUSTER BY article_id, scorer_model
OPTIONS (
  description = "phase-6.1 news-sentiment scorer output (re-scorable)"
)
"""


def main(dry_run: bool) -> int:
    settings = get_settings()
    project = settings.gcp_project_id
    dataset = (
        getattr(settings, "bq_dataset_observability", None)
        or "pyfinagent_data"
    )

    ddls = [
        ("news_articles", DDL_NEWS_ARTICLES.format(project=project, dataset=dataset)),
        ("news_sentiment", DDL_NEWS_SENTIMENT.format(project=project, dataset=dataset)),
    ]

    for table, sql in ddls:
        banner = f"== {table} ({'dry-run' if dry_run else 'live'}) =="
        print(banner)
        print(sql.strip())
        print()

    if dry_run:
        print("dry-run: no BigQuery writes executed.")
        return 0

    from google.cloud import bigquery  # type: ignore
    client = bigquery.Client(project=project)
    for table, sql in ddls:
        print(f"executing DDL for {table}...")
        job = client.query(sql)
        job.result(timeout=60)
        print(f"OK: {project}.{dataset}.{table} ready.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
