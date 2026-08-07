"""phase-6.1 migration: create `news_articles` + `news_sentiment` BQ tables.

Two tables in `pyfinagent_data` (configurable via
`settings.bq_dataset_observability` fallback):

`news_articles` -- append-only raw fact table:
    article_id STRING NOT NULL       -- uuid4 surrogate
    published_at TIMESTAMP           -- source-asserted timestamp; phase-83.0.1: NULLABLE --
                                     -- a missing/unparseable publication time is stored as
                                     -- NULL and quarantined, NEVER wall-clock-fabricated.
                                     -- NULL rows land in the __NULL__ partition.
    ingested_at TIMESTAMP NOT NULL   -- our ingestion time (phase-83.0: renamed from fetched_at)
    effective_trade_date DATE        -- phase-83.0.1: first session STRICTLY AFTER the
                                     -- publication UTC date (one-session embargo); NULL
                                     -- when quarantined or calendar unresolvable (fail-CLOSED)
    provenance STRING NOT NULL       -- phase-83.0: {live, backfill}; REQUIRED can only be
                                     -- created WITH the table (BQ forbids adding REQUIRED
                                     -- to an existing schema), hence the one-shot DDL here
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
    ingested_at TIMESTAMP NOT NULL   -- phase-83.0: when the score row was written
    provenance STRING NOT NULL       -- phase-83.0: {live, backfill}
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
  published_at TIMESTAMP,
  ingested_at TIMESTAMP NOT NULL,
  provenance STRING NOT NULL,
  effective_trade_date DATE,
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
  ingested_at TIMESTAMP NOT NULL,
  provenance STRING NOT NULL,
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


# phase-83.0: post-condition map per table. CREATE TABLE IF NOT EXISTS is a
# NO-OP on an existing table, so the DDL's exit code proves nothing on re-run;
# the migration reads the schema back and fails LOUD on drift instead of
# printing "OK: ... ready." over a table it never touched.
REQUIRED_MODES: dict[str, dict[str, str]] = {
    "news_articles": {
        # phase-83.0.1: published_at relaxed to NULLABLE (quarantine rows store
        # NULL, never a wall-clock fabrication). Relaxation is one-way in BQ.
        "published_at": "NULLABLE",
        "ingested_at": "REQUIRED",
        "provenance": "REQUIRED",
        "effective_trade_date": "NULLABLE",
    },
    "news_sentiment": {
        "scored_at": "REQUIRED",
        "ingested_at": "REQUIRED",
        "provenance": "REQUIRED",
    },
}


def verify_post_condition(client, fq_table: str, required: dict[str, str]) -> None:
    """Read the live schema back and assert the required column modes.

    A REQUIRED column cannot be added to an existing table (BQ docs), so a
    failure here is not retryable by re-running: the table must be dropped
    and recreated, which needs owner approval per CLAUDE.md BQ rule 4.
    """
    schema = {f.name: f.mode for f in client.get_table(fq_table).schema}
    if not schema:
        raise SystemExit(f"MIGRATION POST-CONDITION FAILED {fq_table}: empty schema read")
    for col, mode in required.items():
        got = schema.get(col)
        if got != mode:
            raise SystemExit(
                f"MIGRATION POST-CONDITION FAILED {fq_table}.{col}: "
                f"want mode={mode}, got {got!r}. "
                "CREATE TABLE IF NOT EXISTS is a NO-OP on an existing table and "
                "a REQUIRED column cannot be added afterwards -- the table must "
                "be dropped and recreated (owner approval required)."
            )
    print(f"post-condition OK: {fq_table} carries {sorted(required)} as required")


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
    for table in REQUIRED_MODES:
        verify_post_condition(client, f"{project}.{dataset}.{table}", REQUIRED_MODES[table])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
