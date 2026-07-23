"""phase-25.B7 Migration: create pyfinagent_data.data_source_events.

Idempotent CREATE TABLE IF NOT EXISTS. Defaults to dry-run (prints SQL).
Pass --apply to execute via the BigQuery client (requires ADC).

The table records per-cycle data-source provenance events: each time the
analysis pipeline falls back from the primary source (Alpha Vantage) to a
secondary source (yfinance), one row is appended. The aggregable counter
`pct_yfinance_fallback_dominance` is then a trivial query:

    SELECT COUNTIF(source = 'yfinance_fallback') / COUNT(*)
    FROM `sunny-might-477607-p8.pyfinagent_data.data_source_events`
    WHERE DATE(event_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)

Schema:
    event_id      STRING    NOT NULL  (UUID per event)
    event_time    TIMESTAMP NOT NULL  (when the fallback fired)
    ticker        STRING    NOT NULL  (analysis target)
    source        STRING    NOT NULL  ("yfinance_fallback" | "alphavantage" | ...)
    kind          STRING    NOT NULL  ("primary" | "fallback")
    article_count INT64               (nullable; rows fetched from this source)
    notes         STRING              (optional context)

Partitioned by DATE(event_time). Clustered by source for the
"% yfinance_fallback over window" query pattern.

Usage:
    python scripts/migrations/create_data_source_events_table.py            # dry-run
    python scripts/migrations/create_data_source_events_table.py --apply
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "pyfinagent_data"
TABLE = "data_source_events"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_FQN}` (
    event_id      STRING    NOT NULL OPTIONS(description="UUID for this event"),
    event_time    TIMESTAMP NOT NULL OPTIONS(description="UTC timestamp when the data-source event fired"),
    ticker        STRING    NOT NULL OPTIONS(description="Analysis target ticker"),
    source        STRING    NOT NULL OPTIONS(description="Source label e.g. yfinance_fallback / alphavantage"),
    kind          STRING    NOT NULL OPTIONS(description="primary | fallback"),
    article_count INT64              OPTIONS(description="Rows / articles fetched from this source (nullable)"),
    notes         STRING             OPTIONS(description="Optional context")
)
PARTITION BY DATE(event_time)
CLUSTER BY source
OPTIONS(description="phase-25.B7 per-cycle data-source provenance log; powers pct_yfinance_fallback_dominance metric");
""".strip()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create pyfinagent_data.data_source_events (idempotent)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the CREATE statement against BQ (default: dry-run only).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    logging.info("target table: %s", TABLE_FQN)
    logging.info("SQL:\n%s", CREATE_SQL)

    if not args.apply:
        logging.info("DRY-RUN: not executing. Pass --apply to write to BQ.")
        return 0

    try:
        from google.cloud import bigquery
    except Exception as exc:
        logging.error("google-cloud-bigquery not installed: %r", exc)
        return 2

    try:
        client = bigquery.Client(project=PROJECT)
        job = client.query(CREATE_SQL)
        job.result(timeout=60)
        logging.info("APPLIED: %s created/already-exists", TABLE_FQN)
        return 0
    except Exception as exc:
        logging.error("BQ migration failed: %r", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
