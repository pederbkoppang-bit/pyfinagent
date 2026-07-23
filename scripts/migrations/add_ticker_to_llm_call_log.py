"""phase-25.S.1 Migration: add `ticker STRING` column to
`pyfinagent_data.llm_call_log`.

Idempotent `ALTER TABLE ADD COLUMN IF NOT EXISTS`. Defaults to dry-run
(prints SQL). Pass --apply to execute via the BigQuery client (requires
ADC).

Why: 25.S (cycle 88) shipped proportional-split-by-trade-count
per-ticker attribution as a first-pass. With this column, exact
per-ticker LLM-cost attribution becomes possible:

    SELECT ticker,
           SUM(input_tok * pricing.input_per_mtok / 1e6 +
               output_tok * pricing.output_per_mtok / 1e6) AS cost_usd
    FROM `pyfinagent_data.llm_call_log`
    WHERE DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
      AND ticker IS NOT NULL
    GROUP BY ticker
    ORDER BY cost_usd DESC;

Closes the cost-denominator side of the auto-switch goal-c at the
ticker level (north-star: maximize profit at lowest cost; auto-prune
tickers where LLM cost > realized profit).

Usage:
    python scripts/migrations/add_ticker_to_llm_call_log.py            # dry-run
    python scripts/migrations/add_ticker_to_llm_call_log.py --apply
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "pyfinagent_data"
TABLE = "llm_call_log"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

ALTER_SQL = f"""
ALTER TABLE `{TABLE_FQN}`
ADD COLUMN IF NOT EXISTS ticker STRING OPTIONS(description="phase-25.S.1: per-call ticker tag enabling exact per-ticker cost attribution. NULL for non-ticker-scoped calls (e.g., meta-coordinator decisions).");
""".strip()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add ticker STRING column to pyfinagent_data.llm_call_log (idempotent)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the ALTER TABLE against BQ (default: dry-run only).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    logging.info("target table: %s", TABLE_FQN)
    logging.info("SQL:\n%s", ALTER_SQL)

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
        job = client.query(ALTER_SQL)
        job.result(timeout=60)
        logging.info("APPLIED: %s now has `ticker` column", TABLE_FQN)
        return 0
    except Exception as exc:
        logging.error("BQ migration failed: %r", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
