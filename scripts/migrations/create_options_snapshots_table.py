"""phase-5.6 Migration: create pyfinagent_hdw.options_snapshots.

Idempotent CREATE TABLE IF NOT EXISTS. Defaults to dry-run (prints SQL).
Pass --apply to execute via the BigQuery client (requires ADC).

Schema:
    snapshot_ts: TIMESTAMP (when this snapshot was captured)
    underlying:  STRING    (e.g. "SPY")
    occ_symbol:  STRING    (21-char OCC symbol, e.g. "AAPL  240119C00150000")
    strike:      FLOAT64
    expiration:  DATE
    dte:         INT64     (days to expiration at snapshot_ts)
    option_type: STRING    ("call" or "put")
    bid:         FLOAT64
    ask:         FLOAT64
    mid:         FLOAT64
    iv:          FLOAT64   (implied volatility, decimal e.g. 0.20)
    delta:       FLOAT64
    gamma:       FLOAT64
    theta:       FLOAT64   (per-day)
    vega:        FLOAT64   (per-1%-vol)

Partitioned by DATE(snapshot_ts) for cost containment on the 30-day
range queries. Clustered by underlying + option_type for chain-fetch
locality.

Usage:
    # Default dry-run -- prints SQL only:
    python scripts/migrations/create_options_snapshots_table.py

    # Execute against BQ:
    python scripts/migrations/create_options_snapshots_table.py --apply
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "pyfinagent_hdw"
TABLE = "options_snapshots"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_FQN}` (
    snapshot_ts TIMESTAMP NOT NULL OPTIONS(description="Snapshot capture timestamp"),
    underlying  STRING    NOT NULL OPTIONS(description="Underlying ticker, e.g. SPY"),
    occ_symbol  STRING    NOT NULL OPTIONS(description="21-char OCC option symbol"),
    strike      FLOAT64   NOT NULL,
    expiration  DATE      NOT NULL,
    dte         INT64     NOT NULL,
    option_type STRING    NOT NULL OPTIONS(description="'call' or 'put'"),
    bid         FLOAT64,
    ask         FLOAT64,
    mid         FLOAT64,
    iv          FLOAT64   OPTIONS(description="Implied volatility, decimal (0.20 = 20%)"),
    delta       FLOAT64,
    gamma       FLOAT64,
    theta       FLOAT64   OPTIONS(description="Per-day theta"),
    vega        FLOAT64   OPTIONS(description="Per-1%-vol vega")
)
PARTITION BY DATE(snapshot_ts)
CLUSTER BY underlying, option_type
OPTIONS(description="phase-5.6 options chain snapshots for backtest + live monitoring");
""".strip()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create pyfinagent_hdw.options_snapshots (idempotent)."
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
        job.result(timeout=60)  # block until complete
        logging.info("APPLIED: %s created/already-exists", TABLE_FQN)
        return 0
    except Exception as exc:
        logging.error("BQ migration failed: %r", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
