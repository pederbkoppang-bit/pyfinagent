"""phase-30.4 Migration: add `external_flow_today` column to paper_portfolio_snapshots.

Required for phase-30.4 (GIPS-correct return series). The column carries
the external cash flow (deposit/withdrawal) on each snapshot_date so
`paper_metrics_v2._nav_to_returns` can subtract it before computing
daily returns -- preventing the Sharpe-pollution observed in
phase-30.0 Anomaly A (the 5/13 $5K deposit produced a +32.12% phantom
daily return that polluted the Sharpe denominator).

Audit basis: handoff/archive/phase-30.0/experiment_results.md Anomaly A
+ phase-30.4 contract.

Idempotent via `ADD COLUMN IF NOT EXISTS`. Safe to re-run.

Usage:
    python scripts/migrations/add_external_flow_today_column.py            # dry-run
    python scripts/migrations/add_external_flow_today_column.py --apply
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "financial_reports"
TABLE = "paper_portfolio_snapshots"
COLUMN = "external_flow_today"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

DDL = f"""
ALTER TABLE `{TABLE_FQN}`
ADD COLUMN IF NOT EXISTS {COLUMN} FLOAT64
OPTIONS (description = 'phase-30.4: external cash flow (deposit/withdrawal) on snapshot_date. Subtracted in paper_metrics_v2._nav_to_returns to satisfy GIPS time-weighted return computation. NULL means no flow recorded (legacy rows pre-30.4 migration).')
""".strip()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add external_flow_today column to paper_portfolio_snapshots (idempotent)."
    )
    parser.add_argument("--apply", action="store_true", help="Execute DDL (default: dry-run).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Project: %s", PROJECT)
    logger.info("Target table: %s", TABLE_FQN)
    logger.info("Adding column: %s FLOAT64 (idempotent IF NOT EXISTS)", COLUMN)
    logger.info("DDL:\n%s", DDL)

    if not args.apply:
        logger.info("DRY-RUN -- pass --apply to execute.")
        return 0

    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT)
    job = client.query(DDL)
    job.result()
    logger.info("Migration applied. Job ID: %s", job.job_id)

    # Verify.
    rows = list(client.query(
        f"SELECT column_name, data_type FROM `{PROJECT}.{DATASET}.INFORMATION_SCHEMA.COLUMNS` "
        f"WHERE table_name='{TABLE}' AND column_name='{COLUMN}'"
    ).result())
    if not rows:
        logger.error("Verification FAILED -- column not present post-migration.")
        return 1
    logger.info("Verification OK: %s", [(r["column_name"], r["data_type"]) for r in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
