"""phase-32.4 Migration: add `company_name` column to paper_positions.

Required for phase-32.4 backfill_missing_company_names helper. Backfill
itself is NOT done by this migration -- it is handled at runtime by the
helper inside the autonomous loop's Step 5.6 region. The migration just
provides the column.

Audit basis: operator dashboard observation 2026-05-20: 9 of 11 current
paper_positions rows display ticker-as-company (MU, KEYS, GEV, COHR, ON,
DELL, GLW, LITE, WDC). Same legacy pattern as phase-25.2 stop_loss_price
backfill. Cosmetic gap, not safety-critical.

Idempotent via `ADD COLUMN IF NOT EXISTS`. Safe to re-run.

Usage:
    python scripts/migrations/phase_32_4_add_company_name.py            # dry-run
    python scripts/migrations/phase_32_4_add_company_name.py --apply
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase_32_4_add_company_name")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Execute the migration (default: dry-run).",
    )
    args = ap.parse_args()

    sys.path.insert(0, "/Users/ford/.openclaw/workspace/pyfinagent")
    from backend.config.settings import get_settings  # type: ignore
    from google.cloud import bigquery  # type: ignore

    settings = get_settings()
    client = bigquery.Client(project=settings.gcp_project_id)
    table = f"`{settings.gcp_project_id}.{settings.bq_dataset_reports}.paper_positions`"

    ddl = f"""
        ALTER TABLE {table}
        ADD COLUMN IF NOT EXISTS company_name STRING
        OPTIONS(description='phase-32.4: yfinance shortName/longName for the position ticker. NULL or equal to ticker on legacy rows; populated by paper_trader.backfill_missing_company_names() on the next autonomous-loop cycle.')
    """

    logger.info("=== phase-32.4 migration: add company_name column ===")
    logger.info("Project: %s", settings.gcp_project_id)
    logger.info("Target table: %s", table)
    logger.info("DDL:\n%s", ddl.strip())

    if not args.apply:
        logger.info("DRY-RUN -- ALTER TABLE not executed. Pass --apply to run.")
        return 0

    job = client.query(ddl)
    job.result(timeout=30)
    logger.info("ALTER TABLE done. Job ID: %s", job.job_id)

    # Verify.
    verify_sql = (
        f"SELECT column_name, data_type FROM "
        f"`{settings.gcp_project_id}.{settings.bq_dataset_reports}.INFORMATION_SCHEMA.COLUMNS` "
        f"WHERE table_name='paper_positions' AND column_name='company_name'"
    )
    rows = list(client.query(verify_sql).result(timeout=30))
    if not rows:
        logger.error("Verification FAILED -- column not present post-migration.")
        return 1
    logger.info("Verification OK: %s", [(r["column_name"], r["data_type"]) for r in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
