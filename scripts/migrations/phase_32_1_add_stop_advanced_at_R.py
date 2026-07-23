"""phase-32.1 Migration: add `stop_advanced_at_R` column to paper_positions.

Required for phase-32.1 breakeven-stop ratchet at +1R. The column carries
an ISO timestamp marking when the position's stop was advanced from
entry-anchored (entry * 0.92) up to entry_price after MFE crossed
settings.paper_default_stop_loss_pct (1R, default 8%). NULL means the
ratchet has not yet fired for this position. Once populated, the ratchet
is idempotent and will not re-fire.

Audit basis: handoff/archive/phase-31.0/experiment_results.md section 4
P1.1 (the breakeven ratchet remediation) + handoff/current/contract.md
(this cycle's spec) + the masterplan entry phase-32.1.implementation_plan.

Idempotent via `ADD COLUMN IF NOT EXISTS`. Safe to re-run.

Usage:
    python scripts/migrations/phase_32_1_add_stop_advanced_at_R.py            # dry-run
    python scripts/migrations/phase_32_1_add_stop_advanced_at_R.py --apply
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "financial_reports"
TABLE = "paper_positions"
COLUMN = "stop_advanced_at_R"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

DDL = f"""
ALTER TABLE `{TABLE_FQN}`
ADD COLUMN IF NOT EXISTS {COLUMN} STRING
OPTIONS (description = 'phase-32.1: ISO timestamp when breakeven-stop ratchet fired (mfe_pct >= settings.paper_default_stop_loss_pct, default 8%). NULL = ratchet has not yet fired. Once populated, ratchet is idempotent and will not re-fire for the position.')
""".strip()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add stop_advanced_at_R column to paper_positions (idempotent)."
    )
    parser.add_argument("--apply", action="store_true", help="Execute DDL (default: dry-run).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Project: %s", PROJECT)
    logger.info("Target table: %s", TABLE_FQN)
    logger.info("Adding column: %s STRING (idempotent IF NOT EXISTS)", COLUMN)
    logger.info("DDL:\n%s", DDL)

    if not args.apply:
        logger.info("DRY-RUN -- pass --apply to execute.")
        return 0

    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT)
    job = client.query(DDL)
    job.result(timeout=60)
    logger.info("Migration applied. Job ID: %s", job.job_id)

    # Verify.
    rows = list(client.query(
        f"SELECT column_name, data_type FROM `{PROJECT}.{DATASET}.INFORMATION_SCHEMA.COLUMNS` "
        f"WHERE table_name='{TABLE}' AND column_name='{COLUMN}'"
    ).result(timeout=60))
    if not rows:
        logger.error("Verification FAILED -- column not present post-migration.")
        return 1
    logger.info("Verification OK: %s", [(r["column_name"], r["data_type"]) for r in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
