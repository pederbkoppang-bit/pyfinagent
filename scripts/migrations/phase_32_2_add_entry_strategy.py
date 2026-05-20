"""phase-32.2 Migration: add `entry_strategy` column to paper_positions.

Required for phase-32.2 HWM-trailing stop + Kaminski-Lo adversarial guard.
The trailing branch in `paper_trader._advance_stop` skips positions whose
entry_strategy is 'mean_reversion' or 'pairs' (Kaminski-Lo Proposition 2:
trailing stops degrade expected return for mean-reverting return processes).

This migration:
  1. ALTER TABLE ADD COLUMN IF NOT EXISTS entry_strategy STRING (idempotent).
  2. Backfill existing rows with `'momentum'` (fail-CLOSED-conservative
     default per phase-32.2 contract: when entry_strategy is unknown, treat
     as momentum so the trail IS applied -- "more protection" is the safer
     side to err on).

Going forward, `paper_trader.execute_buy` is the canonical write-site for
this field; wiring it to read from `strategy_decisions.decided_strategy` at
BUY time is a phase-32.x followup -- NOT in scope for this cycle.

Idempotent: re-running with the column already present + all rows already
populated is a no-op.

Audit basis: handoff/archive/phase-31.0/experiment_results.md section 4
P1.2 + handoff/current/contract.md (phase-32.2 spec) +
.claude/masterplan.json::phase-32.2.implementation_plan.migration_steps.

Usage:
    python scripts/migrations/phase_32_2_add_entry_strategy.py            # dry-run
    python scripts/migrations/phase_32_2_add_entry_strategy.py --apply
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase_32_2_add_entry_strategy")

DEFAULT_BACKFILL_VALUE = "momentum"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Execute the migration (default: dry-run).",
    )
    ap.add_argument(
        "--backfill-value",
        default=DEFAULT_BACKFILL_VALUE,
        help=f"Value to backfill into rows with NULL entry_strategy "
             f"(default: '{DEFAULT_BACKFILL_VALUE}', fail-CLOSED-conservative).",
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
        ADD COLUMN IF NOT EXISTS entry_strategy STRING
        OPTIONS(description='phase-32.2: strategy that produced the entry signal (momentum / mean_reversion / pairs / triple_barrier / quality_momentum / factor_model). Drives the Kaminski-Lo Proposition 2 adversarial guard in paper_trader._advance_stop -- positions whose entry_strategy is mean_reversion or pairs SKIP the HWM-trailing branch.')
    """

    logger.info("=== phase-32.2 migration: add entry_strategy column ===")
    logger.info("Project: %s", settings.gcp_project_id)
    logger.info("Target table: %s", table)
    logger.info("Backfill value (fail-CLOSED-conservative): %r", args.backfill_value)
    logger.info("DDL:\n%s", ddl.strip())

    if not args.apply:
        logger.info("DRY-RUN -- ALTER TABLE not executed. Pass --apply to run.")
        return 0

    client.query(ddl).result(timeout=30)
    logger.info("ALTER TABLE done (column added or already present).")

    # Backfill rows where entry_strategy is NULL or empty.
    select_sql = (
        f"SELECT ticker FROM {table} WHERE entry_strategy IS NULL OR entry_strategy = ''"
    )
    tickers = [r["ticker"] for r in client.query(select_sql).result(timeout=30)]
    logger.info("Rows needing backfill: %d -- %s", len(tickers), tickers)

    if not tickers:
        logger.info("No rows need backfill. Done.")
        return 0

    update_sql = (
        f"UPDATE {table} "
        f"SET entry_strategy = @backfill_value "
        f"WHERE entry_strategy IS NULL OR entry_strategy = ''"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("backfill_value", "STRING", args.backfill_value)
        ]
    )
    job = client.query(update_sql, job_config=job_config)
    job.result(timeout=30)
    logger.info(
        "Backfill applied. Job ID: %s. Rows updated: %d.",
        job.job_id, job.num_dml_affected_rows,
    )

    # Verify post-state.
    verify_sql = (
        f"SELECT entry_strategy, COUNT(*) AS n FROM {table} GROUP BY entry_strategy"
    )
    summary = list(client.query(verify_sql).result(timeout=30))
    logger.info("Post-backfill summary: %s", [(r["entry_strategy"], r["n"]) for r in summary])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
