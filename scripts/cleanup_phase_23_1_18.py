"""phase-23.1.18 cleanup: dedupe `paper_portfolio_snapshots` so each
snapshot_date has exactly one row.

BQ inspection on 2026-04-29 confirmed multiple rows per snapshot_date:
  - 2026-04-29: 2 rows ($14,153 stale, $15,647 post-repair)
  - 2026-04-27: 3 rows
  - 2026-04-26: 3 rows
  - others affected too

The Red Line Monitor's `_fetch_snapshots` used ANY_VALUE(total_nav) which
is non-deterministic — empirically picked the older/lower row.

Strategy: rewrite the table keeping ROW_NUMBER() = 1 per snapshot_date,
ordered by `total_nav DESC`. Heuristic since the schema has no
created_at column. In our data, the post-repair row always has the
highest total_nav (mark_to_market ran before that write), so MAX(nav)
is the closest proxy to "most recent / most complete".

Modes:
  --dry-run (default): print SQL + diff, do nothing
  --apply              execute the rewrite (gated on confirmation)
  --yes                skip interactive confirmation (headless mode)

Idempotent: re-runs after success no-op (the rewrite is to the same
deduplicated state).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleanup_23_1_18")


def _bq_client():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from backend.config.settings import get_settings  # type: ignore
    from backend.db.bigquery_client import BigQueryClient  # type: ignore
    settings = get_settings()
    return BigQueryClient(settings), settings


def show_state(bq, settings) -> dict:
    project = settings.gcp_project_id
    dataset = settings.bq_dataset_reports
    table = f"`{project}.{dataset}.paper_portfolio_snapshots`"

    q_total = f"SELECT COUNT(*) AS n FROM {table}"
    n_total = list(bq.client.query(q_total).result())[0]["n"]

    q_unique = f"SELECT COUNT(DISTINCT snapshot_date) AS n FROM {table}"
    n_unique = list(bq.client.query(q_unique).result())[0]["n"]

    q_dup = f"""
        SELECT snapshot_date, COUNT(*) AS n
        FROM {table}
        GROUP BY snapshot_date
        HAVING COUNT(*) > 1
        ORDER BY snapshot_date DESC
    """
    dup_dates = [(r["snapshot_date"], r["n"]) for r in bq.client.query(q_dup).result()]

    logger.info("paper_portfolio_snapshots: %d total rows / %d unique dates",
                n_total, n_unique)
    if dup_dates:
        logger.info("Duplicate-date counts:")
        for d, n in dup_dates:
            logger.info("  %s: %d rows", d, n)
    else:
        logger.info("No duplicate snapshot_dates -- already clean.")
    return {"total": n_total, "unique": n_unique, "dups": dup_dates}


def show_sql(project: str, dataset: str) -> str:
    table = f"`{project}.{dataset}.paper_portfolio_snapshots`"
    return f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            snapshot_date,
            total_nav, cash, positions_value,
            daily_pnl_pct, cumulative_pnl_pct,
            benchmark_pnl_pct, alpha_pct,
            position_count, trades_today, analysis_cost_today
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY snapshot_date
                    ORDER BY total_nav DESC
                ) AS rn
            FROM {table}
        )
        WHERE rn = 1
    """.strip()


def apply_rewrite(bq, settings) -> None:
    sql = show_sql(settings.gcp_project_id, settings.bq_dataset_reports)
    logger.info("Executing rewrite (CREATE OR REPLACE TABLE)...")
    bq.client.query(sql).result()
    logger.info("Rewrite complete.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="phase-23.1.18 snapshot dedup")
    parser.add_argument("--apply", action="store_true",
                        help="Execute the dedup rewrite (default: dry-run)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive confirmation")
    args = parser.parse_args(argv)

    bq, settings = _bq_client()

    logger.info("=== phase-23.1.18 snapshot cleanup ===")
    logger.info("Pre-state:")
    pre = show_state(bq, settings)

    if not pre["dups"]:
        logger.info("Nothing to do.")
        return 0

    logger.info("")
    logger.info("Planned SQL:")
    logger.info(show_sql(settings.gcp_project_id, settings.bq_dataset_reports))

    if not args.apply:
        logger.info("")
        logger.info("DRY RUN -- no changes made. Re-run with --apply to execute.")
        return 0

    if not args.yes:
        confirm = input("Type 'APPLY' to confirm rewrite: ")
        if confirm.strip() != "APPLY":
            logger.info("Aborted.")
            return 1

    apply_rewrite(bq, settings)

    logger.info("")
    logger.info("Post-state:")
    post = show_state(bq, settings)

    if post["dups"]:
        logger.error("UNEXPECTED: duplicates remain after rewrite. Manual review needed.")
        return 2

    logger.info("ok phase-23.1.18 cleanup complete (was %d rows / %d unique dates -> %d rows / %d unique)",
                pre["total"], pre["unique"], post["total"], post["unique"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
