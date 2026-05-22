"""phase-36.1 migration: add scale_out_levels_hit STRING column to paper_positions.

Idempotent. Safe to re-run. The column stores a JSON-encoded list of strings
like '["2R"]' or '["2R", "3R"]' indicating which scale-out levels have
fired for the position (one-shot, idempotent across cycles).

Default value: NULL (paper_trader.check_scale_out_fires defaults NULL -> [] in code).

phase-36.1 success criterion #5: 'scale_out_levels_hit_column_added_via_idempotent_migration'.
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("add_scale_out_levels_hit_column")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


PROJECT = "sunny-might-477607-p8"
DATASET = "financial_reports"
TABLE = "paper_positions"
COLUMN = "scale_out_levels_hit"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    ap.add_argument("--verify", action="store_true", help="Check column exists; exit 0 if yes, 1 if no")
    args = ap.parse_args()

    if args.verify:
        try:
            from google.cloud import bigquery
        except Exception as e:
            logger.error("google-cloud-bigquery not importable: %s", e)
            return 1
        client = bigquery.Client(project=PROJECT)
        table_ref = f"{PROJECT}.{DATASET}.{TABLE}"
        try:
            table = client.get_table(table_ref)
        except Exception as e:
            logger.error("table fetch failed (%s): %s", table_ref, e)
            return 1
        col_names = {f.name for f in table.schema}
        if COLUMN in col_names:
            logger.info("verify OK: %s.%s exists on %s", TABLE, COLUMN, table_ref)
            return 0
        logger.warning("verify FAIL: %s.%s NOT present on %s", TABLE, COLUMN, table_ref)
        return 1

    sql = f"""
    ALTER TABLE `{PROJECT}.{DATASET}.{TABLE}`
    ADD COLUMN IF NOT EXISTS {COLUMN} STRING
    OPTIONS(description='phase-36.1 scale-out idempotency: JSON-encoded list of fired levels (e.g. ["2R", "3R"]). NULL/empty -> no fires yet. Set by paper_trader.check_scale_out_fires when an execute_sell partial-close fires. Read defensively (NULL -> [] in code) for backward compat with pre-migration positions.')
    """
    if args.dry_run:
        logger.info("dry-run -- would execute:\n%s", sql.strip())
        return 0

    try:
        from google.cloud import bigquery
    except Exception as e:
        logger.error("google-cloud-bigquery not importable: %s", e)
        return 1
    client = bigquery.Client(project=PROJECT)
    logger.info("executing: %s", sql.strip().replace("\n", " "))
    job = client.query(sql)
    job.result(timeout=30)
    logger.info("%s column added (or already present) on %s.%s.%s",
                COLUMN, PROJECT, DATASET, TABLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
