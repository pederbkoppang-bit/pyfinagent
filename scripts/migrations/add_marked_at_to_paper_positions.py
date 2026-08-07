"""phase-61.3 migration: add marked_at STRING column to paper_positions.

Idempotent. Safe to re-run. The column stores the ISO-8601 UTC timestamp of the
mark_to_market run that last wrote this row's unrealized_pnl / unrealized_pnl_pct /
market_value.

WHY: those three columns are written ONLY by mark_to_market, which runs once per
scheduled cycle, while the positions table renders a LIVE local price next to them. For
a non-US row the P&L can therefore be up to ~24h stale (a weekend ~72h) beside a live
price, with nothing on screen saying so -- the retail-facing form of the stale-valuation
risk the FCA's 2025 private-market review named ("an asset's valuation no longer
accurately reflects its current market value"). marked_at is what lets the UI label the
P&L as-of rather than imply it is current.

Additive and NULLABLE: existing rows keep NULL and are not rewritten. The column changes
no order, stop, or size -- it is observability only.

Backward compatibility does NOT depend on this migration having run:
paper_trader._POSITION_RT_FIELDS includes marked_at, so _safe_save_position prunes it and
retries on a pre-migration table (see test_phase_61_3_addon_currency.py::
test_61_3_marked_at_currency_is_prunable_on_a_pre_migration_schema).

Usage:
    python scripts/migrations/add_marked_at_to_paper_positions.py --dry-run
    python scripts/migrations/add_marked_at_to_paper_positions.py
    python scripts/migrations/add_marked_at_to_paper_positions.py --verify
"""
from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("add_marked_at_to_paper_positions")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


PROJECT = "sunny-might-477607-p8"
DATASET = "financial_reports"
TABLE = "paper_positions"
COLUMN = "marked_at"


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
    OPTIONS(description='phase-61.3 as-of indicator: ISO-8601 UTC timestamp of the mark_to_market run that last wrote unrealized_pnl/unrealized_pnl_pct/market_value on this row. NULL for rows never marked since the migration. Observability only -- gates nothing, moves no order/stop/size. Consumed by the positions table to label a stale non-US P&L instead of implying it is live.')
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
