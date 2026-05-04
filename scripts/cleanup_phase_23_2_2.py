"""phase-23.2.2 cleanup script — STX orphan trade refund.

Found in phase-23.2.0 audit walkthrough: STX has 1 BUY trade on
2026-04-26 23:44 for $949.48 + $0.95 fee, NO matching paper_positions
row. Same bug class as WDC/XOM cleaned up in phase-23.1.15 (cycle
crashed after trade+cash debit but BEFORE position write landed
visibly). Pre-MERGE-upsert (phase-23.1.15) legacy artifact.

Researcher (phase-23.2.2-fix-internal-codebase-audit.md) confirmed:
- Exactly 1 orphan: STX trade_id 04c6f356-2a5c-47df-8891-bea686cd444f
- All other 14 tickers have matching positions
- No partial-close orphans

Modes:
  --dry-run (default): print SQL + diff, do nothing
  --apply              execute DELETE + UPDATE (gated on confirmation)
  --yes                skip interactive confirmation (headless mode)

Idempotent: if STX trade row already gone, DELETE returns 0 affected
rows and the refund UPDATE short-circuits.

REMINDER for future authors: any raw BQ UPDATE to current_cash MUST
be followed by `mark_to_market()` so total_nav and the Red Line
Monitor stay consistent. Phase-23.1.15 shipped without that and
produced the phase-23.1.17 stale-NAV bug.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleanup_23_2_2")

STX_ORPHAN_TRADE_ID = "04c6f356-2a5c-47df-8891-bea686cd444f"
STX_REFUND_VALUE = 949.48
STX_REFUND_FEE = 0.95
TOTAL_REFUND = STX_REFUND_VALUE + STX_REFUND_FEE  # $950.43


def _bq_client():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from backend.config.settings import get_settings  # type: ignore
    from backend.db.bigquery_client import BigQueryClient  # type: ignore
    settings = get_settings()
    return BigQueryClient(settings), settings


def show_pre_state(bq) -> dict:
    project = bq.settings.gcp_project_id
    dataset = bq.settings.bq_dataset_reports

    q_trade = f"""
        SELECT trade_id, ticker, action, total_value, transaction_cost, created_at, reason
        FROM `{project}.{dataset}.paper_trades`
        WHERE trade_id = '{STX_ORPHAN_TRADE_ID}'
    """
    rows = list(bq.client.query(q_trade).result())
    if rows:
        r = rows[0]
        logger.info(
            "STX orphan trade present: %s %s %s total=$%.2f fee=$%.4f reason=%s created=%s",
            r["trade_id"][:8], r["ticker"], r["action"],
            r["total_value"] or 0, r["transaction_cost"] or 0,
            r["reason"], r["created_at"],
        )
    else:
        logger.info("STX orphan trade NOT present (already cleaned)")

    q_cash = f"""
        SELECT portfolio_id, current_cash FROM `{project}.{dataset}.paper_portfolio`
        WHERE portfolio_id = 'default' LIMIT 1
    """
    cash_row = list(bq.client.query(q_cash).result())
    cash = float(cash_row[0]["current_cash"]) if cash_row else 0.0
    logger.info("Current cash: $%.2f", cash)

    return {
        "trade_present": bool(rows),
        "current_cash": cash,
    }


def apply_changes(bq) -> dict:
    project = bq.settings.gcp_project_id
    dataset = bq.settings.bq_dataset_reports
    table_trades = f"`{project}.{dataset}.paper_trades`"
    table_portfolio = f"`{project}.{dataset}.paper_portfolio`"

    job1 = bq.client.query(
        f"DELETE FROM {table_trades} WHERE trade_id = '{STX_ORPHAN_TRADE_ID}'"
    )
    job1.result()
    deleted = job1.num_dml_affected_rows or 0
    logger.info("Step 1: STX orphan DELETE -> %d rows", deleted)

    refund = TOTAL_REFUND if deleted > 0 else 0.0
    if refund > 0:
        job2 = bq.client.query(
            f"""
            UPDATE {table_portfolio}
            SET current_cash = current_cash + {refund:.2f},
                updated_at = FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S+00:00', CURRENT_TIMESTAMP())
            WHERE portfolio_id = 'default'
            """
        )
        job2.result()
        logger.info("Step 2: refunded $%.2f to current_cash", refund)
    else:
        logger.info("Step 2: no row deleted this run, no refund applied (idempotent no-op)")

    return {"deleted": deleted, "refund_applied": refund}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="phase-23.2.2 STX orphan cleanup")
    parser.add_argument("--apply", action="store_true",
                        help="Actually mutate BQ (default: dry-run)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive confirmation")
    args = parser.parse_args(argv)

    bq, _settings = _bq_client()

    logger.info("=== phase-23.2.2 STX orphan cleanup ===")
    logger.info("Pre-state:")
    pre = show_pre_state(bq)

    logger.info("")
    logger.info("Planned SQL:")
    logger.info("  DELETE FROM paper_trades WHERE trade_id = '%s'", STX_ORPHAN_TRADE_ID)
    logger.info("  UPDATE paper_portfolio SET current_cash = current_cash + %.2f, updated_at = ... WHERE portfolio_id = 'default'", TOTAL_REFUND)
    logger.info("")
    logger.info("Total refund (if row present): $%.2f", TOTAL_REFUND)

    if not args.apply:
        logger.info("")
        logger.info("DRY RUN -- no changes made. Re-run with --apply to execute.")
        return 0

    if not args.yes:
        confirm = input("Type 'APPLY' to confirm: ")
        if confirm.strip() != "APPLY":
            logger.info("Aborted by user.")
            return 1

    logger.info("")
    logger.info("Applying changes...")
    result = apply_changes(bq)

    logger.info("")
    logger.info("Post-state:")
    post = show_pre_state(bq)
    cash_delta = post["current_cash"] - pre["current_cash"]
    logger.info("Cash delta: $%+.2f", cash_delta)

    if abs(cash_delta - result["refund_applied"]) > 0.01:
        logger.error(
            "Mismatch: expected refund $%.2f but cash delta is $%.2f",
            result["refund_applied"], cash_delta,
        )
        return 2

    logger.info("ok phase-23.2.2 cleanup complete")
    logger.info("REMINDER: run scripts/repair_phase_23_1_17.py --apply --yes to refresh total_nav post-refund")
    return 0


if __name__ == "__main__":
    sys.exit(main())
