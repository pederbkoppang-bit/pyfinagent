"""phase-23.1.15 cleanup script.

Refunds two leaked cash debits and removes their orphan trade rows:

1. WDC duplicate trade (trade_id e5447bd9-9cb0-437b-b2a2-c851703b77b1)
   created at 2026-04-26T21:17:41 — cycle 2's phantom retry. The canonical
   row at 21:12:28 (trade_id 56072f0c-568b-4bfb-9278-e5da3588dbe3) is kept.
   Refund: $949.95 + $0.95 fee = $950.90.

2. XOM `test_paper_trade` (trade_id a8e6b00e-e39b-4a00-9eb4-540097b2212a)
   created at 2026-03-28 — has no matching paper_positions row. Refund:
   $500.00 + $0.50 fee = $500.50.

Total cash refund: $1,451.40

Modes:
  --dry-run (default): print SQL + diff, do nothing
  --apply: execute the DELETEs and UPDATE inside an explicit confirmation gate
  --yes:   bypass interactive confirmation prompt (for headless harness use)

Idempotent: if rows already gone, the DELETEs are no-ops and the UPDATE
short-circuits via a guarded WHERE clause that includes a one-time marker.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleanup_23_1_15")

# Hard-coded forensics from BQ. Do not parameterize — these are the exact
# rows we are removing.
WDC_DUPLICATE_TRADE_ID = "e5447bd9-9cb0-437b-b2a2-c851703b77b1"
WDC_KEEP_TRADE_ID = "56072f0c-568b-4bfb-9278-e5da3588dbe3"
XOM_TEST_TRADE_ID = "a8e6b00e-e39b-4a00-9eb4-540097b2212a"

WDC_REFUND_VALUE = 949.95
WDC_REFUND_FEE = 0.95
XOM_REFUND_VALUE = 500.00
XOM_REFUND_FEE = 0.50

TOTAL_REFUND = WDC_REFUND_VALUE + WDC_REFUND_FEE + XOM_REFUND_VALUE + XOM_REFUND_FEE


def _bq_client():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from backend.config.settings import get_settings  # type: ignore
    from backend.db.bigquery_client import BigQueryClient  # type: ignore
    settings = get_settings()
    return BigQueryClient(settings), settings


def show_pre_state(bq) -> dict:
    """Read and print current state."""
    project = bq.settings.gcp_project_id
    dataset = bq.settings.bq_dataset_reports

    q_trades = f"""
        SELECT trade_id, ticker, action, total_value, transaction_cost, created_at, reason
        FROM `{project}.{dataset}.paper_trades`
        WHERE trade_id IN (
            '{WDC_DUPLICATE_TRADE_ID}',
            '{WDC_KEEP_TRADE_ID}',
            '{XOM_TEST_TRADE_ID}'
        )
        ORDER BY created_at
    """
    rows = list(bq.client.query(q_trades).result())
    logger.info("Targeted trade rows currently in BQ:")
    for r in rows:
        logger.info(
            "  %s %s %s qty?  total=%.2f fee=%.2f reason=%s created=%s",
            r["trade_id"][:8], r["ticker"], r["action"],
            r["total_value"] or 0, r["transaction_cost"] or 0,
            r["reason"], r["created_at"],
        )

    q_cash = f"""
        SELECT portfolio_id, current_cash FROM `{project}.{dataset}.paper_portfolio`
        WHERE portfolio_id = 'default' LIMIT 1
    """
    cash_row = list(bq.client.query(q_cash).result())
    cash = float(cash_row[0]["current_cash"]) if cash_row else 0.0
    logger.info("Current cash: $%.2f", cash)
    return {
        "trade_rows_present": len(rows),
        "current_cash": cash,
    }


def show_sql() -> list[tuple[str, str]]:
    """Return the SQL we will execute. Pure printable form for dry-run."""
    return [
        (
            "Step 1 — DELETE WDC duplicate (21:17:41 phantom retry)",
            f"DELETE FROM paper_trades WHERE trade_id = '{WDC_DUPLICATE_TRADE_ID}'",
        ),
        (
            "Step 2 — DELETE XOM test_paper_trade orphan",
            f"DELETE FROM paper_trades WHERE trade_id = '{XOM_TEST_TRADE_ID}'",
        ),
        (
            f"Step 3 — Refund ${TOTAL_REFUND:.2f} to current_cash",
            (
                "UPDATE paper_portfolio "
                f"SET current_cash = current_cash + {TOTAL_REFUND:.2f}, "
                "updated_at = CURRENT_TIMESTAMP() "
                "WHERE portfolio_id = 'default'"
            ),
        ),
    ]


def apply_changes(bq) -> dict:
    """Execute the cleanup. Idempotent: if the trade rows are already gone,
    the DELETEs return 0 affected rows and the UPDATE is gated on actual
    deletion happening this run."""
    project = bq.settings.gcp_project_id
    dataset = bq.settings.bq_dataset_reports
    table_trades = f"`{project}.{dataset}.paper_trades`"
    table_portfolio = f"`{project}.{dataset}.paper_portfolio`"

    # Step 1: delete WDC duplicate, count affected
    job1 = bq.client.query(
        f"DELETE FROM {table_trades} WHERE trade_id = '{WDC_DUPLICATE_TRADE_ID}'"
    )
    job1.result()
    wdc_deleted = job1.num_dml_affected_rows or 0
    logger.info("Step 1: WDC duplicate DELETE -> %d rows", wdc_deleted)

    # Step 2: delete XOM test orphan
    job2 = bq.client.query(
        f"DELETE FROM {table_trades} WHERE trade_id = '{XOM_TEST_TRADE_ID}'"
    )
    job2.result()
    xom_deleted = job2.num_dml_affected_rows or 0
    logger.info("Step 2: XOM test DELETE -> %d rows", xom_deleted)

    # Step 3: refund only the cash amount for rows we ACTUALLY deleted this run
    refund = (
        (WDC_REFUND_VALUE + WDC_REFUND_FEE) * (1 if wdc_deleted else 0)
        + (XOM_REFUND_VALUE + XOM_REFUND_FEE) * (1 if xom_deleted else 0)
    )
    if refund > 0:
        job3 = bq.client.query(
            f"""
            UPDATE {table_portfolio}
            SET current_cash = current_cash + {refund:.2f},
                updated_at = FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S+00:00', CURRENT_TIMESTAMP())
            WHERE portfolio_id = 'default'
            """
        )
        job3.result()
        logger.info("Step 3: refunded $%.2f to current_cash", refund)
    else:
        logger.info("Step 3: no rows deleted this run, no refund applied (idempotent no-op)")

    return {
        "wdc_deleted": wdc_deleted,
        "xom_deleted": xom_deleted,
        "refund_applied": refund,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="phase-23.1.15 cleanup")
    parser.add_argument("--apply", action="store_true",
                        help="Actually mutate BQ (default: dry-run)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive confirmation (headless mode)")
    args = parser.parse_args(argv)

    bq, _settings = _bq_client()

    logger.info("=== phase-23.1.15 cleanup ===")
    logger.info("Pre-state:")
    pre = show_pre_state(bq)

    logger.info("")
    logger.info("Planned SQL:")
    for label, sql in show_sql():
        logger.info("  %s", label)
        logger.info("    %s", sql)
    logger.info("")
    logger.info("Total cash refund (if all rows present): $%.2f", TOTAL_REFUND)

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

    logger.info("ok phase-23.1.15 cleanup complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
