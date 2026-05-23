"""phase-23.1.17 repair script.

Forces a mark_to_market() pass + saves a fresh paper_portfolio_snapshot
row so:
  1. paper_portfolio.total_nav reflects the post-phase-23.1.15 cleanup
     refund (current_cash + sum(live position MV)).
  2. paper_portfolio_snapshots gets a row for today reflecting the
     repaired NAV — feeds the Red Line Monitor on the home page so the
     chart's most-recent point matches the hero scoreboard.

Idempotent: mark_to_market is itself idempotent; running this script
multiple times in the same minute writes near-identical snapshots
(deduped on snapshot_date by the snapshot logic).

REMINDER for future cleanup-script authors: ANY raw BQ UPDATE that
mutates `paper_portfolio.current_cash` (deposits, refunds, manual
corrections) MUST be followed by a `mark_to_market()` call so
`total_nav` and the Red Line Monitor stay consistent. Phase-23.1.15
shipped without this and produced the home/paper-trading discrepancy
that this script repairs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("repair_23_1_17")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="phase-23.1.17 repair")
    parser.add_argument("--apply", action="store_true",
                        help="Execute mark_to_market + save snapshot")
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive confirmation")
    args = parser.parse_args(argv)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from backend.config.settings import get_settings  # type: ignore
    from backend.db.bigquery_client import BigQueryClient  # type: ignore
    from backend.services.paper_trader import PaperTrader  # type: ignore

    settings = get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)

    portfolio = trader.get_or_create_portfolio()
    pre_nav = portfolio.get("total_nav")
    pre_cash = portfolio.get("current_cash")
    logger.info("Pre-repair: cash=$%.2f, total_nav=$%.2f", pre_cash, pre_nav or 0)

    if not args.apply:
        logger.info("DRY RUN -- would call mark_to_market() then save_daily_snapshot().")
        logger.info("Re-run with --apply to execute.")
        return 0

    if not args.yes:
        confirm = input("Type 'APPLY' to execute mark_to_market + snapshot: ")
        if confirm.strip() != "APPLY":
            logger.info("Aborted.")
            return 1

    logger.info("Calling mark_to_market()...")
    mtm = trader.mark_to_market()
    logger.info(
        "mark_to_market done: nav=$%.2f cash=$%.2f positions_value=$%.2f",
        mtm.get("nav", 0), mtm.get("cash", 0), mtm.get("positions_value", 0),
    )

    logger.info("Saving daily snapshot...")
    trader.save_daily_snapshot(trades_today=0, analysis_cost_today=0.0)

    portfolio_after = trader.get_or_create_portfolio()
    post_nav = portfolio_after.get("total_nav")
    logger.info("Post-repair: cash=$%.2f, total_nav=$%.2f",
                portfolio_after.get("current_cash"), post_nav or 0)
    logger.info("NAV delta: $%+.2f", (post_nav or 0) - (pre_nav or 0))
    logger.info("ok phase-23.1.17 repair complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
