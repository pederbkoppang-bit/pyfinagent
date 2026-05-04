"""phase-23.2.6-fix: add `sector STRING` column to paper_positions + backfill.

Phase-23.1.13/14 added in-memory sector enrichment via _fetch_ticker_meta
but BQ paper_positions schema had NO sector column. Every cycle paid the
yfinance fallback cost. This migration:

1. ALTER TABLE ADD COLUMN IF NOT EXISTS sector STRING (idempotent)
2. Backfill the existing 14 rows via yfinance Ticker(t).info["sector"]
3. Going forward, execute_buy persists sector at trade time
   (see paper_trader.py + autonomous_loop.py changes in same commit)

Idempotent: re-running with all rows already populated is a no-op.

Modes:
  --dry-run (default): print SQL + would-be backfill, do nothing
  --apply              execute the migration
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("add_sector_paper_positions")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Execute the migration (default: dry-run)")
    args = ap.parse_args()

    sys.path.insert(0, "/Users/ford/.openclaw/workspace/pyfinagent")
    from backend.config.settings import get_settings  # type: ignore
    from google.cloud import bigquery  # type: ignore

    settings = get_settings()
    client = bigquery.Client(project=settings.gcp_project_id)
    table = f"`{settings.gcp_project_id}.{settings.bq_dataset_reports}.paper_positions`"

    ddl = f"""
        ALTER TABLE {table}
        ADD COLUMN IF NOT EXISTS sector STRING
        OPTIONS(description='GICS sector from yfinance/analysis_results. Populated by phase-23.2.6-fix migration and on every new BUY via execute_buy.')
    """

    logger.info("=== phase-23.2.6-fix migration ===")
    logger.info("DDL:\n%s", ddl.strip())

    if not args.apply:
        logger.info("DRY RUN -- ALTER TABLE not executed")
    else:
        client.query(ddl).result(timeout=30)
        logger.info("ALTER TABLE done (column added or already present)")

    # Backfill rows where sector is NULL or empty
    select_sql = f"SELECT ticker FROM {table} WHERE sector IS NULL OR sector = ''"
    if not args.apply:
        # In dry-run, the column may not exist yet; fall through with empty list.
        try:
            tickers = [r["ticker"] for r in client.query(select_sql).result(timeout=30)]
        except Exception:
            logger.info("(column doesn't exist yet -- dry-run can't list rows)")
            tickers = []
    else:
        tickers = [r["ticker"] for r in client.query(select_sql).result(timeout=30)]

    logger.info("Tickers needing backfill: %d -- %s", len(tickers), tickers)

    if not tickers:
        logger.info("No rows need backfill. Done.")
        return 0

    import yfinance as yf
    backfilled = 0
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector") or ""
        except Exception as e:
            logger.warning("yfinance failed for %s: %s", ticker, e)
            sector = ""

        if not sector:
            logger.warning("No sector resolved for %s; skipping", ticker)
            continue

        if not args.apply:
            logger.info("dry-run -- would UPDATE %s sector='%s'", ticker, sector)
            continue

        update_sql = f"""
            UPDATE {table}
            SET sector = @sector
            WHERE ticker = @ticker AND (sector IS NULL OR sector = '')
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("sector", "STRING", sector),
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        ])
        client.query(update_sql, job_config=job_config).result(timeout=30)
        logger.info("Backfilled %s -> sector='%s'", ticker, sector)
        backfilled += 1

    logger.info("Backfill complete: %d rows updated.", backfilled)
    logger.info("ok phase-23.2.6-fix migration done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
