#!/usr/bin/env python
"""phase-50.1: create financial_reports.historical_fx_rates (idempotent).

Stores usd_value per currency for point-in-time / backtest FX reads:
  pair = "{CCY}USD" (e.g. EURUSD, KRWUSD), rate = USD value of 1 unit of CCY.
Mirrors the historical_macro shape (unpartitioned, `date` as STRING).

Dataset is `financial_reports` which lives in us-central1 -- do NOT pin
--location US; the BigQuery client resolves the dataset's region automatically.

Idempotent CREATE TABLE IF NOT EXISTS. Dry-run by default; pass --apply.

  python scripts/migrations/create_historical_fx_rates_table.py --apply
"""
from __future__ import annotations

import argparse
import logging
import os

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "financial_reports"
TABLE = "historical_fx_rates"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_FQN}` (
    pair   STRING  NOT NULL OPTIONS(description="{{CCY}}USD, e.g. EURUSD / KRWUSD -- USD value of 1 unit of CCY"),
    date   STRING  NOT NULL OPTIONS(description="ISO date (STRING, matches historical_prices/_macro)"),
    rate   FLOAT64 NOT NULL OPTIONS(description="usd_value: USD per 1 unit of the base currency (mid/daily-close)"),
    source STRING           OPTIONS(description="yfinance-backfill / yfinance-fred-live")
)
CLUSTER BY pair
OPTIONS(description="phase-50.1 multi-currency FX rates; point-in-time as-of reads for backtest + live mark write-through");
""".strip()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s | %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create financial_reports.historical_fx_rates (idempotent).")
    parser.add_argument("--apply", action="store_true",
                        help="Execute the CREATE against BQ (default: dry-run).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    logging.info("target table: %s", TABLE_FQN)
    logging.info("SQL:\n%s", CREATE_SQL)
    if not args.apply:
        logging.info("DRY-RUN: not executing. Pass --apply to write to BQ.")
        return 0

    try:
        from google.cloud import bigquery
    except Exception as exc:
        logging.error("google-cloud-bigquery not installed: %r", exc)
        return 2
    try:
        client = bigquery.Client(project=PROJECT)  # no --location: client resolves dataset region
        client.query(CREATE_SQL).result()
        logging.info("APPLIED: %s created/already-exists", TABLE_FQN)
        return 0
    except Exception as exc:
        logging.error("BQ migration failed: %r", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
