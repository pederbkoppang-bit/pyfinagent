"""
BigQuery schema migration -- round-trip performance columns + paper_round_trips table.

Phase 4.5 Step 4.5.2.

Idempotent:
  - ALTER TABLE ADD COLUMN IF NOT EXISTS for paper_positions (mfe_pct, mae_pct).
  - ALTER TABLE ADD COLUMN IF NOT EXISTS for paper_trades
    (mfe_pct, mae_pct, holding_days, round_trip_id, realized_pnl_pct, capture_ratio).
  - CREATE TABLE IF NOT EXISTS paper_round_trips with full attribution schema.

Run: source .venv/bin/activate && python scripts/migrations/add_round_trip_schema.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv(Path(__file__).parents[2] / "backend" / ".env")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = os.environ.get("BQ_DATASET_REPORTS", "financial_reports")


def _alter_add(client: bigquery.Client, table: str, col: str, col_type: str) -> None:
    sql = f"ALTER TABLE `{table}` ADD COLUMN IF NOT EXISTS {col} {col_type}"
    try:
        client.query(sql).result()
        logger.info(f"ok  {table}.{col} ({col_type})")
    except Exception as e:
        logger.warning(f"skip {table}.{col}: {e}")


def run() -> None:
    client = bigquery.Client(project=PROJECT_ID)

    trades = f"{PROJECT_ID}.{DATASET}.paper_trades"
    positions = f"{PROJECT_ID}.{DATASET}.paper_positions"
    round_trips = f"{PROJECT_ID}.{DATASET}.paper_round_trips"

    # Positions: MFE/MAE tracked live across holding period
    _alter_add(client, positions, "mfe_pct", "FLOAT64")
    _alter_add(client, positions, "mae_pct", "FLOAT64")

    # Trades: round-trip attribution on SELL rows
    _alter_add(client, trades, "round_trip_id", "STRING")
    _alter_add(client, trades, "holding_days", "INT64")
    _alter_add(client, trades, "realized_pnl_pct", "FLOAT64")
    _alter_add(client, trades, "mfe_pct", "FLOAT64")
    _alter_add(client, trades, "mae_pct", "FLOAT64")
    _alter_add(client, trades, "capture_ratio", "FLOAT64")
    # 4.5.5: JSON-serialized agent attribution (signals array). STRING so the
    # dynamic INSERT path stays parameterizable; drawer deserializes at read.
    _alter_add(client, trades, "signals", "STRING")

    # New table: paper_round_trips -- canonical exit-quality ledger
    schema = [
        bigquery.SchemaField("round_trip_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("buy_trade_id", "STRING"),
        bigquery.SchemaField("sell_trade_id", "STRING"),
        bigquery.SchemaField("entry_date", "TIMESTAMP"),
        bigquery.SchemaField("exit_date", "TIMESTAMP"),
        bigquery.SchemaField("entry_price", "FLOAT64"),
        bigquery.SchemaField("exit_price", "FLOAT64"),
        bigquery.SchemaField("quantity", "FLOAT64"),
        bigquery.SchemaField("realized_pnl_usd", "FLOAT64"),
        bigquery.SchemaField("realized_pnl_pct", "FLOAT64"),
        bigquery.SchemaField("holding_days", "INT64"),
        bigquery.SchemaField("mfe_pct", "FLOAT64"),
        bigquery.SchemaField("mae_pct", "FLOAT64"),
        bigquery.SchemaField("capture_ratio", "FLOAT64"),
        bigquery.SchemaField("exit_reason", "STRING"),
    ]
    tbl = bigquery.Table(round_trips, schema=schema)
    try:
        client.create_table(tbl, exists_ok=True)
        logger.info(f"ok  table {round_trips}")
    except Exception as e:
        logger.warning(f"skip table {round_trips}: {e}")


if __name__ == "__main__":
    run()
