"""
BigQuery schema migration — create paper trading tables.
Idempotent: skips tables that already exist.

Run: python migrate_paper_trading.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

load_dotenv(Path(__file__).parent / "backend" / ".env")

PROJECT_ID = "sunny-might-477607-p8"
DATASET = "financial_reports"

# ── Table definitions ────────────────────────────────────────────

PAPER_PORTFOLIO_REF = f"{PROJECT_ID}.{DATASET}.paper_portfolio"
PAPER_PORTFOLIO_SCHEMA = [
    bigquery.SchemaField("portfolio_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("starting_capital", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("current_cash", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("total_nav", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("total_pnl_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("benchmark_return_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("inception_date", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("updated_at", "STRING", mode="NULLABLE"),
]

PAPER_POSITIONS_REF = f"{PROJECT_ID}.{DATASET}.paper_positions"
PAPER_POSITIONS_SCHEMA = [
    bigquery.SchemaField("position_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("quantity", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("avg_entry_price", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("cost_basis", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("current_price", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("market_value", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("unrealized_pnl", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("unrealized_pnl_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("entry_date", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("last_analysis_date", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("recommendation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("risk_judge_position_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("stop_loss_price", "FLOAT64", mode="NULLABLE"),
]

PAPER_TRADES_REF = f"{PROJECT_ID}.{DATASET}.paper_trades"
PAPER_TRADES_SCHEMA = [
    bigquery.SchemaField("trade_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("action", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("quantity", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("price", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("total_value", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("transaction_cost", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("reason", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("analysis_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("risk_judge_decision", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "STRING", mode="REQUIRED"),
]

PAPER_SNAPSHOTS_REF = f"{PROJECT_ID}.{DATASET}.paper_portfolio_snapshots"
PAPER_SNAPSHOTS_SCHEMA = [
    bigquery.SchemaField("snapshot_date", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("total_nav", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("cash", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("positions_value", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("daily_pnl_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("cumulative_pnl_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("benchmark_pnl_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("alpha_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("position_count", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("trades_today", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("analysis_cost_today", "FLOAT64", mode="NULLABLE"),
]

ALL_TABLES = [
    ("paper_portfolio", PAPER_PORTFOLIO_REF, PAPER_PORTFOLIO_SCHEMA),
    ("paper_positions", PAPER_POSITIONS_REF, PAPER_POSITIONS_SCHEMA),
    ("paper_trades", PAPER_TRADES_REF, PAPER_TRADES_SCHEMA),
    ("paper_portfolio_snapshots", PAPER_SNAPSHOTS_REF, PAPER_SNAPSHOTS_SCHEMA),
]


def main():
    creds_json = os.environ.get("GCP_CREDENTIALS_JSON", "")
    credentials = None
    if creds_json:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=["https://www.googleapis.com/auth/bigquery",
                    "https://www.googleapis.com/auth/cloud-platform"],
        )
    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

    for name, ref, schema in ALL_TABLES:
        try:
            client.get_table(ref)
            print(f"Table {name} already exists. Skipping.")
        except Exception:
            table = bigquery.Table(ref, schema=schema)
            client.create_table(table)
            print(f"Created table {name}")

    print("\nPaper trading migration complete.")


if __name__ == "__main__":
    main()
