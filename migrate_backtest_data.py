"""
BigQuery schema migration — create historical data tables for backtesting.
Idempotent: skips tables that already exist.

Tables created:
  - historical_prices      (~378K rows for S&P 500 × 3yr)
  - historical_fundamentals (~6K rows for S&P 500 × ~12 quarters)
  - historical_macro        (~252 rows for 7 series × 36 months)

Run: python migrate_backtest_data.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

load_dotenv(Path(__file__).parent / "backend" / ".env")

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "financial_reports"

# ── Table definitions ────────────────────────────────────────────

HISTORICAL_PRICES_REF = f"{PROJECT_ID}.{DATASET}.historical_prices"
HISTORICAL_PRICES_SCHEMA = [
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("date", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("open", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("high", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("low", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("close", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("volume", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="NULLABLE"),
]

HISTORICAL_FUNDAMENTALS_REF = f"{PROJECT_ID}.{DATASET}.historical_fundamentals"
HISTORICAL_FUNDAMENTALS_SCHEMA = [
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("report_date", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("filing_date", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("total_revenue", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("net_income", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("total_debt", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("total_equity", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("total_assets", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("operating_cash_flow", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("shares_outstanding", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("sector", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("industry", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="NULLABLE"),
]

HISTORICAL_MACRO_REF = f"{PROJECT_ID}.{DATASET}.historical_macro"
HISTORICAL_MACRO_SCHEMA = [
    bigquery.SchemaField("series_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("date", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("value", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="NULLABLE"),
]

ALL_TABLES = [
    ("historical_prices", HISTORICAL_PRICES_REF, HISTORICAL_PRICES_SCHEMA),
    ("historical_fundamentals", HISTORICAL_FUNDAMENTALS_REF, HISTORICAL_FUNDAMENTALS_SCHEMA),
    ("historical_macro", HISTORICAL_MACRO_REF, HISTORICAL_MACRO_SCHEMA),
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

    print("\nBacktest data migration complete.")


if __name__ == "__main__":
    main()
