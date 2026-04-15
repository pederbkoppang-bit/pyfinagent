"""
BigQuery schema migration -- create signals_log table.
Idempotent: skips the table if it already exists.

Phase 4.2.4: durable persistence scaffold for the in-memory
SignalsServer.signal_history list. Append-only event log; one row
per publish event this cycle (event_kind="publish"). Outcome events
will append a SECOND row (event_kind="outcome") in a future cycle.

Run: python scripts/migrations/migrate_signals_log.py
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

# -- Table definition ---------------------------------------------

SIGNALS_LOG_REF = f"{PROJECT_ID}.{DATASET}.signals_log"
SIGNALS_LOG_SCHEMA = [
    # -- Identity --
    bigquery.SchemaField("signal_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("signal_type", "STRING", mode="REQUIRED"),
    # -- Prediction state --
    bigquery.SchemaField("confidence", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("signal_date", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("entry_price", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("factors_json", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    # -- Outcome placeholders (NULL for publish events, populated for outcome events) --
    bigquery.SchemaField("outcome", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("scored", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("hit", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("exit_price", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("exit_date", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("forward_return_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("holding_days", "INT64", mode="NULLABLE"),
    # -- Audit metadata --
    bigquery.SchemaField("recorded_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("event_kind", "STRING", mode="REQUIRED"),
]

ALL_TABLES = [
    ("signals_log", SIGNALS_LOG_REF, SIGNALS_LOG_SCHEMA),
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

    print("\nsignals_log migration complete.")


if __name__ == "__main__":
    main()
