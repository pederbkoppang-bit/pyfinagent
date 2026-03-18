"""
BigQuery migration — create agent_memories table for FinancialSituationMemory.
Idempotent: skips if table already exists.

Run: python migrate_agent_memories.py
"""

import json
import os

from google.cloud import bigquery
from google.oauth2 import service_account

PROJECT_ID = "sunny-might-477607-p8"
DATASET = "financial_reports"
TABLE = "agent_memories"
TABLE_REF = f"{PROJECT_ID}.{DATASET}.{TABLE}"

SCHEMA = [
    bigquery.SchemaField("agent_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("situation", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("lesson", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]


def main():
    creds_json = os.environ.get("GCP_CREDENTIALS_JSON", "")
    credentials = None
    if creds_json:
        credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json))
    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

    try:
        client.get_table(TABLE_REF)
        print(f"Table {TABLE_REF} already exists. Nothing to do.")
        return
    except Exception:
        pass

    table = bigquery.Table(TABLE_REF, schema=SCHEMA)
    table = client.create_table(table)
    print(f"Created table {table.full_table_id} ({len(table.schema)} columns).")


if __name__ == "__main__":
    main()
