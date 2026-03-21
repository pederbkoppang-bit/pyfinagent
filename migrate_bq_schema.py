"""
BigQuery schema migration — add ML-training-ready columns to analysis_results.
Idempotent: skips columns that already exist.

Run: python migrate_bq_schema.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

# Load SA credentials from backend/.env (works regardless of ADC state)
load_dotenv(Path(__file__).parent / "backend" / ".env")

# ── Config ──────────────────────────────────────────────────────
PROJECT_ID = "sunny-might-477607-p8"
DATASET = "financial_reports"
TABLE = "analysis_results"
TABLE_REF = f"{PROJECT_ID}.{DATASET}.{TABLE}"

# ── New columns to add (name, BQ type) ─────────────────────────
NEW_COLUMNS = [
    # Phase 1: Financial fundamentals
    ("price_at_analysis", "FLOAT64"),
    ("market_cap", "FLOAT64"),
    ("pe_ratio", "FLOAT64"),
    ("peg_ratio", "FLOAT64"),
    ("debt_equity", "FLOAT64"),
    ("sector", "STRING"),
    ("industry", "STRING"),
    # Phase 2: Risk metrics
    ("annualized_volatility", "FLOAT64"),
    ("var_95_6m", "FLOAT64"),
    ("var_99_6m", "FLOAT64"),
    ("expected_shortfall_6m", "FLOAT64"),
    ("prob_positive_6m", "FLOAT64"),
    ("anomaly_count", "INT64"),
    # Phase 3: Debate & reasoning
    ("bull_confidence", "FLOAT64"),
    ("bear_confidence", "FLOAT64"),
    ("bull_thesis", "STRING"),
    ("bear_thesis", "STRING"),
    ("contradiction_count", "INT64"),
    ("dissent_count", "INT64"),
    ("recommendation_confidence", "FLOAT64"),
    ("key_risks", "STRING"),
    # Phase 4: Enrichment signals
    ("insider_signal", "STRING"),
    ("options_signal", "STRING"),
    ("social_sentiment_score", "FLOAT64"),
    ("nlp_sentiment_score", "FLOAT64"),
    ("patent_signal", "STRING"),
    ("earnings_confidence", "FLOAT64"),
    ("sector_signal", "STRING"),
    # Phase 5: Bias & conflict audit
    ("bias_count", "INT64"),
    ("bias_adjusted_score", "FLOAT64"),
    ("conflict_count", "INT64"),
    ("overall_reliability", "STRING"),
    ("decision_trace_count", "INT64"),
    # Phase 6: Macro context
    ("fed_funds_rate", "FLOAT64"),
    ("cpi_yoy", "FLOAT64"),
    ("unemployment_rate", "FLOAT64"),
    ("yield_curve_spread", "FLOAT64"),
    # Phase 7: Multi-round debate, DA, info-gap, risk assessment
    ("debate_rounds_count", "INT64"),
    ("devils_advocate_challenges", "INT64"),
    ("info_gap_count", "INT64"),
    ("info_gap_resolved_count", "INT64"),
    ("data_quality_score", "FLOAT64"),
    ("risk_judge_decision", "STRING"),
    ("risk_adjusted_confidence", "FLOAT64"),
    ("aggressive_analyst_confidence", "FLOAT64"),
    ("conservative_analyst_confidence", "FLOAT64"),
    # Phase 8: Cost tracking
    ("total_tokens", "INT64"),
    ("total_cost_usd", "FLOAT64"),
    ("deep_think_calls", "INT64"),
    # Phase 9: Reflection loop + quality gates
    ("synthesis_iterations", "INT64"),
    # Phase 10: Model tracking
    ("standard_model", "STRING"),
    ("deep_think_model", "STRING"),
]


OUTCOME_TRACKING_REF = f"{PROJECT_ID}.{DATASET}.outcome_tracking"
OUTCOME_TRACKING_SCHEMA = [
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("analysis_date", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("recommendation", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("price_at_recommendation", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("current_price", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("return_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("holding_days", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("beat_benchmark", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("evaluated_at", "STRING", mode="NULLABLE"),
]


def ensure_outcome_tracking_table(client: bigquery.Client) -> None:
    """Create outcome_tracking table if it doesn't exist (idempotent)."""
    try:
        client.get_table(OUTCOME_TRACKING_REF)
        print(f"Table {OUTCOME_TRACKING_REF} already exists. Nothing to do.")
    except Exception:
        table = bigquery.Table(OUTCOME_TRACKING_REF, schema=OUTCOME_TRACKING_SCHEMA)
        client.create_table(table)
        print(f"Created table {OUTCOME_TRACKING_REF}")


def main():
    # Build client
    creds_json = os.environ.get("GCP_CREDENTIALS_JSON", "")
    credentials = None
    if creds_json:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=["https://www.googleapis.com/auth/bigquery",
                    "https://www.googleapis.com/auth/cloud-platform"],
        )
    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

    # Ensure outcome_tracking table exists
    ensure_outcome_tracking_table(client)

    table = client.get_table(TABLE_REF)
    existing = {field.name for field in table.schema}
    print(f"Existing columns ({len(existing)}): {sorted(existing)}")

    to_add = [(name, bq_type) for name, bq_type in NEW_COLUMNS if name not in existing]
    if not to_add:
        print("All columns already exist. Nothing to do.")
        return

    print(f"\nAdding {len(to_add)} new columns:")
    new_schema = list(table.schema)
    for name, bq_type in to_add:
        print(f"  + {name} ({bq_type})")
        new_schema.append(bigquery.SchemaField(name, bq_type, mode="NULLABLE"))

    table.schema = new_schema
    client.update_table(table, ["schema"])

    # Verify
    table = client.get_table(TABLE_REF)
    final_cols = {f.name for f in table.schema}
    print(f"\nDone. Table now has {len(final_cols)} columns.")
    missing = [name for name, _ in NEW_COLUMNS if name not in final_cols]
    if missing:
        print(f"WARNING: These columns are still missing: {missing}")
    else:
        print("All new columns verified.")


if __name__ == "__main__":
    main()
