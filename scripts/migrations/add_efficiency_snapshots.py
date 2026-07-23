"""phase-25.Q Migration: create pyfinagent_data.efficiency_snapshots.

Idempotent CREATE TABLE IF NOT EXISTS. Defaults to dry-run (prints SQL).
Pass --apply to execute via the BigQuery client (requires ADC).

The table persists daily snapshots of the first-mover red-line goal-d
metric: profit_per_llm_dollar = realized_pnl_usd / llm_cost_usd over a
configurable window (default 30 days). NULL ratio when llm_cost == 0
(no-divide-by-zero contract).

Schema:
    snapshot_date         DATE      (UTC date the snapshot was computed)
    window_days           INT64     (lookback window e.g. 7, 30, 90)
    profit_per_llm_dollar FLOAT64   (NULLable; None when llm_cost == 0)
    realized_pnl_usd      FLOAT64   (numerator)
    llm_cost_usd          FLOAT64   (denominator; anthropic+vertex+openai)
    anthropic_cost_usd    FLOAT64   (per-provider breakdown)
    vertex_cost_usd       FLOAT64
    openai_cost_usd       FLOAT64
    computed_at           TIMESTAMP (UTC compute timestamp)

Partitioned by snapshot_date. Clustered by window_days for the
"latest 30d snapshot" pattern.

Usage:
    # Default dry-run:
    python scripts/migrations/add_efficiency_snapshots.py

    # Execute:
    python scripts/migrations/add_efficiency_snapshots.py --apply
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "pyfinagent_data"
TABLE = "efficiency_snapshots"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_FQN}` (
    snapshot_date         DATE      NOT NULL OPTIONS(description="UTC date the snapshot was computed"),
    window_days           INT64     NOT NULL OPTIONS(description="Lookback window in days"),
    profit_per_llm_dollar FLOAT64            OPTIONS(description="realized_pnl / llm_cost; NULL when llm_cost is 0"),
    realized_pnl_usd      FLOAT64   NOT NULL OPTIONS(description="Sum of realized P&L (USD) over the window"),
    llm_cost_usd          FLOAT64   NOT NULL OPTIONS(description="Sum of LLM cost (USD): anthropic + vertex + openai"),
    anthropic_cost_usd    FLOAT64   NOT NULL OPTIONS(description="Anthropic cost component"),
    vertex_cost_usd       FLOAT64   NOT NULL OPTIONS(description="Vertex/Gemini cost component"),
    openai_cost_usd       FLOAT64   NOT NULL OPTIONS(description="OpenAI cost component"),
    computed_at           TIMESTAMP NOT NULL OPTIONS(description="UTC timestamp of the computation")
)
PARTITION BY snapshot_date
CLUSTER BY window_days
OPTIONS(description="phase-25.Q efficiency snapshots; closes red-line goal-d");
""".strip()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create pyfinagent_data.efficiency_snapshots (idempotent)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the CREATE statement against BQ (default: dry-run only).",
    )
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
        client = bigquery.Client(project=PROJECT)
        job = client.query(CREATE_SQL)
        job.result(timeout=60)
        logging.info("APPLIED: %s created/already-exists", TABLE_FQN)
        return 0
    except Exception as exc:
        logging.error("BQ migration failed: %r", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
