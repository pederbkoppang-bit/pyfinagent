"""phase-26.5 Migration: create the `strategy_decisions` BigQuery table.

Stores per-cycle strategy-router decisions including the phase-26.5
alpha-decay early-warning signal that the phase-25.R policy consumes
upstream of realized-P&L reactive triggers.

Schema:
  ts                 TIMESTAMP NOT NULL    -- decision time (UTC)
  cycle_id           STRING                -- links to llm_call_log (phase-26.1)
  decided_strategy   STRING NOT NULL       -- which strategy is now active
  prior_strategy     STRING                -- what was active before
  trigger            STRING NOT NULL       -- "decay_signal" | "manual" | "performance_threshold"
  decay_signal       FLOAT64               -- alpha-decay strength 0-1 (NULL if not driven by decay)
  decay_attribution  STRING                -- which upstream signal flagged decay
  rationale          STRING                -- 1-2 sentence LLM rationale

Partitioned on DATE(ts); clustered by (trigger, decided_strategy).

Idempotent CREATE TABLE IF NOT EXISTS. Run with:
    python scripts/migrations/add_strategy_decisions_table.py            # dry-run
    python scripts/migrations/add_strategy_decisions_table.py --apply
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "pyfinagent_data"
TABLE = "strategy_decisions"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

DDL = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_FQN}` (
  ts TIMESTAMP NOT NULL,
  cycle_id STRING,
  decided_strategy STRING NOT NULL,
  prior_strategy STRING,
  trigger STRING NOT NULL,
  decay_signal FLOAT64,
  decay_attribution STRING,
  rationale STRING
)
PARTITION BY DATE(ts)
CLUSTER BY trigger, decided_strategy
OPTIONS (
  description = "phase-26.5 strategy-router decisions including alpha-decay early-warning signal feeding phase-25.R reactive policy"
)
""".strip()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create strategy_decisions table (idempotent).")
    parser.add_argument("--apply", action="store_true", help="Execute DDL (default: dry-run).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    logging.info("target: %s", TABLE_FQN)
    logging.info("DDL:\n%s", DDL)

    if not args.apply:
        logging.info("DRY-RUN: pass --apply to execute against BQ.")
        return 0

    try:
        from google.cloud import bigquery
    except Exception as exc:
        logging.error("google-cloud-bigquery not installed: %r", exc)
        return 2

    try:
        client = bigquery.Client(project=PROJECT)
        job = client.query(DDL)
        job.result(timeout=60)
        logging.info("APPLIED: %s ready (idempotent CREATE TABLE IF NOT EXISTS).", TABLE_FQN)
        return 0
    except Exception as exc:
        logging.error("BQ migration failed: %r", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
