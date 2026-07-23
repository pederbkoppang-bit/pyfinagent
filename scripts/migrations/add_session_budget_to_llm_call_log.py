"""phase-26.1 Migration: add `cycle_id STRING` and `session_cost_usd FLOAT64`
columns to `pyfinagent_data.llm_call_log`.

Idempotent `ALTER TABLE ADD COLUMN IF NOT EXISTS`. Defaults to dry-run
(prints SQL). Pass --apply to execute via the BigQuery client (requires
ADC).

Why: phase-26.1 adds a per-cycle LLM-cost ceiling (`_SESSION_BUDGET_USD`
in `backend/services/autonomous_loop.py`). The masterplan step's
live_check requires operator-auditable BQ evidence of a session-budget
trip, which means each `log_llm_call` row needs the cycle's UUID-suffix
id plus the running cumulative cost at the moment the row was written.

    SELECT cycle_id,
           MAX(session_cost_usd) AS final_session_cost,
           COUNT(*) AS n_calls,
           MAX(ts) AS last_call_ts
    FROM `pyfinagent_data.llm_call_log`
    WHERE cycle_id IS NOT NULL
      AND DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    GROUP BY cycle_id
    ORDER BY final_session_cost DESC;

A trip row will show `session_cost_usd >= _SESSION_BUDGET_USD` on the
final entry for its cycle_id, with no subsequent rows (because the
cycle was halted by the budget check).

Usage:
    python scripts/migrations/add_session_budget_to_llm_call_log.py            # dry-run
    python scripts/migrations/add_session_budget_to_llm_call_log.py --apply
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "pyfinagent_data"
TABLE = "llm_call_log"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

ALTER_SQL = f"""
ALTER TABLE `{TABLE_FQN}`
ADD COLUMN IF NOT EXISTS cycle_id STRING OPTIONS(description="phase-26.1: 8-char cycle UUID suffix from autonomous_loop.run_daily_cycle. NULL for calls outside an active cycle (manual/test contexts)."),
ADD COLUMN IF NOT EXISTS session_cost_usd FLOAT64 OPTIONS(description="phase-26.1: running cumulative LLM cost in USD at the moment this row was logged. Compare against _SESSION_BUDGET_USD ceiling. NULL outside a cycle.");
""".strip()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add cycle_id + session_cost_usd columns to pyfinagent_data.llm_call_log (idempotent)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the ALTER TABLE against BQ (default: dry-run only).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    logging.info("target table: %s", TABLE_FQN)
    logging.info("SQL:\n%s", ALTER_SQL)

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
        job = client.query(ALTER_SQL)
        job.result(timeout=60)
        logging.info("APPLIED: %s now has cycle_id + session_cost_usd columns", TABLE_FQN)
        return 0
    except Exception as exc:
        logging.error("BQ migration failed: %r", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
