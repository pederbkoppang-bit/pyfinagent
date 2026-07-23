"""phase-25.A3 Migration: create pyfinagent_data.promoted_strategies.

Idempotent CREATE TABLE IF NOT EXISTS. Defaults to dry-run (prints SQL).
Pass --apply to execute via the BigQuery client (requires ADC).

Schema:
    strategy_id      STRING    (trial_id from the promotion gate)
    week_iso         STRING    (ISO week string, e.g. 2026-W20)
    params           JSON      (strategy hyperparameters dict)
    dsr              FLOAT64   (Deflated Sharpe Ratio at promotion time)
    pbo              FLOAT64   (Probability of Backtest Overfitting)
    status           STRING    (pending | active | paused | superseded | rolled_back)
    allocation_pct   FLOAT64   (starting allocation fraction, e.g. 0.05)
    promoted_at      TIMESTAMP (UTC timestamp of promotion fire)
    sortino_monthly  FLOAT64   (monthly Sortino at promotion time)
    rejection_reason STRING    (reserved for future rejected-candidate rows)

Partitioned by DATE(promoted_at) for cost containment on the per-week
queries. Clustered by strategy_id + week_iso for the MERGE natural key.

The 25.A3 writer in `backend/autoresearch/friday_promotion.py` uses MERGE
on (week_iso, strategy_id) so re-running a Friday promotion is idempotent.

Usage:
    # Default dry-run -- prints SQL only:
    python scripts/migrations/create_promoted_strategies_table.py

    # Execute against BQ:
    python scripts/migrations/create_promoted_strategies_table.py --apply
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "pyfinagent_data"
TABLE = "promoted_strategies"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS `{TABLE_FQN}` (
    strategy_id      STRING    NOT NULL OPTIONS(description="trial_id from the promotion gate"),
    week_iso         STRING    NOT NULL OPTIONS(description="ISO week string, e.g. 2026-W20"),
    params           JSON               OPTIONS(description="Strategy hyperparameters dict"),
    dsr              FLOAT64   NOT NULL OPTIONS(description="Deflated Sharpe Ratio at promotion time"),
    pbo              FLOAT64   NOT NULL OPTIONS(description="Probability of Backtest Overfitting"),
    status           STRING    NOT NULL OPTIONS(description="pending | active | paused | superseded | rolled_back"),
    allocation_pct   FLOAT64            OPTIONS(description="Starting allocation fraction, e.g. 0.05"),
    promoted_at      TIMESTAMP NOT NULL OPTIONS(description="UTC timestamp of promotion fire"),
    sortino_monthly  FLOAT64            OPTIONS(description="Monthly Sortino at promotion time"),
    rejection_reason STRING             OPTIONS(description="Reserved for future rejected-candidate rows")
)
PARTITION BY DATE(promoted_at)
CLUSTER BY strategy_id, week_iso
OPTIONS(description="phase-25.A3 promoted strategy registry");
""".strip()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create pyfinagent_data.promoted_strategies (idempotent)."
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
        job.result(timeout=60)  # block until complete
        logging.info("APPLIED: %s created/already-exists", TABLE_FQN)
        return 0
    except Exception as exc:
        logging.error("BQ migration failed: %r", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
