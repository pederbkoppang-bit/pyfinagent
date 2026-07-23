"""phase-10.7.1 migration: create `pyfinagent_pms.alpha_velocity_samples` table.

Append-only time-series table for the Alpha Velocity metric (Sharpe-
slope-per-day). Partitioned by DATE(window_start), clustered on
(strategy_id, macro_regime).

CLI:
    python scripts/migrations/create_alpha_velocity_table.py             # apply
    python scripts/migrations/create_alpha_velocity_table.py --apply
    python scripts/migrations/create_alpha_velocity_table.py --verify
    python scripts/migrations/create_alpha_velocity_table.py --dry-run

Mirrors `scripts/migrations/create_strategy_deployments_view.py:1-196`
patterns: same idempotency, same flag set, same DATASET_FQN convention.
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "pyfinagent_pms"
TABLE = "alpha_velocity_samples"

DATASET_FQN = f"{PROJECT}.{DATASET}"
TABLE_FQN = f"{DATASET_FQN}.{TABLE}"


def _create_dataset_sql() -> str:
    return f"CREATE SCHEMA IF NOT EXISTS `{DATASET_FQN}` OPTIONS(location='US');"


def _create_table_sql() -> str:
    return f"""
CREATE TABLE IF NOT EXISTS `{TABLE_FQN}` (
  strategy_id           STRING NOT NULL,
  window_start          TIMESTAMP NOT NULL,
  window_end            TIMESTAMP NOT NULL,
  n_obs                 INT64,
  sharpe_start          FLOAT64,
  sharpe_end            FLOAT64,
  alpha_velocity_score  FLOAT64,
  window_days           INT64,
  macro_regime          STRING,
  components_json       STRING,
  computed_at           TIMESTAMP
)
PARTITION BY DATE(window_start)
CLUSTER BY strategy_id, macro_regime
OPTIONS(
  description="phase-10.7.1: per-window Alpha Velocity samples (Sharpe-slope-per-day). Append-only."
);
""".strip()


def _verify_table_sql() -> str:
    return (
        f"SELECT COUNT(*) AS row_count FROM `{TABLE_FQN}` LIMIT 1;"
    )


def _print_dry_run() -> int:
    print("DRY RUN -- no BigQuery mutations will be performed.")
    print()
    print(_create_dataset_sql())
    print()
    print(_create_table_sql())
    return 0


def _apply() -> int:
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError as e:
        print(f"google-cloud-bigquery not installed: {e}", file=sys.stderr)
        return 2
    client = bigquery.Client(project=PROJECT)
    for sql in (_create_dataset_sql(), _create_table_sql()):
        print(f"[apply] running:\n{sql}\n")
        client.query(sql).result(timeout=60)
    print(f"[apply] PASS: table created or already exists at {TABLE_FQN}")
    return 0


def _verify() -> int:
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError as e:
        print(f"google-cloud-bigquery not installed: {e}", file=sys.stderr)
        return 2
    client = bigquery.Client(project=PROJECT)
    sql = _verify_table_sql()
    print(f"[verify] running:\n{sql}\n")
    job = client.query(sql)
    rows = list(job.result(timeout=60))
    if not rows:
        print(f"[verify] FAIL: query returned no rows")
        return 1
    count = rows[0].row_count
    print(f"[verify] PASS: table_exists=true, row_count={count}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true", help="(default) apply migration")
    g.add_argument("--verify", action="store_true", help="verify table exists")
    g.add_argument("--dry-run", action="store_true", help="print SQL without applying")
    args = parser.parse_args(argv)

    if args.dry_run:
        return _print_dry_run()
    if args.verify:
        return _verify()
    return _apply()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
