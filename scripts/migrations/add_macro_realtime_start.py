"""phase-82.0 migration: add `realtime_start` (vintage) to `historical_macro`.

WHY THIS EXISTS
---------------
`financial_reports.historical_macro` is a VINTAGE MOSAIC. Rows are appended
whenever an ingest happens to run, carrying only the observation `date` and an
`ingested_at` write timestamp. Neither records the fact that actually matters
for backtesting: the date on which a value BECAME KNOWN.

FRED publishes macro series with substantial lag, and revises them afterwards.
Measured on this table 2026-07-31: a GDP row dated 2026-04-01 was not published
by FRED until 2026-07-30 -- roughly 120 days of look-ahead. Any backtest that
conditions on macro features and joins on `date <= cutoff` is therefore reading
numbers that did not exist at the cutoff. That is a silent look-ahead bias in
every macro-conditioned result the engine has ever produced.

`realtime_start` is the standard FRED/ALFRED name for this vintage field. With
it, a point-in-time join becomes `date <= cutoff AND realtime_start <= cutoff`.

WHY NOW AND NOT LATER
---------------------
A row's vintage cannot be reconstructed after the fact. Once a row is written
without `realtime_start`, the information about when we first saw it is gone
permanently -- there is no way to retro-attribute it. So the column has to land
WITH the backfill that phase-82.0 runs, or that backfill's rows are
un-attributable forever.

BACKFILL SEMANTICS FOR PRE-EXISTING ROWS
----------------------------------------
Existing rows are backfilled with `realtime_start = DATE(ingested_at)` -- the
earliest date we can PROVE we had the value. This is deliberately conservative
and is NOT the true vintage: a value observed on 2026-03-25 may well have been
published months earlier. Treating our write date as the vintage can only make
a backtest more pessimistic (it withholds data we might legitimately have had),
never more optimistic. It cannot manufacture look-ahead. Rows written from
phase-82.0 onward carry a true first-observation vintage.

Column is NULLABLE: BigQuery cannot add a REQUIRED column to a populated table,
and consumers must handle pre-migration NULLs anyway.

IDEMPOTENT: safe to re-run. `ADD COLUMN IF NOT EXISTS` is a no-op when the
column is present, and the backfill only touches rows where it is still NULL.

Usage:
    python scripts/migrations/add_macro_realtime_start.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config.settings import get_settings


DDL_ADD_COLUMN = """
ALTER TABLE `{project}.{dataset}.historical_macro`
ADD COLUMN IF NOT EXISTS realtime_start DATE
  OPTIONS(description="phase-82.0 vintage: the date this observation became known to us. Point-in-time joins must filter on date <= cutoff AND realtime_start <= cutoff. NULL only for rows written before the migration.")
"""

# Conservative backfill -- see the module docstring. DATE(ingested_at) is the
# earliest provable knowledge date, never earlier than the truth, so it cannot
# introduce look-ahead.
DML_BACKFILL = """
UPDATE `{project}.{dataset}.historical_macro`
SET realtime_start = DATE(ingested_at)
WHERE realtime_start IS NULL
"""

VERIFY = """
SELECT
  COUNT(*) AS total_rows,
  COUNTIF(realtime_start IS NULL) AS null_vintage,
  MIN(realtime_start) AS earliest_vintage,
  MAX(realtime_start) AS latest_vintage
FROM `{project}.{dataset}.historical_macro`
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print SQL, execute nothing")
    args = ap.parse_args()

    settings = get_settings()
    project = settings.gcp_project_id
    # historical_macro lives in the reports dataset (financial_reports),
    # NOT pyfinagent_data -- see backend/db/bigquery_client.py::_pt_table.
    dataset = settings.bq_dataset_reports

    ctx = {"project": project, "dataset": dataset}
    stmts = [
        ("add realtime_start column", DDL_ADD_COLUMN.format(**ctx)),
        ("backfill vintage from ingested_at", DML_BACKFILL.format(**ctx)),
    ]

    if args.dry_run:
        for label, sql in stmts:
            print(f"-- {label}\n{sql}")
        print(f"-- verify\n{VERIFY.format(**ctx)}")
        return 0

    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    for label, sql in stmts:
        print(f"[migration] {label} ...")
        client.query(sql).result(timeout=120)
        print(f"[migration] {label}: OK")

    print("[migration] verifying ...")
    for row in client.query(VERIFY.format(**ctx)).result(timeout=60):
        print(
            f"  total_rows={row.total_rows} null_vintage={row.null_vintage} "
            f"earliest={row.earliest_vintage} latest={row.latest_vintage}"
        )
        if row.null_vintage:
            print(
                f"[migration] WARNING: {row.null_vintage} rows still have a NULL "
                "vintage -- investigate before relying on point-in-time joins."
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
