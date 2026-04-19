"""phase-6.6 migration: create `calendar_events` BQ table.

One table in `pyfinagent_data` (configurable via `bq_dataset_observability`
fallback). Forward-looking + historical FOMC / earnings / macro calendar
events. Dedup anchor is `event_id = sha256(event_type + "|" + ticker + "|"
+ (fiscal_period_end or DATE(scheduled_at)))` -- computed client-side in
`backend/calendar/normalize.py`.

Design choices (see `handoff/current/phase-6.6-research-brief.md`):
- `calendar_events` is NET-NEW. Zero `*_calendar*` or `*_events*` tables
  exist in `pyfinagent_data` today (verified via BQ MCP audit).
- `event_id` as dedup key rather than composite `(ticker, fiscal_period_end)`
  because macro/FOMC events have no ticker. Single primary-key pattern
  simplifies MERGE upserts downstream.
- Partition by `DATE(scheduled_at)` -- we query by forward window.
- Cluster by `event_type, ticker` -- typical filter path is "earnings for
  AAPL in the next 30 days" or "all fomc_meeting in 2026".
- Blackout columns are NULLable -- only populated for FOMC events.

Run:
    python scripts/migrations/add_calendar_events_schema.py           # execute
    python scripts/migrations/add_calendar_events_schema.py --dry-run # print DDL only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config.settings import get_settings


DDL_CALENDAR_EVENTS = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.calendar_events` (
  event_id STRING NOT NULL,
  event_type STRING NOT NULL,
  ticker STRING,
  scheduled_at TIMESTAMP NOT NULL,
  window STRING,
  fiscal_period_end DATE,
  source STRING NOT NULL,
  confidence STRING NOT NULL,
  blackout_start TIMESTAMP,
  blackout_end TIMESTAMP,
  eps_estimate FLOAT64,
  revenue_estimate FLOAT64,
  fetched_at TIMESTAMP NOT NULL,
  metadata JSON
)
PARTITION BY DATE(scheduled_at)
CLUSTER BY event_type, ticker
OPTIONS (
  description = "phase-6.6 calendar events (FOMC + earnings + macro releases)"
)
"""


def main(dry_run: bool) -> int:
    settings = get_settings()
    project = settings.gcp_project_id
    dataset = (
        getattr(settings, "bq_dataset_observability", None)
        or "pyfinagent_data"
    )

    ddls = [
        (
            "calendar_events",
            DDL_CALENDAR_EVENTS.format(project=project, dataset=dataset),
        ),
    ]

    for table, sql in ddls:
        banner = f"== {table} ({'dry-run' if dry_run else 'live'}) =="
        print(banner)
        print(sql.strip())
        print()

    if dry_run:
        print("dry-run: no BigQuery writes executed.")
        return 0

    from google.cloud import bigquery  # type: ignore
    client = bigquery.Client(project=project)
    for table, sql in ddls:
        print(f"executing DDL for {table}...")
        job = client.query(sql)
        job.result(timeout=60)
        print(f"OK: {project}.{dataset}.{table} ready.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
