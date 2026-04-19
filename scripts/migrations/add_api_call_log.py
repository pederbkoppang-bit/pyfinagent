"""phase-6.7 migration: create `api_call_log` BQ table.

Non-LLM API telemetry. Parallels `llm_call_log` (phase-4.14.23) but with a
different column set because the schemas do not overlap (OneUptime 2026
guidance: separate AI telemetry from general API telemetry).

Columns:
    ts TIMESTAMP NOT NULL
    source STRING NOT NULL           (finnhub | benzinga | alpaca | fred | alphavantage | ...)
    endpoint STRING                  (URL path or full URL)
    http_status INT64
    latency_ms FLOAT64
    response_bytes INT64
    cost_usd_est FLOAT64             (0.0 for free tier)
    ok BOOL                          (http_status < 400)
    error_kind STRING                (Timeout | HTTPError | RateLimited | ConnError | None)
    request_id STRING                (server-issued id, when present)

Partition / cluster:
    PARTITION BY DATE(ts) -- time-series query shape
    CLUSTER BY source, ok -- typical filter is "finnhub failures in the last 24h"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config.settings import get_settings


DDL_API_CALL_LOG = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.api_call_log` (
  ts TIMESTAMP NOT NULL,
  source STRING NOT NULL,
  endpoint STRING,
  http_status INT64,
  latency_ms FLOAT64,
  response_bytes INT64,
  cost_usd_est FLOAT64,
  ok BOOL,
  error_kind STRING,
  request_id STRING
)
PARTITION BY DATE(ts)
CLUSTER BY source, ok
OPTIONS (
  description = "phase-6.7 non-LLM external API call telemetry (cost + rate-limit attribution)"
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
            "api_call_log",
            DDL_API_CALL_LOG.format(project=project, dataset=dataset),
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
