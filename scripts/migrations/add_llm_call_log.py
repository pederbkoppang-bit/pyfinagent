"""phase-4.14.23 migration: create the `llm_call_log` BigQuery table.

Stores per-call Claude/Gemini/OpenAI latency telemetry so the harness
tab can render p95 trend lines. Written to satisfy masterplan
4.14.23 success_criterion `BQ_llm_call_log_table_created`.

Schema:
  ts         TIMESTAMP NOT NULL     -- request start (UTC)
  provider   STRING    NOT NULL     -- anthropic | gemini | openai | github-models
  model      STRING    NOT NULL     -- canonical model id (e.g. claude-opus-4-7)
  agent      STRING                 -- caller label (e.g. "Bull R1/2")
  latency_ms FLOAT64   NOT NULL     -- wall-clock ms, perf_counter bracket
  ttft_ms    FLOAT64                -- time to first token; == latency_ms on non-streaming
  input_tok  INT64                  -- prompt tokens
  output_tok INT64                  -- completion tokens
  request_id STRING                 -- Anthropic request_id for cross-ref
  ok         BOOL      NOT NULL     -- true on 2xx, false on exception

Partitioned on DATE(ts); clustered by (provider, model) for cheap
p50/p95 rollups per model per day.

Run with: `python scripts/migrations/add_llm_call_log.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make backend importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config.settings import get_settings


TABLE_ID_SUFFIX = "llm_call_log"

DDL = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` (
  ts TIMESTAMP NOT NULL,
  provider STRING NOT NULL,
  model STRING NOT NULL,
  agent STRING,
  latency_ms FLOAT64 NOT NULL,
  ttft_ms FLOAT64,
  input_tok INT64,
  output_tok INT64,
  request_id STRING,
  ok BOOL NOT NULL
)
PARTITION BY DATE(ts)
CLUSTER BY provider, model
OPTIONS (
  description = "phase-4.14.23 per-call LLM latency telemetry"
)
"""


def main() -> int:
    settings = get_settings()
    project = settings.gcp_project_id
    dataset = getattr(settings, "bq_dataset_observability", None) or "pyfinagent_data"
    table = TABLE_ID_SUFFIX
    sql = DDL.format(project=project, dataset=dataset, table=table)
    print("Running DDL:\n" + sql)

    from google.cloud import bigquery  # type: ignore
    client = bigquery.Client(project=project)
    job = client.query(sql)
    job.result(timeout=60)
    print(f"OK: {project}.{dataset}.{table} ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
