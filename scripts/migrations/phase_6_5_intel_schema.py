"""phase-6.5.1 migration: create the 5 intel-ingestion BQ tables.

Dataset: `pyfinagent_data` (configurable via `settings.bq_dataset_observability`).
Tables:

`intel_sources` -- source registry + kill-switch
`intel_documents` -- append-only raw fact (dedup anchors: canonical_url, content_hash)
`intel_chunks` -- paragraph/sentence chunks with inline embedding ARRAY<FLOAT64>
`intel_novelty_scores` -- re-scorable enrichment keyed on chunk_id + scorer_model
`intel_prompt_patches` -- pending/approved LLM prompt-patch queue

Design choices (see `handoff/current/phase-6.5.1-research-brief.md` +
`handoff/current/phase-6.5.1-contract.md`):
- All tables partitioned on DATE(...) + clustered on the highest-cardinality
  filter columns. Each DDL is `CREATE TABLE IF NOT EXISTS` so the script
  is idempotent (re-runnable).
- `embedding ARRAY<FLOAT64>` lives inline on `intel_chunks` for BQ
  `VECTOR_SEARCH` compatibility (see GCP Dataflow vector-ingestion doc).
- Dedup logic is NOT here; ingestion cron (phase-6.5.2) uses
  `canonical_url` + `content_hash` as anchors.
- Google Cloud BigQuery import is deferred to the live branch so dry-run
  has zero external dependencies.

Run:
    python scripts/migrations/phase_6_5_intel_schema.py           # execute
    python scripts/migrations/phase_6_5_intel_schema.py --dry-run # print DDL only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config.settings import get_settings


DDL_INTEL_SOURCES = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.intel_sources` (
  source_id STRING NOT NULL,
  source_name STRING NOT NULL,
  source_type STRING NOT NULL,
  kill_switch BOOL NOT NULL,
  rate_limit_per_day INT64,
  last_scanned_at TIMESTAMP,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP,
  metadata JSON
)
PARTITION BY DATE(created_at)
CLUSTER BY source_type, source_name
OPTIONS (
  description = "phase-6.5.1 intel source registry + kill-switch"
)
"""


DDL_INTEL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.intel_documents` (
  doc_id STRING NOT NULL,
  source_id STRING NOT NULL,
  source_type STRING NOT NULL,
  doc_type STRING,
  published_at TIMESTAMP,
  ingested_at TIMESTAMP NOT NULL,
  title STRING,
  authors ARRAY<STRING>,
  url STRING,
  canonical_url STRING,
  content_hash STRING,
  raw_text STRING,
  language STRING,
  raw_payload JSON
)
PARTITION BY DATE(ingested_at)
CLUSTER BY source_type, doc_type
OPTIONS (
  description = "phase-6.5.1 intel documents append-only fact table"
)
"""


DDL_INTEL_CHUNKS = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.intel_chunks` (
  chunk_id STRING NOT NULL,
  doc_id STRING NOT NULL,
  chunk_index INT64 NOT NULL,
  chunk_text STRING NOT NULL,
  embedding ARRAY<FLOAT64>,
  embedding_model STRING,
  tokens INT64,
  ingested_at TIMESTAMP NOT NULL
)
PARTITION BY DATE(ingested_at)
CLUSTER BY doc_id, chunk_index
OPTIONS (
  description = "phase-6.5.1 intel chunks with inline embedding array"
)
"""


DDL_INTEL_NOVELTY_SCORES = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.intel_novelty_scores` (
  chunk_id STRING NOT NULL,
  scorer_model STRING NOT NULL,
  scorer_version STRING,
  scored_at TIMESTAMP NOT NULL,
  novelty_score FLOAT64,
  nearest_neighbor_chunk_id STRING,
  nearest_neighbor_distance FLOAT64,
  latency_ms FLOAT64,
  cost_usd FLOAT64
)
PARTITION BY DATE(scored_at)
CLUSTER BY chunk_id, scorer_model
OPTIONS (
  description = "phase-6.5.1 intel novelty-score enrichment (re-scorable)"
)
"""


DDL_INTEL_PROMPT_PATCHES = """
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.intel_prompt_patches` (
  patch_id STRING NOT NULL,
  chunk_id STRING,
  patch_type STRING NOT NULL,
  patch_text STRING NOT NULL,
  rationale STRING,
  status STRING NOT NULL,
  created_at TIMESTAMP NOT NULL,
  reviewed_at TIMESTAMP,
  reviewed_by STRING,
  applied_at TIMESTAMP,
  metadata JSON
)
PARTITION BY DATE(created_at)
CLUSTER BY status, patch_type
OPTIONS (
  description = "phase-6.5.1 intel prompt-patch queue (pending/approved/applied)"
)
"""


DDLS = (
    ("intel_sources", DDL_INTEL_SOURCES),
    ("intel_documents", DDL_INTEL_DOCUMENTS),
    ("intel_chunks", DDL_INTEL_CHUNKS),
    ("intel_novelty_scores", DDL_INTEL_NOVELTY_SCORES),
    ("intel_prompt_patches", DDL_INTEL_PROMPT_PATCHES),
)


def main(dry_run: bool) -> int:
    settings = get_settings()
    project = settings.gcp_project_id
    dataset = (
        getattr(settings, "bq_dataset_observability", None)
        or "pyfinagent_data"
    )

    rendered = [
        (table, tmpl.format(project=project, dataset=dataset))
        for table, tmpl in DDLS
    ]

    for table, sql in rendered:
        banner = f"== {table} ({'dry-run' if dry_run else 'live'}) =="
        print(banner)
        print(sql.strip())
        print()

    if dry_run:
        print("dry-run: no BigQuery writes executed.")
        return 0

    from google.cloud import bigquery  # type: ignore
    client = bigquery.Client(project=project)
    for table, sql in rendered:
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
