# Sprint Contract — phase-6.5 / step 6.5.1 (BigQuery intel schema migration)

**Step id:** 6.5.1
**Phase:** phase-6.5 Global Intelligence Directive (Path D; 4 pending + 5 dropped)
**Cycle:** 1
**Date:** 2026-04-19
**Tier:** moderate

**Parallel-safety note:** all handoff files phase-scoped (`phase-6.5.1-*.md`) to avoid the autonomous harness clobbering rolling `research_brief.md`/`contract.md`/`experiment_results.md` (same pattern as phase-2.12 closeout).

## Research-gate summary

Researcher fetched 6 sources in full (GCP BigQuery partitioning docs, BigQuery creating-partitioned-tables, OneUptime partition+cluster combining, GCP Dataflow vector ingestion, GCP blog RAG+BigQuery+LangChain, SemHash semantic dedup GitHub), 16 URLs collected, recency scan performed (2024–2026), three-variant queries run. Internal audit anchored on `scripts/migrations/add_news_sentiment_schema.py:110-147` (dry-run pattern + deferred BQ import) and `backend/tests/test_bq_writer.py:27-81` (frozen field-set assertion pattern). Brief at `handoff/current/phase-6.5.1-research-brief.md`. `gate_passed: true`.

## Hypothesis

The intel pipeline's alpha-producing backbone is five tables: a source registry (governance/kill-switch), an append-only document fact table (dedup via `content_hash` + `canonical_url`), a chunks table with inline embedding arrays, a re-scorable novelty-score enrichment, and a prompt-patch queue. Schema can be created in one idempotent migration (`CREATE TABLE IF NOT EXISTS`), partitioned on DATE + clustered on the highest-cardinality filter columns, with no live BQ write in the dry-run path. A fail-open pytest asserts every required column is declared.

## Immutable success criteria (copied verbatim from .claude/masterplan.json)

- `migration_dry_run_exit_0`
- `all_intel_tables_defined_in_script`
- `schema_test_green`

**Do not edit.** Interpretation:
- `migration_dry_run_exit_0` ← `python scripts/migrations/phase_6_5_intel_schema.py --dry-run` returns 0.
- `all_intel_tables_defined_in_script` ← every table named in the schema proposal appears as a `CREATE TABLE IF NOT EXISTS …` in the script, with every documented column present.
- `schema_test_green` ← `pytest backend/tests/test_intel_schema.py -q` exits 0 with every asserted field set matching.

## Tables (design — final)

Dataset: `pyfinagent_data` (via `settings.bq_dataset_observability` with fallback, matching news-sentiment precedent).

1. **`intel_sources`** — source registry + kill-switch. Columns: `source_id STRING NOT NULL`, `source_name STRING NOT NULL`, `source_type STRING NOT NULL` (institutional|academic|ai_frontier|player_driven|news), `kill_switch BOOL NOT NULL`, `rate_limit_per_day INT64`, `last_scanned_at TIMESTAMP`, `created_at TIMESTAMP NOT NULL`, `updated_at TIMESTAMP`, `metadata JSON`. PARTITION BY DATE(created_at) CLUSTER BY source_type, source_name.
2. **`intel_documents`** — append-only raw fact table. Columns: `doc_id STRING NOT NULL`, `source_id STRING NOT NULL`, `source_type STRING NOT NULL`, `doc_type STRING`, `published_at TIMESTAMP`, `ingested_at TIMESTAMP NOT NULL`, `title STRING`, `authors ARRAY<STRING>`, `url STRING`, `canonical_url STRING`, `content_hash STRING`, `raw_text STRING`, `language STRING`, `raw_payload JSON`. PARTITION BY DATE(ingested_at) CLUSTER BY source_type, doc_type.
3. **`intel_chunks`** — chunked document rows with inline embedding. Columns: `chunk_id STRING NOT NULL`, `doc_id STRING NOT NULL`, `chunk_index INT64 NOT NULL`, `chunk_text STRING NOT NULL`, `embedding ARRAY<FLOAT64>`, `embedding_model STRING`, `tokens INT64`, `ingested_at TIMESTAMP NOT NULL`. PARTITION BY DATE(ingested_at) CLUSTER BY doc_id, chunk_index.
4. **`intel_novelty_scores`** — re-scorable enrichment. Columns: `chunk_id STRING NOT NULL`, `scorer_model STRING NOT NULL`, `scorer_version STRING`, `scored_at TIMESTAMP NOT NULL`, `novelty_score FLOAT64`, `nearest_neighbor_chunk_id STRING`, `nearest_neighbor_distance FLOAT64`, `latency_ms FLOAT64`, `cost_usd FLOAT64`. PARTITION BY DATE(scored_at) CLUSTER BY chunk_id, scorer_model.
5. **`intel_prompt_patches`** — pending/approved LLM prompt-patch queue. Columns: `patch_id STRING NOT NULL`, `chunk_id STRING`, `patch_type STRING NOT NULL` (strategy_hint|risk_flag|regime_flag|data_source), `patch_text STRING NOT NULL`, `rationale STRING`, `status STRING NOT NULL` (pending|approved|rejected|applied|expired), `created_at TIMESTAMP NOT NULL`, `reviewed_at TIMESTAMP`, `reviewed_by STRING`, `applied_at TIMESTAMP`, `metadata JSON`. PARTITION BY DATE(created_at) CLUSTER BY status, patch_type.

## Plan steps

1. Write `scripts/migrations/phase_6_5_intel_schema.py` mirroring `add_news_sentiment_schema.py`:
   - One DDL constant per table (5 total)
   - `main(dry_run: bool)` prints each DDL block with a banner; if `dry_run` → exit 0 without BQ import
   - Live path defers `from google.cloud import bigquery` to inside the function (no cost on dry-run)
2. Write `backend/tests/test_intel_schema.py`:
   - Frozen field-set constants, one per table
   - Parse each DDL constant with a regex that extracts column names inside the parens
   - Assert extracted set == expected set (no missing, no unexpected)
   - Assert each DDL contains `CREATE TABLE IF NOT EXISTS` (idempotency predicate)
   - Assert each DDL contains `PARTITION BY` and `CLUSTER BY` (partition/cluster predicate)
   - Assert `main(dry_run=True)` returns 0 and does NOT import `google.cloud.bigquery` (lazy-import predicate)
3. Run the immutable command; capture verbatim output.
4. Run full regression `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` (expect 103 → ~108 passed, 1 skipped).
5. Write `handoff/current/phase-6.5.1-experiment-results.md`.
6. Spawn `qa` subagent.
7. On PASS: append cycle block to `handoff/harness_log.md`, then flip `phase-6.5.1.status: pending → done`.

## Out of scope

- No live BigQuery writes (migration only gets dry-run exercised). The live path is implicit (same pattern as `add_news_sentiment_schema.py`) but is NOT tested in this cycle.
- No ingestion code — that's phase-6.5.2.
- No novelty-scoring code — that's phase-6.5.7.
- No Slack/digest code — dropped under Path D.
- ASCII-only logger messages (per `.claude/rules/security.md`).

## References

- `handoff/current/phase-6.5.1-research-brief.md`
- `handoff/current/phase-6.5-decision-contract.md` (Path D)
- `scripts/migrations/add_news_sentiment_schema.py:110-147` (house pattern)
- `backend/tests/test_bq_writer.py:27-81` (frozen-field-set test pattern)
- `.claude/masterplan.json` → phase-6.5 / 6.5.1 (immutable verification)
- Evidence: GCP partitioning docs (canonical), OneUptime partition+cluster cost analysis (2026), GCP Dataflow vector ingestion (RAG/embedding pattern).
