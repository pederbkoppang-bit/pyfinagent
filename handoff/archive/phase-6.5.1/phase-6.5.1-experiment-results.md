# Experiment Results — phase-6.5 / step 6.5.1 (BigQuery intel schema migration)

**Step:** 6.5.1 — first executable step under Path D (4-step reduced phase-6.5).
**Date:** 2026-04-19.
**Cycle:** 1.

## What was built

Two files (100% greenfield; no existing code changed):

1. **`scripts/migrations/phase_6_5_intel_schema.py`** (~180 lines). Mirrors `scripts/migrations/add_news_sentiment_schema.py` (phase-6.1 precedent) with five idempotent `CREATE TABLE IF NOT EXISTS` DDLs. Dry-run prints each DDL with a banner and exits 0 before any BQ import. Live branch defers `from google.cloud import bigquery` until after the `if dry_run: return 0` early-exit (same pattern as the news-sentiment migration at line 133).

2. **`backend/tests/test_intel_schema.py`** (~115 lines). Frozen field-set constants per table; a column extractor that walks paren-depth to isolate the column list from later `OPTIONS (...)` blocks; 8 tests asserting declared tables, expected columns (none missing, none extra), idempotency, partition+cluster discipline, lazy BQ import on dry-run, `embedding ARRAY<FLOAT64>` inline on chunks, JSON-typed metadata columns, and ASCII-only DDLs (per `.claude/rules/security.md`).

Tables created (all in `pyfinagent_data` via `settings.bq_dataset_observability`):

| Table | Partition | Cluster | Column count |
|---|---|---|---|
| `intel_sources` | `DATE(created_at)` | `source_type, source_name` | 9 |
| `intel_documents` | `DATE(ingested_at)` | `source_type, doc_type` | 14 |
| `intel_chunks` | `DATE(ingested_at)` | `doc_id, chunk_index` | 8 |
| `intel_novelty_scores` | `DATE(scored_at)` | `chunk_id, scorer_model` | 9 |
| `intel_prompt_patches` | `DATE(created_at)` | `status, patch_type` | 11 |

## File list

Created:
- `scripts/migrations/phase_6_5_intel_schema.py`
- `backend/tests/test_intel_schema.py`

Modified: none. No production backend module was touched.

## Verification command output

### Immutable (masterplan 6.5.1)

```
$ source .venv/bin/activate && python scripts/migrations/phase_6_5_intel_schema.py --dry-run && pytest backend/tests/test_intel_schema.py -q
== intel_sources (dry-run) ==
CREATE TABLE IF NOT EXISTS `sunny-might-477607-p8.pyfinagent_data.intel_sources` (
  source_id STRING NOT NULL, ...
  PARTITION BY DATE(created_at) CLUSTER BY source_type, source_name ...
)
== intel_documents (dry-run) ==  ...
== intel_chunks (dry-run) ==  ...
== intel_novelty_scores (dry-run) ==  ...
== intel_prompt_patches (dry-run) ==  ...
dry-run: no BigQuery writes executed.

........                                                                 [100%]
8 passed in 0.02s
CHAIN_EXIT=0
```

Chain exit 0. All 5 DDL banners printed. All 8 schema tests pass.

### Full regression (no_regressions check)

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
...
111 passed, 1 skipped, 1 warning in 6.25s
```

Baseline was 103 passed, 1 skipped (phase-2.12 closeout). Delta = +8 passed = the 8 new `test_intel_schema.py` tests. Zero regressions on the previously-green surface.

### Syntax check (defensive)

```
$ python -c "import ast; ast.parse(open('scripts/migrations/phase_6_5_intel_schema.py').read()); ast.parse(open('backend/tests/test_intel_schema.py').read()); print('SYNTAX OK')"
SYNTAX OK
```

## Contract criterion check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `migration_dry_run_exit_0` | PASS | Dry-run returned 0; "dry-run: no BigQuery writes executed." confirmed in stdout. |
| 2 | `all_intel_tables_defined_in_script` | PASS | `test_all_tables_declared_in_ddls_constant` + `test_each_ddl_has_expected_columns` (no missing, no extra). 51 columns total matched. |
| 3 | `schema_test_green` | PASS | 8 passed / 0 failed in `backend/tests/test_intel_schema.py`. |

## Known caveats (transparency)

1. **Live BQ branch untested this cycle.** The `else` path that imports `google-cloud-bigquery` and calls `client.query(sql).result(...)` is the same shape as the proven news-sentiment migration but has not been exercised against a live BQ project in this step. That's intentional per the contract's out-of-scope list and matches the house pattern.
2. **Column extractor is regex-based.** It works by walking paren-depth and matching `^\s*IDENT\s+` before a type keyword. It does NOT parse BigQuery's full grammar. If a future DDL ever uses generated columns, check constraints, or non-standard quoting, the extractor could miss or mis-attribute — test has a `test_each_ddl_has_expected_columns` that would FAIL loudly in that case rather than silently passing.
3. **First-draft test-extractor bug caught mid-cycle.** Initial extractor used `ddl.rfind(")")` and wrapped the `OPTIONS (description="...")` block, so `description` leaked into the column set and failed `test_each_ddl_has_expected_columns` on `intel_sources`. Fixed to walk paren-depth and exit on the first balanced close. Documented here for auditability; no "all green on first try" rubber-stamp.
4. **ASCII-only discipline honored.** `test_no_unicode_in_ddls` uses `.encode("ascii")` which raises `UnicodeEncodeError` if any non-ASCII slips in. No stray em-dashes / arrows in the migration file.

## Pre-Q/A self-check

- Immutable command chain exit 0.
- All 8 schema tests pass.
- Full regression 111p/1s (was 103p/1s; +8 = new tests; zero existing tests broken).
- `git status --short` shows 2 new files only (migration + test); no production backend module touched.
- `masterplan.json` NOT flipped yet; log-last discipline preserved.
- Handoff files phase-scoped: `phase-6.5.1-contract.md`, `phase-6.5.1-experiment-results.md`, `phase-6.5.1-research-brief.md`.
