# Experiment Results — phase-27.4 (B-2 BQ schema migration)

Generated: 2026-05-16T22:12:00+00:00
Step id: 27.4
Owner: Main

## What was built/changed

### 1. New migration script

`scripts/migrations/add_phase27_columns.py` — idempotent migration that adds 5 FLOAT64 NULLABLE columns to `sunny-might-477607-p8.financial_reports.analysis_results` (us-central1):

| Column | Description |
|---|---|
| `consumer_sentiment` | composite (-1..+1) from Reddit + news + alt-data sources |
| `revenue_growth_yoy` | trailing-twelve-month revenue growth YoY (decimal fraction) |
| `quality_score` | composite quality score (1-10) from gross margin / ROIC / leverage |
| `momentum_6m` | 6-month price momentum (decimal return) |
| `rsi_14` | 14-day Relative Strength Index (0-100) |

Pattern follows `scripts/migrations/add_sector_to_paper_positions.py`. One `ALTER TABLE … ADD COLUMN IF NOT EXISTS` per column (BigQuery rejects multi-column ADD). Pre-flight type-mismatch check (would abort if any column existed with a non-FLOAT64 type). Post-flight verification confirms all 5 present.

### 2. Files modified

| File | Change |
|------|--------|
| `scripts/migrations/add_phase27_columns.py` | new file (~110 lines) |
| `handoff/current/contract.md` | rewritten for 27.4 |
| `handoff/current/experiment_results.md` | this file |
| BigQuery `financial_reports.analysis_results` schema | +5 columns (live change) |

No code changes to `bigquery_client.py::save_report` — the writer was already correct; only the schema was wrong.

## Verification — dry-run output

```
=== phase-27.4 migration ===
Target: `sunny-might-477607-p8.financial_reports.analysis_results`
Already present (no-op): none
To add: ['consumer_sentiment', 'revenue_growth_yoy', 'quality_score', 'momentum_6m', 'rsi_14']
DRY RUN — would execute:
ALTER TABLE `sunny-might-477607-p8.financial_reports.analysis_results`
ADD COLUMN IF NOT EXISTS consumer_sentiment FLOAT64
OPTIONS(description='phase-27.4: consumer-sentiment composite (-1..+1) ...')
... [4 more identical-shape DDLs] ...
DRY RUN complete. Re-run with --apply to execute.
```

## Verification — apply output

```
=== phase-27.4 migration ===
Already present (no-op): none
To add: ['consumer_sentiment', 'revenue_growth_yoy', 'quality_score', 'momentum_6m', 'rsi_14']
Added column: consumer_sentiment
Added column: revenue_growth_yoy
Added column: quality_score
Added column: momentum_6m
Added column: rsi_14
ok phase-27.4 migration: all 5 columns now present
```

Each ALTER TABLE took ~800ms-1100ms. Total wall time: ~5 seconds.

## Verification command output (verbatim from masterplan 27.4)

```bash
$ eval "$(jq -r '.phases[] | select(.id=="phase-27") | .steps[] | select(.id=="27.4") | .verification.command' .claude/masterplan.json)"
PASS, all 5 columns present
$ echo $?
0
```

## Idempotency confirmation

Re-running `--apply` after the columns exist:

```
=== phase-27.4 migration ===
Already present (no-op): ['consumer_sentiment', 'momentum_6m', 'quality_score', 'revenue_growth_yoy', 'rsi_14']
To add: none — fully idempotent re-run
ok phase-27.4 migration is a no-op; all 5 columns already exist
```

Exit 0. No DDL issued. Safe to re-run.

## Live check — next persist should succeed (deferred to 27.5)

Pre-fix evidence (live observation 2026-05-16 cycle 756a19c7):
```
22:31:15 W Failed to persist lite analysis for FIX: Failed to save report:
  [{'index':0,'errors':[{'reason':'invalid','location':'momentum_6m','message':'no such field: momentum_6m.'}]}]
```

Post-fix: the schema now has `momentum_6m`. The persist that previously failed will now succeed. End-to-end evidence is captured by phase-27.5 (`run-now` cycle producing a row in `analysis_results` with non-null values for at least one of the 5 columns).

## Artifact shape

- Script: `python scripts/migrations/add_phase27_columns.py [--apply]`
- Default mode: dry-run (lists DDLs, no execution)
- With `--apply`: executes the 5 DDLs and verifies post-state
- Pre-flight type-check: aborts before any DDL if any of the 5 already exists with a wrong type
- Post-flight verification: re-reads schema and asserts all 5 are present (exits 5 if not)

## Risks / known limits

- Backfill of existing rows: NOT performed. Existing rows have NULL in the 5 new columns; future writes from `save_report()` will populate them. The audit's expectation was forward-looking; no historical backfill required.
- The `consumer_sentiment` and `quality_score` features are PLANNED but not yet computed by any Layer-1 skill — the columns exist for downstream wiring (a separate phase). The migration creates the table structure; populating those columns end-to-end is masterplan post-launch work.
