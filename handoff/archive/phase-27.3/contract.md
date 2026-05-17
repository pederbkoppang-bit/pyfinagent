# Sprint Contract — phase-27.4 (B-2: BQ schema migration, 5 missing analysis_results columns)

Generated: 2026-05-16T22:08:00+00:00
Owner: Main
Step id: 27.4
Depends on: 27.0 (research gate done)

## Research-gate summary

`handoff/current/research_brief.md` §"C4 (B-2) — BQ ADD COLUMN idempotent pattern" (lines 326-380). Authoritative: `https://cloud.google.com/bigquery/docs/managing-table-schemas`. Internal precedent: `scripts/migrations/add_sector_to_paper_positions.py:43-55` (idempotent `ALTER TABLE … ADD COLUMN IF NOT EXISTS … OPTIONS(description=…)`). BigQuery DDL: **ONE column per `ALTER TABLE` statement** — multi-column ADD-IF-NOT-EXISTS is NOT supported (unlike PostgreSQL).

## Hypothesis

A new idempotent migration script `scripts/migrations/add_phase27_columns.py` that issues 5 separate `ALTER TABLE ADD COLUMN IF NOT EXISTS` statements (one per column) and exits 0 will unblock B-2. Once the columns exist, `BigQueryClient.save_report()` (which writes them per `bigquery_client.py:113-117`) will succeed instead of raising `no such field: <col>` on every lite-path persist. No code change to `bigquery_client.py` needed — the writer is correct; the schema is what was wrong.

Falsifier: if any of the 5 columns already exists with a different type (not `FLOAT64`), `ALTER TABLE ADD COLUMN IF NOT EXISTS` would silently no-op and the writer would still fail. Check pre-migration: `describe-table` confirms all 5 are simply absent (not type-mismatched).

## Immutable success criteria (verbatim from `.claude/masterplan.json` step 27.4)

```bash
source .venv/bin/activate && python -c "
from google.cloud import bigquery
c=bigquery.Client(project='sunny-might-477607-p8')
t=c.get_table('sunny-might-477607-p8.financial_reports.analysis_results')
names={f.name for f in t.schema}
missing={'consumer_sentiment','revenue_growth_yoy','quality_score','momentum_6m','rsi_14'} - names
assert not missing, f'still missing: {missing}'
print('PASS, all 5 columns present')"
```

Plus the live_check: `run-now` after migration produces ≥1 row in `financial_reports.analysis_results` with non-null values for at least one of the 5 newly-added columns. (Verified in 27.5 cycle, not here.)

## Plan steps

1. Write `scripts/migrations/add_phase27_columns.py` mirroring `add_sector_to_paper_positions.py` structure:
   - `--apply` flag (default dry-run)
   - 5 separate ALTER TABLE statements (one per column)
   - Each with `OPTIONS(description=…)` citing the audit
   - Logs each DDL + outcome
2. Dry-run the script (assert it lists 5 DDLs without executing).
3. Apply the migration (`--apply`).
4. Run the immutable verification command above.
5. Q/A spawn.
6. harness_log append.
7. Flip 27.4 to done.

## Anti-patterns to avoid

- Do NOT try multi-column `ADD COLUMN` — BigQuery rejects it.
- Do NOT add a backfill step — the columns are NULLABLE; existing rows can remain NULL. Future writes will populate them.
- Do NOT change `bigquery_client.py::save_report` — the writer's signature is correct.

## References

- `handoff/current/research_brief.md` lines 326-380 (C4 section)
- `scripts/migrations/add_sector_to_paper_positions.py` (idempotent pattern reference)
- `backend/db/bigquery_client.py:113-117` (writer expecting these 5 columns)
- `docs/audits/smoke_test_preprod_2026-05-16.md` §A B-2 (origin bug)
- `.claude/masterplan.json` phase-27 step 27.4 verification command (immutable)
