# Sprint Contract -- phase-25.A3 -- Write promoted strategies to BQ table

**Cycle:** phase-25 cycle 14 (P1 sprint)
**Date:** 2026-05-12
**Step ID:** 25.A3
**Priority:** P1
**Audit basis:** bucket 24.3 F-3 -- `friday_promotion.py:121-131` writes only to flat `weekly_ledger.tsv` with no BQ subscriber

## Research-gate

Researcher spawned this cycle (agent a5d3f536215a3aa1f). Brief at
`handoff/current/research_brief.md`. Gate envelope: tier=moderate,
external_sources_read_in_full=5, urls_collected=13, recency_scan_performed=true,
internal_files_inspected=8, gate_passed=true.

Key research conclusions:
- Mirror `scripts/migrations/create_options_snapshots_table.py` skeleton (argparse + CREATE_SQL + main + `--apply`).
- Schema: 10 columns -- `strategy_id`, `week_iso`, `params` (JSON), `dsr`, `pbo`, `status`, `allocation_pct`, `promoted_at`, `sortino_monthly`, `rejection_reason`. PARTITION BY DATE(promoted_at), CLUSTER BY strategy_id, week_iso.
- Idempotency: MERGE on natural key `(week_iso, strategy_id)`. Mirror existing `save_paper_snapshot` shape at `bigquery_client.py:700-740`.
- JSON column: pass as STRING param `@v_params`, use `PARSE_JSON(@v_params)` in the MERGE SQL (no native JSON BQ parameter type).
- Dataset: hardcoded `pyfinagent_data` (precedent: `slot_accounting.py:26`); NOT `settings.bq_dataset_reports` which is "financial_reports".
- Wire point: after the `if not ok:` block in `friday_promotion.run_friday_promotion` (line ~140), add a loop over `top` calling `bq_client.save_promoted_strategy(row)`. When `bq_client is None`, skip silently (preserves existing test behavior + library purity).
- Cost: negligible (max 156 rows/year).

## Hypothesis

Adding (a) an idempotent BQ migration, (b) a `save_promoted_strategy`
helper on `BigQueryClient`, and (c) an optional `bq_client` kwarg on
`run_friday_promotion` that triggers a per-promotion MERGE -- closes
phase-24.3 F-3 without changing the existing TSV ledger behavior or
breaking any existing caller. New subscribers (25.B3 `load_promoted_params`)
can then read from BQ instead of parsing TSV.

## Success criteria (verbatim from masterplan)

1. `bigquery_promoted_strategies_table_exists`
2. `friday_promotion_writes_row_per_promotion`
3. `schema_includes_strategy_id_params_json_dsr_pbo_status`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_A3.py`

Live check (per masterplan):
`BQ promoted_strategies row visible after next Friday promotion run`

## Plan

1. **Migration** -- `scripts/migrations/create_promoted_strategies_table.py` (new file):
   - Mirror the canonical migration skeleton from `create_options_snapshots_table.py`.
   - `PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")`, `DATASET = "pyfinagent_data"`, `TABLE = "promoted_strategies"`.
   - CREATE TABLE IF NOT EXISTS with 10 columns, PARTITION BY DATE(promoted_at), CLUSTER BY strategy_id, week_iso.
   - Argparse `--apply` (else dry-run prints SQL).
2. **BQ client** -- `backend/db/bigquery_client.py`:
   - New method `save_promoted_strategy(self, row: dict) -> None` that MERGEs on `(week_iso, strategy_id)`.
   - FQN hardcoded: `f"{self.settings.gcp_project_id}.pyfinagent_data.promoted_strategies"`.
   - Use `PARSE_JSON(@v_params)` for the params column on both INSERT and UPDATE SET arms.
   - Pass `params` as STRING param (the source dict serialized via `json.dumps`); pass numeric fields as FLOAT64; status/timestamps as STRING.
   - Per CLAUDE.md rule, append `result(timeout=30)` on the query.
3. **friday_promotion wiring** -- `backend/autoresearch/friday_promotion.py`:
   - Function signature gains `bq_client: Any | None = None` kwarg (default None preserves all existing callers).
   - After the `if not ok:` block (~line 140), if `bq_client is not None`, loop over `top` candidates and call `bq_client.save_promoted_strategy(row)` with the dict from the research brief.
   - Per-row try/except so one BQ write failure doesn't poison the others or block the function return.
   - Import `json` + `datetime, timezone` at module top.
4. **Verifier** -- `tests/verify_phase_25_A3.py` -- 9+ claims:
   - Claim 1: migration script exists at the canonical path AND contains `pyfinagent_data.promoted_strategies` table name.
   - Claim 2: migration CREATE_SQL declares all 10 columns (strategy_id, week_iso, params, dsr, pbo, status, allocation_pct, promoted_at, sortino_monthly, rejection_reason).
   - Claim 3: migration uses `CREATE TABLE IF NOT EXISTS` (idempotent).
   - Claim 4: `params` column is `JSON` (not STRING).
   - Claim 5: PARTITION BY DATE(promoted_at) + CLUSTER BY strategy_id, week_iso.
   - Claim 6: `BigQueryClient.save_promoted_strategy(row: dict) -> None` exists in `bigquery_client.py`.
   - Claim 7: MERGE on `week_iso AND strategy_id` (natural key idempotency) + `result(timeout=30)`.
   - Claim 8: `run_friday_promotion` signature contains `bq_client` kwarg (default None).
   - Claim 9: **Behavioral round-trip** -- pass a fake `bq_client` with a `save_promoted_strategy(row)` mock to `run_friday_promotion`; with one passing candidate, the mock receives exactly one call with the expected row shape (strategy_id/week_iso/params/dsr/pbo/status="pending"/allocation_pct/promoted_at/sortino_monthly).
   - Claim 10: with `bq_client=None`, the function still succeeds and ledger TSV write still happens (no regression).

## Non-goals

- No code changes in `weekly_ledger.tsv` writer or reader -- ledger keeps its existing role.
- No daily-loop consumer (`load_promoted_params`) -- that's 25.B3.
- No strategy state-machine (status flips beyond initial `pending`) -- that's 25.C3.
- No live execution of the migration script against BQ in this cycle (would require operator approval per CLAUDE.md MCP rules); dry-run SQL is verified via the migration's argparse default.

## References

- `handoff/current/research_brief.md` -- full brief this cycle
- `scripts/migrations/create_options_snapshots_table.py` -- canonical migration skeleton to mirror
- `backend/db/bigquery_client.py:700-740` (`save_paper_snapshot`) -- canonical MERGE shape to mirror
- `backend/autoresearch/friday_promotion.py:30-148` -- wire point + signature edit site (line ~140 after ledger append)
- `backend/services/slot_accounting.py:26` -- precedent for hardcoded `pyfinagent_data` dataset
- CLAUDE.md `Critical Rules` -- 30s BQ timeout; idempotent migrations; live BQ migrations require operator approval
