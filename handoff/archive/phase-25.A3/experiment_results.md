---
step: phase-25.A3
cycle: 70
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_A3.py'
title: Write promoted strategies to pyfinagent_data.promoted_strategies BQ table (P1)
audit_basis: phase-24.3 F-3 (friday_promotion.py wrote only to TSV ledger; no BQ subscriber)
---

# Experiment Results -- phase-25.A3

## Code changes

### `scripts/migrations/create_promoted_strategies_table.py` (new file)
- Mirrors the canonical migration skeleton from `create_options_snapshots_table.py` exactly: argparse + `CREATE_SQL` + `main()` + `--apply` flag.
- `PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")`, `DATASET = "pyfinagent_data"`, `TABLE = "promoted_strategies"`.
- CREATE TABLE IF NOT EXISTS with 10 columns + PARTITION BY DATE(promoted_at) + CLUSTER BY strategy_id, week_iso.
- Default dry-run prints SQL; `--apply` executes against BQ via the user's ADC.

### `backend/db/bigquery_client.py`
- New method `save_promoted_strategy(self, row: dict) -> None`:
  - Validates required MERGE-key fields (`strategy_id`, `week_iso`).
  - Target FQN hardcoded: `f"{self.settings.gcp_project_id}.pyfinagent_data.promoted_strategies"` (precedent: `slot_accounting.py:26`).
  - Field type registry maps each column to (BQ type, Python coercion).
  - `params` column is handled with `PARSE_JSON(@v_params)` in both SOURCE SELECT and INSERT VALUES because there's no native JSON BQ parameter type.
  - MERGE on `(week_iso, strategy_id)` -- natural-key idempotency; re-running a Friday promotion overwrites the row instead of duplicating.
  - `result(timeout=30)` per CLAUDE.md 30s BQ timeout rule.

### `backend/autoresearch/friday_promotion.py`
- New imports: `json`, `from datetime import datetime, timezone`.
- `run_friday_promotion` signature gains `bq_client: Any | None = None` kwarg (default None preserves all existing callers).
- After the ledger-append `if not ok:` block, when `bq_client is not None`:
  - Loops over `top` (the actually-promoted candidates).
  - Per-row try/except: a BQ write failure logs a warning but does NOT poison sibling rows or block the function return (the TSV ledger already holds the canonical record at this point).
  - Builds the BQ row dict: `strategy_id, week_iso, params (JSON string), dsr, pbo, status="pending", allocation_pct, promoted_at (UTC ISO), sortino_monthly`.

### `tests/verify_phase_25_A3.py` (new file)
- 10 immutable claims:
  - Claims 1-5: migration structure (table FQN, all 10 columns, idempotent CREATE, params JSON, partition + cluster).
  - Claims 6-7: BQ client method signature + MERGE key + 30s timeout.
  - Claim 8: friday_promotion signature has `bq_client` kwarg.
  - Claim 9: **Behavioral round-trip** -- pass fake bq_client + fake ledger + always-promoting fake gate. Assert `save_promoted_strategy` called exactly once with the expected shape (strategy_id, week_iso, status="pending", params JSON-round-trips, etc.).
  - Claim 10: with `bq_client=None`, the function still succeeds AND the ledger TSV write still happens -- no regression.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_A3.py
PASS: bigquery_promoted_strategies_table_exists
PASS: schema_includes_strategy_id_params_json_dsr_pbo_status
PASS: migration_is_idempotent_create_table_if_not_exists
PASS: params_column_is_json_type
PASS: migration_partition_and_cluster_correct
PASS: bigquery_client_save_promoted_strategy_method_exists
PASS: save_promoted_strategy_merge_with_timeout_30
PASS: run_friday_promotion_accepts_bq_client_kwarg_default_none
PASS: friday_promotion_writes_row_per_promotion
PASS: bq_client_none_preserves_existing_ledger_write_no_regression

10/10 claims PASS, 0 FAIL
```

## Backend gates

- `python -c "import ast; ast.parse(open('backend/autoresearch/friday_promotion.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/db/bigquery_client.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('scripts/migrations/create_promoted_strategies_table.py').read())"` -- OK
- Migration dry-run executes and prints the full 10-column DDL (no `--apply` flag -> no BQ write).
- Behavioral round-trip (claim 9): the fake bq_client's `save_promoted_strategy` receives the dict with the expected keys + JSON-round-trippable params. End-to-end shape verified without touching real BQ.

## Hypothesis verdict

CONFIRMED. Three immutable success criteria mapped to verifier claims (1 covers criterion 1, 2 covers criterion 3, 9 covers criterion 2). Behavioral round-trip exercises the actual function with a fake bq_client, so a mutation that breaks the wire (e.g., not calling save_promoted_strategy when bq_client is non-None) would fail claim 9. Backward-compat round-trip (claim 10) ensures existing call sites are unaffected.

## Live-check

Per masterplan: "BQ promoted_strategies row visible after next Friday promotion run". Live evidence will land in `handoff/current/live_check_25.A3.md` after:
1. Operator runs the migration with `--apply` (since CLAUDE.md gates write-class BQ operations on owner approval).
2. The next Friday promotion fires (or is triggered manually via the autoresearch routine) with `bq_client=BigQueryClient(settings)` passed in.
3. A row is visible via `SELECT * FROM \`sunny-might-477607-p8.pyfinagent_data.promoted_strategies\` ORDER BY promoted_at DESC LIMIT 5`.

## Non-regressions

- All existing callers of `run_friday_promotion` (with no `bq_client` kwarg) unaffected.
- TSV `weekly_ledger` write path unchanged; the BQ write is an ADDITIONAL subscriber, not a replacement.
- Migration is idempotent; safe to re-run.

## Downstream

Unblocks 25.B3 (daily loop reads `load_promoted_params()` from the new BQ table), which unblocks 25.C3 (strategy registry status flips), which unblocks 25.R (strategy auto-switching policy).

## Next phase

Q/A pending.
