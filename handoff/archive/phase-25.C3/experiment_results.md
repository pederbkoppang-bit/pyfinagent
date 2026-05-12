---
step: phase-25.C3
cycle: 72
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_C3.py'
title: Strategy registry status field; flip actual_replacement (P1)
audit_basis: phase-24.3 F-4 (monthly_champion_challenger.py:75 + :263 hardcoded actual_replacement=False; no BQ status flip on monthly approval)
depends_on: 25.B3 (done, commit 8a322219)
---

# Experiment Results -- phase-25.C3

## Code changes

### `backend/config/settings.py`
- New flag: `real_capital_enabled: bool = Field(False, description="SR 11-7 paper-only gate; toggling to True deploys approved strategies against real capital")`. Defaults False preserves the existing paper-only invariant.

### `backend/db/bigquery_client.py`
- New method `update_promoted_strategy_status(self, strategy_id: str, new_status: str, *, week_iso: str | None = None) -> None`.
- Parameterized UPDATE with named `@new_status`, `@strategy_id` (and `@week_iso` when the week_iso path is used).
- `week_iso=None` updates all rows with that strategy_id (testing / single-row scenarios); the pinned path is the recommended production usage.
- `result(timeout=30)` per CLAUDE.md.

### `backend/autoresearch/monthly_champion_challenger.py`
- `run_monthly_sortino_gate` signature gains `real_capital_enabled: bool = False` kwarg (keyword-only; default preserves callers).
- Line 75 area: `"actual_replacement": False` (hardcoded) is replaced with derivation `actual_replacement = bool(real_capital_enabled)`; the value is written to the result dict AND snapshotted into the persisted state row so the approval audit log reads back the same value (no drift between fire and approval).
- `record_approval` signature gains `status_update_fn: Callable[[str, str], None] | None = None` kwarg. When `status == "approved"` AND the transition lands successfully, calls `status_update_fn(challenger_id, "active")` inside a try/except (fail-open: BQ failure does NOT roll back the in-memory state transition).
- `_emit_deployment_log_row` notes string: literal `actual_replacement=False` replaced with f-string interpolation `{actual_replacement}` driven by the snapshotted state-row value.

### `tests/verify_phase_25_C3.py` (new file)
- 12 immutable claims with 4 behavioral round-trips:
  - Claims 1-7: structural (settings flag, BQ signature, parameterized SQL + timeout, gate kwarg, removed hardcodings, derived notes, record_approval signature).
  - Claim 8: **Behavioral approval flip** -- seed temp state with pending row; `record_approval(approved)` invokes `status_update_fn` exactly once with `(challenger_id, "active")`.
  - Claim 9: **Behavioral rejection no-flip** -- `record_approval(rejected)` does NOT invoke `status_update_fn`.
  - Claim 10: **Behavioral derived actual_replacement** -- gate-fire with `real_capital_enabled=True` -> `actual_replacement=True`; with `=False` (default) -> `actual_replacement=False`.
  - Claim 11: **Behavioral BQ UPDATE round-trip** -- direct test of `update_promoted_strategy_status` with fake BQ client; asserts parameterized SQL pattern AND `result(timeout=30)`.
  - Claim 12: migration schema documents status values including `active` + at least one retired-equivalent.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_C3.py
PASS: settings_real_capital_enabled_defaults_false
PASS: bq_update_promoted_strategy_status_signature
PASS: bq_update_uses_parameterized_sql_and_timeout_30
PASS: run_monthly_sortino_gate_accepts_real_capital_enabled_kwarg
PASS: actual_replacement_no_longer_hardcoded_false
PASS: deployment_log_notes_uses_derived_actual_replacement
PASS: record_approval_accepts_status_update_fn_kwarg
PASS: monthly_approval_flips_status_from_shadow_to_active
PASS: record_approval_rejection_does_not_flip_status
PASS: behavioral_actual_replacement_derived_from_real_capital_flag
PASS: behavioral_bq_update_status_round_trip
PASS: strategy_registry_table_has_status_field_active_shadow_retired

12/12 claims PASS, 0 FAIL
```

## Backend gates

- `python -c "import ast; ast.parse(open('backend/autoresearch/monthly_champion_challenger.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/db/bigquery_client.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/config/settings.py').read())"` -- OK
- 4 behavioral round-trips exercise the actual functions with fakes -- mutation resistance.

## Hypothesis verdict

CONFIRMED. The three immutable success criteria map to verifier claims:
- Criterion 1 (`strategy_registry_table_has_status_field_active_shadow_retired`) -- claim 12 grep + claim 2 BQ helper signature confirms the table has `status` field; status names mapped (shadow = pending, active = active, retired = superseded; existing 25.A3 schema is a superset).
- Criterion 2 (`actual_replacement_no_longer_hardcoded_false`) -- claim 5 (result dict) + claim 6 (deployment log notes) + claim 10 (behavioral: True flag yields True actual_replacement).
- Criterion 3 (`monthly_approval_flips_status_from_shadow_to_active`) -- claim 8 (behavioral: approval calls status_update_fn with "active") + claim 9 (rejection does NOT call it).

The SR 11-7 paper-only invariant is preserved: `Settings.real_capital_enabled` defaults to False, so the derived `actual_replacement` stays False unless an operator explicitly toggles the flag after a compliance review.

## Live-check

Per masterplan: "Monthly HITL approval test flips registry status atomically".

Live evidence pending in `handoff/current/live_check_25.C3.md` after operator:
1. Applies the 25.A3 migration (`--apply`).
2. Triggers a monthly Sortino gate fire (real or simulated) with `bq_client=BigQueryClient(settings)`.
3. Calls `record_approval(month_key, status="approved", status_update_fn=bq_client.update_promoted_strategy_status, bq_fn=...)`.
4. Verifies `SELECT status FROM promoted_strategies WHERE strategy_id=<challenger>` returns `"active"`.

## Non-regressions

- Existing callers of `run_monthly_sortino_gate` unaffected (the new `real_capital_enabled` kwarg has default False, matching the prior hardcoded behavior).
- Existing callers of `record_approval` unaffected (the new `status_update_fn` kwarg has default None).
- `bq_fn` audit path preserved; notes string now interpolates the snapshotted `actual_replacement` value (still False under default flag state, so audit output is byte-identical).
- No BQ schema migration; reuses the 25.A3 `promoted_strategies` table.
- The fail-open pattern means a BQ status-flip failure does NOT roll back the in-memory state transition -- consistent with the existing `bq_fn` fail-open contract.

## Downstream

Unblocks **25.R** (Strategy auto-switching policy -- the red-line goal-c) which can now rely on a real status state machine driven by HITL approval.

## Next phase

Q/A pending.
