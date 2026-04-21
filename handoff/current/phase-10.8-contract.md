# Sprint Contract — phase-10.8 (Slot accounting to harness_learning_log)

**Step id:** 10.8 **Date:** 2026-04-20 **Tier:** moderate **Harness-required:** true

## Why

Every phase-10 routine (10.3 Thu, 10.4 Fri, 10.6 monthly, 10.7 rollback) already persists its own state (ledger + JSON + JSONL), but the BQ observability surface is empty. This step wires one canonical BQ sink so the Harness tab (phase-10.9) can query a single table for phase-10 activity.

## Research-gate summary

Fresh researcher (moderate): `handoff/current/phase-10.8-research-brief.md` — 6 in full, 16 URLs, three-variant, recency, gate_passed=true.

Key findings:
- BQ write via `insert_rows_json` (matches `bigquery_client.py:251` existing idiom); Storage Write API is overkill for 2-4 rows/week
- **Critical pitfall found:** `backend/services/learning_logger.py:70` uses `project.trading.harness_learning_log` — wrong dataset. New module must hard-default to literal `pyfinagent_data.harness_learning_log`
- Idempotency handled upstream per-routine via `already_fired` guards — no insertId dedup needed here
- Weekly invariant: `thu_batch + fri_promotion == 2` per `week_iso` (monthly_gate and rollback don't count against the weekly slot budget)

## Immutable success criteria (masterplan-verbatim)

Test command: `python scripts/harness/phase10_slot_accounting_test.py`

1. `every_phase10_routine_logged` — calling `log_slot_usage` for each of 10.3/10.4/10.6/10.7 produces exactly one row per call, capturing slot_id, routine, result
2. `label_phase_10_applied` — every row has `phase == "phase-10"`
3. `weekly_invariant_sum_equals_2` — after logging Thursday and Friday for the same `week_iso`, `verify_weekly_invariant` returns `{sum: 2, satisfied: true}`
4. `bq_writes_go_to_pyfinagent_data_harness_learning_log` — default `table` argument is literally `"pyfinagent_data.harness_learning_log"`; the test asserts the BQ insert was called with that exact table name

## Plan

1. Create `backend/autoresearch/slot_accounting.py`:
   - Public `log_slot_usage(*, week_iso, slot_id, routine, result, phase="phase-10", bq_insert_fn=None, table="pyfinagent_data.harness_learning_log", now=None) -> dict` returning `{inserted, row_id, table, row}`
   - Row schema: `{logged_at, row_id (uuid), week_iso, slot_id, phase, routine, result_json (str), status, error_msg}`
   - Default `bq_insert_fn` uses `google.cloud.bigquery.Client().insert_rows_json(table, rows)`; fail-open on exception (returns `{inserted: False, ...}` with error logged)
   - Public `verify_weekly_invariant(week_iso, *, bq_query_fn=None, table=...) -> dict` returning `{week_iso, sum, satisfied}`; default query fn runs `SELECT COUNT(*) FROM {table} WHERE week_iso = @week_iso AND phase = 'phase-10' AND slot_id IN ('thu_batch', 'fri_promotion')`
   - Valid `slot_id` set: `{"thu_batch", "fri_promotion", "monthly_gate", "rollback"}` — raise `ValueError` on unknown (defensive)
   - ASCII-only logs
2. Create `scripts/harness/phase10_slot_accounting_test.py` — 4 cases mapping to masterplan success_criteria verbatim; inject `bq_insert_fn=capture_list.append`
3. Create `tests/autoresearch/test_slot_accounting.py` with ≥6 pytest cases
4. Verify: ast + immutable CLI + pytest new + neighbors
5. Spawn fresh Q/A. Cycle-2 flow if gaps surfaced.
6. Log, flip, close task.

## References

- `handoff/current/phase-10.8-research-brief.md`
- `backend/db/bigquery_client.py:251` (insert_rows_json idiom)
- `backend/services/learning_logger.py:70` (wrong-dataset pitfall — avoid)
- 10.3-10.7 return dicts (what we log)

## Carry-forwards

- Wire 10.3/10.4/10.6/10.7 callers to actually invoke `log_slot_usage` (done in this step via the test's assertion but the CLI scripts for those steps don't call it yet — wiring is a downstream concern for phase-10.9 backend dashboard)
- Migrate monthly_gate state + rollback state from local JSON to BQ rows — out of scope
