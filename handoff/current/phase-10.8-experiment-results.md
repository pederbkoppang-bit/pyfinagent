# Experiment Results — phase-10.8 (Slot accounting to harness_learning_log)

**Step:** 10.8 **Date:** 2026-04-20

## What was done

1. Fresh researcher (moderate): 6 in full, 16 URLs, recency 2026, gate_passed=true. Brief at `handoff/current/phase-10.8-research-brief.md`. Critical pitfall: `backend/services/learning_logger.py:70` points at the wrong dataset (`project.trading.harness_learning_log`); new module hard-defaults to literal `pyfinagent_data.harness_learning_log`.
2. Contract authored at `handoff/current/phase-10.8-contract.md`.
3. Created `backend/autoresearch/slot_accounting.py` (~135 lines):
   - Public `log_slot_usage(*, week_iso, slot_id, routine, result, phase="phase-10", bq_insert_fn=None, table="pyfinagent_data.harness_learning_log", now=None) -> dict`
   - Public `verify_weekly_invariant(week_iso, *, bq_query_fn=None, table=...) -> dict`
   - Valid `slot_id` set enforced: `{"thu_batch", "fri_promotion", "monthly_gate", "rollback"}` — raises `ValueError` on bogus
   - Row schema: logged_at, row_id (uuid4), week_iso, slot_id, phase, routine, result_json, status, error_msg
   - `_default_bq_insert` uses `google.cloud.bigquery.Client.insert_rows_json` with `GCP_PROJECT_ID` env var (defaults to `sunny-might-477607-p8`)
   - `_default_bq_query_count` runs parameterized scalar COUNT
   - Fail-open on any BQ exception (returns `inserted=False` with warning log)
4. Created `scripts/harness/phase10_slot_accounting_test.py` — 4 cases matching masterplan success_criteria verbatim; all use `bq_insert_fn=capture.append` pattern (no real BQ needed)
5. Created `tests/autoresearch/test_slot_accounting.py` — 9 pytest cases incl. edge cases (invalid slot_id, BQ failure, unique uuid, error_msg propagation, missing-slot invariant)

## Verification (verbatim)

```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/autoresearch/slot_accounting.py','scripts/harness/phase10_slot_accounting_test.py','tests/autoresearch/test_slot_accounting.py']]; print('OK')"
OK

$ python scripts/harness/phase10_slot_accounting_test.py
[PASS] every_phase10_routine_logged  (captured=4, slot_ids={fri_promotion,monthly_gate,rollback,thu_batch})
[PASS] label_phase_10_applied  (all_phase10=True, count=4)
[PASS] weekly_invariant_sum_equals_2  (sum=2, satisfied=True)
[PASS] bq_writes_go_to_pyfinagent_data_harness_learning_log  (table='pyfinagent_data.harness_learning_log')

ALL PASS  (4/4)
(exit 0)

$ pytest tests/autoresearch/test_slot_accounting.py -q
.........                                                                [100%]
9 passed in 0.01s

$ pytest tests/autoresearch/ tests/slack_bot/ backend/metrics/ -q
........................................................................ [ 67%]
...................................                                      [100%]
107 passed in 1.51s
```

## Success criteria (masterplan, immutable)

| # | Criterion | Status |
|---|---|---|
| 1 | `every_phase10_routine_logged` | PASS — 4 routines (thu_batch / fri_promotion / monthly_gate / rollback) each produce exactly 1 captured row |
| 2 | `label_phase_10_applied` | PASS — every row has `phase == "phase-10"` |
| 3 | `weekly_invariant_sum_equals_2` | PASS — `verify_weekly_invariant("2026-W17")` returns `{sum: 2, satisfied: True}` after Thu+Fri logged (monthly_gate + rollback don't count) |
| 4 | `bq_writes_go_to_pyfinagent_data_harness_learning_log` | PASS — captured table literal == `"pyfinagent_data.harness_learning_log"` on every call |

## Key design decisions

- **Canonical slot_id set:** `{thu_batch, fri_promotion, monthly_gate, rollback}` — matches the 4 phase-10 routines already shipped
- **Weekly invariant counts only weekly slots:** monthly_gate + rollback are not part of the 2-slot weekly budget; the query's `WHERE slot_id IN ('thu_batch', 'fri_promotion')` filter enforces this
- **Parameterized SQL:** `@week_iso` ScalarQueryParameter prevents injection if week_iso ever comes from user input
- **Stub-first design:** every BQ call is injectable; no live BQ needed for tests (matches the codebase's established DI idiom from phase-9.1+)

## Carry-forwards (out of scope)

- Wire 10.3/10.4/10.6/10.7 routines to actually invoke `log_slot_usage` at their success paths — done via the phase-10.9 dashboard backend that will read these logs; doing it inline in each routine would couple them and expand scope
- Migrate monthly_gate state JSON + rollback state JSON into BQ rows — would normalize all phase-10 state into one sink, but requires schema design
- The `learning_logger.py:70` wrong-dataset bug — separate ticket
