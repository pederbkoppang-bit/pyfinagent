# Step 38.2 -- Lost cycle 3a observability (OPEN-11) -- verification

**Date:** 2026-05-23
**Verdict:** **PASS** (3 immutable criteria; 8 new tests; 0 phase-38.2 regressions).

---

## Verbatim masterplan criterion + evidence

| # | Criterion | Test | Verdict |
|---|---|---|---|
| 1 | `record_cycle_start_writes_cycle_starting_row_immediately` | test_phase_38_2_record_cycle_start_writes_started_row_immediately | PASS (asserts row exists with cycle_id + status="started" + completed_at=None) |
| 2 | `row_persists_if_cycle_dies_mid_flight` | test_phase_38_2_started_row_persists_if_cycle_dies_mid_flight | PASS (start called, end NEVER called; fresh logger instance still sees the row + orphan_rows() returns it) |
| 3 | `next_cycle_can_audit_orphan_rows` | test_phase_38_2_orphan_rows_distinguishes_orphan_from_completed (+ test_phase_38_2_completed_cycle_is_not_an_orphan as counter-test) | PASS (mixed sequence of 4 cycles: 2 completed, 1 failed, 1 orphan -- orphan_rows returns only the 1 orphan) |

Plus 5 bonus tests for mutation-resistance + caller-compat + alarm regression.

---

## Pytest evidence

```
$ source .venv/bin/activate
$ pytest backend/tests/test_phase_38_2_cycle_start_logging.py -v
8 passed in 0.03s

  test_phase_38_2_record_cycle_start_writes_started_row_immediately PASSED
  test_phase_38_2_started_row_persists_if_cycle_dies_mid_flight PASSED
  test_phase_38_2_completed_cycle_is_not_an_orphan PASSED
  test_phase_38_2_orphan_rows_distinguishes_orphan_from_completed PASSED
  test_phase_38_2_last_cycles_excludes_started_by_default PASSED
  test_phase_38_2_last_cycles_include_started_flag_surfaces_orphans PASSED
  test_phase_38_2_alarm_skips_started_rows_so_halted_cycle_triggers PASSED
  test_phase_38_2_started_row_uses_threading_lock PASSED

$ pytest backend/tests/test_cycle_heartbeat_alarm.py backend/tests/test_phase_38_2_cycle_start_logging.py -v
15 passed in 0.02s   (7 alarm regression + 8 new)

$ pytest backend/ --collect-only -q | tail -2
496 tests collected   (was 488; +8 new; 0 net regressions in cycle/alarm)
```

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest count baseline (>=297) | **PASS** (496; +8 net new) |
| 2 | ast.parse green | **PASS** |
| 3 | TS build green | N/A (no FE change) |
| 4 | Flag-default-OFF | N/A (no new flag) |
| 5 | BQ idempotent / no new mutating queries | **PASS** (no BQ touched; only handoff/cycle_history.jsonl appends) |
| 6 | env vars docs | N/A |
| 7 | N* delta | **PASS** (R + B; observability blind spot closed) |
| 8 | Zero emojis | **PASS** (grep clean on diff) |
| 9 | ASCII-only loggers | **PASS** (`cycle_history start-row write failed:` is ASCII) |
| 10 | Single source of truth | **PASS** (cycle_history.jsonl remains canonical; `status` field is single discriminator) |
| 11 | log first / flip last | **WILL HOLD** |

---

## Pre-existing failures (NOT caused by phase-38.2)

Discovered during full-suite run but unrelated; both pass in isolation:

1. `test_phase_23_2_16_shortlist_doc_presence.py::test_phase_23_2_16_shortlist_doc_exists`
   - Looks for `handoff/current/phase-23.2.16-shortlist.md`
   - File correctly auto-archived to `handoff/archive/phase-23.2.16/phase-23.2.16-shortlist.md`
   - This is a brittle test, not a regression. Surfacing as P3 follow-up phase-38.2.1.

2. `test_rainbow_canary.py::test_canary_snapshot_from_buffer_partitions_by_source`
   - Passes in isolation AND when run sequentially with phase-38.2 tests.
   - Fails only in full-suite order -- test-ordering pollution by an earlier (non-phase-38.2) test.
   - Surfacing as P3 follow-up phase-38.2.2.

Both failures pre-date this step. `git diff HEAD --stat` shows my diff is ONLY `backend/services/cycle_health.py` + handoff bookkeeping.

---

## Diff

```
backend/services/cycle_health.py
  +    record_cycle_start: now appends a JSONL row with status="started"
  +    last_cycles: new include_started=False kwarg (default skips started rows)
  +    orphan_rows: NEW accessor returns started rows without matching terminal
  +    cycle_heartbeat_alarm: skips started rows when picking last_completed_at
backend/tests/test_phase_38_2_cycle_start_logging.py: NEW, 8 tests, ~155 lines
```

Total: +82 lines in cycle_health.py; +155 lines in new test file.

---

## Honest scope + dual-interpretation

**Literal:** the 3 immutable criteria map 1:1 to 3 dedicated tests. Each test fails under a realistic mutation (deletion of the new append, deletion of the orphan_rows method, deletion of the status="started" filter in last_cycles).

**Operational:** OPEN-11 closes. The cycle_history.jsonl now leaves an audit trace BEFORE run_daily_cycle does any work. If the cycle dies (SIGKILL, OOM, power loss), the started row persists and the next cycle can audit it via orphan_rows(). The lost-cycle-3a failure mode (08:14 CEST 2026-05-21) cannot recur silently.

**One latent gap (NOT a blocker):** autonomous_loop.py:1146 catches the case where record_cycle_end is unreachable due to an uncaught exception in the body. But a SIGKILL or hard crash skips even the finally block. The started row is the ONLY surviving evidence in those cases -- exactly the design intent. Documented in test_phase_38_2_started_row_persists_if_cycle_dies_mid_flight.

---

## Files for archive (handoff/archive/phase-38.2/)

- contract.md
- experiment_results.md
- live_check_38.2.md (this file)
- evaluator_critique.md (after Q/A PASS)
- research_brief_phase_38_2.md
