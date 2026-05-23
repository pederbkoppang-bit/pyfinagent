# phase-38.2 -- experiment results (Cycle 45)

**Date:** 2026-05-23
**Cycle:** 45
**Step:** phase-38.2 -- Lost cycle 3a observability (OPEN-11)
**Verdict:** PASS (deterministic; 8/8 new tests + 7/7 alarm regression)

---

## What changed

| File | Change | Lines |
|---|---|---|
| `backend/services/cycle_health.py` | `record_cycle_start` now appends a JSONL row with `status="started"`; `last_cycles` gains `include_started=False` default; new `orphan_rows()` method; `cycle_heartbeat_alarm` skips started rows when picking last_completed_at. | +82 |
| `backend/tests/test_phase_38_2_cycle_start_logging.py` | NEW; 8 tests covering 3 immutable criteria + 5 mutation-resistance / caller-compat / alarm regression. | +155 |

`backend/services/autonomous_loop.py` -- UNCHANGED (call sites already invoke record_cycle_start at top of run_daily_cycle).

---

## Verbatim test output

```
$ source .venv/bin/activate
$ pytest backend/tests/test_phase_38_2_cycle_start_logging.py -v
============================== 8 passed in 0.03s ==============================

  test_phase_38_2_record_cycle_start_writes_started_row_immediately PASSED
  test_phase_38_2_started_row_persists_if_cycle_dies_mid_flight PASSED
  test_phase_38_2_completed_cycle_is_not_an_orphan PASSED
  test_phase_38_2_orphan_rows_distinguishes_orphan_from_completed PASSED
  test_phase_38_2_last_cycles_excludes_started_by_default PASSED
  test_phase_38_2_last_cycles_include_started_flag_surfaces_orphans PASSED
  test_phase_38_2_alarm_skips_started_rows_so_halted_cycle_triggers PASSED
  test_phase_38_2_started_row_uses_threading_lock PASSED

$ pytest backend/tests/test_cycle_heartbeat_alarm.py backend/tests/test_phase_38_2_cycle_start_logging.py -v
============================== 15 passed in 0.02s =============================

$ pytest backend/ --collect-only -q | tail -2
496 tests collected   (was 488; +8 net; 0 phase-38.2 regressions)
```

---

## Immutable success criteria

1. `record_cycle_start_writes_cycle_starting_row_immediately` -- PASS (test 1; fresh JSONL line with cycle_id + status="started")
2. `row_persists_if_cycle_dies_mid_flight` -- PASS (test 2; start called, end NEVER called; row survives + orphan_rows() exposes it)
3. `next_cycle_can_audit_orphan_rows` -- PASS (test 4; mixed sequence with 2 completed + 1 failed + 2 orphans; orphan_rows() returns exactly the 2 orphans)

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest count baseline (>=297) | **PASS** (496; +8 net new) |
| 2 | ast.parse green | **PASS** |
| 3 | TS build | N/A |
| 4 | Flag-default-OFF | N/A |
| 5 | BQ idempotent | **PASS** (no BQ; only handoff/cycle_history.jsonl appends) |
| 6 | env vars docs | N/A |
| 7 | N* delta declared | **PASS** (R + B) |
| 8 | Zero emojis | **PASS** |
| 9 | ASCII-only loggers | **PASS** |
| 10 | Single source of truth | **PASS** (cycle_history.jsonl remains SoT; status field is single discriminator) |
| 11 | log-first / flip-last | **WILL HOLD** |

---

## Honest scope + dual-interpretation

**Literal:** 3 dedicated tests for 3 immutable criteria; each fails under realistic mutation.

**Operational:** OPEN-11 closes. cycle_history.jsonl now writes a started row BEFORE run_daily_cycle does any work. SIGKILL / OOM / power loss -> next cycle can detect the orphan via `orphan_rows()`. Lost cycle 3a (08:14 CEST 2026-05-21) failure mode cannot recur silently.

**Backward compat:** `last_cycles` adds `include_started=False` default kwarg -- existing callers (paper_trading.py:423) see no behavior change (no started rows existed before, and the default still skips any newly-written ones). UI / alarm continue to see only terminal rows.

**Deferred follow-ups (NOT blockers; surfacing as P3 tickets):**
- phase-38.2.1: brittle test `test_phase_23_2_16_shortlist_doc_exists` looks at handoff/current/ but file was correctly auto-archived. Fix: update path to archive/.
- phase-38.2.2: test-order flake `test_canary_snapshot_from_buffer_partitions_by_source` passes in isolation; fails in full-suite order. Some earlier test pollutes state. Fix: add isolation fixture.

Both failures pre-date this step (`git diff HEAD --stat` shows my diff is ONLY `backend/services/cycle_health.py` + handoff bookkeeping; no test_rainbow_canary or test_phase_23_2_16 changes).

---

## Research-gate

Researcher SPAWNED FIRST this cycle (no SKIP -- lesson from cycle 44 honored). Brief at `handoff/current/research_brief_phase_38_2.md`. Tier=simple, 6 sources read in full, gate_passed=true, recency scan present. Recommended design (a) implemented verbatim: append-then-append with status discriminator; POSIX O_APPEND atomicity + threading.Lock retained.

---

## Files for archive (handoff/archive/phase-38.2/)

- contract.md
- experiment_results.md (this file)
- live_check_38.2.md
- evaluator_critique.md (after Q/A PASS)
- research_brief_phase_38_2.md
