# phase-38.2 -- Lost cycle 3a observability (OPEN-11)

**Step id:** `38.2`
**Date:** 2026-05-23
**Mode:** EXECUTION (cycle 45).
**Cycle:** Cycle 45 (after Cycle 44 phase-38.6.1 wiring).

---

## North-star delta

**Terms:** R (operational integrity -- close observability blind spot) + B (no $ cost; saved diagnostic time on future halted cycles).

**R:** Orphan cycle 3a (08:14 CEST 2026-05-21) wrote NO history row and is invisible to the harness. After this fix every cycle leaves a `status="started"` row before run_daily_cycle does any work; if the cycle dies, the orphan persists for the next cycle's audit. Closes the "lost cycle 3a" portion of OPEN-11.

**B:** Mechanical -- reduces "where did the 08:14 cycle go?" diagnostic time from ~30 min to ~30 sec (grep cycle_id without a matching terminal row).

**P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** new pytest module `test_phase_38_2_cycle_start_logging.py` exercises 3 immutable criteria; existing tests still pass; cycle_history.jsonl produced by a real run shows both a started row and a terminal row joined by cycle_id.

---

## Research-gate compliance

**Researcher SPAWNED FIRST** -- brief at `handoff/current/research_brief_phase_38_2.md`. Tier=simple. 6 sources read in full, gate_passed=true, recency scan present. Recommended design (a): append-then-append with status="started" + status="completed/failed" rows joined by cycle_id. POSIX O_APPEND atomicity guaranteed. Existing threading.Lock retained as belt-and-braces. Follow-on caveat surfaced: `cycle_heartbeat_alarm` (cycle_health.py:182-217) and `last_cycles` reader MUST filter `status="started"` rows when picking the most recent row, else alarm mis-evaluates on halted cycles.

**Honest follow-on disclosure:** the alarm/reader filter IS in scope for this step -- it's the third criterion `next_cycle_can_audit_orphan_rows` operationalized. Without the filter, the orphan-detection logic returns the started row as "most recent" and reports a still-running cycle as still-running forever.

---

## Hypothesis

> 1. `CycleHealthLog.record_cycle_start` appends a JSONL row with `status="started"`, `completed_at=null`, `duration_ms=null` to cycle_history.jsonl (in addition to the existing heartbeat write).
> 2. Existing `record_cycle_end` continues to write the terminal row (`status="completed"`/`"failed"`) -- behavior unchanged.
> 3. `last_cycles` and `cycle_heartbeat_alarm` filter out `status="started"` rows when picking the "most recent completed cycle" (a separate orphan-detection accessor surfaces them for audit).
> 4. New `orphan_rows()` accessor returns started rows whose cycle_id has no matching terminal row -- next cycle can audit them.

---

## Immutable success criteria (verbatim from masterplan 38.2.verification)

1. `record_cycle_start_writes_cycle_starting_row_immediately` -- a fresh `record_cycle_start("X")` call must produce a JSONL line containing `"cycle_id": "X"` and `"status": "started"` in cycle_history.jsonl.
2. `row_persists_if_cycle_dies_mid_flight` -- if `record_cycle_end` is never called, the started row remains. (Test simulates by calling start without end.)
3. `next_cycle_can_audit_orphan_rows` -- new accessor (`orphan_rows()`) returns started rows whose `cycle_id` has no matching terminal row.

Plus /goal integration gates 1-10.

---

## Files this step touches

- `backend/services/cycle_health.py` -- ~30 lines: extend `record_cycle_start` to write a started row; add `orphan_rows()` accessor; filter started rows from `last_cycles` (or surface them via a separate method); add per-row `status="started"` constant.
- `backend/tests/test_phase_38_2_cycle_start_logging.py` (NEW, ~120 lines, >=3 tests covering each immutable criterion + orphan-audit happy path + mutation-resistance for the filter).

**NOT changed:** `backend/services/autonomous_loop.py` (call sites already invoke record_cycle_start at the right line; signature unchanged).

---

## /goal integration gates (declared)

| # | Gate | Plan |
|---|---|---|
| 1 | pytest count >= 297 | will INCREASE by ~3-5 tests; baseline 488 -> ~491+; 0 regressions |
| 2 | ast.parse green | will hold |
| 3 | TS build green | N/A (no FE change) |
| 4 | flag-default-OFF | N/A (no new flag) |
| 5 | BQ idempotent / no new mutating queries | N/A (no BQ touched; only handoff/cycle_history.jsonl) |
| 6 | env vars docs | N/A |
| 7 | N* delta declared | DONE (R + B above) |
| 8 | zero emojis | will hold |
| 9 | ASCII-only loggers | will hold |
| 10 | single source of truth | cycle_history.jsonl remains the SoT; alarm + reader filter on a single canonical `status` field |
| 11 | log-first / flip-last | will hold |

---

## References

- closure_roadmap.md §3 OPEN-11
- handoff/current/research_brief_phase_38_2.md (cycle 45; design (a) approved + POSIX O_APPEND citation)
- backend/services/cycle_health.py:259-262, :264-297, :182-217, :299-318
- backend/services/autonomous_loop.py:199, :1146
- /goal directive
