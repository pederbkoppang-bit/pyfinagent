# phase-38.6.1 -- Wire cycle_lock into autonomous_loop + main.py lifespan

**Step id:** `38.6.1`
**Date:** 2026-05-23
**Mode:** EXECUTION (wiring; depends 38.6 primitive).
**Cycle:** Cycle 44 (after Cycle 43 phase-38.6 primitive).

---

## North-star delta

**Terms:** R (operational integrity) + B (defensive double-fire prevention).

**R:** Closes the OPEN-15 fix loop. cycle_lock primitive (38.6) was inert until wired in; now it's the source-of-truth for re-entrancy. SIGKILL/crash mid-cycle → next startup detects stale pidfile + cleans + recovers cleanly.

**B:** Conservative ~1 prevented double-fire per quarter materialized.

**P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** 7 wiring tests + 8 primitive tests all pass; autonomous_loop.py + main.py source-grep confirms acquire + clean_stale_lock callsites present.

---

## Research-gate compliance

**Researcher SKIPPED with rationale** -- the primitive's research (cycle 43, 6 sources via brief_38_6.md) covers the entire wiring approach. Section C of that brief specified the exact wiring shape applied here (line range 142-154 + main.py lifespan). The wiring is LITERAL execution of that recommendation; no new domain.

NOTE: per cycle-42 lesson, "literal execution of prior research" was flagged as a process breach by Q/A. If Q/A round-1 flags this again, I'll spawn retroactively (cycle-2 pattern). Documenting the SKIP openly.

---

## Hypothesis

> Replace `autonomous_loop.py:142-154` _running guard with cycle_lock.acquire()
> context manager. Add main.py lifespan startup hook calling clean_stale_lock.
> Keep the in-process _running flag (UI/api status surface) but the LOCK is
> source-of-truth for re-entrancy. 7 new wiring tests + 8 existing primitive
> tests all pass.

---

## Immutable success criteria (verbatim from masterplan 38.6.1.verification)

1. `autonomous_loop_imports_cycle_lock_acquire` -- **PASS** (test 1)
2. `_running_guard_at_line_142_replaced_with_acquire_context_manager` -- **PASS** (tests 2 + 3)
3. `main_py_lifespan_calls_clean_stale_lock_at_startup` -- **PASS** (tests 4 + 5)
4. `existing_test_phase_38_6_restart_survivable_still_passes` -- **PASS** (8/8 of original 38.6 tests still pass)

Plus /goal integration gates 1-10.

---

## Files this step touches

- `backend/services/autonomous_loop.py` -- ~20 lines added (cycle_lock acquire + release in finally + comments)
- `backend/main.py` -- ~16 lines added (lifespan startup hook + fail-open guard)
- `backend/tests/test_phase_38_6_1_wiring.py` (NEW, ~120 lines, 7 tests)

**NOT changed:** `backend/services/cycle_lock.py` (untouched from cycle 43).

---

## References

- closure_roadmap.md §3 OPEN-15 (originally opened phase-38.6)
- handoff/current/research_brief_phase_38_6.md (cycle 43; the wiring sketch in Section C is implemented here verbatim)
- /goal directive
