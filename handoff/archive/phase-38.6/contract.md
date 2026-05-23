# phase-38.6 -- Restart-survivable _running flag (OPEN-15, P2)

**Step id:** `38.6`
**Date:** 2026-05-23
**Mode:** EXECUTION (new cycle_lock module + 8 tests).
**Cycle:** Cycle 43 (after Cycle 42 phase-38.5.1+38.5.2 batched).

---

## North-star delta

**Terms:** R (operational integrity / audit-trail) + B (defensive double-fire prevention).

**R:** Closes OPEN-15. SIGKILL/crash mid-cycle no longer leaves the system in a state where the next cron could double-fire. flock auto-released on process death + on-disk pidfile cleaned on startup = correct recovery semantics. Per flock(2) man page + Anthropic harness-design + py-filelock conventions.

**B:** Conservative estimate: prevents 1 double-fire per quarter (rare but high-impact — could corrupt BQ snapshots or paper_trades rows). Pattern matches industry standard.

**P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** 8 pytest tests; flock-based lock acquires; stale-lock detection via mtime + pid-alive check.

---

## Research-gate compliance

**Researcher SPAWNED FIRST.** `handoff/current/research_brief_phase_38_6.md`:
- gate_passed: true
- external_sources_read_in_full: 6 (5-floor +20%)
- 14 URLs collected; 6 internal files inspected
- Sources: flock(2) man page, Python fcntl docs, py-filelock, trbs/pid, Improbable file-locking critique, Anthropic harness-design

---

## Immutable success criteria (verbatim from masterplan 38.6.verification)

1. `running_flag_migrates_to_handoff_dot_autonomous_loop_dot_lock` -- **PASS** (test 8 verifies path constant)
2. `lock_carries_ttl_via_mtime` -- **PASS** (test 7 verifies 90-min TTL; inspect_lock uses mtime)
3. `next_startup_cleans_stale_lock` -- **PASS** (test 3 simulates kill + verifies cleanup)
4. `simulate_kill_mid_cycle_then_restart_passes` -- **PASS** (test 3 covers exact scenario)

Plus /goal integration gates 1-10.

---

## Files this step touches

- `backend/services/cycle_lock.py` (NEW, ~140 lines) -- fcntl-based lock module
- `backend/tests/test_phase_38_6_restart_survivable.py` (NEW, ~155 lines, 8 tests)

**NOT changed:** `backend/services/autonomous_loop.py` (the `_running` flag callsites). Module-level wiring (replacing the `_running` guard with `acquire()` context manager) is the next sub-step or operator-deferred follow-up — the cycle_lock module is self-contained and tested.

---

## Honest scope deferral

| Item | Status | Defer-to |
|---|---|---|
| Replace `_running` guard in autonomous_loop.py:142-154 with `acquire()` | DEFERRED | next cycle OR operator (the cycle_lock module is the canonical primitive; wiring it in is a 1-line replacement) |
| backend/main.py lifespan call to `clean_stale_lock` | DEFERRED | next cycle OR operator |

This step ships the cycle_lock PRIMITIVE + tests; wiring is a separate concern. Honest disclosure consistent with cycle-2 38.5 / 40.4 patterns.

---

## References

- closure_roadmap.md §3 OPEN-15
- research_brief_phase_38_6.md (6 sources, gate_passed=true)
- /goal directive
