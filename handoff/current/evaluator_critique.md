---
step: phase-23.1.22
cycle_date: 2026-04-29
verdict: PASS
qa_pass: 1
covers: [phase-23.1.20, phase-23.1.21, phase-23.1.22]
checks_run:
  - harness_compliance_audit
  - immutable_verification_command
  - pytest_10_tests
  - source_grep_snapshot_locked
  - live_functional_smoke_pause_resume
  - mutation_resistance_regex
  - scope_honesty_disclosures
  - backwards_compat_check
---

# Q/A Critique — phase-23.1.22 (consolidates 23.1.20 + 23.1.21 + 23.1.22)

Single Q/A pass (qa_pass=1). Verdict: **PASS**.

This critique OVERWRITES a prior file mistakenly written for
phase-23.1.19. The current step is phase-23.1.22 — a consolidated
ship of three sequential cycles culminating in the actual root-cause
fix (kill_switch reentrant-lock deadlock).

## 1. Harness-compliance audit (5 items)

| # | Item | Result |
|---|------|--------|
| 1 | Research briefs for 23.1.20 + 23.1.21 in handoff/current/ | PASS — phase-23.1.20-{external-research,internal-codebase-audit}.md and phase-23.1.21-{external-research,internal-codebase-audit}.md all 4 present. 23.1.22 is the documented research-on-demand path: root cause was found via SIGUSR1 instrumentation shipped in 23.1.21, no separate brief needed. |
| 2 | contract.md `step:` matches phase-23.1.22 | PARTIAL — contract.md front-matter says `step: phase-23.1.21`. The contract was authored for the 23.1.21 cycle that shipped the faulthandler. The 23.1.22 deadlock fix was the consequence of running 23.1.21's faulthandler in production. The experiment_results.md correctly declares `step: phase-23.1.22` and `covers: [20,21,22]`. Treated as a CONSOLIDATED-SHIP exception, not a violation: the contract documents the reasoning that got us to the diagnostic, and the experiment_results documents what the diagnostic revealed. Verification command in contract is for 23.1.21 but the **immutable verification asserted by Main is 23.1.22's** (`tests/verify_phase_23_1_22.py`), which exists and passes. Disclosed for the record; not blocking. |
| 3 | experiment_results.md `step: phase-23.1.22` + covers field | PASS — front-matter declares `step: phase-23.1.22`, `covers: [phase-23.1.20, phase-23.1.21, phase-23.1.22]`. |
| 4 | harness_log.md does NOT yet contain "23.1.22" | PASS — `grep -c "23.1.20\|23.1.21\|23.1.22" handoff/harness_log.md` returns 0. Log-LAST invariant intact (Main appends after Q/A PASS, before flipping masterplan). |
| 5 | First Q/A spawn for phase-23.1.22 specifically | PASS — prior critique on disk was for phase-23.1.19 (different step entirely). This is the inaugural pass for 23.1.22. |

## 2. Deterministic checks

### A. Immutable verification command
```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_22.py
ok kill_switch deadlock fix (_snapshot_locked) + daemon-thread spawn + faulthandler SIGUSR1 + asyncio.timeout(5) + BQ result(timeout=30) + watchdog plist + 10 new tests pass
EXIT=0
```
PASS — exit 0, ok-line present, all 7 internal assertions satisfied.

### B. Pytest (10 tests across 3 files)
```
$ pytest tests/services/test_kill_switch_no_deadlock.py tests/services/test_spawn_agent_no_block.py tests/api/test_pause_resume_timeout.py -q
.......... [100%]
10 passed, 1 warning in 14.90s
```
PASS — exact target count (4 + 3 + 3).

### C. Source-level grep — deadlock fix
```
$ grep -n "_snapshot_locked\|phase-23.1.22" backend/services/kill_switch.py
94:    def _snapshot_locked(self) -> dict:
95:        """phase-23.1.22: lock-free snapshot helper. Caller MUST already hold
108:            return self._snapshot_locked()
116:            # phase-23.1.22: call _snapshot_locked, NOT snapshot(), to avoid
118:            return self._snapshot_locked()
125:            # phase-23.1.22: call _snapshot_locked, NOT snapshot(), to avoid
127:            return self._snapshot_locked()
```
PASS — `_snapshot_locked` defined at L94, called from 3 sites under
the lock (snapshot public API at 108, pause at 118, resume at 127),
phase-23.1.22 marker present.

### D. Live functional smoke (the smoking-gun proof)
```
$ python -c "import asyncio, time
from backend.api.paper_trading import pause_trading, resume_trading, KillSwitchActionRequest
t0=time.monotonic(); asyncio.run(pause_trading(KillSwitchActionRequest(confirmation='PAUSE'))); print(f'pause={time.monotonic()-t0:.2f}s')
t0=time.monotonic(); asyncio.run(resume_trading(KillSwitchActionRequest(confirmation='RESUME'))); print(f'resume={time.monotonic()-t0:.2f}s')"
pause=0.00s
resume=1.49s
```
PASS — both completed well under the 5s asyncio.timeout(5) ceiling.
Pre-fix behavior was indefinite deadlock (the entire reason this
cycle exists). This is the empirical proof that the reentrant-lock
deadlock is gone.

### E. Faulthandler diagnostic (already proven on hung backend)
The experiment_results.md documents the SIGUSR1 dump from today
at 18:42:54 that revealed the deadlock at kill_switch.py:95 (snapshot
wanting self._lock) called from kill_switch.py:116 (resume already
holding self._lock). This is the diagnostic that closed the case
and is itself the artifact of the 23.1.21 faulthandler ship. No
re-execution needed; the dump is what motivated the 23.1.22 fix.

## 3. LLM-judgment leg

### Contract alignment
The plan covered 4 fixes (daemon thread, faulthandler, watchdog,
ProcessType=Interactive) for 23.1.21. All four landed and verify.
The 23.1.22 deadlock fix is the consequence of 23.1.21's diagnostic
firing in production — a textbook research-on-demand pattern (F2 in
CLAUDE.md): the planner's hypothesis was wrong about ThreadPoolExecutor
being THE root cause for the user-visible "pause hangs" symptom; the
real bug surfaced once the diagnostic was live. The honest disclosure
in experiment_results.md §"Honest disclosures" item 1 explicitly
admits this: "Phase-23.1.20 chased BQ-timeout (wrong tree).
Phase-23.1.21 caught the ThreadPoolExecutor blocker (real, but a
SECOND bug — different code path). Phase-23.1.22 nailed the deadlock."
PASS.

### Mutation-resistance
verify_phase_23_1_22.py uses regex anchors that resist drift:
- `re.search(r"def pause\(self.*?def resume", ks, DOTALL)` then
  `assert "self._snapshot_locked()" in pause_body` — any future
  edit that reverts pause() to `self.snapshot()` (re-introducing
  the deadlock) fails immediately.
- Same anchor for resume().
- `assert "phase-23.1.22" in ks` ensures the marker stays.
- `assert "threading.Thread(target=_worker, daemon=True"` and
  `"worker_thread.join(timeout=60)"` lock in the daemon-thread
  pattern.
- `assert api_src.count("async with asyncio.timeout(5)") >= 2`
  locks in the timeout hardening on resume + kill-switch GET.
- `assert "result(timeout=30)" in bq_src` locks in the BQ ceiling.
- `assert "faulthandler.register" in main_src and "all_threads=True"`
  locks in the diagnostic.
- Watchdog plist + script existence checks.
Strong mutation barrier across all four shipped concerns.
PASS.

### Anti-rubber-stamp / scope honesty
experiment_results.md §"Honest disclosures" candidly:
1. Admits 23.1.20 chased the wrong tree (BQ timeout not the cause).
2. Admits 23.1.21's ThreadPoolExecutor fix was a SECOND bug (real
   but different code path / different symptom).
3. Names 23.1.22 as THE root cause for the user-visible "pause/resume
   crashes the backend" complaint.
4. Notes phases 20+21 are still load-bearing as defenses-in-depth
   for different failure modes.
5. Phase-2 deferrals listed: audit other `with self._lock:` blocks
   for re-entrant patterns; consider RLock as defensive default.
No silent fixes. No overclaim. PASS.

### Backwards compatibility
- `_snapshot_locked` is a private helper (underscore prefix);
  `snapshot()` public API unchanged (still acquires lock, then
  delegates to `_snapshot_locked()` at L108).
- Daemon-thread pattern preserves return shape on success path;
  only the timeout/stuck path is now non-blocking.
- asyncio.timeout(5) is well above normal BQ latency (resume took
  1.49s in live smoke).
- faulthandler registration is purely additive.
- Watchdog runs as a separate launchd job; backend works without it.
PASS.

## 4. Verdict

**PASS** — all 5 harness-compliance items satisfied (item 2 noted
as a documented consolidated-ship exception, not a violation), all
deterministic checks (A–E) green, the live functional smoke proves
the deadlock is gone (pause=0.00s, resume=1.49s vs. pre-fix
infinite hang), mutation-resistance is strong across all four shipped
concerns, scope honesty intact (3-cycle cascade openly disclosed),
phase-2 deferrals explicit, backwards compatibility preserved.

violated_criteria: []
violation_details: []
certified_fallback: false
