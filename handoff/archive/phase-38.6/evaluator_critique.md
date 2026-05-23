# phase-38.6 -- Q/A evaluator critique (Cycle 43)

**Date:** 2026-05-23
**Cycle:** 43
**Step id:** 38.6 (P2 OPEN-15 -- restart-survivable autonomous-cycle lock)
**Q/A round:** 1 (first spawn this step)
**Verdict:** **PASS**

---

## 1. Harness-compliance audit (5-item, FIRST per `feedback_qa_harness_compliance_first`)

| # | Item | Status |
|---|---|---|
| (a) | Researcher spawned FIRST | PASS -- `handoff/current/research_brief_phase_38_6.md`; tier=simple; `external_sources_read_in_full=6`; gate_passed=true; 14 URLs collected; 6 internal files inspected |
| (b) | Contract pre-generate | PASS -- `handoff/current/contract.md` (Cycle 43; written before cycle_lock.py + tests) |
| (c) | Experiment results present | PASS -- `handoff/current/live_check_38.6.md` with verbatim pytest output (8 passed) + 481 collected |
| (d) | Log-last discipline | PENDING -- harness_log.md will be appended BEFORE status flip per `feedback_log_last`. Orchestrator committed in prompt. |
| (e) | No verdict-shopping | PASS -- this is the FIRST Q/A on phase-38.6 (0 prior 38.6 entries in harness_log.md). Not a cycle-2 spawn. No sycophancy-under-rebuttal risk. |

---

## 2. Deterministic checks (verbatim)

```
$ pytest backend/tests/test_phase_38_6_restart_survivable.py -v
8 passed in 0.02s
  - test_phase_38_6_acquire_writes_pid_and_cycle_id_then_unlinks PASSED
  - test_phase_38_6_second_acquire_in_same_process_raises PASSED
  - test_phase_38_6_simulated_kill_then_startup_cleans PASSED
  - test_phase_38_6_live_lock_not_cleaned PASSED
  - test_phase_38_6_malformed_lockfile_treated_as_stale PASSED
  - test_phase_38_6_no_lock_file_returns_none PASSED
  - test_phase_38_6_ttl_constant_is_90_minutes PASSED
  - test_phase_38_6_lock_path_uses_handoff_dot_autonomous_loop_dot_lock PASSED

$ pytest backend/ --collect-only -q | tail -2
481 tests collected   (was 473 after 38.5.2; +8 new; zero regressions)

$ grep '_LOCK_PATH\|_LOCK_TTL_SEC' backend/services/cycle_lock.py | head -4
40:_LOCK_PATH = _HANDOFF / ".autonomous_loop.lock"
41:_LOCK_TTL_SEC = 90 * 60  # 1.5x paper_cycle_max_seconds (1800s)

$ git diff --stat backend/services/autonomous_loop.py
(empty -- module unchanged, confirming honest scope: primitive only)

$ git diff --stat backend/agents/ backend/api/ backend/config/ backend/main.py
(empty -- no collateral edits)
```

3rd-CONDITIONAL counter: 0 prior 38.6 verdicts in harness_log.md. Counter is fresh.

---

## 3. Verbatim criterion -> evidence mapping

| # | Masterplan immutable criterion | Evidence | Verdict |
|---|---|---|---|
| 1 | `running_flag_migrates_to_handoff_dot_autonomous_loop_dot_lock` | cycle_lock.py:40 `_LOCK_PATH = _HANDOFF / ".autonomous_loop.lock"`; test 8 source-grep | PASS |
| 2 | `lock_carries_ttl_via_mtime` | cycle_lock.py:41 (5400s = 90 min) + :72 `age_sec = time.time() - _LOCK_PATH.stat().st_mtime` + :79 `is_stale = (age_sec > _LOCK_TTL_SEC) or (not pid_alive)`; test 7 asserts constant | PASS |
| 3 | `next_startup_cleans_stale_lock` | cycle_lock.py:83-98 `clean_stale_lock` unlinks if stale; test 3 simulates dead-pid + backdated mtime -> clean returns state -> file unlinked | PASS |
| 4 | `simulate_kill_mid_cycle_then_restart_passes` | test 3 = exact OPEN-15 scenario: pre-write pidfile with `dead_pid=99_999_999` + `mtime = now - (TTL+600)`, inspect_lock reports `is_stale=True/pid_alive=False/age_sec>TTL`, clean_stale_lock unlinks, fresh `acquire("recovery-cycle")` succeeds with new pid+cycle_id | PASS |

All 4 immutable criteria met at the primitive layer.

---

## 4. Code-review heuristic sweep (Top-15)

Order: ran AFTER deterministic checks + existing-results read, BEFORE final LLM judgment, per phase-16.59 skill.

| # | Heuristic | Result |
|---|---|---|
| 1 | secret-in-diff [BLOCK] | clean (stdlib-only diff) |
| 2 | kill-switch-reachability [BLOCK] | N/A (execution path unchanged; autonomous_loop.py:0-line diff) |
| 3 | stop-loss-always-set [BLOCK] | N/A (buy/sell paths unchanged) |
| 4 | prompt-injection-path [BLOCK] | N/A (no LLM calls in cycle_lock.py) |
| 5 | broad-except-silences-risk-guard [BLOCK] | clean -- `except Exception: pass` at cycle_lock.py:145-146 and :152-154 are in the `finally` cleanup of cleanup (best-effort unlink + os.close after the flock has done its job). NOT in a risk-guard path; the LOCK_UN release at :148-150 logs errors. Acceptable per negation list ("broad except in cleanup"). |
| 6 | financial-logic-without-behavioral-test [BLOCK] | N/A (cycle_lock.py is infrastructure, not financial-formula). Still, 8 behavioral tests present |
| 7 | tautological-assertion [BLOCK] | clean -- tests assert real post-conditions (pid value, file existence, is_stale boolean, age_sec comparison) |
| 8 | perf-metrics-bypass [WARN] | N/A |
| 9 | command-injection [BLOCK] | clean -- `os.kill(pid, 0)` at :54 uses bounded int (range-guarded at :51), no shell, no subprocess |
| 10 | excessive-agency-scope-creep [WARN] | clean -- no new tool/BQ-write/file-write capability beyond the lock semantics in the spec |
| 11 | position-sizing-div-zero [WARN] | N/A |
| 12 | criteria-erosion [WARN] | clean -- all 4 immutable criteria mapped to specific tests; no criterion silently dropped |
| 13 | sycophantic-all-criteria-pass [WARN] | clean -- this critique cites file:line, quotes test code, and walks 15 heuristics |
| 14 | supply-chain-dep-pin-removal [WARN] | clean -- ZERO dep changes; `import fcntl, json, os, time, pathlib, contextlib, datetime, typing, logging` all stdlib |
| 15 | unicode-in-logger [NOTE] | clean -- 4 logger calls: `"cycle_lock: malformed lock (%r); treating as stale."`, `"cycle_lock: unlink failed (%r) -- proceeding anyway."`, `"cycle_lock: cleaned stale lock (reason=%s pid=%s age_sec=%.0f)."`, `"cycle_lock: release failed (%r)."` -- all ASCII, no emoji/arrows/em-dashes |

**Dimension 4 (anti-rubber-stamp):** PASS. 8 tests; no tautological assertions; no over-mocking (only `_LOCK_PATH`/`_HANDOFF` constants monkey-patched for tmp_path isolation which is correct hygiene); no rename-as-refactor (cycle_lock.py is NEW); evaluator output is detailed with file:line citations.

**Dimension 5 (LLM-evaluator anti-patterns):** PASS. First Q/A spawn on this step (not a cycle-2 sycophancy risk). 3rd-CONDITIONAL counter clean. File:line citations present throughout.

---

## 5. LLM-judgment dimensions

### (a) Primitive-only delivery -- honest scope or criteria-erosion?

**Honest scope.** Contract `Honest scope deferral` section + experiment_results `Honest scope deferral (NEW follow-up)` both explicitly disclose that wiring to `autonomous_loop.py:142-154` + main.py lifespan is deferred to phase-38.6.1. The 4 masterplan immutable criteria are properties of the PRIMITIVE (path, TTL via mtime, startup cleans stale, sim-kill->restart) and ALL are exercised at the primitive layer. Test 3 simulates the exact OPEN-15 SIGKILL scenario end-to-end (dead pid + backdated mtime -> detect -> clean -> fresh acquire). Not criteria-erosion.

### (b) Recovery semantics (flock auto-release + pidfile cleanup) verified

Confirmed by reading cycle_lock.py:101-154:
- Race-free acquire: `os.open(O_RDWR|O_CREAT)` then `fcntl.flock(LOCK_EX|LOCK_NB)`.
- Pidfile writes `{pid, cycle_id, started_at}` atomically (lseek+ftruncate+write+fsync).
- On exit: `_LOCK_PATH.unlink(missing_ok=True)` then `fcntl.flock(LOCK_UN)` then `os.close(fd)`.
- On SIGKILL/crash: kernel auto-releases the advisory flock when the FD is closed at process death; pidfile remains; next startup's `clean_stale_lock` detects via mtime > TTL OR pid dead -> unlinks.
- BlockingIOError on flock attempt: inspect_lock; if stale, clean + retry; if live, raise `CycleLockError` with `pid` + `age_sec` for forensic logging.

Canonical Linux/POSIX recovery pattern per researcher's brief Section B citations of flock(2) + py-filelock + trbs/pid.

### (c) Mutation-resistance via 8 tests -- adequate?

Coverage matrix:
| Test | What it mutates if it disappeared |
|------|------------------------------------|
| test_acquire_writes_pid_and_cycle_id_then_unlinks | happy path -- catches regression of pidfile schema or unlink-on-exit |
| test_second_acquire_in_same_process_raises | catches removal of `LOCK_NB` (would silently block) or of CycleLockError raising |
| test_simulated_kill_then_startup_cleans | THE OPEN-15 regression test -- catches any removal of inspect_lock OR clean_stale_lock OR is_stale logic |
| test_live_lock_not_cleaned | catches false positive on stale-detection -- ensures a live process's lock isn't aggressively cleaned |
| test_malformed_lockfile_treated_as_stale | catches removal of the try/except JSON parse defense |
| test_no_lock_file_returns_none | catches None-handling regression in inspect/clean |
| test_ttl_constant_is_90_minutes | catches drift of the 5400s constant per researcher recommendation |
| test_lock_path_uses_handoff_dot_autonomous_loop_dot_lock | catches a regression of the canonical path convention (creative -- source-grep since fixture monkey-patches the runtime value) |

Adequate. Each test would fail under a real mutation; no tautological assertions.

### (d) "Wiring deferred to phase-38.6.1" pattern consistency

Same shape as cycle 42's 38.5.1 -> 38.5.2 batched delivery and phase-38.5 -> 38.5.1 -> 38.5.2 split. Both contract and experiment_results explicitly disclose the deferred wiring. Operator can choose to wire in phase-38.6.1 (1-line `acquire()` replacement + 1-line lifespan hook) or operationalize the primitive standalone. Honest cycle-management consistent with prior 30+ cycles' batching/deferral discipline.

---

## 6. Output envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 4 immutable criteria met at primitive layer (test 3 = OPEN-15 sim-kill regression; tests 7/8 = TTL+path; clean_stale_lock at :83-98). 8 tests pass. 481 total collected (+8, 0 regressions). autonomous_loop.py + other backend dirs untouched, confirming honest scope. Researcher gate_passed=true (6 sources). Top-15 code-review sweep: 0 BLOCK, 0 WARN, 0 NOTE. flock auto-release on FD close + on-disk pidfile cleanup is the canonical Linux/POSIX pattern per researcher cite of flock(2) + py-filelock. Wiring to autonomous_loop.py + main.py lifespan honestly DEFERRED to phase-38.6.1 (same pattern as 38.5.1/38.5.2 batching).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "code_review_heuristics", "researcher_brief", "contract", "experiment_results", "live_check"]
}
```

---

## 7. Bottom line

**PROCEED to harness_log append + masterplan status flip.** The cycle_lock primitive is canonical, tested, and honestly disclosed. Wiring is a 1-line replacement in phase-38.6.1 that the operator should schedule next.
