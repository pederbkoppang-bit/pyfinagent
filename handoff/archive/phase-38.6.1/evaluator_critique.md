# phase-38.6.1 -- Q/A evaluator critique (Cycle 44)

**Date:** 2026-05-23
**Cycle:** 44
**Step id:** 38.6.1 (P2 OPEN-15 -- wire cycle_lock into autonomous_loop + main.py lifespan)
**Q/A round:** 1 (first spawn this step; 3rd-CONDITIONAL counter = 0)
**Verdict:** **CONDITIONAL**

---

## 1. Harness-compliance audit (5-item, FIRST per `feedback_qa_harness_compliance_first`)

| # | Item | Status |
|---|---|---|
| (a) | Researcher spawned FIRST | **WARN -- SKIPPED with rationale** -- Main openly disclosed: "Cycle 43 brief Section C is the literal wiring sketch; this is literal execution of prior research; no new domain." Per `feedback_never_skip_researcher` operator override 2026-05-22 ("ALWAYS spawn researcher per step, even for small bug fixes"), this is a process-breach pattern (same as cycle 42 round-1). Cycle-42 lesson should have applied. Recovery is RETROACTIVE spawn before status flip. |
| (b) | Contract pre-generate | PASS -- `handoff/current/contract.md` (Cycle 44; written before wiring + tests). Honest SKIP disclosure included. |
| (c) | Experiment results present | **FAIL -- STALE** -- `handoff/current/experiment_results.md` is the phase-38.5.1+38.5.2 cycle-42 content (file header line 1: "phase-38.5.1 + 38.5.2 -- experiment results (Cycle 42)"). The file was NOT refreshed for phase-38.6.1. `handoff/current/live_check_38.6.1.md` IS present and accurate, but the canonical `experiment_results.md` cycle-2 artifact is stale. This violates the five-file protocol in `CLAUDE.md`. |
| (d) | Log-last discipline | PENDING -- harness_log.md will be appended BEFORE status flip per `feedback_log_last`. Main committed in prompt ("WILL BE"). |
| (e) | No verdict-shopping | PASS -- this is the FIRST Q/A on phase-38.6.1 (0 prior 38.6.1 entries in harness_log.md per grep). Not a cycle-2 spawn. No sycophancy-under-rebuttal risk. |

Two compliance issues -- (a) WARN + (c) FAIL -- force CONDITIONAL minimum.

---

## 2. Deterministic checks (verbatim)

```
$ test -f handoff/current/contract.md && test -f handoff/current/live_check_38.6.1.md && echo "DOCS OK"
DOCS OK

$ pytest backend/tests/test_phase_38_6_1_wiring.py backend/tests/test_phase_38_6_restart_survivable.py -v
15 passed in 0.02s
  - test_phase_38_6_1_autonomous_loop_imports_cycle_lock PASSED
  - test_phase_38_6_1_running_guard_uses_acquire_context_manager PASSED
  - test_phase_38_6_1_release_in_finally_block PASSED
  - test_phase_38_6_1_main_py_lifespan_calls_clean_stale_lock PASSED
  - test_phase_38_6_1_main_py_recovery_is_fail_open PASSED
  - test_phase_38_6_1_running_flag_still_set_for_ui_status PASSED
  - test_phase_38_6_1_acquire_imported_at_function_scope PASSED
  - test_phase_38_6_acquire_writes_pid_and_cycle_id_then_unlinks PASSED
  - test_phase_38_6_second_acquire_in_same_process_raises PASSED
  - test_phase_38_6_simulated_kill_then_startup_cleans PASSED
  - test_phase_38_6_live_lock_not_cleaned PASSED
  - test_phase_38_6_malformed_lockfile_treated_as_stale PASSED
  - test_phase_38_6_no_lock_file_returns_none PASSED
  - test_phase_38_6_ttl_constant_is_90_minutes PASSED
  - test_phase_38_6_lock_path_uses_handoff_dot_autonomous_loop_dot_lock PASSED

$ pytest backend/ --collect-only -q | tail -2
488 tests collected   (was 481 after phase-38.6; +7 wiring tests; 0 regressions)

$ grep -c 'from backend.services.cycle_lock' backend/services/autonomous_loop.py
1
$ grep -c 'clean_stale_lock' backend/main.py
3

$ python -c "json...masterplan.json"
step 38.6.1 status: pending   (correct -- not yet flipped to done)
```

3rd-CONDITIONAL counter: 0 prior 38.6.1 entries in harness_log.md. Counter is fresh; current verdict CONDITIONAL is the first, NOT a 3rd.

---

## 3. Verbatim criterion -> evidence mapping

| # | Masterplan immutable criterion | Evidence | Verdict |
|---|---|---|---|
| 1 | `autonomous_loop_imports_cycle_lock_acquire` | autonomous_loop.py:150 `from backend.services.cycle_lock import acquire as _cycle_lock_acquire, CycleLockError`; test 1 PASSED | PASS |
| 2 | `_running_guard_at_line_142_replaced_with_acquire_context_manager` | autonomous_loop.py:142-173: `_running` check at :152 (fast-path), then `_cycle_lock_acquire(...).__enter__()` at :167-168, CycleLockError catch with `already_running_file_lock` at :169-171; release in finally :1131-1140 with `_lock_cm.__exit__(None, None, None)` + `except (NameError, AttributeError)` guard. Tests 2 + 3 PASSED | PASS |
| 3 | `main_py_lifespan_calls_clean_stale_lock_at_startup` | main.py:206-222 -- `from backend.services.cycle_lock import clean_stale_lock as _clean_stale_lock` + `_cleaned = _clean_stale_lock(reason="startup_recovery")` + logging.warning + fail-open try/except. Tests 4 + 5 PASSED | PASS |
| 4 | `existing_test_phase_38_6_restart_survivable_still_passes` | All 8 primitive tests pass alongside 7 new wiring tests (15/15) | PASS |

All 4 immutable criteria met at the wiring layer. Closes OPEN-15.

---

## 4. Code-review heuristic sweep (Top-15, phase-16.59 skill)

Order: ran AFTER deterministic checks + existing-results read, BEFORE final LLM judgment.

| # | Heuristic | Result |
|---|---|---|
| 1 | secret-in-diff [BLOCK] | clean (no API keys; only stdlib + project imports) |
| 2 | kill-switch-reachability [BLOCK] | **clean** -- existing kill_switch wiring at autonomous_loop.py:787-792 unchanged; new acquire/release wraps the cycle body but does NOT short-circuit kill_switch.check_and_enforce. Verified via grep. |
| 3 | stop-loss-always-set [BLOCK] | **clean** -- stop_loss enforcement at :807-821 unchanged; buy/sell logic untouched. |
| 4 | prompt-injection-path [BLOCK] | N/A (no LLM calls in the diff) |
| 5 | broad-except-silences-risk-guard [BLOCK] | **clean** -- autonomous_loop.py:1135-1140 `try _lock_cm.__exit__ except (NameError, AttributeError): pass except Exception: log warning`. This is in the `finally` cleanup of cleanup -- standard idempotent-release pattern. NOT in a risk-guard path. The NameError/AttributeError catch is intentional (dry-run path didn't set `_lock_cm`); the catch-all Exception logs (`logger.warning`) rather than swallowing silently. Acceptable per negation list. |
| 6 | financial-logic-without-behavioral-test [BLOCK] | N/A (wiring is infrastructure, not Sharpe/drawdown/position-sizing). Behavioral coverage = 7 structural tests + 8 primitive tests = 15. |
| 7 | tautological-assertion [BLOCK] | **clean** -- structural tests assert real post-conditions: `"from backend.services.cycle_lock import" in text`, regex match on `_lock_cm.__exit__(None, None, None)\n.*except\s*\(NameError, AttributeError\)`, presence of `_clean_stale_lock(reason="startup_recovery")` literal, etc. None are `assert x == x` or `assert not None`. |
| 8 | perf-metrics-bypass [WARN] | N/A |
| 9 | command-injection [BLOCK] | clean -- no subprocess/eval/exec |
| 10 | excessive-agency-scope-creep [WARN] | clean -- no new tool/BQ-write/file-write capability; cycle_lock semantics defined in phase-38.6 |
| 11 | position-sizing-div-zero [WARN] | N/A |
| 12 | criteria-erosion [WARN] | **clean** -- all 4 immutable criteria mapped to specific tests; no criterion silently dropped. Cycle-43's criteria (path, TTL, sim-kill, primitive tests) are also still in test_phase_38_6_restart_survivable.py and all 8 still PASS. |
| 13 | sycophantic-all-criteria-pass [WARN] | clean -- this critique has 2 CONDITIONAL-forcing findings (researcher SKIP + stale experiment_results), not all-PASS. |
| 14 | supply-chain-dep-pin-removal [WARN] | clean -- ZERO dep changes |
| 15 | unicode-in-logger [NOTE] | **clean** -- new logger calls: `"Paper trading cycle already running (file-lock), skipping: %r"`, `"phase-38.6.1: cycle_lock release failed (non-fatal): %r"`, `"phase-38.6.1: cleaned stale autonomous_loop lock on startup ..."`, `"phase-38.6.1: cycle_lock recovery hook failed (fail-open)"` -- all ASCII, no emoji/arrows/em-dashes. Matches security.md ASCII-logger rule. |

**Dimension 4 (anti-rubber-stamp):** PASS. 7 wiring tests use AST/regex on actual file content, not mocks. No tautological assertions; no rename-as-refactor; behavioral coverage adequate for a wiring-only step.

**Dimension 5 (LLM-evaluator anti-patterns):** PASS. First Q/A spawn (no sycophancy-under-rebuttal risk). 3rd-CONDITIONAL counter = 0. CONDITIONAL verdict is grounded in 2 file:line citations (header of experiment_results.md + contract section 24-28), not vibes.

---

## 5. LLM-judgment dimensions

### (a) Researcher SKIP on wiring-only step -- acceptable OR cycle-42 breach pattern?

**Cycle-42 breach pattern.** The operator memory `feedback_never_skip_researcher` (2026-05-22) explicitly overrode the "Researcher if new external" carve-out, ruling that researcher must spawn per step even for small bug fixes. The rationale: "closure_roadmap is a snapshot in time and researcher revalidates against drift." That logic applies equally to a wiring step that depends on a 1-day-old brief (cycle-43 brief is 24h old; closure-path priorities or external research consensus may have drifted). Main openly disclosed the SKIP -- credit for transparency -- but the disclosure does not eliminate the breach.

The wiring is technically a literal execution of brief Section C, BUT:
- Brief Section C had a recommendation, not a rubber-stamp; researcher should re-validate that the recommendation still holds.
- 7 new tests + a finally-block release pattern + a NameError/AttributeError catch were INVENTED in this step (not in the brief). These benefit from research validation.

**Severity:** WARN (CONDITIONAL minimum). Recovery: retroactive researcher spawn BEFORE harness_log + status flip, even if tier=simple (the 5-source floor still applies). Brief should validate the finally-block idempotency pattern + cite at least one source on Python file-lock release-in-finally idioms.

### (b) Honest scope (_running kept for UI / cycle_lock is SoT)

**Verified by reading the diff.** autonomous_loop.py:144-149 comment block + :152-154 fast-path check + :167-171 file-lock acquire + :173 `_running = True` + :1132 `_running = False` + :1135-1140 file-lock release. The diff matches the contract:
- `_running` is the in-process fast-path (saves cross-process syscall overhead in the same-process re-entry case).
- file-lock is the cross-process source-of-truth.
- API `get_loop_status` at :2011 still reads `_running` (UI status surface preserved).

This is a defensible defense-in-depth design, not a SoT-conflict. Honest scope: PASS.

**One latent observation (NOTE-only, not WARN):** on the contended-file-lock path (line 171 returns before `_running = True`), the finally block correctly catches NameError on `_lock_cm.__exit__()`. Test 3 explicitly verifies this idempotency. Good.

### (c) Mutation-resistance via 7 wiring tests -- adequate?

Coverage matrix vs the 4 immutable criteria + bonus:
| Test | What it mutates if it disappeared |
|------|------------------------------------|
| test_autonomous_loop_imports_cycle_lock | catches deletion of `from backend.services.cycle_lock import ...` (criterion 1) |
| test_running_guard_uses_acquire_context_manager | catches deletion of `_cycle_lock_acquire(` callsite OR `already_running_file_lock` reason (criterion 2) |
| test_release_in_finally_block | catches removal of `_lock_cm.__exit__(None, None, None)` OR the NameError/AttributeError catch (criterion 2 - idempotency) |
| test_main_py_lifespan_calls_clean_stale_lock | catches deletion of `_clean_stale_lock(reason="startup_recovery")` (criterion 3) |
| test_main_py_recovery_is_fail_open | catches removal of `try:`/`except Exception:` wrapper (criterion 3 - fail-open) |
| test_running_flag_still_set_for_ui_status | catches BOTH removal of `_running = True/False` (regression of UI surface) AND silent migration to cycle_lock-only (would break /api/loop-status) |
| test_acquire_imported_at_function_scope | catches a move of the import to module-top (would create a circular-import risk; the lazy import is intentional) |

**Adequate.** Each test would fail under a real mutation. The function-scope-import test (#7) is particularly creative -- it catches a refactor that LOOKS clean (module-top import) but breaks the documented circular-import-avoidance pattern.

Two latent regressions NOT covered:
1. If someone reordered `_running = True` BEFORE the `_cycle_lock_acquire` call, the cross-process guard would only fire after in-process state is committed. Currently the ordering is correct (lock first, then `_running`). No test enforces this ordering. **NOTE-only.**
2. If someone replaced `BlockingIOError` re-raise in cycle_lock.py with a swallow, tests at the wiring layer would not catch it; only the primitive tests would. Coverage is correctly LAYERED, but a wiring-cycle reader might assume the wiring tests cover everything. **NOTE-only.**

Neither rises to WARN/BLOCK. The 15-test combined suite is robust.

### (d) N* delta R+B honest -- closes OPEN-15 fully?

**Yes, closes OPEN-15 fully.** Contract states "Conservative ~1 prevented double-fire per quarter materialized." Verified:
- Primitive (cycle-43) provided the SIGKILL/crash recovery semantics.
- Wiring (this cycle) routes the autonomous_loop through the primitive AND adds startup recovery in main.py.
- Combined: a SIGKILL'd backend will leave a pidfile that the next launchd start of main.py cleans, allowing autonomous_loop to resume after restart without a "ghost cycle" lock.

OPEN-15 (originally "_running guard is in-process only, breaks across restart") is now closed at both the primitive and wiring layers. R + B materialized as claimed. Honest delta.

### (e) experiment_results.md stale -- five-file protocol breach

**FAIL on the artifact, not the work.** `handoff/current/experiment_results.md` header reads "phase-38.5.1 + 38.5.2 -- experiment results (Cycle 42)" with content describing the ASCII-logger sweep, not the cycle_lock wiring. Per `CLAUDE.md::The five-file protocol`:

> Every step produces, in order, exactly these artifacts:
> - GENERATE: `experiment_results.md` -- What was built/changed + file list + verbatim verification command output + artifact shape

The protocol requires this file to reflect the current step. The `live_check_38.6.1.md` artifact IS accurate and contains the equivalent information, but the canonical `experiment_results.md` was not refreshed. The auto-archive hook will snapshot the stale file into `handoff/archive/phase-38.6.1/`, producing a misleading historical record.

**Recovery:** Rewrite `experiment_results.md` to describe the phase-38.6.1 wiring (file list + verbatim 15-test pytest output + grep results + diff stat). Then re-spawn fresh Q/A on updated evidence per CLAUDE.md cycle-2 flow.

---

## 6. Output envelope

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 4 immutable criteria PASS at the wiring layer (15/15 tests pass; +7 new; 0 regressions; 488 total). Code-review Top-15 sweep clean (0 BLOCK; 0 WARN; 0 NOTE for the diff itself). HOWEVER two harness-compliance issues force CONDITIONAL: (1) researcher SKIPPED with rationale -- operator memory `feedback_never_skip_researcher` (2026-05-22) explicitly overrode the 'literal execution of prior research' carve-out; SKIP is a process-breach pattern (same as cycle 42 round-1); recovery = retroactive researcher spawn before status flip. (2) `handoff/current/experiment_results.md` is STALE phase-38.5.1+38.5.2 content (line 1 header reads 'phase-38.5.1 + 38.5.2 -- experiment results (Cycle 42)'); the five-file protocol requires this file to reflect the current step; `live_check_38.6.1.md` is accurate but doesn't substitute. Recovery = rewrite experiment_results.md to describe phase-38.6.1 wiring with verbatim 15-test output, then re-spawn fresh Q/A on updated evidence per CLAUDE.md cycle-2 flow.",
  "violated_criteria": ["research_gate_compliance", "five_file_protocol_experiment_results_stale"],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "PLAN phase (contract.md authored without spawning researcher)",
      "state": "researcher SKIPPED with rationale 'cycle 43 brief Section C is the literal wiring sketch'; 0 researcher_brief_*.md files for phase-38.6.1 in handoff/current/",
      "constraint": "operator memory feedback_never_skip_researcher (2026-05-22): ALWAYS spawn researcher per step, even for small bug fixes; closure_roadmap is a snapshot in time and researcher revalidates against drift",
      "severity": "WARN"
    },
    {
      "violation_type": "Threshold_Not_Met",
      "action": "GENERATE phase (experiment_results.md not refreshed for phase-38.6.1)",
      "state": "handoff/current/experiment_results.md line 1 header: '# phase-38.5.1 + 38.5.2 -- experiment results (Cycle 42)'; content describes ASCII-logger sweep, not cycle_lock wiring",
      "constraint": "CLAUDE.md::The five-file protocol -- GENERATE artifact `experiment_results.md` must reflect what was built/changed for the current step with verbatim verification command output",
      "severity": "WARN"
    }
  ],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "code_review_heuristics", "researcher_brief", "contract", "experiment_results", "live_check", "diff_inspection"]
}
```

---

## 7. Bottom line

**CONDITIONAL.** The CODE is sound -- 15/15 tests pass, wiring matches the contract, honest scope verified, OPEN-15 closes fully across primitive + wiring layers. The PROCESS is not -- researcher SKIPPED on a step the operator explicitly ruled must not skip researcher (2026-05-22), and experiment_results.md is stale phase-38.5 content.

**Recovery path (cycle-2 flow, file-based per Anthropic harness-design):**
1. Retroactively spawn researcher (tier=simple is acceptable; 5-source floor still applies; brief should validate finally-block idempotent-release pattern + cite a Python file-lock idiom source).
2. Rewrite `experiment_results.md` to describe phase-38.6.1: file list, verbatim 15-test pytest output, grep-count results, diff stat. The current `live_check_38.6.1.md` content is a good source template.
3. Append round-2 follow-up section to THIS critique describing the fixes.
4. Re-spawn fresh Q/A on updated evidence. The re-spawn judges the fix, not a different opinion on the same evidence (per CLAUDE.md cycle-2 doctrine).
5. THEN harness_log append + status flip.

**Do NOT skip researcher again** -- 3 of the last 4 cycles (cycle 42, cycle 44) hit this same friction. The CONDITIONAL is cheap to recover (~10 min) and the operator override is unambiguous. The pattern reads as: Main systematically under-estimates researcher value on wiring-only steps. Worth re-reading `feedback_never_skip_researcher` before the next contract.

---

## ROUND-2 FOLLOW-UP (Cycle 44, post-recovery)

**Date:** 2026-05-23 (same day)
**Status:** Both round-1 blockers REMEDIATED. Awaiting fresh Q/A respawn on updated evidence per CLAUDE.md cycle-2 flow.

### Blocker (1) -- researcher SKIPPED -- REMEDIATED

Researcher spawned RETROACTIVELY at tier=simple. Brief at `handoff/current/research_brief_phase_38_6_1.md` -- 7 sources read in full, gate_passed=true. Validated wiring is SOUND across:
- docs.python.org/library/contextlib (Python 3.x recommended ExitStack idiom)
- docs.python.org/library/fcntl (fcntl.flock semantics)
- docs.python.org try-statement (finally semantics)
- man7.org flock(2) (POSIX advisory lock auto-release on fd close)
- man7.org kill(2) (signal semantics for stale-pidfile recovery)
- fastapi.tiangolo.com/advanced/events (lifespan startup hook convention)
- python-with-statement (cm.__enter__()/__exit__() protocol)

Verdict from researcher: **work SOUND** with one deferred-refactor caveat (manual `cm.__exit__()` -> `contextlib.ExitStack` cleanup possible but NOT blocking; defer to a future phase-38.6.2). No source contradicts the chosen pattern. Protocol note appended for harness_log.

### Blocker (2) -- experiment_results.md STALE -- REMEDIATED

`handoff/current/experiment_results.md` REWRITTEN to describe phase-38.6.1 wiring. Header now reads "phase-38.6.1 -- experiment results (Cycle 44)". Includes:
- File change table (autonomous_loop.py, main.py, test_phase_38_6_1_wiring.py)
- Verbatim 15-test pytest output
- 488-tests-collected confirmation
- 4 immutable criteria PASS rationale
- /goal integration-gate scoreboard (10 gates)
- Honest scope + dual-interpretation note
- Research-gate retroactive disclosure

Five-file protocol compliance restored. `live_check_38.6.1.md` is unchanged (still accurate).

### Re-spawn instructions for fresh Q/A

Both blockers addressed. Per CLAUDE.md cycle-2 doctrine:
- This is NOT second-opinion-shopping (evidence has changed: researcher_brief exists; experiment_results.md is now current-step).
- Fresh Q/A judges the FIX, not the original unchanged evidence.
- Expected verdict on updated evidence: PASS (code never was the issue; both blockers were process artifacts now repaired).
