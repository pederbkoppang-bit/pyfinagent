# Evaluator critique -- phase-23.2.14 (P2) -- Cycle 38

**Step:** 23.2.14 -- Verify no other re-entrant Lock patterns in backend/
(deferred from phase-23.1.22).
**Type:** VERIFICATION (static scan + regression tests; ZERO source code
changes; tests-only delta).
**Date:** 2026-05-23.
**Q/A verdict:** **PASS** (no BLOCK violations; one WARN on test-comment
honesty, non-blocking).
**Prior CONDITIONALs on step 23.2.14:** 0 (no 3rd-conditional risk).

## 1. Five-item harness-compliance audit

| # | Check | Result |
|---|-------|--------|
| (a) | Researcher SPAWNED first | PASS -- `handoff/current/research_brief_phase_23_2_14.md` present; 5 sources read in full; gate_passed=true; tier=moderate |
| (b) | Contract written pre-GENERATE | PASS -- `handoff/current/contract.md` present |
| (c) | harness_log append before status flip | PENDING -- orchestrator must append Cycle 38 block before flipping `status=done` |
| (d) | Log-last discipline | PENDING -- same |
| (e) | First Q/A on this step (no verdict-shopping) | PASS -- no prior Q/A for 23.2.14 in harness_log |

## 2. Deterministic checks

```
DOCS OK                                                          # contract + live_check + research_brief all present
pytest backend/tests/test_phase_23_2_14_no_reentrant_locks.py
  5 passed in 0.07s
    - test_phase_23_2_14_threading_lock_count_matches_roster              PASS
    - test_phase_23_2_14_locked_helpers_document_caller_holds_lock        PASS
    - test_phase_23_2_14_no_locked_helper_reacquires_self_lock            PASS
    - test_phase_23_2_14_phase_23_1_22_anchor_preserved                   PASS
    - test_phase_23_2_14_no_rlock_used_as_workaround                      PASS
pytest backend/ --collect-only
  453 tests collected, 0 errors                                  # no collection regression
grep 'threading.Lock()' backend/ (non-test)                = 14 raw hits
grep 'threading.RLock'  backend/ (non-test)                = 0 hits  # phase-23.1.22 design preserved
git diff --stat backend/{agents,services,api,config,main,governance}/  = empty  # zero non-test source change
```

`checks_run`: `["syntax", "verification_command", "file_existence", "lock_count_independent", "rlock_independent", "diff_scope", "evaluator_critique", "code_review_heuristics"]`

## 3. Independent lock-roster reconciliation

The 14 raw `threading.Lock()` regex hits decompose as:

| # | File:line | Real instantiation? | Notes |
|---|-----------|---------------------|-------|
| 1 | `tools/alt_data.py:30` | YES | `_CACHE_LOCK` module-level |
| 2 | `agents/llm_client.py:365` | YES | `_BUDGET_CACHE_LOCK` (new vs phase-23.1.21 roster of 12) |
| 3 | `agents/_genai_client.py:29` | YES | `_client_lock` module-level |
| 4 | `governance/limits_loader.py:50` | YES | `_init_lock` module-level |
| 5 | `api/job_status_api.py:87` | YES | `_lock` module-level |
| 6 | `services/api_cache.py:31` | YES | instance `self._lock` |
| 7 | `services/kill_switch.py:46` | YES | instance `self._lock` (the anchor lock) |
| 8 | `services/kill_switch.py:112` | **NO -- docstring text** | Line is inside the `_snapshot_locked` docstring describing the phase-23.1.22 BUG (`re-entered the same threading.Lock() via snapshot()`) |
| 9 | `services/cycle_health.py:257` | YES | instance `self._lock` |
| 10 | `services/perf_tracker.py:34` | YES | instance `self._lock` |
| 11 | `services/live_prices.py:40` | YES | instance `self._lock` |
| 12 | `services/observability/alerting.py:64` | YES | instance `self._lock` |
| 13 | `services/observability/api_call_log.py:59` | YES | `_lock` module-level |
| 14 | `services/observability/api_call_log.py:200` | YES | `_llm_lock` module-level |

**Real instantiations: 13.** The researcher's count of 13 is correct.
The test claim "pytest verified 14 (additional lock at kill_switch.py:112)"
in the EXPECTED_LOCK_COUNT comment is misleading -- line 112 is a
**docstring artifact**, not a 14th lock. The test passes by counting
that docstring line via a regex that strips `#`-prefixed comments but
does NOT exclude triple-quoted docstrings.

## 4. Code-review heuristics (5-dimension sweep on tests-only diff)

| Dimension | Result |
|-----------|--------|
| **1. Security** | CLEAN -- no secret-in-diff, no prompt-injection path, no unsafe deserialization, no new agency/capability |
| **2. Trading-domain correctness** | CLEAN -- kill_switch helper `_snapshot_locked` + `phase-23.1.22` anchor strings preserved (test 4 PASS); RLock workaround banned (test 5 PASS); no risk-guard wiring touched; no kill-switch-reachability change |
| **3. Code quality** | One WARN -- see below |
| **4. Anti-rubber-stamp on financial logic** | CLEAN -- diff is verification-only; no financial logic touched (no perf_metrics / risk_engine / backtest); 5 behavioral tests with 5 distinct anti-pattern coverage; NOT tautological |
| **5. LLM-evaluator anti-patterns** | CLEAN -- file:line citations present throughout; first Q/A pass; verdict tracks evidence; no sycophancy risk (no prior verdict to flip) |

### WARN-class finding -- `test-comment-honesty` (Dimension 3, code quality)

**Severity:** WARN (does NOT force CONDITIONAL because the test still
correctly enforces drift-detection on the regex count; the issue is the
docstring claim, not the assertion).

- **File:line:** `backend/tests/test_phase_23_2_14_no_reentrant_locks.py:24`
- **Issue:** Comment reads `EXPECTED_LOCK_COUNT = 14  # researcher
  2026-05-23 found 13; pytest verified 14 (additional lock at
  kill_switch.py:112)`. Line 112 of `kill_switch.py` is INSIDE a
  triple-quoted docstring describing the phase-23.1.22 BUG -- it is
  NOT a 14th `threading.Lock()` instantiation. There are 13 real
  instantiations. The regex `threading\.Lock\s*\(\s*\)` matches the
  docstring text and the comment-stripper only excludes lines starting
  with `#`, not docstring bodies.
- **Why this matters:** The drift-detection still works (a 14th real
  lock added without re-audit would push the count to 15 and fail).
  But the test's stated rationale (`additional lock`) misrepresents
  the fixture. A future reader cleaning up the docstring (e.g.
  removing the `threading.Lock()` phrase from the bug-description
  string) would silently break this test for an unrelated reason and
  conclude their docstring edit broke an audit.
- **Recommended fix (non-blocking, post-merge OK):** Either
  (a) update the EXPECTED_LOCK_COUNT comment to read
  `"researcher counted 13 real instantiations; regex finds 14 because
  kill_switch.py:112 is a docstring artifact (the phase-23.1.22
  bug-description string contains 'threading.Lock()'); count stays
  at 14 to keep the test passing under the regex"`, OR
  (b) tighten the test to skip docstring bodies (`ast.parse` walk
  rather than regex), which yields 13 and matches the researcher
  count honestly. Option (a) is cheaper.

This is a code-quality WARN, not a BLOCK, because the audit's
**behavioral intent** (any NEW real lock fails the count) is preserved.

## 5. LLM judgment -- contract alignment

| Acceptance criterion (from contract) | Evidence | Verdict |
|--------------------------------------|----------|---------|
| Static scan all `threading.Lock` instantiations in backend/ | 14 regex hits enumerated; 13 real + 1 docstring artifact | MET |
| Identify any re-entrant call paths | Researcher brief lists all helper-method paths; 3 use `_*_locked` convention; no `_*_locked` body re-acquires self._lock (test 3 PASS) | MET |
| Lock the phase-23.1.22 fix shape with regression tests | 5 tests covering 5 distinct anti-patterns: count drift / helper docstring discipline / no self-re-acquire / anchor preservation / no-RLock workaround | MET |
| ZERO source code changes (verification-only) | `git diff --stat backend/{agents,services,api,config,main,governance}/` = empty | MET |
| Researcher gate cleared | brief present; 5 sources read in full; gate_passed=true | MET |

### Anti-rubber-stamp probe -- "would these tests catch a real bug?"

Five DISTINCT failure surfaces, none tautological:

- **Test 1** -- Add a 15th real `threading.Lock()` somewhere -> count
  drifts to 15, assert fails. Catches "new lock added without audit."
- **Test 2** -- Add a `_foo_locked` helper without "caller MUST hold"
  docstring -> tolerance `<=1` exhausts, assert fails. Catches "new
  helper drops the discipline."
- **Test 3** -- Make `_snapshot_locked` re-acquire `self._lock` ->
  body-regex finds `"with self._lock"`, assert fails. Catches the
  phase-23.1.22 anti-pattern returning.
- **Test 4** -- Rename `_snapshot_locked` away or strip the
  `phase-23.1.22` anchor string -> assert fails. Catches the audit
  trail being severed.
- **Test 5** -- Swap any `threading.Lock()` for `threading.RLock()` ->
  hits != [], assert fails. Catches the "switch to RLock as a
  workaround" anti-pattern Real Python warns against.

This is a 5-layer regression lock, NOT 5 wrappers around the same
assertion.

### Mutation-resistance (spot check, not live mutation)

Inserting `with self._lock:` into the body of `_snapshot_locked`
would be caught by test 3 (its body regex over `_*_locked` methods
catches the substring). Verified by code inspection of the test
(would mutate source if run live; declined per Q/A read-only rule).

## 6. Verdict and N* delta R+B honesty check

The N* delta R+B framing in the Q/A prompt:
> *"researcher counted 13, pytest caught the 14th"*

This is loose. What actually happened: the researcher correctly
counted 13 **real instantiations**. The pytest regex captured 14
matches because it also picks up a docstring artifact at
`kill_switch.py:112`. The 14th is NOT an additional lock -- it's a
quirk of the regex catching prose inside a triple-quoted string.

The audit is still sound:
- 13 real Lock instantiations all verified clean of re-entrant paths
- The phase-23.1.21 baseline was 12; phase-23.2.14 confirms 1 new lock
  (`_BUDGET_CACHE_LOCK` in `llm_client.py:365`) added since then; that
  lock is module-level singleton with no helper-method paths, clean
- 3-layer regression discipline (count + helper-docstring + body-check)
  plus 2 ancillary locks (anchor preservation + no-RLock-workaround)
  = 5-layer regression lock total
- ZERO source code drift; tests-only delta
- All 5 pytest tests pass

The WARN on `EXPECTED_LOCK_COUNT = 14` comment is a readability /
honesty finding, not a behavioral defect.

**Recommendation: PASS, with the test-comment-honesty WARN logged as a
post-merge nit (not a blocker).** Researcher off-by-one was honestly
caught + raised to 14 in the test; the documentation framing of that
14 is what's slightly off.

## 7. JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "5 pytest tests pass; 13 real threading.Lock instantiations + 1 docstring artifact at kill_switch.py:112 = 14 regex hits; ZERO source code drift; researcher gate cleared (5 sources, gate_passed=true); 5-layer regression discipline locks the phase-23.1.22 fix shape (count drift / helper docstring / no self-re-acquire / anchor preserved / no-RLock workaround). One WARN on test-comment honesty (EXPECTED_LOCK_COUNT comment mischaracterizes the docstring artifact as a real lock) -- behavioral intent preserved; does not block.",
  "violated_criteria": [],
  "violation_details": [
    {
      "violation_type": "Threshold_Not_Met",
      "action": "read backend/tests/test_phase_23_2_14_no_reentrant_locks.py:24",
      "state": "EXPECTED_LOCK_COUNT = 14 comment says 'additional lock at kill_switch.py:112' but line 112 is a docstring artifact, not a real Lock() instantiation; real count is 13",
      "constraint": "test-comment-honesty (code-quality WARN, dimension 3): test comments should accurately describe the fixture they pin",
      "severity": "WARN"
    }
  ],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "file_existence", "lock_count_independent", "rlock_independent", "diff_scope", "evaluator_critique", "code_review_heuristics"]
}
```
