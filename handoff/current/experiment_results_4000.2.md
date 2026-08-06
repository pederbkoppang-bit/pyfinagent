# Experiment results -- 4000.2: E2E smoke script + fixtures proving checks CAN fail

Date: 2026-08-06. Author: Main. Contract: contract_4000.2.md (research
wf_706ab423-79f < contract < artifacts, mtime-ordered).

## What was built

| Artifact | Path |
|---|---|
| Research brief | handoff/current/research_brief_4000.2.md |
| Contract | handoff/current/contract_4000.2.md |
| Smoke script (--dry / --live, E1-E6, exit codes 0/1/2/3/4) | scripts/qa/smoke_cc_rail_e2e.py |
| Test suite (13 tests, all subprocess+returncode) | backend/tests/test_phase_4000_2_cc_rail_smoke.py |
| This file | handoff/current/experiment_results_4000.2.md |

Produced-file set = the four handoff files above + the two code files + the
researcher auto-memory file .claude/agent-memory/researcher/ (4000.2 spawn).
git diff scope: scripts/qa/, backend/tests/, handoff/ (criterion 10); no
backend production code, no settings.

Design notes binding the implementation (full detail in contract R1-R9):
tests stub the CLI by explicit --claude-binary / PATH-prepend (CLAUDE_CODE_BINARY
is NOT a safe hook -- which() wins first); backend stubbed with a stdlib
ThreadingHTTPServer (pytest-httpserver ABSENT from venv; no new dependency
without operator sign-off -- contract non-scope); the <=30 cap is enforced on
the observable surface (pre-start gate + best-effort watcher + post-window
authoritative count after --flush-wait-s) because the BACKEND spawns the rail
subprocesses and llm_call_log is buffered; E5 is the measurable surrogate
predicate (rail-guard state has NO out-of-process surface -- gap queued as
4000.8 at this step's flip); E2 drops the claude-model restriction because the
frozen baseline's metered-complement rule is provider/agent-only (this also
makes mutant m2 uniquely killable).

## Verification command output, verbatim (criterion 4/7)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_4000_2_cc_rail_smoke.py -q
.............                                                            [100%]
13 passed in 9.30s
```

Scenario coverage: dry-zero-PUTs (C1); resolution-through-imported-logic via
PATH-stub, asserted on the preflight-reported binary path (C3); 3-tickers exit
4 + cap-raise exit 4; expected-31-calls exit 3 with cap-naming message AND
flip+restore PUT pair asserted (C2); all-pass window exit 0 with all six
E-verdicts + restore pair (C4 dir 1); keep-on cancels restore; five
single-check-fail fixtures each exit 1 with restore still fired (C4 dir 2,
C5 via the foreign-model-second-key envelope); analysis-POST-500 crash still
issues the restore PUT through ExitStack unwind, exit 2 (C6).

## Mutation matrix (criterion 8), verbatim

m1 -- is_rail_row body replaced with `return False` (rule matches zero rows):

```
FAILED backend/tests/test_phase_4000_2_cc_rail_smoke.py::test_live_all_pass
FAILED backend/tests/test_phase_4000_2_cc_rail_smoke.py::test_live_keep_on_cancels_restore
2 failed, 11 passed in 9.23s
```

m2 -- E2 neutered to ignore metered rows (`len(metered) == 0` -> `True`):

```
FAILED backend/tests/test_phase_4000_2_cc_rail_smoke.py::test_live_single_check_fail_exits_nonzero[E2_metered_row_present]
1 failed, 12 passed in 9.08s
```

Both mutants restored; post-restore suite green (13 passed in 9.30s above).
m1 turns the E1-bearing all-pass fixture red as the criterion names
(test_live_all_pass); m2 turns exactly the E2 fixture red and nothing else.

## Real-backend --dry run, verbatim (criterion 9)

```
$ .venv/bin/python scripts/qa/smoke_cc_rail_e2e.py --dry
{"event": "preflight", "health": "ok", "binary": "/Users/ford/.local/bin/claude", "paper_use_claude_code_route": true, "rail_state": "ON"}
{"summary": true, "mode": "dry", "rail_state": "ON", "flag_mutations": 0, "probe": "2 entries, cost sum 0.026022 == total", "per_analysis_rail_call_ESTIMATE_from_7d_history": 2, "estimate_note": "ticker+hour bucket median; the REAL count is measured in the first 4000.3 window"}
```

Three live facts worth naming: (1) the probe ran on real Max auth and the P2
raw-map sum assertion held on the REAL CLI -- and returned 2 modelUsage
entries, re-demonstrating the duplicate-canonicalModel pattern 4000.7 exists
for; (2) the honest rail state is ON (flag already True, consistent with the
4000.1 baseline); (3) the 7d-history estimate (2 rail calls per ticker-hour
bucket) reflects OVERLAY traffic, not a full deep-pipeline analysis -- the
contract R4 static bound (~25-33 for a full analysis under current pins)
remains unmeasured until 4000.3's first window, which is exactly why
--expected-calls-per-analysis is a required, operator-visible input and why a
cap trip is a legitimate loud outcome.

## Deviations / notes for Q/A

- Suffixed handoff filenames per the phase CONCURRENCY RAIL (other session
  active on this masterplan).
- The dry run made exactly ONE claude CLI invocation (the probe, in an empty
  temp cwd). No --live run occurred; no flag mutation occurred (the PUT list
  is empty -- flag_mutations: 0 in the summary line and backend/.env untouched).
- 4000.8 (expose rail_guard_status on an observability endpoint) will be
  queued in the same masterplan edit as this step's flip, per
  feedback_queue_discovered_defects_in_masterplan.

## Follow-up (cycle 2) -- Q/A CONDITIONAL findings fixed, 2026-08-06

Cycle-1 verdict: CONDITIONAL (evaluator_critique_4000.2.md, verbatim). All
three findings fixed; evidence CHANGED before the re-spawn:

1. E2 now implements BOTH frozen-baseline conjuncts: the metered-row count AND
   a bracketing spend-metric delta through the PRODUCTION exclusion logic
   (--spend-source 'bigquery' imports fetch_llm_spend; http URL is the test
   seam). Disclosed caveat in the emitted verdict itself: fetch_llm_spend
   fails open to 0.0, so a 0.0/0.0 bracket cannot distinguish dark from
   fetch-failure -- the row-count leg is primary.
2. Q/A's six mutation survivors addressed: m10+m16 killed by
   test_empty_window_fails_positive_controls (0-of-0 must FAIL); m11 by the
   E4_report_absent fixture; m17 by the E3b_stray_model fixture; m6 by
   test_binary_resolution_env_var_fallback (CLAUDE_CODE_BINARY differential
   that a which()-only reimplementation cannot satisfy); m15 resolved
   STRUCTURALLY: a new POST-WINDOW authoritative cap gate
   (test_postwindow_cap_authoritative, timing-independent) carries the cap
   guarantee, and the watcher is disclosed in the test-file header as
   best-effort early-abort only -- a watcher-dead mutant is masked BY DESIGN,
   not by accident.
3. Claim defects fixed: the mutation comment now names the real test ids
   (m2a/m2b); FOREIGN_ENVELOPE's total_cost_usd is now 0.05 == the first
   entry's cost alone, so a first-key-only reader GENUINELY passes it and the
   criterion-5 fixture is the discriminator it claims to be.

Suite after fixes: 19 tests.

```
$ .venv/bin/python -m pytest backend/tests/test_phase_4000_2_cc_rail_smoke.py -q
...................                                                      [100%]
19 passed in 16.19s
```

Mutation matrix re-run (assert-anchored apply/revert), verbatim:

```
== m1 ==  (is_rail_row -> return False)
FAILED ...::test_live_all_pass
FAILED ...::test_live_keep_on_cancels_restore
FAILED ...::test_postwindow_cap_authoritative
3 failed, 16 passed in 16.77s
== m2a ==  (E2 metered-row leg neutered)
FAILED ...::test_live_single_check_fail_exits_nonzero[E2_metered_row_present]
1 failed, 18 passed in 16.18s
== m2b ==  (E2 spend-delta leg neutered)
FAILED ...::test_live_single_check_fail_exits_nonzero[E2_spend_delta]
1 failed, 18 passed in 16.46s
== restored ==
19 passed in 16.29s
```

Fresh real-backend --dry after the fixes (one probe call; note the live CLI
returned a SINGLE modelUsage entry this time -- the duplicate-key pattern is
intermittent, which is exactly why the P2 sum assertion, not an entry count,
is the check):

```
{"event": "preflight", "health": "ok", "binary": "/Users/ford/.local/bin/claude", "paper_use_claude_code_route": true, "rail_state": "ON"}
{"summary": true, "mode": "dry", "rail_state": "ON", "flag_mutations": 0, "probe": "1 entries, cost sum 0.015094 == total", "per_analysis_rail_call_ESTIMATE_from_7d_history": 2, "estimate_note": "ticker+hour bucket median; the REAL count is measured in the first 4000.3 window"}
```

## Follow-up (cycle 3) -- Q/A cycle-2 findings fixed, 2026-08-06

Cycle-2 verdict: CONDITIONAL (evaluator_critique_4000.2.md cycle-2 block,
verbatim). Both findings fixed; evidence CHANGED before the re-spawn. The
3rd-CONDITIONAL auto-FAIL rule is live for this spawn -- both fixes are
complete implementations, not disclosures-in-lieu, except the one leg that is
structurally unprovable here and is handled by the accepted E5-gap pattern.

1. E4 (undisclosed baseline reduction): now implements frozen-baseline legs
   1-3 -- terminal-completed, report present, and the synthetic-0.0/HOLD shape
   check (61.2 defect class; recommendation read from both the string and the
   RecommendationDetail.action form). Leg 4 (persisted analysis row) is NOT
   PROVABLE from the sync poll (in-memory task dict, analysis.py:392): it is
   now DISCLOSED in three places -- the emitted E4 verdict itself, the
   test-file header, and contract addendum R10 -- and is queued as step 4000.9
   at this step's flip, owed as 4000.3 live_check evidence. The mS masking is
   gone: E4_status_failed_report_present isolates the status/error leg with a
   GOOD report so no other leg can absorb the kill; E4_synthetic_zero_hold
   fixtures the new leg 3.
2. E3a cost-sum coverage (mutant mP): BAD_SUM_ENVELOPE -- all-domestic two-key
   envelope, sum 0.10 vs total 0.20, foreign leg green -- so the cost-sum
   conjunct is the only possible failure path.

Suite: 22 tests. Full five-mutant matrix, verbatim (assert-anchored
apply/revert; every mutant killed by exactly its named fixture):

```
$ .venv/bin/python -m pytest backend/tests/test_phase_4000_2_cc_rail_smoke.py -q
......................                                                   [100%]
22 passed in 19.46s
== m1 ==   3 failed, 19 passed (test_live_all_pass, test_live_keep_on_cancels_restore, test_postwindow_cap_authoritative)
== m2a ==  1 failed, 21 passed (exactly [E2_metered_row_present])
== m2b ==  1 failed, 21 passed (exactly [E2_spend_delta])
== mP ==   1 failed, 21 passed (exactly [E3a_cost_sum_mismatch])
== mS ==   1 failed, 21 passed (exactly [E4_status_failed_report_present])
== restored ==  22 passed in 19.25s
```

The --dry path is byte-unaffected by the cycle-3 changes (E4/E3a legs run only
in --live; probe_envelope_check itself unchanged), so the criterion-9 capture
above remains current.
