# Evaluator critique — phase-85.4 (cycle 183, EVALUATE pass 1)

**Verdict: CONDITIONAL** · `ok: false` · `certified_fallback: false` · `harness_compliance_ok: true`

Launch: Workflow `qa-verdict` rail, run `wf_09336268-c19`, agent `a85d73b1772e38517`.
Evaluated commit `e0287161`. Transcribed VERBATIM from the Q/A's returned
structured output — Main records the verdict, never authors it. Machine copy:
`handoff/current/evaluator_critique_85.4.json`.

---

## reason

Criteria 1-5 are all MET and were independently reproduced by me, not read: the C1 measurement script re-ran to byte-identical per-cycle numbers (8554s/8529s projected vs 7200s budget; 5670s on 08-05), the immutable command returned Counter({'completed': 7, 'timeout': 3}) exit=0, 32/32 new tests pass, Settings().paper_merged_analysis_dispatch_enabled is live-False, and FOUR mutations I constructed myself (not Main's matrix) were all killed with precisely-named tests -- including MUT-A, swapping the merged-path candidate/holding partition, and MUT-B, leaking the dark path ON. Blocked from PASS by one deterministic gate: qa.md §1a ruff exits 1 on a file this step introduces -- "F401 [*] `types` imported but unused --> backend/tests/test_phase_85_4_cycle_loudness.py:36:8 ... Found 1 error." -- and neither experiment_results_85.4.md nor live_check_85.4.md records any lint run at all, so the required gate was never executed by the author. Fix the dead import, run and record the ruff gate, re-spawn a fresh Q/A. Criterion 6 therefore not granted.

## violated_criteria

- qa.md §1a Python lint gate RED: ruff --select F821,F401,F811 exit=1 on backend/tests/test_phase_85_4_cycle_loudness.py:36
- handoff evidence contains no lint-gate run despite a 9-file *.py diff
- criterion 6: fresh Q/A PASS not granted

## violation_details

### 1. Invalid_Precondition

**action**

```
printf '%s\n' "$FILES" | xargs uvx ruff check --select F821,F401,F811   (scope DERIVED: git show --name-only e0287161 + git diff --name-only HEAD + git ls-files --others, 9 files, non-empty guard passed)
```

**state**

ruff_exit=1. Verbatim: "F401 [*] `types` imported but unused --> backend/tests/test_phase_85_4_cycle_loudness.py:36:8 ... Found 1 error. [*] 1 fixable with the `--fix` option." True positive confirmed independently: grep -nE '\btypes\b' on that file returns exactly ONE line, the import itself (line 36). The file is NEW in commit e0287161, so the finding is introduced by this step, not pre-existing.

**constraint**

qa.md §1a: 'Python lint gate (REQUIRED if the diff touches any *.py) ... Non-zero exit = FAIL (quote the finding verbatim).' BLOCKING -- forbids PASS. Recorded as CONDITIONAL rather than the literal FAIL because (i) this is the FIRST EVALUATE for 85.4 (0 prior `phase=85.4` cycle headers in harness_log.md, so the 3rd-CONDITIONAL escalation does not bind), (ii) the hit is F401-only in a test file with zero functional reach -- no F821/F811, i.e. not the undefined-name class the gate exists to kill, and (iii) the remediation path is identical under either label. NEXT Q/A: if this gate is still red, it MUST be FAIL.

### 2. Missing_Assumption

**action**

```
grep -rniE 'ruff|F821|F401|lint' handoff/current/experiment_results_85.4.md handoff/current/live_check_85.4.md
```

**state**

ZERO matches. §9 of experiment_results records only test totals; live_check §5 records only the full-suite pytest. The diff touches 9 *.py files (4 production, 3 test, 2 script) and no lint gate appears anywhere in the GENERATE evidence.

**constraint**

qa.md §1a makes the lint gate REQUIRED for any *.py-touching diff. An author-side gate that was never run is an unmeasured claim, not a passed one (feedback_gate_scope_and_disclosure_completeness). WARN-level; it is the direct cause of the blocking finding above.

### 3. Missing_Assumption

**action**

```
Read backend/services/cycle_health.py:250-285 (cycle_heartbeat_alarm, success leg) and search the 12 completed-age tests for coverage of an unparseable success timestamp
```

**state**

If the most recent status=='completed' row EXISTS but its completed_at fails _parse_iso, then success_dt is None AND the `elif last_success_row is None and lines:` fallback does not fire, so success_stale stays False and should_alarm_success stays False -- the completed-age P1 goes SILENT on a corrupt-timestamp ledger. No test in test_phase_85_4_completed_age_alarm.py covers this branch (the covered no-completion case is last_success_row is None, which correctly sets stale=True).

**constraint**

WARN / NOTE-level, NOT a criterion-4 miss: criterion 4's normal path is proven behaviourally and MUT-C + MUT-D both died. But this is a fail-OPEN hole in an alarm whose entire purpose is loudness. One-line fix for the next cycle: treat an unparseable success timestamp as stale rather than as healthy.

## checks_run

- read qa.md from disk at runtime (STEP 0)
- harness-compliance audit, all 5 items: research gate (gate_passed:true, 9 sources, 25 URLs, recency scan true, 11 internal files); contract-before-generate by mtime (contract 11:12:47 < every prod/test file 21:45:08-21:57:38); experiment_results present; log-last (0 `phase=85.4` cycle headers in harness_log.md, masterplan status still `pending`); no-verdict-shopping (first EVALUATE for 85.4)
- immutable verification command -- Counter({'completed': 7, 'timeout': 3}), exit=0
- git show --stat e0287161 scope review: 13 files, 4 production + 3 test + 2 script + 4 handoff; no unintended production change; git status --short shows zero uncommitted *.py
- full read of the production diff (autonomous_loop.py, cycle_health.py, scheduler.py, settings.py)
- python lint gate §1a -- ruff F821,F401,F811 over a DERIVED 9-file scope with the non-empty guard, via xargs (RED, exit=1)
- backend runtime smoke §1d -- import autonomous_loop / cycle_health / scheduler; dispatch_analyses, fire_cycle_completed_stale_alarm, check_cycle_health_alarms all callable; _CYCLE_COMPLETED_STALE_SEC=345600.0
- criterion-5 live flag read -- Settings().paper_merged_analysis_dispatch_enabled = False
- scoped pytest, 3 new files -- 32 passed in 17.53s
- INDEPENDENT re-run of the C1 measurement script (scripts/diagnostics/measure_analysis_phase.py) -- all per-cycle, per-ticker, parallelism and projection figures reproduce byte-identically
- INDEPENDENT mutation matrix, 4 mutations of my own construction via sys.modules runtime injection (zero tree writes), with an in-process CONTROL first -- 4/4 KILLED
- live-file pollution check: md5 + line counts of kill_switch_audit.jsonl / .cycle_heartbeat.json / cycle_history.jsonl before and after the new-test run -- all three UNCHANGED
- consumer grep for cycle-status string switches across backend/, frontend/src, scripts/
- watchdog wiring check -- scheduler.py:271-274 add_job(_watchdog_health_check, interval=watchdog_interval_minutes) -> check_cycle_health_alarms()
- steps.append ordering audit -- 'analyzing' appended at :1158, dispatch at :1227
- code_review_heuristics (5 dimensions)
- 3rd-CONDITIONAL check -- 0 prior 85.4 entries

## notes

CRITERION MAP (all evidence re-derived by me unless stated).

C1 MET. I re-ran scripts/diagnostics/measure_analysis_phase.py myself: every figure reproduces exactly (08-06 projected 8554s / +1354s over; 08-07 8529s / +1329s; 08-05 5670s within). Only lines_parsed differs (162,395 -> 163,342) because backend.log has grown since -- which corroborates the artifact as a real capture rather than a transcription. The question the criterion asks is answered explicitly and in the required direction: 7200s is too short at 6 tickers / semaphore 3 / mean 2310-2320s per ticker; the phase does not hang. The 08-05 counterfactual (that cycle only fit because CRWD and DDOG PARSE-FAILED at ~176s/197s) is disclosed against interest and is the strongest single piece of scope honesty in the artifact.

C2 MET. Root cause to file:line (autonomous_loop.py:1157 and :1164 pre-fix -- two sequential asyncio.gather calls over ONE Semaphore(3)), with the production log excerpt showing the freed slot idling 1923s on 08-07 and NTAP starting 4517s in. On Main's flagged point (c): the reproduction runs against dispatch_analyses, which is now the PRODUCTION call site (line 1227 calls it), not a copy -- so it is not the phase-75.14 re-implemented-test shape. It is a quantised-clock mechanism reproduction, and it is corroborated by the real-log observation; I judge that to satisfy "with a reproduction". (b) and (c) are each ruled out by a test rather than an opinion, and the residual (no inner per-ticker timeout in _run_single_analysis) is stated plainly and queued rather than smuggled in.

C3 MET. The fault-injected tests drive the real run_daily_cycle. I verified the load-bearing mechanism myself: summary["steps"].append("analyzing") is at :1158, BEFORE the dispatch at :1227, so _steps[-1] genuinely names the phase the cycle died in for the proven case (also true for screening :516 / mark_to_market :1342 / deciding :1490 / executing :1557). On Main's flagged point (f): "proven by a fault-injected cycle" is satisfied -- the fault is injected into the real production coroutine, not into a stand-in; requiring a live 2-hour cron cycle would make the criterion unprovable within any step.

C4 MET, and the wiring is real, not scanned. scheduler.py:271-274 registers _watchdog_health_check on an interval job, and it calls the extracted check_cycle_health_alarms(). Main's own M8 caught its first-draft source-scan guard surviving `if False and ...` and it was replaced with tests that CALL the seam -- that is the phase-36.12 lesson correctly applied, and it is disclosed against interest.

C5 MET. No order/sizing/risk logic touched. Flag verified DARK by live read. On Main's flagged point (b) -- the un-gated status-fidelity + alert-title change is NOT a criterion-5 violation: criterion 3 REQUIRES the alert to fire naming the phase and to be proven live by fault injection, so reading criterion 5 to force that dark makes 3 and 5 mutually unsatisfiable. I independently checked the blast radius rather than taking the claim: no consumer switches on cycle_history status strings (paper_trading.py:1120 status["running"] is a dict key on a different object; scheduler.py:390 is APScheduler JOB status; no frontend type pins it), and cycle_health._COMPLETED_STATUSES == {"completed"} correctly counts halted_kill_switch as a non-completion.

MY OWN MUTATIONS (4/4 killed, runtime sys.modules injection, no tree writes, CONTROL run first = 16 passed):
- MUT-A merged-path partition SWAPPED (holdings returned as candidates): 4 failed, incl. test_both_paths_return_the_same_results. This is the money-relevant one -- a mis-partition on the promoted flag would feed holdings into the new-candidate path -- and the equivalence guard is genuine.
- MUT-B legacy branch ignored, always merged (dark flag leaks ON): 2 failed, incl. test_a_legacy_two_gather_path_idles_a_slot_and_starts_the_reeval_late.
- MUT-C halted_kill_switch counted as a completion: 2 failed, incl. test_c3_halted_status_is_not_counted_as_a_completion.
- MUT-D completed-stale threshold widened 96h -> 30d: 6 failed. Attribution caveat, stated so it is not over-credited: my patch also wrapped the function signature, so part of this kill may be harness artifact rather than the threshold pin. I credit MUT-A/B/C as clean kills and MUT-D as suggestive only.

ON MAIN'S FLAGGED POINT (a) -- the immutable verification command cannot fail: agreed, and Main states it in BOTH artifacts. I ran it (exit=0) and treated it as a reporter. The gate is criteria 1-5, judged on the evidence above. This is the correct handling of feedback_immutable_criteria_must_be_green_able.

ON (d) -- live-file pollution: the disclosure is adequate AND I corroborated it independently. tail -1 of handoff/kill_switch_audit.jsonl is the disclosed 2026-08-08T19:59:35Z manual PAUSE row, file at 52 lines (51 -> 52 as stated); zero resumes, so the direction is fail-safe. Critically, I md5'd all three live control-plane files before and after running the three new test files: ALL THREE UNCHANGED. That independently proves Main's claim that the new tests do not touch the operator's journal, and proves the M9 fix is real rather than asserted. The heartbeat/journal writes came from the full-suite run, are correctly attributed to the pre-existing 36.28 class, and Main correctly declined to widen 36.28 mid-EVALUATE (freeze-the-tree). Not a taint on this step.

ON (e) -- the 26-failure claim: stated with its correct limits. Main explicitly says it did NOT derive the pre-existing 26 node ids and inherited "identical set" from cycle 182, and records the current 26 ids so the next session can diff a SET. That is the honest form. I did not re-run the full suite deliberately: doing so would itself write to the live kill_switch_audit.jsonl and .cycle_heartbeat.json, which a read-only evaluator must not do. The +32 arithmetic reconciles against my own 32-passed run.

BLOCKING FINDING is the ruff red plus the absent lint evidence (details above). One WARN-level code gap recorded for the same fix cycle (the unparseable-success-timestamp fail-open in cycle_health). One NOTE: the "_died_in" comment asserts every phase appends on entry -- true for all five phases I checked, but unasserted in general.

REMEDIATION for the fresh Q/A cycle: (1) remove `import types` at test_phase_85_4_cycle_loudness.py:36; (2) run the §1a gate over a git-derived non-empty scope and paste the verbatim exit code into experiment_results_85.4.md; (3) optionally close the cycle_health fail-open. Then re-spawn on CHANGED evidence.

---

## Follow-up — Main's cycle-2 remediation (written by Main, NOT by the Q/A)

Everything above this line is the Q/A's verbatim return. This section records
what Main changed in response, so the fresh Q/A grades CHANGED evidence rather
than re-reading the same tree (the documented cycle-2 flow; spawning a fresh
Q/A on UNCHANGED evidence would be verdict-shopping and is forbidden).

| Q/A finding | Severity | Action |
|---|---|---|
| ruff F401 `import types` unused, `test_phase_85_4_cycle_loudness.py:36` | BLOCKING | **Fixed.** Import removed; symbol proven unreferenced (`grep -nE '\btypes\b'` returns zero lines). |
| No lint-gate evidence in the handoff despite a 9-file `*.py` diff | WARN (cause of the blocker) | **Fixed.** `experiment_results_85.4.md` §8b records the gate over a git-DERIVED, non-empty-guarded scope, its verbatim `ruff_exit=0`, a proof it can still go RED via `--stdin-filename`, and an honest accounting of the 225 default-rule findings (the repo has no ruff config; 23 of 225 are on files this step created, none correctness-class). Two real `EXE001` findings introduced by this step were fixed (`chmod 0755`). |
| Fail-open: an unparseable `completed_at` on a `completed` row silences the completed-age P1 | WARN | **Fixed** in `cycle_health.py` (`elif last_success_row is not None: success_stale = True`). Two new tests + mutation **M10**. Age stays `None`; the P1 renders `never` rather than a number it cannot support. |
| `_died_in` comment asserts an un-asserted general invariant | NOTE | **Accepted, not actioned** — recorded as a known limit in §11. |

Re-verified after the changes:

```
pytest (3 new files)                  34 passed  (was 32; +2 from cycle-2)
scripts/qa/mutation_matrix_85_4.py    10/10 mutations killed, tree restored, suite green
uvx ruff check --select F821,F401,F811 (9-file derived scope)   All checks passed! (exit 0)
```

**Not done, deliberately:** 36.28 was not widened, even though this step
surfaced a second instance of its class (the live `.cycle_heartbeat.json`
overwrite). A gap found mid-EVALUATE belongs to the next cycle, not to the tree
under grade.
