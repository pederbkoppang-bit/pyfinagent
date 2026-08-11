# Evaluator critique -- step 86.38

# CYCLE 1 -- RAIL DROP. NO VERDICT.

**Run `wf_2881574d-de2` (task `wj3a88q66`), 2026-08-11 07:17-07:25Z. Terminated
with `agent({schema}): subagent completed without calling StructuredOutput`
after 162,182 subagent tokens and 40 tool uses.**

**NO VERDICT. Never PASS, never CONDITIONAL.** No counter is advanced; 86.38 has
had zero completed Q/A cycles.

This is the THIRD drop of the day (86.34 cycle 3 at 185,745 tokens, 86.29 cycle 1
at 197,098, this one at 162,182). Three of seven completed runs -- a 43% drop
rate on my spawns today.

**RECOVERY NOTE, because I got it wrong first and the resolver caught me.** The
WIP path is now run-stamped (phase-86.36), so `cat verdict_wip_86.38.md` would
have returned a different cycle. I used the resolver -- and my FIRST call passed
a spawn time taken from `stat -f%SB`, which prints LOCAL time, and I labelled it
`Z`. The resolver answered `STALE / recoverable: false` and refused the record.
The artifact was fine; my timestamp was two hours wrong. Re-derived from the
epoch (`date -u -r $(stat -f%B ...)`) the same record resolves `INCOMPLETE /
recoverable: true`. **That refusal is the mechanism working**, and it is the
second time this exact mistake has been made on this project.

The rescued record follows verbatim. It is EVIDENCE for the re-run, never a
verdict.

```
STATUS: INCOMPLETE -- not a verdict
STEP: 86.38
WRITTEN: 2026-08-11T07:17:22Z

# Q/A write-first record -- step 86.38, CYCLE 1

Read qa.md in full at 07:17Z. Workflow rail. No prior verdict for 86.38 (cycle 1,
no CONDITIONAL counter armed per the spawn prompt -- to be verified against
handoff/harness_log.md myself).

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable ast.parse command; git status/diff scope; ruff lint gate;
   backend import smoke; scoped pytest
C. Claim auditing (4b) -- re-derive every number in experiment_results.md
D. Guard vacuity / mutation matrix (4c) -- especially the `_degradation_summary_fields`
   seam the author flags in (a), and M6 paging-pin
E. Judge the three refutations of the step text + the "drought does not correlate with
   degradation" claim

## Findings (appended as established)

### D1. Immutable verification command
`bash -c 'source .venv/bin/activate && python -c "import ast;ast.parse(open(...autonomous_loop.py...))"'`
-> `parsed`, **EXIT=0**. (Parse-only; proves syntax and nothing else, as the spawn prompt states.)

### D2. Commit ordering -- CONTRACT-BEFORE-GENERATE VERIFIED FROM GIT
c116e63a (census instrument, pre-contract) -> cef76c3b (contract + research brief)
-> fd419038 (production code) -> 5e97ca27 (seam extraction) -> 07fd7c07 (experiment_results
+ live_check + queue 86.41). Ordering is exactly as claimed. `git log --oneline` confirms
the ancestry chain.

### D3. Masterplan not tampered
07fd7c07 touched .claude/masterplan.json but the diff is PURELY ADDITIVE: a new step
86.41 object. 86.38's own `name`/`verification.success_criteria`/`live_check` are
untouched in that diff. No immutable-criteria edit.

### D4. Ruff lint gate (F821,F401,F811) over the derived scope
Scope derived with `git diff --name-only c116e63a~1 HEAD -- '*.py'` (12 files, spans
peer-session files too), passed through `xargs -0` (NOT an unquoted var -- zsh does
not word-split). Non-empty set asserted. Result: `All checks passed!` exit=0.

### D5. Consumer-contract check on the changed summary keys
`grep -rn "fallback_rate|fallback_reasons|fallback_alarm_fired"` across backend/,
frontend/src/, scripts/ (quoted --include patterns; the unquoted form FAILED in zsh
with "no matches found" and I re-ran it -- a false-clean I nearly accepted).
Result: ZERO consumers outside autonomous_loop.py itself, the new test file, the
mutation matrix, and a settings.py docstring. So making `fallback_rate` /
`fallback_reasons` unconditional cannot change any downstream reader's behaviour.
`fallback_alarm_fired` is a NEW key -- purely additive.
```

## What the dropped run had established before it died

Nothing in it is adopted as a finding, but it had completed the deterministic
half and none of it contradicts the artifacts:

- immutable `ast.parse` command exit=0;
- **contract-before-generate VERIFIED FROM GIT**: `c116e63a` -> `cef76c3b` ->
  `fd419038` -> seam extraction -> `07fd7c07`;
- the masterplan diff is **purely additive** (the new 86.41 step); 86.38's own
  criteria untouched;
- ruff F821/F401/F811 clean over a 12-file derived scope;
- **a consumer-contract check I had not done myself**: zero consumers of
  `fallback_rate` / `fallback_reasons` / `fallback_alarm_fired` anywhere outside
  `autonomous_loop.py`, the new test, the matrix and a settings docstring -- so
  making those keys unconditional cannot change a downstream reader's behaviour.

It also recorded a near-miss worth keeping: its first consumer grep used an
UNQUOTED `--include` pattern, which zsh rejected with "no matches found", and it
nearly accepted that as a clean result before re-running it quoted. **Same
false-clean shape that has cost this project a FAIL.**


---

# CYCLE 1 VERDICT (FIRST COMPLETED) -- Q/A, run `wf_468907a8-b13`

**Transcribed VERBATIM from the captured return value.** 157,947 tokens / 27 tool
uses / 531s. Two prior runs dropped; this is the first completed grade, so the
FAIL is on the evidence and no escalation counter was involved.

## VERDICT: FAIL   (ok=false)

## violated_criteria

- criterion_6_mutation_test_every_new_guard
- criterion_2_per_cycle_table_over_10_cycles
- illusory-guard
- anti-rubber-stamp

## reason

Criterion 6 MISS: I ran my own in-memory mutation matrix (controls first, no writes to the tree) and FOUR mutants survived the full 9-test suite, two of them sole-coverage on the step's central properties. (1) DEGRADATION_RECORD_KEYS: dropping 'fallback_reasons', 'meta_scorer_degraded', 'degraded' or 'degraded_analyses' leaves 9 passed, because `assert set(got) == set(DEGRADATION_RECORD_KEYS)` is SELF-REFERENTIAL -- `_degradation_record` derives `got` from the same tuple, so both sides shrink together; only fallback_rate and fallback_alarm_fired are truly pinned. The author's cell MY hides this: replayed exactly, it is killed by `KeyError: 'fallback_rate'` at test line 218, not by the set assertion at line 213 (mis-attributed kill mechanism). (2) Calling the record-always seam with neutered args (`_fb_fire, 0, 0, _fb_reasons`) makes `_degradation_summary_fields` return `{}` on every cycle -- the exact defect restored -- and the suite stays green, because `test_the_seam_is_actually_wired_into_the_cycle` pins the call substring and source ORDER, never the ARGUMENTS. That is "reverting the fix at the call site", which criterion 6 names explicitly. Also surviving: `_degradation = {}` inserted after the seam call (the AST guard pins the NAME, not the value), and a decoy dead helper carrying a correct-looking `record_cycle_end(..., degradation=_degradation)` (the AST guard does not pin WHICH call site). Criterion 2 is PARTIAL: the table is PER-DAY over 10 DAYS under a header reading "Per-cycle ... over >=10 cycles", substitution undisclosed, and 2026-08-04's 11 analyses show day-level aggregation can merge a degraded cycle with a clean one. One artifact claim does not reproduce: "2026-08-03..09 ran 54 full-pipeline analyses with ZERO fallbacks" is contradicted by the same artifact's table (2026-08-05: 2 lite fallbacks); the conclusion still holds on the 5 clean days (48 full / 0 lite, no trades), the stated evidence does not. VERIFIED GOOD: immutable cmd exit=0, ruff exit=0, 9 passed, imports clean, no unintended production change; the 429 body exists verbatim in backend.log and is complete; my independent re-run of the census reproduces 67/9/11.8%/10 days byte-for-byte; F2 is genuinely closed (every boundary claim in autonomous_loop.py:1335/1343, cycle_health.py:479 and both test docstrings carries the ticker-ratio/denominator-not-measured qualifier); `_intended_path` removal is justified (only 2 `_fallback_reason` write sites, :2235 adjacent to the removed line and :3401 a copy-forward, so the sets cannot differ); paging is byte-identical (`_fallback_rate_check` has zero diff lines); NOT-IN-FORCE is correct (pid 66306 started 19:33:01Z, GENERATE commit 07:04:35Z next day). Criteria 1, 3, 4, 5 MET. Harness compliance clean; 0 prior CONDITIONALs logged so the 3rd-CONDITIONAL rule did not fire -- this FAIL is on the evidence.

## violation_details

### 1. Circular_Reasoning

**action** -- pytest backend/tests/test_phase_86_38_degradation_visibility.py with al.DEGRADATION_RECORD_KEYS mutated to drop 'fallback_reasons' (also run for 'meta_scorer_degraded' and for 'degraded'+'degraded_analyses')

**state** -- 9 passed, SURVIVED in all three cells. test line 213 `assert set(got) == set(DEGRADATION_RECORD_KEYS)` compares the function output against the same tuple the function iterates, so both sides shrink identically. Control cell (no mutation) 9 passed; cell A4 dropping 'fallback_rate' KILLED, confirming only the two explicitly-asserted keys are pinned. 4 of 6 keys can vanish from the persisted degradation record with the suite green -- including 'fallback_reasons', which carries the 429 causes. The test's own failure message claims to catch exactly this ('the persisted degradation record dropped or gained a key -- a key silently missing here is invisible to every downstream reader'), and _degradation_record's docstring names key-dropping as the cycle-1 survivor this extraction was written to close.

**constraint** -- Immutable criterion 6: 'Mutation-test every new guard, including reverting the fix at the call site; a guard whose mutant survives does not count.' qa.md 4c: sole-coverage vacuity on a behavioural criterion is BLOCKING.

### 2. Missing_Assumption

**action** -- pytest with backend/services/autonomous_loop.py source mutated so the record-always call reads `summary.update(_degradation_summary_fields(_fb_fire, 0, 0, _fb_reasons,))`

**state** -- 9 passed, SURVIVED. _degradation_summary_fields returns {} for every cycle (its `if not n_total: return {}` branch), so no fallback_rate is ever recorded and the exact defect this step removes is restored. test_the_seam_is_actually_wired_into_the_cycle asserts only that the substring 'summary.update(_degradation_summary_fields(' is present and appears before 'if _fb_fire:' -- it never inspects the arguments. Control cell (comment-only edit) 9 passed; cell C1 moving the call inside the paging branch was KILLED, so the guard covers position but not payload. Not contrived: the step's own stated open question is what the alarm denominator should be, making an argument-level edit at this call site a likely future change. The sibling seam's AST guard does pin its value name; this one does not.

**constraint** -- Immutable criterion 6 explicitly includes 'reverting the fix at the call site'. qa.md 4c shape 1/2: a guard that cannot fail when its subject is broken does not count.

### 3. Unjustified_Inference

**action** -- Replay of the author's mutation cell MY (delete the tuple line '    "fallback_rate", "fallback_alarm_fired", "fallback_reasons",') against the suite

**state** -- rc=1, but the failure is `KeyError: 'fallback_rate'` raised at test_phase_86_38_degradation_visibility.py:218, i.e. by the follow-up assert on one explicitly-named key -- NOT by the set-equality assertion at line 213 that the cell exists to exercise. Because MY removes three keys at once and one of them is the pinned key, the matrix reports KILLED and experiment_results.md concludes the persisted key SET is 'behaviourally testable'. It is not: single-key drops of the other four survive. This is the mis-attributed-kill-mechanism shape (qa.md 4c #11) and it is the reason the gap above went undetected.

**constraint** -- qa.md 4c #11: a mutation genuinely killed but by a different assertion than credited must name WHICH assertion killed; a matrix result licenses only 'these N mutations were killed'.

### 4. Overgeneralization

**action** -- Read handoff/current/live_check_86.38.md section B header vs the table it presents, then re-ran scripts/qa/derive_lite_fallback_census_86_38.py myself

**state** -- Section header reads 'Per-cycle full-versus-lite over >=10 cycles (required item 2)'; the output it presents is headed 'PER-DAY full-pipeline vs lite-fallback' and closes with 'days covered: 10'. The unit is DAYS, not CYCLES, and the substitution is disclosed nowhere in live_check_86.38.md or experiment_results_86.38.md (which records it as '10 days, 67 vs 9'). The aggregation is lossy: 2026-08-04 shows 11 full analyses, more than one cycle's ticker count, so a day containing a fully-degraded cycle and a clean one reports as partially degraded. handoff/cycle_history.jsonl carries cycle boundaries, so per-cycle derivation was available. My independent re-run reproduces the table byte-for-byte (67/9, 11.8%, 10 days), so the instrument is sound -- the granularity claim is not.

**constraint** -- Immutable criterion 2 / masterplan success_criteria[1]: 'the degradation RATE is derived over at least the last 10 completed CYCLES: for each, how many tickers ran the full orchestrator versus fell back to lite.'

### 5. Contradiction

**action** -- Cross-checked the claim 'ZERO fallbacks' against the per-day census table printed ~40 lines above it in the same artifact, and against my own independent re-run of derive_lite_fallback_census_86_38.py

**state** -- live_check_86.38.md:127 and experiment_results_86.38.md:24 both state '2026-08-03..09 ran 54 full-pipeline analyses with ZERO fallbacks'. The table in the same artifact, reproduced exactly by my re-run, shows 2026-08-05 with full=6, lite=2, cause 'code defect: QuantAgent NoneType x2' -- inside that window. The 54 total is arithmetically correct (8+11+6+11+10+8); 'zero fallbacks' is not. By the artifact's own framing those QuantAgent fallbacks ARE degradation ('6 of the 9 fallbacks are a different defect... more of the degradation than the 429 is'). The load-bearing conclusion -- that the trade drought does not correlate with degradation -- still holds on the 5 genuinely clean days (48 full analyses, 0 fallbacks, no trades), so this is a defective figure supporting a sound conclusion, not a wrong conclusion.

**constraint** -- qa.md 4b: every numeric or set-membership claim in a measured artifact must reproduce; a number in a verbatim-labelled artifact that its own evidence contradicts is a Contradiction finding.

## checks_run (18)

- harness_compliance_audit_5_item
- immutable_verification_command_exit_0
- ruff_F821_F401_F811_derived_scope
- scoped_pytest_9_passed
- backend_import_runtime_smoke
- git_scope_no_unintended_production_change
- independent_mutation_matrix_11_cells_2_controls
- author_mutation_cell_MY_replay_kill_mechanism
- census_script_independent_rerun
- 429_body_verified_against_backend_log
- fallback_reason_write_site_enumeration
- F2_boundary_qualifier_resweep
- not_in_force_pid_vs_commit_time
- paging_predicate_zero_diff
- claim_auditing_4b
- guard_vacuity_check_4c
- code_review_heuristics
- harness_log_conditional_count

## notes

No UI claim in this step, so gate 1c does not apply and no Playwright capture was taken. All mutations were performed IN MEMORY (patching al.DEGRADATION_RECORD_KEYS, which the function reads as a module global, and patching inspect.getsource for the source-based guards) via pytest.main with an in-memory plugin -- zero writes to the tree, nothing to restore, and two control cells were run first and confirmed green. I deliberately did NOT run scripts/qa/mutation_matrix_86_38.py because it edits production files; I replayed its cells MX and MY in memory instead and both are genuinely killed, so the '9 cells, 9 killed' claim is not itself false -- the matrix is incomplete in exactly the place experiment_results.md claims it closed. I did not re-run the BigQuery paper_trades query (criterion 3); I accepted it on the agreement of the shown query output, the step text, and my own prior independent record of 2026-07-31T18:47:37Z. The 'Vertex has no per-day quota' refutation rests on the research brief and I did not independently fetch Google's documentation. Write-first record at /Users/ford/.openclaw/workspace/pyfinagent/.claude/agent-memory/qa/verdicts/verdict_wip_86.38__20260811T075940Z.md (COMPLETE) -- it is a crash-survival record, not a verdict. Named fixes for the next cycle, so this does not cost another blind round: (1) replace the self-referential set assertion with a literal expected-key set written out in the test, so the tuple and the assertion are two independent statements; (2) give the record-always seam an argument-level guard -- the cleanest form mirrors what was already done for the second seam, compose _degradation_for(analyses, threshold) and drive it end-to-end so C2 dies; (3) add single-key-drop mutation cells, not only the three-key line delete; (4) correct 'ZERO fallbacks' to name the 2 QuantAgent fallbacks on 2026-08-05 and restate the refutation on the 5 clean days; (5) either derive the table per-cycle from cycle_history.jsonl or state plainly in section B that per-day was substituted for per-cycle and why. The production change itself is correct, fail-safe and non-scope-clean; I found no defect in it.
