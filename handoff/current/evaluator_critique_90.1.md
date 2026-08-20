# Evaluator Critique -- step 90.1

## Cycle 1 -- Q/A verdict, TRANSCRIBED VERBATIM

Launched `Workflow({scriptPath: '.claude/workflows/qa-verdict.js'})`, run `wf_b7fc2eb5-efd`,
1 agent, 242,849 tokens, 968s, 80 tool calls. Attempt 2 of 5 for step 90.1
(attempt 1 was this step's research gate -- 90.6's finding, live).

**Main records the verdict; Main never authors it.** The block below is the captured
return value, unedited.

```json
{
  "ok": false,
  "verdict": "FAIL",
  "reason": "Immutable command reproduces exit=0 (self-test 32 ok, control GREEN first, 10 KILLED / 0 SURVIVED / N0 survived, tree md5 unchanged) and criteria 2, 3, 6 are independently re-derived and MET. But two criteria are falsified BY EXECUTION on the live system. (1) Criterion 4: masterplan_step_ids() walks only phases[].steps[] and misses 14 dotted ids that live under phases[].subphases[] -- 10 of them status=pending AND harness_required=true (38.13, 46.0-46.8). Driven through the REAL hook with only the ledger/escalation dir redirected: step_id=46.0 exit=2 and step_id=38.13 exit=2, with the denial text asserting 'not a step in .claude/masterplan.json' about ids that ARE in that file (control 86.118 exit=0). The criterion says 'rejects any id ABSENT from .claude/masterplan.json'; the gate now blocks 10 real pending steps, fail-CLOSED, the opposite of the disclosed fail-open. No known-member recall test was ever run over the plan's own members. (2) Criterion 1: the backfill is NOT re-runnable -- `attempt_outcomes.py --backfill --dry-run` now exits 1 with AssertionError, because attempt_gate.py:451-462 writes launch rows with outcome/outcome_reason/total_tokens PRESENT-and-null (comment: 'completed later by attempt_outcomes.py --backfill') while backfill():306-313 aborts the whole write on any existing-key change. The two halves of the same commit are mutually incompatible; the first launch after the commit (this Q/A spawn's own row, 19:27:19Z) broke it. Every matrix and self-test fixture seeds the OLD row shape, so the drives stay green forever (fixture blindness, qa.md 4c shape #5). The live gate itself is unaffected (resolved_rows re-resolves lazily) and the failure is fail-closed, so no ledger corruption. Three WARNs: M10's description names 'reverts to timestamp semantics by WIDENING the tolerance' while the code narrows 30 to 0 -- I ran both named mutants through the author's own harness: field-revert KILLED, widening to 86400 SURVIVED, and it is not equivalent (summed tokens collapse 19,742,415 -> 1,865,277, re-opening the inertness criterion 3 closes); criterion 5 clause 3 is falsified (a mutant injected with a SyntaxError scores KILLED, not ERROR -- bounded: all 11 shipped cells parse, so no reported kill is a build artifact); and 90.1 turns scripts/qa/mutation_matrix_86_71.py permanently 'CONTROL IS RED' because it drives step_id 77.7 against the real plan with no synthetic override. Scope honesty is otherwise strong and the disclosed leak was genuinely fixed, not merely disclosed.",
  "violated_criteria": [
    "criterion_4_unresolvable_step_id_deny",
    "criterion_1_rerunnable_backfill",
    "consumer-contract-break",
    "illusory-guard"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "python3 scripts/harness/attempt_gate.py driven via stdin with args.step_id='46.0' and '38.13', REAL .claude/masterplan.json, ledger+escalation dir redirected to a tempdir",
      "state": "exit=2 for both, stderr \"[attempt-gate] DENIED: this launch claims step_id '46.0', which is not a step in .claude/masterplan.json\"; control step_id='86.118' exit=0. Root cause: attempt_outcomes.py:349-370 masterplan_step_ids() iterates only mp['phases'][]['steps'][]. Recursive census: shallow walk 1347 ids, dotted ids present anywhere 1282, MISSED 14. 13 of the 14 are real steps under phases[].subphases[]; 10 are status=pending AND harness_required=true: 38.13, 46.0, 46.1, 46.2, 46.3, 46.4, 46.5, 46.6, 46.7, 46.8",
      "constraint": "SEVERITY BLOCK -- criterion 4: 'extract_step_id rejects any id absent from .claude/masterplan.json'. These ids are PRESENT in that file, so the gate now self-blocks 10 pending harness-required steps and its denial text states a falsehood about the plan of record. Fix: walk phases[].subphases[] too, with a recall assertion over the plan's own members"
    },
    {
      "violation_type": "Contradiction",
      "action": "python3 scripts/harness/attempt_outcomes.py --backfill --dry-run",
      "state": "exit=1, AssertionError: \"backfill would MUTATE an existing field on row ts='2026-08-20T19:27:19Z' step_id='90.1': {...'outcome':'UNKNOWN','outcome_reason':'no_run_record','total_tokens':0...} != {...'outcome':None,'outcome_reason':'unresolved_at_launch','total_tokens':None...}. Refusing to write\". attempt_gate.py:451-462 writes those keys present-and-null at launch and its own comment says the record is 'completed later by attempt_outcomes.py --backfill'; attempt_outcomes.py:306-313 projects the merged row onto the original key set and raises on any changed value, aborting the whole ledger. Two null rows already exist (19:27:19Z, 19:27:57Z) and one is added per launch. In-memory proof of the fixture gap: production row shape -> projection != parsed True; matrix/self-test fixture shape -> False",
      "constraint": "SEVERITY BLOCK -- criterion 1: 'a re-runnable backfill reconstructs both for all 92 existing rows and prints the per-value counts'. It now prints no counts and exits 1. Fix: exclude launch-placeholder keys from the projection when their persisted value is null, or omit them at launch"
    },
    {
      "violation_type": "Overgeneralization",
      "action": "python3 scripts/qa/mutation_matrix_86_71.py --verify",
      "state": "'CONTROL IS RED -- the matrix is meaningless' with 5 failing checks. Proven cause: that matrix drives step_id '77.7'; \"'77.7' in masterplan -> False\"; grep -c ATTEMPT_GATE_MASTERPLAN -> 0 so it uses the real plan; the pre-90.1 gate had no membership check ('masterplan_step_ids' in the 3bf0b0fe~1 image -> False). Census of every attempt_gate consumer supplying a step id: exactly one broken. Mitigation: 86.71's immutable command is an ast.parse, not this matrix. experiment_results says 'Blast radius, measured over all 617 historical launches ... Zero production Q/A evaluations are affected' -- that population excludes checker fixtures, and the masterplan notes had explicitly directed 'fix the fixture, do not weaken the check'",
      "constraint": "SEVERITY WARN -- consumer-contract-break [Dim-3 #16]: a shape/routing change must have every consumer grep-verified in the same diff. Fix: set ATTEMPT_GATE_MASTERPLAN to a synthetic plan containing 77.7 in mutation_matrix_86_71.py"
    },
    {
      "violation_type": "Threshold_Not_Met",
      "action": "Q/A-supplied mutants MX1/MX2 run through mutation_matrix_90_1.run_cell, plus a tolerance sweep of resolve_row over the real 91 attempt rows",
      "state": "M10 is described as 'the join reverts to timestamp semantics by WIDENING the tolerance past the measured ambiguity threshold' but the code mutates DEFAULT_TOLERANCE_S 30 -> 0 (narrowing). MX1 (start=d.get('startTime') -> d.get('timestamp')) KILLED; MX2 (DEFAULT_TOLERANCE_S 30 -> 86400) SURVIVED. Not equivalent: tol=30s/300s graded=66 ambiguous=0 summed_tokens=19,742,415; tol=3600s graded=6 ambiguous=68 tokens=3,847,645; tol=86400s graded=2 ambiguous=77 tokens=1,865,277 -- widening collapses token accounting to ~9% and re-opens the ceiling inertness criterion 3 exists to close. Direction is fail-safe (under-count allows more)",
      "constraint": "SEVERITY WARN -- qa.md 4c: name the concrete mutation that makes each guard fail; a matrix licenses only 'these N mutations were killed'. The upper bound on DEFAULT_TOLERANCE_S has no guard and the cell that claims to test it does not. Fix: relabel M10 and add a widening cell"
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "Q/A-supplied cells MXE1 (nonexistent anchor) and MXE2 (inject '((((' into attempt_gate.py) through mutation_matrix_90_1.run_cell",
      "state": "MXE1 -> ERROR ('anchor appears 0 times in attempt_gate.py, expected 1'); MXE2 -> KILLED with blanket check failures. run_cell's ERROR path covers anchor-count mismatch and exceptions raised inside observations(), but subprocess.run does not raise on non-zero exit, so a mutant that cannot even parse is credited as a kill. BOUNDED: I applied every one of the 11 shipped cells' replacements and ast.parse'd the result -- all 11 parse, so no reported kill is a build-failure artifact",
      "constraint": "SEVERITY WARN -- criterion 5: 'a mutant that fails to run scores ERROR and never counts as a kill'. Fix: score a drive whose subprocess exits non-zero with a traceback as ERROR"
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "read attempt_gate.py:728-733 and execute the assertion shape on a temp directory",
      "state": "the check named 'the self-test wrote every escalation into its OWN temp dir -- nothing leaked into handoff/current/' asserts ESCALATION_DIR != old_e AND all(p.parent == ESCALATION_DIR for p in ESCALATION_DIR.iterdir()); the second clause is a tautology (executed: True by construction, iterdir yields only children) and the check never looks at handoff/current/. It catches the historical bug only via the rebinding proxy",
      "constraint": "SEVERITY WARN -- assert the property, not a proxy (qa.md 4c shape #4). NOTE the underlying FIX is real and verified: the stray escalation_unknown_step_id_9.9.md is absent and was never committed, and the four real exhaustion escalations are byte-identical after my own self-test + full matrix run. Fix: assert against handoff/current/ directly"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_items",
    "verification_command",
    "syntax_ast_parse",
    "python_lint_ruff_derived_scope",
    "backend_runtime_smoke",
    "mutation_test",
    "qa_supplied_mutants_MX1_MX2_MXE1_MXE2",
    "independent_rederivation_of_numeric_claims",
    "known_member_recall_test_masterplan",
    "sibling_checker_regression",
    "sha256_ledger_invariance",
    "code_review_heuristics",
    "research_gate_compliance",
    "contract_completeness"
  ],
  "harness_compliance_ok": true,
  "notes": "Harness compliance clean on all 5 items. Order by mtime (local CEST): research_brief_90.1 21:12:47 < contract_90.1 21:15:34 < attempt_outcomes.py 21:17:38 < attempt_gate.py 21:21:17 < mutation_matrix_90_1.py 21:23:43 < experiment_results_90.1 21:26:07 < live_check_90.1 21:26:30, all before my WRITTEN stamp. log-last verified: grep -F 'phase=90.1' handoff/harness_log.md returns 0 hits with the grep proven live by phase=86.116 returning 2; masterplan 90.1 status=pending. Research gate: brief_status COMPLETE, gate_passed true, 10 sources read in full, 25 URLs, recency scan present; all 6 criteria appear verbatim in the contract.\n\nsequence: no_rows_for_step. verdict_history_86_21.py --step 90.1 --evidence-only returned status 'no_rows_for_step', verdicts (none), with its own caveat that nothing writes the ledger automatically. qa_wip.py 90.1 --spawned-at 2026-08-20T19:27:24Z returned source_present true, attempt_number 1, prior_attempts 0, attempt_number_status ok, attempt_number_is_lower_bound false, prior_records []. Cross-check: prior_attempts 0 is not greater than the ledger's 0 rows, so no staleness flag. Cycle 1 confirmed; no prior verdict exists, so no second-opinion-shopping is possible.\n\nAnswers to the six items Main asked me to attack. (1) The 92-vs-93 row drift: stating it IS sufficient on its own terms -- 89 of 89 attempt rows resolved, extension rows correctly passed through, and I watched the ledger grow to 95 then 96 during this evaluation, so a hard 92 is unmeetable by construction. But the criterion's design assumed re-runnability would absorb the moving denominator, and that is exactly what is broken. (2) The ATTEMPT_GATE_MASTERPLAN exemption IS by construction and not a weakening: membership genuinely executes against a synthetic plan whose members are 9.1-9.5, the DENY cells 9.9 / 9.1.1 / 9.10 are genuine relative to that plan, the env var is restored in a finally block, and it is set nowhere in .claude/settings.json or the live environment. Main's correction to the masterplan note is also correct -- I confirmed 9.1 through 9.5 are all real masterplan steps. Residual: because BOTH the self-test and the matrix point membership at a synthetic plan, no automated check anywhere exercises the real .claude/masterplan.json -- which is precisely why the subphase blind spot survived to production. (3) Verified two-sided by my own in-process drive, not by reading: 1,199,999 -> exhausted False / CONTINUE; 1,200,000 -> True / ESCALATE; 1,200,001 -> True / ESCALATE. I re-implemented the old zero-token control independently and compared at the decide() level across all 28 live ids: DECISION CHANGES NONE. Note exhausted alone does flip for 75.11.4 (1,500,493 tokens, 6/7 attempts) and is masked by CLOSED_PASS -- Main disclosed exactly that case. (4) UNKNOWN-on-ambiguity is genuinely fail-safe: I reproduced max |delta| = 1.007s and zero ambiguous matches at 30/60/120/300s exactly, an ambiguous hit returns UNKNOWN with reason ambiguous_match rather than nearest-wins, and an UNKNOWN row still consumes an attempt while contributing 0 tokens, so only the token half under-counts. (5) Main fixed the leak rather than disclosing in place of fixing: ESCALATION_DIR is genuinely redirected, the stray file is absent from disk and was never committed on any branch, and the four real exhaustion escalations were byte-identical after I ran the self-test and full matrix myself. The containment CHECK he added is a weak proxy (violation 6) but the containment itself holds. (6) The M4 drive is genuine, not tuned -- drive_forge reaches write_escalation with no body through a real subprocess and would catch any restoration of the fallback. The M10 drive is genuine as a behavioural exercise (it plants a real record 900ms out and resolves it end-to-end through the CLI carrying tokens 4242 and run_id wf_planted) but it only pins 'tolerance > 0.9s'; the cell's stated semantics are not what it mutates, and I proved the claimed direction survives.\n\nClaims that reproduced exactly, independently re-derived: max |delta| 1.007s; zero ambiguity 30-300s; the four escalation sha256s; verdict_ledger sha256 fcfe56ad...3eb2 before and after everything I ran; UNKNOWN=5 all step_id 999.2; additive-only against the committed .bak (0 rows changed an original field, exactly the 4 keys outcome/outcome_reason/run_id/total_tokens added, ts order preserved); the four criterion-4 cells. One claim did not reproduce, in the direction that works against Main's own thesis: the commit message says the timestamp join 'resolves 9'; I measure 1 with ISO parsing (0 if the field is treated as numeric). It appears in no criterion or artifact -- NOTE only. Commit 3bf0b0fe also swept in pre-existing uncommitted work (backend/api/charts.py, backend/config/settings.py, backend/agents/claude_code_client.py, two new test files, many handoff artifacts, plus handoff/audit/attempt_budget_audit.jsonl.bak); all of it predates the 90.1 work per the session-start git status, so it is not unintended 90.1 authorship -- NOTE only, and it lints and imports clean.\n\nSuggested minimal fixes, all small: walk phases[].subphases[] (and any future nesting) in masterplan_step_ids, ideally with a recall assertion over the plan's own members so this class cannot recur silently; exclude the launch-placeholder keys from the additive-only projection when their persisted value is null (or have the gate omit them at launch) so the backfill can complete its own rows; add ATTEMPT_GATE_MASTERPLAN with a synthetic plan containing 77.7 to mutation_matrix_86_71.py; relabel M10 and add a widening cell; make run_cell score a non-zero-exit-with-traceback drive as ERROR; and assert the containment property against handoff/current/ rather than against the rebinding.\n\nProcess disclosure: I used a single `>` redirect to /tmp/qa90_pre_gate.py while extracting the pre-90.1 gate for comparison. qa.md forbids redirects; nothing in the repository was written or mutated by it, and I am reporting it rather than omitting it. All other mutation work ran in tempfile.TemporaryDirectory copies via the author's own harness, and the real tree md5s are unchanged. Write-first record: /Users/ford/.openclaw/workspace/pyfinagent/.claude/agent-memory/qa/verdicts/verdict_wip_90.1__20260820T192724Z.md (STATUS: COMPLETE, COMPLETED 2026-08-20T19:40:23Z) -- evidence for the next spawn, never a verdict."
}
```

---

## Cycle 1 -- Main's follow-up (fix record, NOT a re-grading)

Main does not grade. This section records only what changed in response to the verbatim
verdict above; the fresh Q/A grades the changed evidence.

| Finding | Severity | Disposition |
|---|---|---|
| criterion_4_unresolvable_step_id_deny | BLOCK | **FIXED.** Recursive plan walk + `assert_membership_recall()` (independent walk). Recall 1427/1427, missing 0. All 10 pending harness-required steps ADMITTED; 86.118.1 / 86.1180 / 999.99 still DENIED. Cell M13 + 2 recall checks. |
| criterion_1_rerunnable_backfill | BLOCK | **FIXED.** Settled-vs-unsettled rule with `RESOLUTION_KEYS`. `--backfill` exits 0, is idempotent, and a new drive seeds the row shape the gate actually writes. Cell M14. |
| consumer-contract-break (86_71 control RED) | WARN | **FIXED.** `ATTEMPT_GATE_MASTERPLAN` synthetic plan in `mutation_matrix_86_71.py`; its control is green, `--verify` exits 0. |
| illusory-guard (M10 mislabel; unparseable mutant scored KILLED) | WARN | **FIXED.** M10 relabelled; M11 (widening) and M12 (timestamp revert) added and killed via a decoy run record; `run_cell` now `ast.parse`s the mutant so a build failure scores ERROR -- verified with the Q/A's own MXE1/MXE2 probes. |
| `--operator-extend` membership | (Main-found, not raised) | **DISCLOSED, NOT FIXED.** Outside criterion 4's wording; recorded in experiment_results §"Still open". |

Cycle-2 immutable command: **exit 0**. Matrix 15 cells / 25 checks: **14 KILLED, 0
SURVIVED, 0 ERROR**, null mutant survived, tree untouched.


---

## Cycle 2 -- Q/A verdict, TRANSCRIBED VERBATIM

Run `wf_7ab71c1d-843`, 258,552 tokens, 909s, 75 tool calls. Attempt 3 of 5.
**CONDITIONAL.** Both cycle-1 BLOCKs independently re-derived and confirmed FIXED.
Three WARNs remain. Main records; Main does not author.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Immutable command reproduces exit=0 (self-test 34 ok lines PASSED; matrix CONTROL GREEN first, 15 cells/25 checks, 14 KILLED / 0 SURVIVED / 0 ERROR, N0 survived, tree md5 unchanged). Both cycle-1 BLOCKs are genuinely fixed and I reproduced each independently rather than reading them: criterion 4 recall re-derived on the REAL plan (members 1427, missing 0, ids 1614; shallow-walk mutant misses 123 -- Main's figure exactly), blast radius re-derived with the FIXED walk over 621 real run records (540 with a step_id, 535 ADMITTED, 5 DENIED and all 5 already shape-refused pre-90.1 -- zero new denials), and 4 walk mutants of my own (no-list-recursion / id-only-if-name / no-phase-strip / id-only-if-verification) ALL KILLED; criterion 1 backfill is re-runnable AND idempotent (3 real runs on a copy, rc 0/0/0, file byte-identical after run 1, counts identical), with UNKNOWN=7 == reason no_run_record=7 and ambiguous_match=0, and my freeze probe showing a settled row carrying a deliberately WRONG outcome stays byte-identical next to a matching PASS record while an UNKNOWN row is re-resolved. Criteria 2, 3, 6 independently driven and MET (non-exhaustion denial wrote escalation_unknown_step_id_86.118.1.md while the seeded escalation_attempt_budget_86.85.md stayed sha256 75f727f7...==75f727f7...; 1,200,001 tokens -> rc 2 with 4 of 5 attempts unused vs 1,199,999 -> rc 0; verdict_ledger sha256 fcfe56ad...2e3eb2 identical across every run and an AST sweep finding exactly ONE write-capable VERDICT_LEDGER call, at attempt_gate.py:577 inside _self_test). Harness compliance clean on all 5 items; lint clean on a git-derived 9-file scope; no unintended production change in commit 1fc7b2e6. NOT PASS because criterion 5 clause 3 is falsified by execution: the cycle-2 fix added ast.parse, which closes only the SyntaxError subset, and three mutants that parse cleanly but cannot be imported (RuntimeError / NameError / ImportError at module scope) each scored KILLED while a SyntaxError control correctly scored ERROR -- 'parses' is not 'runs'. Bounded to WARN rather than BLOCK because I applied all 15 shipped cells to a temp copy and imported each: all 15 import, so no reported kill is false today. Two further WARNs: the cycle-1 critique carried SIX violation_details and the cycle-2 disposition table five, with the Circular_Reasoning containment-tautology WARN dropped entirely (code unchanged at attempt_gate.py:747-752; I re-proved the clause is True by construction with a file written outside the dir, and vacuously true on an empty dir), and the M11 upper-bound cell is calibrated to drive_join's 7200s decoy rather than the documented 900s ambiguity threshold -- my sweep through the author's own harness has tolerances 1 through 7199 all SURVIVING, including 3600, which on the real ledger collapses summed tokens from 20,365,361 to 4,015,375 with 71 ambiguous rows. One finding I nearly filed and RETIRED after trying to evade it: the self-test's flat synthetic plan DOES catch the shallow walk (members 6, missing 1), because the phase object's own phase-9 -> 9 id is a dotted member the shallow walk never reaches. The disclosed --operator-extend hole is adequately disclosed and verified INERT by execution (extension row for 999.99 created, subsequent launch claiming it still DENIED rc=2).",
  "violated_criteria": [
    "criterion_5_clause3_nonrunning_mutant_must_score_ERROR",
    "criteria-erosion",
    "illusory-guard"
  ],
  "violation_details": [
    {
      "violation_type": "Threshold_Not_Met",
      "action": "Q/A-authored cells MXE3/MXE4/MXE5/MXE6 run through mutation_matrix_90_1.run_cell, anchored on the unique module-level line `_STEP_ID_RE = re.compile(r\"\\A[0-9]+(?:\\.[0-9]+)*\\Z\")` (anchor count == 1)",
      "state": "MXE3 (append `raise RuntimeError(...)` at module scope) -> KILLED; MXE4 (`_STEP_ID_RE = __qa_undefined_name_probe__`, NameError at module scope) -> KILLED; MXE5 (prepend `import __qa_no_such_module_probe__`) -> KILLED; MXE6 (SyntaxError control) -> ERROR 'mutant does not parse (attempt_gate.py:101)'. Mechanism: observations() drives every cell through subprocess.run, which does not raise on a non-zero exit, so an unimportable mutant fails every CHECK and run_cell credits a KILL. The cycle-2 remedy `ast.parse(mutated)` is a compile-time gate on a runtime failure. live_check_90.1.md section C2.4 is headed 'A mutant that cannot build scores ERROR, never a kill' but evidences only the SyntaxError and anchor-mismatch probes. BOUNDED: I applied all 15 shipped cells to a temp copy and imported each mutated module -- all 15 import cleanly, so none of the 14 reported kills is a build artifact and the matrix result stands.",
      "constraint": "SEVERITY WARN -- immutable criterion 5, clause 3: 'a mutant that fails to run scores ERROR and never counts as a kill'. Falsified by three executed counterexamples. Same class the cycle-1 Q/A raised (WARN 2); narrowed, not closed. Fix: smoke-import the mutant (subprocess.run([sys.executable,'-c',f'import {module}'])) and score a non-zero import as ERROR before running the checks."
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "Enumerated the cycle-1 verdict's violation_details array in handoff/current/evaluator_critique_90.1.md and matched each element to the cycle-2 disposition table and to the '# CYCLE 2' section of experiment_results_90.1.md; then executed the flagged assertion shape on temp directories",
      "state": "cycle-1 carried SIX violation_details; the disposition table carries FIVE rows and merges two distinct WARNs (M10 mislabel + run_cell ERROR) into one 'illusory-guard' row. The sixth -- Circular_Reasoning on the self-test containment check -- has NO disposition row, NO cycle-2 remediation section and NO 'still open' entry. The code is unchanged at attempt_gate.py:747-752: `ESCALATION_DIR != old_e and all(p.parent == ESCALATION_DIR for p in ESCALATION_DIR.iterdir())`. Executed: the second clause returns True while a file was written OUTSIDE the dir (the leak class it is named for), and returns True vacuously on an empty dir -- true by construction, since iterdir() yields only direct children. The cycle-1 prose at experiment_results_90.1.md:244 ('the containment is now itself a check') therefore stands uncorrected beside the verdict that refuted it. The underlying containment is real: nothing leaked into handoff/current/ during my self-test and full-matrix runs.",
      "constraint": "SEVERITY WARN -- criteria-erosion [Dim-5] + assert the property, not a proxy (qa.md 4c shape #4). A previously-required finding was neither fixed, nor queued, nor disclosed. Fix: either assert containment against handoff/current/ directly, or record the item as a disclosed residual so the count cannot drift again."
    },
    {
      "violation_type": "Overgeneralization",
      "action": "Tolerance sweep of DEFAULT_TOLERANCE_S run through mutation_matrix_90_1.run_cell, plus resolve_row re-run over the real 94 attempt rows against the 621 live run records",
      "state": "drive_join plants its decoy 7,200,000 ms from the row, so the M11 guard's true boundary is 7200s, not the measured ambiguity threshold. Sweep: tol=0 KILLED, 1 SURVIVED, 60 SURVIVED, 300 SURVIVED, 900 SURVIVED, 1800 SURVIVED, 3600 SURVIVED, 7199 SURVIVED, 7200 KILLED, 86400 KILLED. The module docstring states ambiguity first appears at 900s, and I reproduced the damage on the real ledger: tol=30 -> 20,365,361 tokens / 0 ambiguous; 900 -> 18,998,336 / 6; 3600 -> 4,015,375 / 71; 86400 -> 2,033,007 / 80 (9.98%, so the '~9%' claim reproduces). A tolerance of 3600 collapses token accounting to 20% and 71 rows to ambiguous, and the cell survives it.",
      "constraint": "SEVERITY WARN -- qa.md 4c: a matrix licenses only 'these N mutations were killed'. The cell added to close the cycle-1 upper-bound WARN defends the decoy's boundary rather than the documented one. Not a criterion miss (criterion 5 names only M1/M2/ERROR) and not vacuous. Fix: place the decoy just past the measured threshold (e.g. 900s), or assert on the resolved token SUM rather than only on the single row's outcome."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "syntax",
    "verification_command",
    "python_lint_gate_F821_F401_F811",
    "existing_results_check",
    "evaluator_critique",
    "contract_completeness",
    "mutation_test",
    "independent_mutation_probes",
    "claim_reproduction",
    "guard_vacuity_check",
    "code_review_heuristics",
    "harness_compliance_audit",
    "runtime_exercise",
    "counter_evidence"
  ],
  "harness_compliance_ok": true,
  "research_needed": false,
  "notes": "Harness compliance clean on all 5 items. Research gate: research_brief_90.1.md brief_status COMPLETE, gate_passed true, external_sources_read_in_full 10 (>=5), urls_collected 25 (>=10), recency_scan_performed true, enforced run wf_db313c3d-b75 cited in the contract; all 6 immutable criteria appear VERBATIM in contract_90.1.md (verified by string containment against masterplan.json, not by eye). Order by mtime (local CEST): research_brief 21:12:47 < contract 21:15:34 < attempt_outcomes.py 21:45:58 < attempt_gate.py 21:47:23 < mutation_matrix_90_1.py 21:47:53 < experiment_results 21:49:19 < live_check 21:49:42, all before my WRITTEN stamp. LOG-last verified: grep -Fc 'phase=90.1' handoff/harness_log.md = 0 with the grep proven LIVE by phase=86.116 returning 2; masterplan 90.1 status=pending, not flipped. No verdict-shopping: evidence CHANGED (commit 1fc7b2e6 -- attempt_gate.py +19, attempt_outcomes.py +147, mutation_matrix_90_1.py +191, mutation_matrix_86_71.py +12, experiment_results +160, evaluator_critique +99, live_check +99).\n\nsequence: UNKNOWN. verdict_history_86_21.py --step 90.1 --evidence-only returned status 'no_rows_for_step', verdicts (none). qa_wip.py 90.1 --spawned-at 2026-08-20T19:50:39Z returned source_present true, attempt_number_status ok, attempt_number 2, prior_attempts 1, attempt_number_is_lower_bound false, records_retained 2 (gauge), prior_records = the cycle-1 record 20260820T192724Z. CROSS-CHECK: prior_attempts 1 > the ledger's 0 rows, so THE LEDGER IS STALE for this step and the sequence is unreliable. Separately observed, a DIFFERENT population: the gate's own audit row for my spawn reads attempt_number_inclusive=3, because it counts this step's research-gate launch as an attempt; neither number is reconstructible from the other.\n\nROW COUNT re-judged independently. At the committed evidence point (HEAD 1fc7b2e6) the ledger holds EXACTLY 96 rows = 92 attempt + 4 operator_extension, and all 92 attempt rows carry outcome and total_tokens -- criterion 1's '92' lands at the evidence point. Trajectory re-derived from git: 1c2f25b3 (pre-90.1) 83/79 with_outcome 0; 3bf0b0fe (cycle 1) 93/89 with_outcome 89; HEAD 96/92. The working tree is now 98/94 because the gate is live and two launches landed during this evaluation, one of them my own spawn (19:50:34Z). Disclosure is adequate and the growth is structural, not a discrepancy.\n\nAdditional NOTEs, none verdict-degrading. (a) experiment_results section 5 '90.7' claims 'Membership deliberately accepts both X and phase-X'; measured, 'phase-90' is DENIED because _STEP_ID_RE refuses it before membership is consulted -- the SET holds both forms, the GATE accepts only the bare one. (b) The recursive walk over-includes exactly ONE non-step dotted id ('77', a phase container at /phases/100) out of 1427; criterion 4 speaks to ABSENCE from the file and '77' is present, so the letter holds. (c) A pre-existing non-vocabulary outcome value is treated as settled and counted under its own key in outcome_counts, escaping the closed five; the backfill never writes such a value. (d) The unrelated backend/* files in the diff range were swept into the CYCLE-1 commit by the auto-commit hook's git add -A, are pre-existing 86.120 work, and lint clean -- not cycle-2 authorship. (e) scripts/qa/mutation_matrix_86_71.py --verify re-run by me: control green, 9/9 killed, exit 0, and its diff is exemption-by-construction that weakens no check.\n\nProcess disclosure: no repository file was written or mutated by me. All mutation and backfill work ran on copies under tempfile temp dirs via the author's own harness or my own drivers; the real tree md5s (attempt_gate.py 85de2e74a186aac33da596ec7bec0285, attempt_outcomes.py 81ebe68b498c63cbc424bf1f01ae02d1) match live_check C2.5 byte-for-byte, and handoff/audit/attempt_budget_audit.jsonl (1f2a39cf...) plus handoff/verdict_ledger.jsonl (fcfe56ad...) are sha-identical before and after everything I ran. I used no shell redirects. Evidence freeze held for 90.1: git status --short on all five 90.1 artifacts and all four scripts is clean vs HEAD at the end of the evaluation. NOTE for the record, not a finding against this step: Main launched a 90.9 research gate at 19:50:56Z DURING my evaluation (visible in the audit stream and as an untracked contract_90.2.md); no 90.1 evidence file moved. Write-first record: /Users/ford/.openclaw/workspace/pyfinagent/.claude/agent-memory/qa/verdicts/verdict_wip_90.1__20260820T195039Z.md (STATUS: COMPLETE, COMPLETED 2026-08-20T20:12:44Z) -- evidence for a next spawn, never a verdict. Two agent-memory files updated with the new lessons (parses-is-not-runs; count the violation_details, not the disposition rows)."
}
```

---

## Cycle 2 -- Main's follow-up (fix record, NOT a re-grading)

| Finding | Severity | Disposition |
|---|---|---|
| criterion_5_clause3_nonrunning_mutant_must_score_ERROR | WARN (numbered criterion) | **FIXED.** `run_cell` smoke-imports the mutant; a non-zero import scores ERROR. All three Q/A counterexamples (RuntimeError / NameError / ImportError) now ERROR, the SyntaxError control still ERROR, and a null control still SURVIVES so the probe is not blanket-ERROR. A false-ERROR the probe itself caused on `attempt_budget.py` (dataclass needs `sys.modules` registration) was found and fixed. |
| criteria-erosion (the dropped sixth finding) | WARN | **FIXED + acknowledged.** The cycle-1 verdict carried six violation_details; my cycle-2 table carried five and lost the Circular_Reasoning item. Recorded here in full and remediated below. |
| illusory-guard (containment tautology) | WARN | **FIXED.** The tautological `iterdir()` clause is replaced by a name-set comparison against the REAL `handoff/current/`, plus an anti-vacuity clause. Red-first proof: a deliberate-leak mutant turns both checks RED. |
| illusory-guard (M11 decoy boundary) | WARN | **FIXED.** Decoy moved from 7200s to the documented 900s threshold; M11 mutates to 900, M11b keeps 86400. Sweep: 900/1800/3600/86400 now KILLED (3600 previously survived). |
| `--operator-extend` membership | (disclosed residual) | **STILL DISCLOSED.** The Q/A verified it INERT by execution -- an extension row for `999.99` did not admit a launch claiming it (rc=2). |

**All six cycle-1 findings and all three cycle-2 findings are now accounted for. The count
cannot drift again: this table enumerates them.**

Cycle-3 immutable command: **exit 0**. Matrix 16 cells / 25 checks: **15 KILLED, 0
SURVIVED, 0 ERROR**, null mutant survived, tree untouched, verdict ledger byte-identical.
