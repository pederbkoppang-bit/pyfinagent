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
