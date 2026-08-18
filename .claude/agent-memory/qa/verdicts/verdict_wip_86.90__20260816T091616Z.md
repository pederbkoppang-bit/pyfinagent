STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.90
WRITTEN: 2026-08-16T09:16:16Z

# Q/A cycle-3 evaluation of step 86.90 (write-first record)

## Attempt / sequence evidence
- `qa_wip.py 86.90 --spawned-at 2026-08-16T09:16:16Z`: source_present=True, attempt_number=3,
  prior_attempts=2, attempt_number_status='ok', attempt_number_is_lower_bound=True,
  records_retained=3 (GAUGE), records_pruned_known=None, is_verdict=False.
- `verdict_history_86_21.py --step 86.90 --evidence-only`: status=`no_rows_for_step`, verdicts=(none).
- CROSS-CHECK: attempt_number (3) > ledger count (0). **LEDGER IS STALE**; sequence per the
  ledger = UNKNOWN. Main's advisory [C, C] recorded as advisory only. No aggregate derived.

## A. HARNESS COMPLIANCE -- 5/5 CLEAN (verified on disk, not from the contract)
1. research gate: brief 43,014 B, envelope on disk `brief_status: COMPLETE`, gate_passed true,
   external_sources_read_in_full 12 (>=5), urls_collected 45 vs 46 distinct URLs actually in the
   file (no over-claim), recency_scan_performed true + a real "## Recency scan (last 2 years)"
   section at line 208, coverage.dry true. mtime 09:59:05 < contract 10:01:10. OK
2. contract-before-generate: contract 10:01:10 < qa-verdict.js 10:42:14 < guard 11:09:19 <
   experiment_results 11:14:29. ORDER HOLDS.
3. experiment_results + live_check present.
4. log-last: `grep -cF "phase=86.90" handoff/harness_log.md` = **0**; masterplan 86.90 = pending.
5. no verdict-shopping: evidence CHANGED between 98c5b6ab (10:50:45) and 468c7908 (11:14:29) --
   9 files, +602/-59. Documented cycle-2 flow, not a re-ask.

## B. DETERMINISTIC
- IMMUTABLE CMD -> `parses`, **exit 0**.
- verify_prompt_render_86_90.mjs -> ALL GREEN: 83 passed, 0 failed (exit 0)
- verify_research_gate_workflow.mjs -> ALL GREEN: 124 passed, 0 failed (exit 0)
- verify_escalation_86_78.mjs -> ALL CHECKS PASS (exit 0)
- verify_rail_retry.mjs -> ALL GREEN: 38 passed, 0 failed (exit 0)
- verify_workflow_args_boundary.mjs -> **FAILED: 84 passed, 3 failed**, exit 1 -- reproduces the
  figure live_check quotes. Pre-existing, filed 86.92.
- LINT: scope derived from `git diff --name-only a21a5889^ 468c7908 -- '*.py'` = 2 files.
  FIRST ATTEMPT WAS A FALSE PASS (zsh does not word-split `$FILES`; ruff got one newline-joined
  path, printed "All checks passed!", exit 0 while linting NOTHING). Redone via `| xargs`:
  **All checks passed! exit=0** on both files, existence asserted first.
- 1b frontend / 1c UI / 1d backend smoke: NOT TRIGGERED. The step's 3 commits touch 19 files,
  none under frontend/ or backend/. Uncommitted tree edits to backend/api/sovereign_api.py and
  5 frontend files are mtime **2026-08-14**, appear in none of the step's commits -- pre-existing
  peer work. No UI claims in any artifact (grep = 0).

### Pre-existing-RED claim (cycle-3 finding 4) -- REPRODUCED, all three legs
- `git log -S'carries NO brief_status marker' -- research-gate.js` -> `d3bb1dfb 2026-08-10` (one commit)
- fixture `research_brief_86.17.md` mtime 9 aug 17:24, `grep -c brief_status` = **0**
- `git diff a21a5889 98c5b6ab -- research-gate.js | grep -c enforceGate` = **0**; the cycle-1
  "occurrence" is the hunk header `@@ ... function enforceGate(env, verification, opts) {`
RULING: the replacement instrument DOES establish the claim; the retired worktree argument did not.

## C. FIGURE RE-DERIVATION (judge_these D)
- 583 run records REPRODUCES when restricted to mtime < the cycle-3 commit (593 now; +10 is
  corpus growth incl. my own spawn).
- `('qa-verdict','object')` args = **31** -- reproduces EXACTLY. ("409 strings" measures 411
  all-workflow pre-commit; 2-record drift.)
- COERCED CENSUS, independent superset instrument (first user message of all 1,394
  `agent-*.jsonl`, header line whose value is exactly `[object Object]`): 23 runs, of which the
  23rd is `wf_4588d8a7-e70` = Main's own declared pre-fix probe. **SYMMETRIC DIFFERENCE vs the
  22-row table = 0** (known-member recall complete). 6 also lost `extra` -- matches.
- VERDICT SPLIT re-derived from the 22 records' own `result.verdict`:
  **7 CONDITIONAL + 7 NO_VERDICT + 4 PASS + 4 FAIL = 22.** Matches. Nine step-ids match.
  One drop is `status=killed`; Main classified by ABSENCE OF VERDICT -- correct instrument.
- §8 zeros: strict header scan gives OBJECTIVE coerced 0, INTERNAL SCOPE coerced 0. A LOOSE
  substring scan gives 4 and 4 -- prose contamination from 86.90's own documents (the trap
  Main documents in §6). Denominators 75/78 reproduce under the broader jsonl glob.
- `[3]` = 52 assertions, 4 CONTROL, 48/2 scripts = **12 cases x 2 scripts** -- reproduces.
- 84/3 for the args-boundary checker -- reproduces.

### THE ONE FIGURE THAT DOES NOT REPRODUCE (the fourth instance Main asked me to look for)
`experiment_results_86.90.md:15` (the FILES-CHANGED table, §1) still reads
`NEW, **78** assertions`, while :180, :488, :518 and :544 all state **83**, and :544 is the
cycle-3 remediation row that claims "Every capture REGENERATED ... Guard is now 83".
The live count is 83. A correction that did not REPLACE everywhere it was carried -- the same
class as cycle-3 finding 2, and the exact rule 86.94's own criterion 5 states. WARN.

## D. NEW FINDING -- criterion 5 is not closed for the `criteria` CONTAINER
`qa-verdict.js:278` `const criteria = Array.isArray(a.criteria) ? a.criteria : []`, and :346-348
substitute `'(none passed in args -- read them from .claude/masterplan.json ...)'` when empty.
The ELEMENTS were routed through `renderArgField` by this step (the removed line proves the
author edited exactly this expression); the CONTAINER was not.
MEASURED by driving the real shipped script, CONTROL FIRST:
  ARRAY (control)      -> criterion text present, no fallback, 1 spawn
  STRING               -> criterion text ABSENT, fallback substituted, 0 logs, 0 throw, 1 spawn
  OBJECT               -> same
  OBJECT numeric keys  -> same
So a present-but-non-array `criteria` -- trivially JSON-reachable, unlike all six holes found so
far -- is SILENTLY REPLACED BY A PLACEHOLDER on the single most load-bearing field in the prompt
(qa.md: "The step's immutable success criteria ARE the rubric"). `criteria` is in
KNOWN_ARG_KEYS, so the unknown-key warning does not fire either.
BOUNDING IT HONESTLY: measured 384 list / 1 absent / **0 wrong-shaped** across the whole
run-record corpus -- latent, not live; the fallback still points the evaluator at masterplan.json,
so harm is bounded unlike `[object Object]`; and the container guard is PRE-EXISTING, not a
regression introduced here. WARN, not BLOCK.
Not raised by cycle 1 or cycle 2 (grep = 0 in all three artifacts).

## E. judge_these
A. **86.94 criterion edit -- LEGITIMATE on the evidence, with a NOTE.** Verified by execution:
   86.94 entered the masterplan at 98c5b6ab (10:50:45 today) and the pin left at 468c7908
   (11:14:29, 24 min later, same session); ZERO handoff artifacts, ZERO qa_wip records, ZERO
   harness_log rows for 86.94 -- "no cycle has run" is measured, not asserted. The old text
   embedded a false premise (a criterion cannot be met by correct work); the new text demands a
   live differential with no pinned figures, i.e. MORE demanding and answerable. NOTE: the
   doctrine "a filing no cycle has run against may be repaired" is nowhere in CLAUDE.md, whose
   never-amend rule is unqualified. It should be codified or operator-ratified, else the
   exception is author-defined.
B. **The five per-cell CONTROLs are REAL.** Each requires `expect()` FALSE on the unmutated
   source and TRUE on the mutant -- the two answers must DIFFER. Verified independently: all 5
   anchors unique (1/1/1/1/1) and all 5 mutants BUILD. DECISIVE TEST: I replayed the cycle-2 M3
   form (`... + violation + ''\n return '(unrenderable)'\n void ('`) against the current source
   under the cycle-3 harness -- it now scores `UNSCORABLE: the mutant did not build (Invalid or
   unexpected token)` and the check FAILS. Under the cycle-2 harness it scored KILLED. The
   repair is real and measured, not accepted.
C. **Leaving the Proxy unfixed is RIGHT, not convenient.** I reproduced the hole: the walk saw
   call1/call2 and the rendered JSON carried `"real": "call4"` with NO throw. I then tested the
   reachability bound rather than accepting it: `JSON.parse` cannot produce a Proxy
   (util.types.isProxy false on the JSON path), and `classifyArgs` accepts only a JSON string it
   parses or a runtime object that is itself JSON-derived. Residual, honestly stated in the
   artifact: the bound rests on an assumption about the runtime the script cannot enforce.
D. Answered above: one non-reproducing figure (the 78 at :15).

## F. CANDIDATE FINDING I RETIRED BY MEASUREMENT
I hypothesised that `research-gate.js` `auditClass = a.audit_class === true` silently reads
`"true"`/`1` as false and so disables the loop-until-dry requirement. **REFUTED**: driving the
real script with audit_class as `true`, `"true"`, `1` and ABSENT all produced
`gate_passed: false` with `"audit-class step but coverage.dry is not true"` -- enforceGate
recomputes audit-class from the AGENT's returned `coverage`, not from `args`. Retired rather
than reported.

## G. CRITERIA
1 MET · 2 MET · 3 MET · 4 MET · 5 **NOT FULLY MET** (D above) · 6 MET · 7 MET
Criterion 7 proven BY EXECUTION: unrenderable evidence -> throws, `spawns: 0`, return
`undefined` (no verdict at all); renderable -> 1 spawn, verdict returned. The only diff line
matching verdict-logic patterns is the new `log()` warning string.

WORST-OF-N LENSES: correctness = CONDITIONAL (D), does-it-reproduce = CONDITIONAL (the 78),
scope-honesty = PASS (every bound in the artifact is stated, incl. the census floor and the
unfixed sixth hole). min = CONDITIONAL.

COMPLETED: 2026-08-16T09:52:41Z
