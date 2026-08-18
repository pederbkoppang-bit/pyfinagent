STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.78
WRITTEN: 2026-08-17T13:36:59Z
COMPLETED: 2026-08-17T13:54:23Z

# Q/A cycle-3 evaluation of step 86.78 (consequence blinding)

Spawn: Workflow rail (runId wf_5038e340-6ca), post-restart session, 2026-08-17.
Instructions read: .claude/agents/qa.md IN FULL (runtime read, live).

## Prior-attempt evidence (gathered, NOT applied)
- `qa_wip.py 86.78 --spawned-at 2026-08-17T13:36:59Z`: source_present=True,
  attempt_number=4 (status ok, is_lower_bound=True), prior_attempts=3,
  records_retained=4 (GAUGE). Priors: 20260814T090443Z / 090544Z / 101054Z.
- `verdict_history_86_21.py --step 86.78 --evidence-only`: status
  `no_rows_for_step`; verdicts (none).
- CROSS-CHECK: attempt_number 4 > ledger 0 => LEDGER IS STALE for this step.
  sequence: UNKNOWN from the governing source. Secondary only (harness_log):
  Cycle 1 CONDITIONAL, Cycle 2 CONDITIONAL, Cycle 2 NO_VERDICT (rail drop).
  Not guessed, not aggregated.

## A. Harness compliance -- CLEAN
- research_brief_86.78.md (30,269 B, 14 aug 09:47) < contract (10:56) <
  experiment_results (17 aug 15:25) / live_check (17 aug 14:51). Order OK.
- masterplan 86.78 status = pending (not flipped) -- log-last OK.
- Evidence CHANGED since the last verdict (no verdict-shopping): qa.md scrub,
  --evidence-only mode, computed recorder, census section 10, commits
  9b4d5281 / 77f15b4d / 54c6ec51.
- NOTE: criterion 5 in MY spawn prompt is a PARAPHRASE ("...when its absence is
  load-bearing") of the masterplan verbatim ("...when its absence was grounds to
  delete the rubric"). The CONTRACT quotes the masterplan correctly. Graded
  against the masterplan.

## B. Deterministic
- IMMUTABLE COMMAND `bash -c 'source .venv/bin/activate && node --check
  .claude/workflows/qa-verdict.js && echo parses'` -> `parses`, EXIT=0. GREEN.
- ruff F821,F401,F811 over DERIVED scope (git diff HEAD *.py + the step's own
  commits' *.py = backend/api/sovereign_api.py, scripts/qa/verdict_history_86_21.py):
  "All checks passed!" exit=0.
- node --check GREEN on qa-verdict.js, research-gate.js, verify_escalation_86_78.mjs,
  mutation_matrix_86_78.mjs, verify_prompt_render_86_90.mjs.
- frontend gate (fires because the diff touches .claude/agents/qa.md):
  `npx tsc --noEmit` EXIT=0; `npx eslint src` EXIT=0 (0 errors, 55 warnings).
- No unintended production change in the step's commits: fedcffff/2cade52f/
  9b4d5281/61e359b4/ccb29804/140f1ac3/77f15b4d/54c6ec51 touch only
  .claude/workflows/*.js, .claude/agents/qa.md, scripts/qa/*, docs/adr/*, handoff/*.
  Uncommitted frontend/* + backend/api/sovereign_api.py are pre-existing, other work.

### FINDING B1 (BLOCKING): the step's OWN checker is RED at evaluation time
`node scripts/qa/verify_escalation_86_78.mjs` -> EXIT=1
```
  checks run : 51   (cardinality floor 49)
  failed     : 1
    FAIL escalation is NESTED in the return, not spread into it -- the mutation
         that flattens it must not survive
  CHECKER RED
```
Root cause DERIVED: `verify_escalation_86_78.mjs:158-161` asserts the LITERAL
`SRC.includes('const merged = { ...verdict, escalation }')`. Commit **77f15b4d**
(2026-08-17 15:02) changed `.claude/workflows/qa-verdict.js:835` to
`const merged = { ...verdict, escalation, research_routing }`.
`git log -S` on both literals -> both resolve to 77f15b4d. The SAME commit that
shipped this step's cycle-3 criterion-5 fix broke this step's own checker.
The artifact (mtime 15:25, i.e. 23 min AFTER the break) reports greens for a
family -- prompt-render 136 / research-gate 124 / args-boundary 96 -- that
EXCLUDES both scripts this step built.

### FINDING B2 (BLOCKING): criterion 6's matrix ABORTS on its own control
`node scripts/qa/mutation_matrix_86_78.mjs` -> EXIT=1, **ZERO cells executed**:
```
subject : .claude/workflows/qa-verdict.js  sha256[:16]=62e682b9f6aad93d
checker : scripts/qa/verify_escalation_86_78.mjs
[CONTROL] unmutated checker -> exit 1
ABORT -- the control is RED, so no kill below would mean anything.
```
Criterion 6 requires the control "observed GREEN first". Today it is RED and no
cell runs; the recorded "13/13 killed" does not reproduce through the shipped
instrument.

### The PROPERTY still holds -- so B1/B2 are instrument defects, not product defects
- qa-verdict.js:835 still NESTS escalation; runtime `leaked` throw :839-843 intact;
  the checker's own 2nd conjunct `!/\{[^}]*\.\.\.escalation/` passes.
- INDEPENDENT REPRODUCTION (mine, in-memory via data: URL import of the real
  source span -- tree never written): CONTROL GREEN FIRST on an 18-check battery,
  then **12 mutations, 12 KILLED**, each with a named killing assertion:
  MX1 threshold off-by-one / MX2 no-reset-on-PASS / MX3 fail-open-with-zero /
  MX4 arm-on-any-verdict / MX5 NO_VERDICT-resets / MX6 budget-fails-open /
  MX7 burden-removed / MX8 override-defaults-applied / MX9 input-not-echoed /
  MX10 recorder-neutered / MX11 recorder-hardcoded-false / MX12 recorder-over-broad.
  => the relocated counter IS genuinely mutation-resistant.
- The recorder half is ALSO covered by a currently-GREEN instrument:
  verify_prompt_render_86_90.mjs [8] -- 8-recorder-neutered KILLED,
  8-recorder-hardcoded-false KILLED, with 4 control drives. 136 passed, 0 failed.

## C. Criteria
- **C1 MET.** Census independently REPRODUCED with the shipped POSITIONAL_CLAIM_RE
  over the stated population rule: 418 qa spawn prompts (was 413 at 14:49 -- growth),
  9 positional hits (was 8). **KNOWN-MEMBER RECALL 8 of 8, missing []**. qa.md scrub
  verified live: probes "return **FAIL** instead of a third" / "recommend operator
  escalation" / "You MUST state the derived attempt number" all = 0.
- **C2 MET.** Direction (leniency, ERRJ 0) stated; asymmetry reasoned; both framed
  PASSes (wf_20a27baa 2026-08-11, wf_4e01adc8 = 86.71 close) QUEUED for a re-grade
  decision in 86.93 with mitigation stated. Verified wf_4e01adc8 ran a PRE-recorder
  script (script_has_RECORDER=False), so its told=false was the hardcode -- exactly
  as the artifact discloses.
- **C3 MET, verified on LIVE run records.** 86.84 cycle 10 (wf_6339897a-3c4, cc=1)
  and cycle 11 (wf_f5aac4c6-176, cc=2): `attempt_number` arg absent on BOTH; the
  envelope computed consecutive_conditionals from args.verdict_sequence alone.
  Across all 62 records carrying an envelope, `attempt_number` is None in every one.
- **C4 MET, verified on LIVE run records.** 86.85 cycle 11 (wf_8b8d1bb5-245):
  cc=3, verdict FAIL -- the loop terminated; the preceding cycle (wf_aa138724-102)
  shows cc=2 arm=True. And `verdict_unmodified` is True on **62 of 62** records.
  My MX1-MX5 kills cover the semantics cells.
- **C5 MET with residual (F4).** burden_on shipped and named on the departing party;
  override/override_reason shipped defaulting to null; the double standard answered by
  putting the field on the CALLER side (judge schema is additionalProperties:false);
  sourcing gap stated honestly. Recorder converted from a hardcoded false to a
  COMPUTED measurement -- corroborated live: wf_9bc5bc53-351 (86.72 c1, 15:17) has
  script_has_RECORDER=True and returned told=false on a clean prompt.
- **C6 substantively MET, instrument DOWN (B1/B2).**

### FINDING F3 (measured): the forward-discipline claim is falsified by THIS spawn
experiment_results tail: "Caller discipline forward: sequence-as-data only; no
positional prose in evidence/extra (the 86.21 cycle-7 spawn ... is the first under
the discipline and carries none)."
My census flagged a NINTH member the 14:49 census could not have seen:
`wf_5038e340-6ca` (2026-08-17 15:44) -- **this evaluation's own spawn**, step_id 86.78,
matched string `PASS-or-FAIL`, from the EVIDENCE section:
  "(C4) the rule TERMINATED a loop today: 86.85 cycle 11 was bound PASS-or-FAIL by
   three consecutive CONDITIONALs and returned FAIL"
Driving the shipped detector on that verbatim sentence:
`judge_was_told_consequence = true, evidence = "PASS-or-FAIL"`.
That sentence delivers the threshold's VALUE (three), UNIT (consecutive
CONDITIONALs) and OUTCOME (bound PASS-or-FAIL) -- precisely what qa.md withholds.
Per 2604.15224 (ERRJ 0.000) I cannot certify my own grading was unaffected, and I
disclose it rather than claim immunity.

### FINDING F4 (WARN): `escalation.override` is structurally unsettable
`enforceEscalation` reads only opts.max_attempts / opts.caller_text /
opts.attempt_number. Driven: `{override:'operator accepted'}` and
`{override:true, override_reason:'because'}` BOTH yield override=null.
So safeguard 2's field has no writer; the checker's "an override SLOT exists" +
"defaults to null" pass on a field that can never be anything but null. The claim
"the CALLER records it here, following research-gate.js's self_report_disagreed
pattern" overstates -- research-gate.js POPULATES its field. The workable channel
is the ledger's free-text `note` (62 of 76 rows populated, 0 with any override key).

### FINDING F5 (NOTE): the recorder's false-negative surface is undisclosed
POSITIONAL_CLAIM_RE is a fixed literal list. Driven, 6 realistic rewordings all
MISSED ("this is your third consecutive CONDITIONAL", "two prior CONDITIONALs are
on record", "a further CONDITIONAL would terminate the loop", "the budget is nearly
exhausted", "you have graded this step four times already", "this cycle is the last
one before escalation"); the literal control "attempt 5 of 5" DETECTED. So the
census's 8/413 and the recorder's boolean are LOWER BOUNDS under paraphrase.

## VERDICT: CONDITIONAL
All six criteria substantively MET on independently reproduced evidence; blocked
on the step's own two shipped verification assets being non-functional (B1/B2 --
one stale literal at verify_escalation_86_78.mjs:159), plus F3/F4/F5.
No write was blocked. No unintended production change.

### FINDING B3: the live_check's own stated reproduction recipe does not reproduce
`live_check_86.78.md:6-7` (the how-to-re-run header):
```
node scripts/qa/verify_escalation_86_78.mjs    # 37 checks, exit 0
node scripts/qa/mutation_matrix_86_78.mjs      # 10 cells, exit 0
```
Measured today: exit **1** / exit **1**, 51 checks (1 failed) / **0 cells**.
The cardinalities are also stale against the artifact's own later sections
(37 -> 51, 10 -> 13). Both documented commands are false as printed.

### Fix that closes B1+B2+B3 (one literal + a re-run)
`scripts/qa/verify_escalation_86_78.mjs:159` -> assert the NESTING property
rather than an exact whole-line literal (e.g. match
`const merged = { ...verdict, escalation` as a prefix, or key-membership),
keeping the existing `!/\{[^}]*\.\.\.escalation/` conjunct; then re-run the
checker and the matrix and paste the regenerated blocks, and refresh the
:6-7 recipe cardinalities.

COMPLETED: 2026-08-17T13:54:23Z
