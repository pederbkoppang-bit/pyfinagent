STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.90
WRITTEN: 2026-08-16T08:25:15Z

# Q/A write-first record -- step 86.90 (cycle 1 per Main's disclosure)

Step topic: `[object Object]` stringification of nested `evidence`/`extra` objects
in the Workflow rail prompt (qa-verdict.js / research-gate.js).

## Plan
- A. Harness-compliance audit (5 items)
- B. Deterministic: immutable command, git scope, lint, re-runnable checks
- C. Criteria 1-7 with cited evidence; independent mutation testing

## Findings (appended as established)

## A. PRIOR ATTEMPTS / SEQUENCE
- qa_wip.py 86.90 --spawned-at 2026-08-16T08:25:15Z -> source_present=true,
  attempt_number=1 (status ok), prior_attempts=0, records_retained=1 (own record).
- verdict_history_86_21.py --step 86.90 --evidence-only -> status=no_rows_for_step,
  verdicts=(none). prior_attempts(0) == ledger count(0): CONSISTENT. Cycle 1.

## B. DETERMINISTIC
- IMMUTABLE CMD: `bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && echo parses'`
  -> stdout "parses", EXIT=0.
- node --check research-gate.js -> BOTH_PARSE.
- verify_prompt_render_86_90.mjs        -> ALL GREEN: 53 passed, 0 failed.
- verify_research_gate_workflow.mjs     -> ALL GREEN: 124 passed, 0 failed.
- verify_escalation_86_78.mjs           -> 51 checks, 0 failed, ALL CHECKS PASS.
- verify_rail_retry.mjs                 -> ALL GREEN: 38 passed, 0 failed.
- verify_workflow_args_boundary.mjs     -> FAILED: 84 passed, 3 failed.
- No *.py in the diff -> ruff gate N/A. No frontend/**, no backend/** in a21a5889.
- git status: .claude/workflows/* and scripts/qa/verify_prompt_render_86_90.mjs are CLEAN
  (working tree == HEAD). Uncommitted backend/api/sovereign_api.py + frontend/src/* are
  NOT in a21a5889 (8-file stat) nor in 8dc70502/952ed521 -> peer-session work, VERIFIED.

## C. THE 3 "PRE-EXISTING" FAILURES -- INDEPENDENTLY CONFIRMED (different method)
Direct A/B of enforceGate loaded from a21a5889~1, a21a5889 and the working tree,
driven with the checker's exact case-[3] fixture:
  PRE_FIX / POST_FIX / WORKING_TREE -> healthy.gate_passed=false, legacy=false,
  and BYTE-IDENTICAL violation arrays in all three.
  enforceGate source byte-identical pre/post a21a5889: true (11843 == 11843 chars).
Root cause is fixture drift, NOT 86.90: handoff/current/research_brief_86.17.md
(written 2026-08-09) has grep -c brief_status == 0, and phase-86.37 later made the
marker mandatory. Main's claim VERIFIED by an independent route.

## D. CRITERION 4 -- BLAST RADIUS, RE-DERIVED FROM THE RECEIPTS
Strict receipt rule (a coerced field is a WHOLE LINE of the agent's own first user
message): 23 runs. 22 == Main's table EXACTLY (symmetric difference vs Main's list
is EMPTY in both directions); the 23rd is Main's own declared pre-fix probe
wf_4588d8a7-e70. Loose grep found 4 extra (4588d8a7, 9bd7e233, a09930e2, 70a3e2c4)
-- all prose contamination, correctly excluded. Positive control: wf_b1747d75-eec found.
Verdicts re-derived from the run records (args.step_id derived, not typed):
  CONDITIONAL 7 | FAIL 4 | PASS 4 | no-verdict (drop) 7  == 22. Every step id matches.
86.86 receipt verified verbatim: prompt line 61 "EVIDENCE / FILES TO READ: [object Object]",
line 63 "ADDITIONAL CONTEXT: [object Object]".
*** FINDING D1: experiment_results_86.90.md:250 says "13 non-PASS verdicts
(CONDITIONAL/FAIL) and 6 rail drops". The correct split is 11 and 7. 13+6=19 > the 18
non-PASS-or-dropped rows in Main's OWN table. Origin: 6 rows read "*(rail drop)*" and
1 reads "*(rail drop -- no verdict)*", so a literal-count misses the 7th. The TABLE is
right; the ROLLUP SENTENCE is wrong. Disposition unaffected (both sub-buckets are the
"no re-grade needed" class; the 4 PASSes are correctly enumerated). WARN-level. ***

## E. CRITERION 5 -- ADVERSARIAL, PER MAIN'S INVITATION
Five constructions render LOSSILY WITHOUT THROWING (measured against the shipped renderer):
  A1 non-enumerable own data prop -> silently dropped
  A2 non-enumerable toJSON        -> WHOLE value replaced by "REPLACED" (a substitution)
  A4 non-deterministic getter     -> walk reads it once, JSON.stringify reads it AGAIN (TOCTOU)
  A6 nested non-enumerable        -> dropped
  A7 array w/ non-index own prop  -> dropped
Controls: A3 enumerable toJSON THROWS (correct); A5 null-prototype renders losslessly.
REACHABILITY: classifyArgs either JSON.parses a string or passes the runtime object
through untouched; a JSON-derived object has no non-enumerables, no getters, no toJSON.
So NOT reachable from a real caller. The in-code claim "THE RULE IS LOSSLESS-OR-THROW"
is broader than what was measured. WARN-level residual, no verdict impact.

## F. CRITERION 6 -- MY OWN MUTATION MATRIX (12 mutants the author did not use)
Anchor uniqueness of all 4 author cells re-counted independently: 1,1,1,1.
`topic` IS routed through renderArgField (research-gate.js:282), so cell 2 is genuine.
Scored by DIFFERENCE IN FAILING CELL NAMES vs a staged control (the staged control
loses exactly the 2 git-dependent [1]/[4] cells; identical offset every row).
  M-A drop the lossless walk .................. KILLED (9 novel)
  M-B arrays bypass the renderer .............. KILLED (3)
  M-C non-plain-instance check removed ........ KILLED (3)
  M-D finite-number guard removed ............. KILLED (3)
  M-E undefined-member check removed .......... KILLED (3)
  M-F renderer truncates every key ............ KILLED (3)
  M-G2 throw stops NAMING the field, BOTH copies KILLED (12) -- so [6] is not the killer
  M-I identity arg renders objects instead of throwing KILLED (4)
  M-J HARNESS: runDriver reports NO spawns ..... KILLED (16, section [0] goes red)
  M-K HARNESS: scorer marks every mutant KILLED  SURVIVED (0) -- see note
M-G (first attempt) was a CONSTRUCTION ARTIFACT: the violation string is seeded with
`where`, so the field name survived; retired, not reported as a finding.
M-K note: section [5] cannot detect its own scorer being neutered. That regress
terminates at the independent evaluator by design (this matrix), and the anchor-
uniqueness pre-check covers the shape that has actually bitten this repo. NOTE, not a
finding against criterion 6.

## G. CRITERIA 1/2/3/7 -- VERIFIED FROM THE RUN RECORDS, NOT FROM MAIN'S PROSE
- wf_4588d8a7-e70 (pre-fix live probe) result: runtime_typeof_evidence="object",
  script_concat_result="EVIDENCE / FILES TO READ: [object Object]",
  agent_received={"received_line":"[object Object]","is_literal_object_object":true}.
  -> C1 reproduction AND C2 layer localisation are both grounded in this one record:
  marshalling INNOCENT (typeof object), template GUILTY (concat produced the literal),
  transport FAITHFUL (the agent received exactly what the template built).
- wf_9bd7e233-f38 (research gate): gate_passed=true, self_report_disagreed=false.
  Brief envelope: brief_status COMPLETE, 12 sources, 45 URLs, recency true, gate true.
- wf_a09930e2-3d7 (86.86 re-grade): verdict=PASS, ok=true, violated_criteria=[].
- C3 by execution: guard [1] drives the PRE-FIX blob of BOTH scripts from git; [2]/[3]
  drive the CURRENT research-gate.js. My strict receipt census independently found
  ZERO research-gate production receipts -> vulnerable by construction, never triggered.
- C7 MEASURED, not read: VERDICT_SCHEMA identical=True (1205==1205),
  enforceEscalation identical=True (11451==11451) across a21a5889~1 -> a21a5889.
  No verdict-logic lines in the diff (only comments + KNOWN_ARG_KEYS naming, never reading,
  verdict_sequence/attempt_number/max_attempts).
- FIRST-PERSON RECEIPT: this spawn's own prompt carries EVIDENCE / FILES TO READ and
  ADDITIONAL CONTEXT as fenced ```json blocks, not the literal. The fix is live on me.

## H. P2 UNKNOWN-KEY WARNING -- behaviourally verified (not a source read)
  qa-verdict + {questions, bogus} -> "WARNING -- args carried 2 key(s) ... reached NOTHING"
  qa-verdict, all-known keys      -> NO warning (clean negative control)
  research-gate + {questions}     -> "WARNING -- args carried 1 key(s) ..."
Fires correctly. NOT covered by any check() in the committed guard -> NOTE (unguarded
addition), not a criterion miss: criterion 6 is about stringification, which IS guarded.
Main's log-only rationale HOLDS: qa-verdict.js:683 carries the phase-86.78 invariant that
throws if caller-authored fields become siblings of the judge's output. Not a scope dodge.

## I. HARNESS COMPLIANCE (5 items)
1. research-gate-before-contract : PASS (brief 09:59:05 < contract 10:01:10; gate enforced)
2. contract-before-generate      : PASS (contract 10:01:10 < qa-verdict.js 10:02:56 <
                                   guard 10:06:10 < experiment_results 10:22:06)
3. experiment_results present    : PASS
4. log-last                      : PASS (masterplan 86.90 status=pending; 0 harness_log rows)
5. no-verdict-shopping           : PASS (cycle 1, no prior verdict; ledger no_rows_for_step)

*** FINDING D2: FOUR items are asserted "queued" and NONE are in the masterplan.
The masterplan's last commit is c627a810, which PRECEDES a21a5889 -- it has not been
touched since. Unqueued: (a) the 85.5 / 86.25 / 86.34 re-grades; (b) harness-self-audit.js:68
(same concat shape); (c) the stronger unknown-key treatment; (d) the stale
.claude/workflows/qa-verdict.js.export.mjs. The standing rule is "own step per standing
rule" and Main DID follow it for 86.91 (c627a810), so this is an omission, not disregard.
WARN. ***
NOTE: verify_workflow_args_boundary.mjs is live-RED (3 cells) from fixture drift and is
itself unqueued.

## J. CRITERIA ROLL-UP
1 MET | 2 MET | 3 MET | 4 MET (+WARN D1, D2) | 5 MET (+WARN E residual) | 6 MET | 7 MET
Harness compliance clean. No unintended production change (a21a5889 = 8 files, all in
scope + the disclosed 86.91 brief; the uncommitted backend/frontend edits are a peer
session's and appear in none of the three commits).
Worst-of-N lenses: correctness=PASS, does-it-reproduce=CONDITIONAL (D1, D2),
scope-honesty=CONDITIONAL (D2). min = CONDITIONAL.

VERDICT: CONDITIONAL -- every criterion met, fix is real and independently
mutation-resistant, but two claims in the artifacts do not reproduce (D1 numeric rollup,
D2 "queued") and the in-code criterion-5 absolute over-states what was measured.
COMPLETED: 2026-08-16T08:38:34Z
