STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 90.2
WRITTEN: 2026-08-21T07:50:52Z

# Q/A write-first record -- step 90.2 (cycle 3 evidence)

Role: independent Q/A evaluator (merged qa-evaluator + harness-verifier).
Launch: Workflow structured-output rail.
This file is a CRASH-SURVIVAL RECORD, not a verdict.

## Plan
- A. Harness-compliance audit (5 items)
- B. Deterministic: immutable command, git scope, lint, syntax
- C. Mutation attack on the NEW cycle-3 code + cells (M11/M12/M14, governing_severities,
  emitted_severities, comparable gate, negator fixture)
- D. Criterion-by-criterion MET/NOT MET

## Log
(appended as established)

### [1] Attempt / sequence evidence (07:52Z)
- qa_wip.py 90.2 --spawned-at 2026-08-21T07:50:52Z:
  source_present=True, attempt_number=3 (status ok, is_lower_bound=True), prior_attempts=2,
  records_retained=3 (GAUGE), records_pruned_known=None, is_verdict=False.
  prior_records: verdict_wip_90.2__20260821T073230Z.md, verdict_wip_90.2__20260821T071422Z.md
- verdict_history_86_21.py --step 90.2 --evidence-only: status=no_rows_for_step, verdicts=(none).
- CROSS-CHECK: prior_attempts (2) > ledger rows (0) => LEDGER IS STALE. sequence: UNKNOWN.

### [2] Immutable command (07:53Z)
$ bash -c 'node --check .claude/workflows/qa-verdict.js && node scripts/qa/verify_severity_routing_90_2.mjs --self-test'
IMMUTABLE COMMAND EXIT: 0
checks run: 77 (floor 66); failed: 0
16 cells: N0 SURVIVED | M1-M9,M11,M12,M14,L1,L2 KILLED | QX ERROR
REPRODUCED independently by me (not transcribed).

### [3] Working tree scope (07:52Z)
git status --short: only ` M handoff/audit/attempt_budget_audit.jsonl`,
` M handoff/audit/pre_tool_use_audit.jsonl` (hook-written append-only audit streams)
and my own untracked WIP file. NO unintended production change.
HEAD = 69500b8f (auto-changelog) over f4f103c6 (the cycle-3 commit).
f4f103c6 touched 8 files: qa-verdict.js, verify_severity_routing_90_2.mjs,
3 handoff artifacts, 2 audit jsonl, 1 prior WIP record.

### [4] INDEPENDENT MUTATION WORK (08:05Z) -- driver: fs.readFileSync monkeypatch on
the WORKFLOW path, real checker as oracle, NO file written anywhere.
  CONTROL A (null, comment-only)                    -> SURVIVED, 77 checks, 0 failed  [harness sane]
  CONTROL B (verdict guard removed)                 -> KILLED, 6 real failures        [harness sane]
  QX1 emitted_severities truncated in RETURN LITERAL -> ***SURVIVED, 77 checks, 0 failed***
      mutation: `emitted_severities: anyEmitted ? emitted : null,`
             -> `emitted_severities: anyEmitted ? (emitted.length > 1 ? emitted.slice(1) : emitted) : null,`
      PROOF BY ENUMERATION TOO: `emitted_severities` appears in the checker at exactly
      ONE assertion (verify_severity_routing_90_2.mjs:296-298), against a length-1
      fixture, testing only Array.isArray + [0]==='NOTE'. The mutant preserves
      length-1 arrays, so no check can see it.
      *** THIS IS THE CYCLE-2 WARN F1 SHAPE, REPRODUCED ON THE FIELD CYCLE 3 ADDED. ***
  QX2 derived_severities truncated in RETURN LITERAL -> KILLED (4 named failures) [control: shape does kill when guarded]
  QX3 routing fn mutates judge verdict string        -> KILLED (F: 6/24 identical, 18 mutated) [criterion 3 guard NON-vacuous]

### [5] TWO SURVIVING MUTANTS, both with proven behavioural differentials (08:15Z)

**S1 -- `emitted_severities` (the field CYCLE 3 ADDED) has no cardinality/content guard.**
  QX1  `emitted.length > 1 ? emitted.slice(1) : emitted`   -> SURVIVED 77/77
  QX1b `emitted.length > 1 ? emitted.slice(0,-1) : emitted`-> SURVIVED 77/77
  QX7  `emitted_severities: null` (whole field)            -> KILLED (so the field is guarded for EXISTENCE only)
  Enumeration: the checker touches `emitted_severities` at ONE site, verify:296-298,
  against a LENGTH-1 fixture, asserting only Array.isArray + [0]==='NOTE'. Any mutant
  that preserves length-1 arrays is invisible.
  DIFFERENTIAL (control vs QX1), fixture = CONDITIONAL, findings ['(WARN) a','(WARN) b'],
  details [{WARN},{NOTE},{BLOCK}] (3 != 2 -> not comparable -> derivation governs):
     control emitted_severities=["WARN","NOTE","BLOCK"] route=queue_residual
     QX1     emitted_severities=["NOTE","BLOCK"]        route=queue_residual
     QX1b    drops the trailing element -- i.e. the judge's BLOCK
  NOT an equivalent mutant. Route is unchanged, so this is a REPORTING-completeness
  defect, not a routing defect -- but on a queue_residual run `emitted_severities` is
  the ONLY channel carrying a non-comparable judge-emitted BLOCK to the caller.
  THIS IS THE CYCLE-2 WARN F1 SHAPE ON THE FIELD THE CYCLE-2 FIX INTRODUCED. The
  shipped checker's own comment at verify:269 says "GUARD EVERY RETURNED ARRAY, NOT
  JUST THE ONE THE MATRIX HAPPENS TO REACH."

**S2 -- the cycle-3 `reliability` gate change is unguarded.**
  QX6 `reliability: comparable ? null : {` -> `anyEmitted ? null : {` -> SURVIVED 77/77
  This RESTORES the pre-cycle-3 line (git diff ca656466..f4f103c6 shows exactly this
  line changed). DIFFERENTIAL on the emitted-but-not-comparable branch:
     control reliability = OBJECT(derivation_is_authoritative=false)
     QX6     reliability = null
  Same branch on which the DERIVATION governs -- so the mutant ships a derived route
  with no unreliability label, which is precisely what checker section E4's stated
  property forbids. NOT equivalent.

### [6] Attribution checks (no mis-credited kills)
  M14 -> killed by EXACTLY the E3 negator fixture (sole content failure).
  M11 -> killed by the 3 governing_severities checks (E).
  Criterion-3 guard proven NON-vacuous: QX3 (routing mutates the verdict in place)
  KILLED with "6/24 by string equality", "18 mutated".

### [7] Independent replay re-derivation (08:12Z)
  My own 3rd operationalization: 40 vs shipped 41. SYMMETRIC DIFFERENCE = 1 run,
  wf_9b398d19-fa8, entry `illusory-guard (WARN, shape #2): ...`. MY rule was too narrow
  (required the token immediately followed by the closing bracket); the shipped
  `delimited` rule allows <=45 chars inside the bracket. THE SHIPPED MATCHER IS RIGHT
  and mine was the defective control. Residual reported, not averaged away.
  Real runs mixing a tagged WARN/NOTE with an UNTAGGED finding: 20 -- ALL 20 route to
  remediate under the shipped function driven by me.
  24/24 fixture rows are byte-identical to the on-disk run records (verdict +
  violated_criteria + violation_details). E3 fixture phrase IS verbatim in
  wf_7fa0e5d6-c50.

### [8] Harness compliance -- ALL 5 CLEAN (08:20Z)
1. research-gate-before-contract: research_brief_90.2.md envelope brief_status=COMPLETE,
   external_sources_read_in_full=7 (>=5), urls_collected=17 (>=10),
   recency_scan_performed=true with a dedicated "## Recency scan (2024-2026)" section
   (x2, incl. "-- PERFORMED"), gate_passed=true.
2. contract-before-generate (mtime): brief 08-20T21:39:19 < contract 08-20T21:52:18 <
   qa-verdict.js 08-21T09:48:49 < experiment_results 09:50:23. All SIX criteria appear
   VERBATIM in contract_90.2.md by string equality against masterplan.json.
3. experiment_results_90.2.md present, 373 lines, CYCLE 3 section at :332.
4. log-last: grep -cF "phase=90.2" handoff/harness_log.md = 0; masterplan 90.2
   status="pending". Not logged, not flipped.
5. no-verdict-shopping: evidence CHANGED. f4f103c6 touches qa-verdict.js (+43/-16),
   verify_severity_routing_90_2.mjs (+89), experiment_results (+59), evaluator_critique
   (+232), live_check (97 lines). Documented fresh-respawn.
GATE SCOPE, derived not typed: git diff --name-only c09bd96b~1 HEAD -> 0 *.py, 0
frontend/**, 0 backend/**. qa.md gates 1a/1b/1c/1d are N/A on this diff, not skipped.
No UI claim in the step. node --check exits 0 on both .mjs and on qa-verdict.js.
HEAD re-checked at the end: 69500b8f, unchanged throughout.

### [9] NOTES (non-blocking)
N1 EXPECTED_CHECKS floor left at 66 while the suite is now 77. Cycles 1/2 raised it
   in lockstep (50 -> 55 -> 66, suite exactly at the floor); cycle 3 did not. 11 checks
   -- including all four new ones covering F1/F2/negator -- could be deleted and the
   cardinality guard would still pass. The guard's own stated rationale is that a
   checker covering nothing "exits 0 and looks identical to success".
N2 Section J's VERDICT_SCHEMA source scan uses a hardcoded 2200-char window against a
   schema whose span I measured at 2273 chars -- the last 73 chars are unscanned.
N3 Criterion 4's confusion table carries run ids for queue_residual and for both
   disagreement directions, but not for the 247-run remediate bucket. Acceptable.
N4 residual_close_gate.mjs is behaviourally driven by 8 section-H fixtures (each
   clause has a fixture with the opposite expected outcome) but is in NO mutation cell.

### [10] CANDIDATE FINDINGS I RETIRED (reported so they are not re-litigated)
R1 My own independent matcher gave 40 vs the shipped 41. MY rule was too narrow.
   The shipped rule is correct. Withdrawn.
R2 Main's `entries.length > 0` equivalent-mutant claim is CORRECT and I proved it
   rather than accepting it: derivedOnly.length === entries.length, and `comparable`
   itself requires derived.length > 0, so governing.length > 0 => entries.length > 0.
   Honest disclosure, not padding.
R3 The `k !== 'severity_routing'` clause in the leakedS filter is a defensive no-op;
   `k in merged` walking the prototype chain makes the guard MORE likely to throw,
   never less. Neither is a finding.

### [11] CRITERION MAPPING
1 MET  2 MET  3 MET  4 MET  5 MET  6 NOT MET (clause 2's universal has two surviving
mutants on emitted_severities, the field this cycle added; both differentials proven)

VERDICT ISSUED: CONDITIONAL -- 2 WARN (S1 emitted_severities, S2 reliability gate),
4 NOTE. No BLOCK. Criteria 1-5 independently reproduced by me, not read.

COMPLETED: 2026-08-21T08:02:45Z
