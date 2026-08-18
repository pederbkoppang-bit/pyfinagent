STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.79
WRITTEN: 2026-08-17T14:17:02Z

# Q/A cycle-4 evaluation of step 86.79 (records_retained off-by-one / gauge semantics)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, ruff lint, scoped tests
C. Criterion-by-criterion judgment (7 criteria), with mutation testing of the gate

## Findings log (appended as established)

### Context / prior-attempt evidence
- `qa_wip.py 86.79 --spawned-at 2026-08-17T14:17:02Z`: attempt_number=4,
  prior_attempts=3, records_retained=4 (GAUGE), source_present=true,
  attempt_number_status=ok, attempt_number_is_lower_bound=true,
  records_pruned_known=null.
- `verdict_history_86_21.py --step 86.79 --evidence-only`: status=ok,
  3 rows, sequence `CONDITIONAL -> CONDITIONAL -> NO_VERDICT`.
- HEAD at eval start: cb279731. Cycle-4 product commits: ba74813b
  (verify_counter_86_79.py gate pins) + 8ed8ba54 (artifacts).
  NOTE: the spawn prompt attributes the gate change to 8ed8ba54; the gate
  change is actually in ba74813b (8ed8ba54 touched artifacts only).

### DETERMINISTIC
- Immutable command `ast.parse(scripts/qa/qa_wip.py)` -> `qa_wip-parses`, EXIT=0.
- `python scripts/qa/verify_counter_86_79.py` -> ALL CHECKS PASS,
  checks run 60 (floor 53), failed 0, BARE EXIT=0. Reproduced by me.

### FINDING F1 (real, reproduced): the staleness cross-check the step
### rewrote in qa.md is now off by one and fires unconditionally
- qa.md:715 (authored by THIS step's commit 9b4d5281, blame-confirmed):
  "if `attempt_number` (auto) **>** the ledger's verdict count, **the ledger
  is STALE** -- say so in `notes` and treat the sequence as unreliable."
- qa.md:658-660 (same step) defines attempt_number as INCLUSIVE of the
  current attempt. The ledger can only ever hold rows for PRIOR attempts
  (the in-flight verdict is written caller-side after return). So for any
  perfectly-current ledger, attempt_number == rows + 1 > rows and the rule
  fires ALWAYS. The correct operand is `prior_attempts`.
- MEASURED on this very step: attempt_number=4, ledger rows=3, and the 3
  rows are exactly the 3 prior attempts (2 CONDITIONAL + 1 NO_VERDICT).
  The ledger is CURRENT, and the literal rule orders me to call it STALE.
  `prior_attempts (3) > rows (3)` is False -- the correct comparison.
- Pre-fix wording (2e40e8c7) compared `records_retained`, which carried the
  same inclusive off-by-one, so the defect is inherited, not created; but
  this step OWNED and rewrote that sentence (experiment_results explicitly
  lists `:645` as a site it changed) while criterion 4 is "the DOC and the
  CODE are made to agree".
- Harm direction is CONSERVATIVE (over-reports staleness -> judge distrusts
  a good sequence). It cannot invert a verdict.

### HARNESS COMPLIANCE (5 items) -- CLEAN
1. research gate: research_brief_86.79.md, brief_status COMPLETE,
   gate_passed true, 10 sources read in full, 25 URLs, recency scan true.
2. contract-before-generate (mtime chain): brief 08-14T09:05:37 <
   contract 09:14:15 < qa_wip.py 09:18:56. OK.
3. experiment_results_86.79.md present (286 lines) + live_check present.
4. log-last: masterplan 86.79 status=pending; cycle-4 row NOT yet in
   harness_log. OK. (Secondary-source note: the log carries TWO rows headed
   "Cycle 3" for 86.79 -- one CONDITIONAL(ESCALATED), one NO_VERDICT(rail
   drop) -- i.e. 4 rows against 3 attempts. The ledger governs and says 3.)
5. no-verdict-shopping: evidence CHANGED -- ba74813b (15:19) + 8ed8ba54
   (15:31) on 2026-08-17, after the 08-14 cycle-3 drop. OK.

### LINT
- Derived scope `git diff --name-only dc6575b6^ HEAD -- '*.py'` = 37 files
  (non-empty guard passed), `uvx ruff check --select F821,F401,F811` ->
  "All checks passed!", EXIT=0.

### INDEPENDENT RE-DERIVATION of criteria 1/2/3/5/6 (my own drives, not
### the author's checker)
- C1: 2 priors -> records_retained=3, attempt_number=3, prior_attempts=2.
  Producing line quoted from source: `qa_wip.py:507: "records_retained":
  len(records),`  -> off-by-one REPRODUCED.
- C2: BEFORE write-first records_retained=2 / attempt_number=None /
  status=no_record_for_this_spawn; AFTER records_retained=3 /
  attempt_number=3 / status=ok. Coupling demonstrated; new field REFUSES.
- C3: 6 records -> prune(keep=3) -> records_retained=3 (saturation: reports
  3 rather than 6) while attempt_number=6 survives via the loss ledger
  (records_pruned_known=3). Enumeration of prune callers is executed by the
  gate with the grep command PRINTED; 20 hits, 0 non-allowlisted.
- C5: F1b -> CONTINUE at 1..4, ESCALATE at 5/5 and beyond. Verdict-keyed
  boundary discriminates: 1 CONDITIONAL not armed, 2 armed, PASS resets,
  missing ledger -> None (not 0).
- C6: missing sink -> source_present=False, attempt_number=None,
  status=source_missing, no `verdict` key, is_verdict=False. Budget
  exhaustion close_kind over ALL flag combinations -> {'ESCALATE'} only.

### MUTATION TESTING I RAN MYSELF (relocated-repo harness; live tree only READ)
CONTROL (relocated, unmutated): rc=0, 60 checks, 0 fails -> GREEN.
| cell | mutation | result |
|---|---|---|
| N1 | qa.md gauge sentence inverted | KILLED by "4c APPLIED: qa.md carries the gauge correction" |
| N2 | old wrong wording reinstated in qa.md | KILLED by "4c APPLIED: the old wrong wording is GONE" |
| N3 | "null is NEVER 0" removed from qa-verdict.js | KILLED by the 4b null-rule check |
| N4 | "a gauge, not" removed from qa-verdict.js | KILLED by the 4b gauge-distinction check |
| N5 | prior_attempts renamed (2nd half of the ANDed guard) | KILLED |
| N6 | attempt_number renamed (1st half of the ANDed guard) | KILLED |
| N10a | PRE-cycle-4 gate, unmutated tree (control) | GREEN, 55 checks |
| N10b | PRE-cycle-4 gate vs the N1 doc revert | SURVIVED -> the old gate WAS blind; cycle 4 is load-bearing |
| N10c | PRE-cycle-4 gate vs the 4b revert | SURVIVED -> same |
Both halves of the ANDed guard redden independently. Kill mechanism named
per cell (no mis-attribution).

### FINDING F2 (WARN): the new 4b/4c pins are whole-file BYTE-PRESENCE pins
- N7: delete "null is NEVER 0" from the PROMPT PAYLOAD array and leave the
  bytes in a `// vestigial doc note` comment -> gate GREEN (SURVIVED). The
  rule is gone from what the judge is actually sent.
- N9: INVERT the qa.md normative sentence to "Feel free to use
  records_retained as the attempt number" and park the original bytes in an
  HTML comment -> gate GREEN (SURVIVED).
- Not sole-coverage vacuity: N1-N6 prove the pins CAN fail on a natural
  revert. Named fix: scope the pins to the prompt-payload array / the
  normative section instead of the whole file.

### FINDING F3 (WARN): the cardinality floor was not raised with the cycle-4
### checks, so the block cycle 4 added can be silently SKIPPED
- EXPECTED_CHECKS is still 53 while the gate now runs 60 (slack 7).
- N11: comment out the whole cycle-4 doc-pin block (5 checks) -> rc=0,
  "checks run : 55 (cardinality floor 53)" -> SURVIVED.
- N12: comment out the entire C4 doc/patch tail (7 checks) -> rc=0,
  "checks run : 53" -> SURVIVED, sitting exactly on the floor.
- The constant's own comment says it was "raised to sit just under the
  current total so a silently-skipped block is caught rather than absorbed"
  after the cycle-1 Q/A found 12 checks of slack. The same instruction was
  again not followed at cycle 4.

### FINDING F4 (NOTE, evidence-quality): stale headline totals
- experiment_results_86.79.md:11 "**Current totals: 55 checks (floor 53),
  11 mutation cells, 11 killed.**" -- present tense, now FALSE: 60 checks.
- live_check_86.79.md:7 "# 55 checks, exit 0" in the re-run header -- same.
- The cycle-4 capture used `tail -2`, which shows only "ALL CHECKS PASS"
  and therefore cannot expose the drift. Dated historical captures (:88,
  :143, :330, :406) are correctly preserved and NOT a finding.

### CLAIMS THAT DO REPRODUCE (checked, not assumed)
- Anchor pre-verification counts in live_check cycle-4: `null is NEVER 0`
  1x, `a gauge, not` 1x, `prior_attempts` 1x in qa-verdict.js; qa.md gauge
  line 1x; old wrong wording 0x. ALL FIVE reproduce exactly today.
- "A revert of either 4b or 4c now REDDENS the gate" -- reproduced (N1-N6),
  and the converse proved with the pre-cycle-4 gate (N10b/N10c).
- "records_retained_unit is asserted on the healthy ok path" -- confirmed,
  C1 check "records_retained_unit states it is a GAUGE".
- "records_pruned_known carries four assertions" -- counted 4 in the gate
  source (C3 x2, C3b, C3c). Reproduces.
- Enumeration re-derived by ME with the gate's own printed command: 20 hits
  across 5 files (verify_counter x12, verify_wip_retention x2, qa_wip x2,
  mutation_matrix_86_36 x2, mutate_counter_source_86_21 x2). NO production
  caller. Criterion 3's second half reproduces.
- Every field qa.md tells the judge to read exists in the live payload
  (10/10: attempt_number, prior_attempts, attempt_number_status,
  attempt_number_guidance, attempt_number_is_lower_bound,
  records_pruned_known, records_retained, records_retained_unit,
  is_verdict, source_present). No `verdict` key. is_verdict False.
- Sibling gates re-run green: verify_wip_retention_86_36.py EXIT=0,
  mutate_counter_source_86_21.py EXIT=0.

### CRITERION 7 -- the author's matrix, re-run by me
CONTROL GREEN first (60 checks), then 11/11 cells KILLED, each BY A NAMED
assertion (M1..M11), subject sha256[:16]=146600b722a02481 identical before
and after -> the tracked file was never mutated. MATRIX EXIT=0.

### FINDING F5 (NOTE, evidence-quality): the patch record still says
### "NOT APPLIED" and its own stated verification command falsifies it
- handoff/current/qa_md_patch_86.79.md:1 "# PROPOSED - NOT APPLIED";
  :3 "Status: WRITTEN OUT FOR THE OPERATOR, DELIBERATELY NOT APPLIED.";
  :18 "Nothing in `.claude/agents/qa.md` was modified by step 86.79 --
  verify with `git diff --stat .claude/agents/qa.md`".
- I ran that exact command against the step's own commit:
  `git diff --stat 9b4d5281^ 9b4d5281 -- .claude/agents/qa.md` ->
  "1 file changed, 116 insertions(+), 45 deletions(-)". Falsified.
- The gate's replacement check calls this file "the applied correction's
  historical record" but asserts only that it EXISTS and contains the
  literal `records_retained` -- not that its status line is accurate.

### TREE MOTION DURING EVALUATION (rechecked before returning)
- HEAD moved cb279731 -> 9aa2f64e mid-run (651e1f78 phase-86.78/86.37
  cycle-5 + changelog). The only product file touched is
  .claude/workflows/qa-verdict.js, ONE line, the disclosed 86.78 STEP-0
  wording change ("the 3rd-CONDITIONAL auto-FAIL rule" ->
  "the loop-termination rule"). It touches NONE of 86.79's pinned
  literals (all three still 1x) and the gate re-run at the NEW head is
  still ALL CHECKS PASS / 60 checks / exit 0. Grade holds at 9aa2f64e.

### SCOPE
- 86.79's cycle-4 commits touch only: verify_counter_86_79.py,
  experiment_results_86.79.md, live_check_86.79.md (+ 86.72's critique,
  the verdict ledger row and the attempt-budget audit line, co-committed
  and disclosed in ba74813b's message). No unintended production change.

### CONCLUSION
All SEVEN criteria substantively MET and independently reproduced by me.
Residuals: F1 (live doc defect, conservative direction), F2 + F3 (two
DEMONSTRATED holes in the step's own gate), F4 + F5 (evidence-quality
only). Verdict CONDITIONAL, capped by F1/F2/F3; F4/F5 stated explicitly
as evidence-quality for queueing rather than iteration.

COMPLETED: 2026-08-17T14:32:37Z
