STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.79
WRITTEN: 2026-08-14T08:12:02Z

# Q/A write-first record -- step 86.79 (cycle 2, evidence CHANGED per commit dc6575b6)

Role instructions read: .claude/agents/qa.md (full, at runtime).

## ATTEMPT COUNT
`python scripts/qa/qa_wip.py 86.79 --spawned-at 2026-08-14T08:12:00Z`
-> source_present=TRUE (checked FIRST), records_retained=2, prior_records=[073710Z],
   prior_attempts=1, attempt_number=2, attempt_number_status=ok.
ATTEMPT 2 of F1b's 5. Prior verdict sequence: [CONDITIONAL] (cycle 1, wf_61338c26-b90).
3rd-consecutive-CONDITIONAL trigger NOT armed (needs 2 consecutive priors; there is 1).
LEDGER CROSS-CHECK: `verdict_history_86_21.py --step 86.79` -> status=no_rows_for_step,
consecutive=0. prior_attempts (1) > ledger verdict count (0) => THE LEDGER IS STALE for
this step: the cycle-1 CONDITIONAL was never appended. Sequence taken from the
evaluator_critique's own "Verdict ledger" table instead, and flagged as unreliable.

## DETERMINISTIC (all reproduced by me, not read)
- immutable cmd `ast.parse(qa_wip.py) && echo qa_wip-parses` -> `qa_wip-parses`, exit 0.
- verify_counter_86_79.py -> "checks run : 50 (cardinality floor 48) / failed : 0", exit 0.
- mutation_matrix_86_79.py -> CONTROL exit 0 GREEN (50 checks) FIRST, then 9/9 KILLED,
  subject sha256[:16] 146600b722a02481 before==after.
- Lint gate 1a. Working tree has NO uncommitted/untracked .py, so
  `git diff --name-only HEAD -- '*.py'` is EMPTY; scope DERIVED from the step commit
  `git show --name-only --format="" dc6575b6 -- '*.py'` = 3 files, asserted non-empty,
  passed via xargs -0 -> `All checks passed!` exit 0. The cycle-1 F401 is GONE.
- git status: no unintended production change. Only audit jsonl / health.jsonl /
  researcher memory churn + my own WIP file. Step work is committed at dc6575b6.
- Frontend gate 1b NOT triggered (no frontend/** in diff; qa.md zero-line diff).
  Backend smoke 1d NOT triggered (no backend/**). UI gate 1c NOT triggered (no UI claim).

## HARNESS COMPLIANCE -- CLEAN
- Research gate: brief_status COMPLETE, external_sources_read_in_full=10 (floor 5),
  urls_collected=25 (floor 10), recency_scan_performed=true, gate_passed=true,
  run wf_267244ab-91e cited in contract section 1.
- mtime chain: research_brief 07:05:37Z < contract 07:14:15Z < qa_wip.py 07:18:56Z
  < verify_counter 08:01:46Z < experiment_results 08:10:13Z. ORDER OK.
- harness_log: `grep -Fc 86.79` -> 0 rows. masterplan step still `pending`. LOG-LAST OK.
- no-verdict-shopping: evidence CHANGED (commit dc6575b6, +21 lines qa-verdict.js,
  +8 checks, +2 mutation cells, F401 removed). Legitimate cycle-2 respawn.

## CRITERION REPRODUCTION (mine, in a scratch sink)
- C1: priors=1 -> records_retained=2/attempt=2; priors=2 -> 3/3. Producing line
  qa_wip.py:507 `"records_retained": len(records),`. MET.
- C2: BEFORE own write: records_retained=2, attempt_number=None,
  status=no_record_for_this_spawn, prior_attempts=2. AFTER: 3 / 3 / ok / 2.
  The number DIFFERS and the new field REFUSES rather than guessing. MET.
- C3: 6 records -> prune(keep=3) -> records_retained SATURATES at 3 while
  attempt_number stays 6 and the loss ledger reads 3. Enumeration re-run by me with
  NO --include filters: every hit outside the allowlist is handoff/ prose. MET.
- C4: NOT MET (see below).
- C5: C5 drives attempt_budget.BudgetState from records_retained (3 -> CONTINUE) vs
  attempt_number (6 -> ESCALATE), and drives verdict_history read_ledger over
  1C / 2C / 2C+PASS / missing-ledger. Both bounds fire. MET (NOTE: only the F1b half
  is fed by the corrected number; the 3rd-CONDITIONAL half is verdict-keyed by design).
- C6: consumer grep over .claude/hooks, scripts/harness, backend for
  `qa_wip|attempt_number` -> ZERO hits. No `verdict` key in any report() variant;
  every uncomputable path returns None; exhaustion -> {"ESCALATE"} over all flags. MET.
- C7: control GREEN first, 9/9 killed on NAMED assertions, RED-WRONG-REASON
  discrimination is live executed code, subject digest identical. MET.
- Sibling gates re-run by me: 23 passed / 246 passed / 24-of-24 / 5-of-5 / 0-surviving,
  all exit 0. (246 not 244/245: that checker emits one assertion per live
  verdict_wip_*.md, so my own record moved it. Known F4 coupling, not a defect.)

## MY OWN MUTATIONS (6 cells through PYFIN_QA_WIP_OVERRIDE; control GREEN first)
KILLED: Q-C ignore-ledger-entirely; Q-D records_retained-minus-one;
        Q-F spawn-identity-check-disabled.
SURVIVED (with EXECUTED behavioural differentials, so not equivalent mutants):
- Q-A `len(records) >= DEFAULT_KEEP` -> `> DEFAULT_KEEP`. At EXACTLY 3 retained with
  no ledger: baseline is_lower_bound=True, mutant False. Mutant claims exactness in
  the precise state a legacy prune leaves. C3c only drives retained=2 and retained=5,
  so the boundary is uncovered. Same class as cycle-1's N1, which M9 was written for --
  M9 kills only the always-False form. NEW FINDING, capping.
- Q-B `read_loss` `val >= 0` -> `val > 0`. A no-op prune legitimately writes
  {"lost": 0}; baseline records_pruned_known=0/is_lower_bound=False, mutant
  None/True. SAFE direction (over-flags), lower severity. NOTE.
- Q-E move `_record_loss` from BEFORE the unlink to AFTER. Under simulated crash
  (unlink raises): baseline read_loss=3 and prior_attempts=9; mutant read_loss=None
  and prior_attempts=6 -- i.e. the mutant UNDER-counts, the unsafe direction. The
  docstring's explicit "ledger written BEFORE the unlink so a crash over-counts"
  safety claim has NO guard. A guard is feasible (I wrote one in ~10 lines).
  Mitigating: prune_wip_records still has ZERO production callers. NOTE.
C3b is NOT vacuous: it asserts the loss account is non-zero as an explicit
precondition (lost==3, with the detail string naming the vacuity risk), asserts the
branch is reached, asserts prior_attempts==6 exactly, and asserts the ESCALATE /
CONTINUE differential. My Q-C independently dies on that same assertion.

## CLAIM AUDIT -- experiment_results_86.79.md IS INTERNALLY STALE
§0 (new) states the correct post-cycle-2 totals (50 checks / 9 cells). But the
cycle-1 text below it was NOT updated:
- §1 file table: "42-check re-runnable checker", "7-cell mutation matrix"
- §2, headed "Verbatim verification output": "checks run : 42 (cardinality floor 30)"
  and "cells: 7 killed: 7". This block does NOT reproduce -- the real run prints
  50/48 and 9/9.
- §3 criterion-7 row: "7/7 killed"; criterion rows point at live_check §1-§7 (the
  superseded capture) rather than §12-§14.
- §6: "244" (my run: 246; explained by the dynamic assertion count, see above).
- §7: "these 7 mutations were killed".
live_check §12-§14 DO carry the correct verbatim and reproduce exactly.
Classified as superseded-text-accompanying-a-correction, NOT fabrication.

## CRITERION 4 -- NOT MET
`.claude/agents/qa.md` zero-line diff CONFIRMED: `git diff --stat 4efecb87 HEAD --
.claude/agents/qa.md` is empty. qa.md:622 still reads "records_retained is the count
of prior Q/A spawns on this step -- the attempt number, and it is authoritative",
which is false on BOTH halves. I independently found a SECOND stale site at qa.md:645
("if records_retained (auto) > the ledger's verdict count") whose qa-verdict.js
counterpart WAS fixed -- so within qa.md the class is 2 sites, not 1.
qa-verdict.js class VERIFIED EMPTY by me: the only two surviving records_retained
mentions (:156, :158) now correctly say it is NOT the attempt number.
Item (D) assessed: leaving the consequence-framing text is CORRECT scope discipline,
not evasion -- that text is not FALSE and diverges from no code, so it is outside
criterion 4 entirely, and it is sibling step 86.78's subject.

## FINDING ON THE RAIL THAT SPAWNED ME (first-party)
The prompt I received carries the PRE-FIX text verbatim: "count your own prior
attempts by running `python scripts/qa/qa_wip.py <step_id>` and reading
records_retained / prior_records", "records_retained gives the ATTEMPT number
(authoritative)", "If records_retained > the ledger verdict count", and
"CHECK source_present FIRST (phase-86.21): records_retained==0 ...". All four are the
lines dc6575b6 replaced. So the file on disk is fixed, but the prompt actually
delivered to this spawn was the un-fixed text -- the fix was NOT in force for the
spawn that grades it. (Consistent with an inline/equivalent-script launch, which
qa.md explicitly permits.) This bounds the claim "the primary launch rail's prompt is
fixed": true of the checked-in file, not demonstrated of the delivered prompt.

## DISPOSITION
CONDITIONAL (attempt 2). 6 of 7 criteria MET and independently reproduced; criterion 4
NOT MET (author concurs and declines a waiver). Three capping/NOTE findings above.
NOT FAIL: all four cycle-1 findings are genuinely fixed and I verified each by
execution; the residual is an operator-gated file; nothing here can turn a FAIL into
a PASS. NOT PASS: criterion 4 is unmet and a new surviving mutant (Q-A) exists in the
same class the cycle-2 remediation targeted.

COMPLETED: 2026-08-14T08:29:36Z
