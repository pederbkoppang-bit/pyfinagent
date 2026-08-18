STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.71
WRITTEN: 2026-08-17T11:40:07Z

# Q/A write-first record -- step 86.71, cycle 4 re-evaluation

Spawned via Workflow rail. Task: grade 86.71 against 8 immutable criteria.
Cycle-4 claim from Main: three fixes since cycle-3 verdict --
(1) tautological self-test check fixed (before_rows captured BEFORE refused cmd_extend);
(2) crash-detection widened from 3-exception marker list to CLASS test (traceback in stderr
    OR exit code outside {0,2} => ERROR not kill);
(3) verdict_outcomes broad-except no longer silent -- prints loudly, fail-closed direction stated.

## Log

- [11:40:07Z] WIP created. Read qa.md in full (872 lines).

### Prior-attempt evidence
- qa_wip.py 86.71 --spawned-at 2026-08-17T11:40:07Z: attempt_number=4, prior_attempts=3,
  source_present=true, attempt_number_status=ok, is_lower_bound=true, identity_checked=true,
  records_retained=4 (GAUGE), records_pruned_known=null.
- verdict_history_86_21.py --step 86.71 --evidence-only: status=ok, "3 verdict(s)",
  sequence FAIL -> CONDITIONAL -> CONDITIONAL.
- CROSS-CHECK: prior_attempts (3) == ledger count (3). Ledger NOT stale for this step.
- THIRD independent corroboration: the LIVE attempt gate wrote a production row for MY OWN
  launch -- handoff/audit/attempt_budget_audit.jsonl line 16:
  ts 2026-08-17T11:40:02Z, step_id 86.71, workflow qa-verdict.js,
  attempt_number_inclusive=4, session e6b8ec06-... (my session), 5s before my WRITTEN stamp.
  Three counters agree on 4. The gate demonstrably ran on me.

### Harness compliance (5 items)
1. research-gate-before-contract: research_brief_86.71.md envelope brief_status=COMPLETE,
   9 sources in full / 30 URLs / recency scan present (line 102) -> floors cleared.
2. contract-before-generate: mtimes (LOCAL, UTC+2) research 11:47:50 (09:47Z) <
   contract 12:07:27 (10:07Z) < first gate attempt row 10:35Z < artifacts. Order OK.
   All 8 criteria appear VERBATIM in contract_86.71.md lines 43-50 vs masterplan.
3. experiment_results_86.71.md present (10,116 b) with a Cycle 4 GENERATE section.
4. log-last: grep -c "phase=86.71" handoff/harness_log.md = 0; masterplan status=pending. OK.
5. no-verdict-shopping: evidence CHANGED after the cycle-3 verdict (10:29Z ledger row).
   Commit cbbd1566 (13:41 local = 11:41Z) carries the three cycle-4 edits to
   attempt_gate.py (+53) and mutation_matrix_86_71.py (+153) plus +72/+216 artifact lines.
   Documented cycle-2 flow, not a re-spawn on unchanged evidence.
   NOTE: HEAD MOVED DURING MY RUN, cadab378 -> 78a1d071 (cbbd1566 + changelog). The
   cycle-4 work is now COMMITTED; I graded the on-disk state and re-derived scope after.

### Deterministic
- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && python -c "import ast;
  ast.parse(open(\"scripts/harness/attempt_budget.py\").read()); print(\"parses\")"'`
  -> stdout "parses", EXIT=0.
- attempt_gate.py --self-test -> SELF-TEST PASSED, rc=0, 16 checks all ok.
  Production ledger UNCHANGED at 16 lines, md5 1a8aad95f1a5d6cf74c250e6fa724593
  (the cycle-3 def-time-default pollution bug stays fixed).
- mutation_matrix_86_71.py --verify -> cells=7 killed=7 real survivors=0 errors=0,
  CONTROL green first, relocated-unmutated SURVIVES, null-mutant SURVIVES,
  BYTE-IDENTICAL RESTORE md5 e284ecb7f7663274d06f98b1a0d450f8, VERIFY: PASS, rc=0.
  Reproduces Main's stated md5 exactly.
- ruff F821/F401/F811 over a DERIVED 9-file scope (union of
  `git diff --name-only 192ef652^ HEAD -- '*.py'`, `git diff --name-only HEAD -- '*.py'`,
  `git ls-files --others --exclude-standard -- '*.py'`; non-empty guard asserted;
  xargs -0, no unquoted var) -> "All checks passed!", exit 0.
- pytest backend/tests/test_phase_86_32_attempt_budget.py -> 15 passed.
- mutation_matrix_86_32.py --verify -> all 8 cells KILLED, control green, target restored.

### MY OWN mutation work (control GREEN first in every block; tree md5 never changed)

CYCLE-4 FIX #1 (tautology fix at attempt_gate.py:370-377) -- PROVEN LOAD-BEARING.
  CONTROL relocated-unmutated self-test: rc=0, 0 FAIL.
  M-A (blank-reason path APPENDS a row but STILL returns 2 -- the exact defect the
  check names): against the FIXED source rc=1, exactly ONE failing check,
  "FAIL  refused extension appends NO row". Correct, singular attribution.
  M-A against the REVERTED fix (before_rows moved back AFTER the act): rc=0, 0 FAIL
  -- SURVIVES. So the FIX itself is what kills. Cycle-3 blocker #1 CLOSED.

CYCLE-4 FIX #2 (crash-class ERROR in mutation_matrix_86_71.py:271-277) -- PROVEN
LOAD-BEARING, and BOTH halves of the OR are individually reachable:
  Z3 (NameError at module import) with current guard -> ERROR, run_matrix rc=1. (At
    cycle 3 this scored KILLED. Probe Z3 CLOSED.)
  Z3 with the guard REMOVED (broken=False) -> KILLED, rc=0. Guard load-bearing.
  Null mutant (comment only) -> SURVIVED. Harness discriminates.
  Y1 (sys.exit(3) at import: rc outside {0,2}, NO traceback) -> ERROR. rc-half fires alone.
  Y2 (traceback on stderr from an unraisable __del__, exit code normal) -> ERROR.
    traceback-half fires alone. Neither half is dead.
  Y3 (exception INSIDE handle_hook -- swallowed by the documented fail-open handler,
    rc=0, no traceback) -> KILLED "by: below-ceiling launch is COUNTED". Attribution is
    CORRECT: the mutant genuinely stops counting. Not a finding.
  Y4 (cmd_extend raises; crash reachable only via _extend_probe) -> KILLED "by: an
    operator extension WITHOUT --reason is REFUSED". A CRASHED mutant scored a KILL.
    The guard reads only obs["below"]; _extend_probe/_corrupt_probe return no stderr.
    -> the artifact sentence "a crashed mutant is never a kill whatever its exception
    was called" is broader than the implementation (whose own label says "mutant failed
    to import", which IS the honest scope). NOTE-level: no shipped cell is affected.

CYCLE-4 FIX #3 (loud stderr disclosure in verdict_outcomes, attempt_gate.py:168-178)
-- TWO REAL SURVIVORS. This is the capping finding.
  V1: revert the fix (silent `except Exception: return []`) -> matrix 9/9 checks GREEN
      (SURVIVED), self-test 16/16 GREEN rc=0. NOTHING goes red. Criterion 8 says
      "revert it and show the check goes red"; reverting this new guard shows nothing.
  V2: same branch, `return [Outcome.PASS]` (a fail-OPEN budget bypass: any ledger read
      error grants the permanent PASS exception) -> matrix SURVIVED, self-test rc=0.
  ROOT CAUSE, measured: emit_sequence on an ABSENT ledger returns [] quietly (rc=0,
      "absent -> []"), and every drive/self-test points VERDICT_LEDGER at an absent
      path -- so the except branch is UNREACHABLE from every automated check.
  grep corroboration: "verdict-ledger read failed" exists in exactly 2 places, the
      source line and the live_check narrative. No test, no self-test check, no cell.
  Its only evidence is the hand-run §10 demo, which I DID reproduce verbatim
      (rc=0, identical stderr, 1 temp row, production ledger md5 unchanged).
  NAMED FIX: one self-test check + one matrix cell driving the hook with
      ATTEMPT_GATE_VERDICT_LEDGER pointed at a directory; assert the stderr line AND
      that the disposition is unchanged. The same fixture closes V2.

### Criteria re-derived from the production side (not from Main's prose)
C1 MET. Ran live_check §7's script verbatim: 497 runs / 332 repeats = 66.8%,
   qa 314/403 = 77.9%, researcher 18/93 = 19.4%, max qa on one step 36.8 -> 9.
   I also reproduced the DECOMPOSITION exactly: oldest 513 records -> 64.7%;
   oldest 580 -> 481/320 = 66.5% (§1's exact figures -- a real snapshot, not
   fabricated); full 596 -> 66.8%. Disagreement with the filed 58.4% reported, not
   adopted. NOTE: §1 still carries no corpus-size stamp (cycle-3 fix-list item 3).
C2 MET. My OWN positive controls: "BudgetState" and "escalation_summary" each hit
   backend/tests/test_phase_86_32_attempt_budget.py (control passes); negative control
   ZZZ_NO_SUCH_SYMBOL_86_71 -> 0 (search discriminates). At 192ef652^ the only
   referencing files were the test + mutation_matrix_86_32.py + verify_counter_86_79.py
   -- zero runtime callers. attempt_gate.py:84 is the new import. (My first run of this
   search was defeated by zsh globbing an unquoted --include=*.py and printed a false
   0; re-run quoted.)
C3 MET. 15 SEPARATE OS processes on a temp ledger: 7 launch processes each followed by
   an independent --status process reading 1/5,2/5,3/5,4/5,5/5 then DENY at #6 and #7.
C4 MET. Launch #6 rc=2 with the deny message and escalation_attempt_budget_77.9.md
   written; a below-ceiling step (77.10) on the same ledger rc=0 CONTINUE/allow.
   Strongest evidence: the LIVE hook counted MY OWN launch as production row 16.
C5 MET. 4,368 cells (all non-PASS sequences len 1..6 x product_verified x
   evidence_complete): disposition set and close_kind set are BOTH exactly
   {CONTINUE, ESCALATE}. Every at/over-ceiling non-PASS history -> ESCALATE.
   POSITIVE CONTROL discriminates: a PASS reaches CLOSED_PASS / CLOSED_COMPLETE.
   FAIL-only history -> CONTINUE under every flag combination; never a close.
   Escalation body: "THIS IS NOT A PASS AND NOT A FAIL", no verdict key.
C6 MET. Every production attempt row is a qa-verdict.js launch; row 16 is mine.
C7 MET. No .env in the 86.71 commits; the only config change is the +11-line
   PreToolUse/Workflow hook block (the step's product). ASK-1 at contract:88.
   masterplan.json untouched; all 8 criteria still byte-verbatim in the contract.
C8 NOT FULLY MET -- see V1/V2 above.

### Other findings (non-blocking, reported)
- live_check §10 uses `cmd | tail -N; echo EXIT=$?` three times. DEMONSTRATED: that
  shape prints EXIT=0 for a command that exits 7. The three captured EXIT=0 values are
  tail(1)'s status, not the commands'. The underlying facts hold (I re-derived all
  three unpiped: self-test 0, matrix 0, ruff 0), but the capture cannot distinguish
  pass from fail. qa.md 1a calls out exactly this shape.
- live_check §8 line 252 still describes the SUPERSEDED three-name marker list
  (ModuleNotFoundError/ImportError/SyntaxError). Cycle 4 corrected the sentence in
  experiment_results in place but left the live_check copy stale.
- The 5 session_id=pipetest rows for synthetic step 999.2 in the git-tracked production
  audit stream are still not called out as residue (cycle-3 fix-list item 4); the 9.4
  extension row IS disclosed at live_check:292-295.

### Scope / integrity
- HEAD MOVED during my run: cadab378 -> 78a1d071 (cbbd1566 landed the cycle-4 work
  mid-eval). Re-derived everything afterwards; HEAD stable at 78a1d071 since.
- cbbd1566 bundles 86.84 + 86.85 + 86.71 but is SCOPED (13 files), not `git add -A`:
  the dirty frontend/sovereign_api peer work was NOT swept in. All 6 peer .py files
  reference attempt_budget/attempt_gate 0 times each.
- handoff/current/evaluator_critique_86.71.md is UNTRACKED (not gitignored --
  check-ignore rc=1); its 86.84/86.85 siblings are tracked. Hygiene note for Main.
- Tree unchanged by me: attempt_gate e284ecb7f7663274d06f98b1a0d450f8,
  attempt_budget 5511ac7e6f105b6b0716d4b80812a170,
  mutation_matrix_86_71 01e22ddeae90c31ad6d7ef23a4af8ae5,
  production audit ledger 1a8aad95f1a5d6cf74c250e6fa724593 at 16 lines --
  all identical before and after every mutation block. Every mutant ran from a
  Python tempfile copy or an in-memory exec; no write to the repo tree.
- Gates N/A: 1b (no frontend/** in the graded change), 1c (no UI claim; no capture
  taken and none required), 1d (no backend/** source; I exercised the gate live rather
  than merely importing it).
- WRITES: only this file, under .claude/agent-memory/qa/verdicts/. No write blocked.

### Verdict reasoning
Criteria 1-7 MET and independently proven. Criterion 8's two cycle-3 blockers are
genuinely closed and I proved each load-bearing by reverting it. But one of the three
cycle-4 changes ships with zero automated coverage and its revert turns nothing red,
and the branch it edits is unreachable from every check -- so a fail-OPEN budget-bypass
mutation of the same branch also survives. Worst-of-lenses: correctness PASS,
reproduce PASS, scope-honesty CONDITIONAL, criterion-8 NOT MET -> CONDITIONAL.

COMPLETED: 2026-08-17T11:52:10Z
