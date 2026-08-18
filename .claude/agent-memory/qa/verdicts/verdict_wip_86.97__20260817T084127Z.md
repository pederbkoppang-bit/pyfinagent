STATUS: COMPLETE -- write-first record, still NOT a verdict
COMPLETED: 2026-08-17T08:53:57Z
STEP: 86.97
WRITTEN: 2026-08-17T08:41:27Z

# Q/A write-first record -- step 86.97, cycle 4 (HEAD claimed 2d861f5f)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`;
   git status/diff scope; ruff lint gate; scoped tests
C. Independent re-run of `scripts/qa/verify_decision_log_86_97.py` (control 48/0 claimed)
D. Independent mutation matrix N-1..N-5
E. Attack vectors 1-5 from the spawn prompt
F. Criterion-by-criterion MET/NOT MET

## Log
- 08:41:27Z qa.md read in full. Write-first record created.

### Prior-attempt / sequence evidence
- `qa_wip.py 86.97 --spawned-at 2026-08-17T08:41:27Z`: source_present=true,
  attempt_number=4 (status ok, is_lower_bound=true), prior_attempts=3,
  records_retained=4 (gauge, not a counter), records_pruned_known=null.
- `verdict_history_86_21.py --step 86.97 --evidence-only`: status=`no_rows_for_step`,
  verdicts=(none). CROSS-CHECK: attempt_number (4) > ledger count (0) -> LEDGER IS STALE.
  sequence: UNKNOWN from the ledger. harness_log carries exactly ONE 86.97 result row
  (Cycle 229, 2026-08-16, result=FAIL, PARKED at the 3-attempt cap) -- secondary only.

### HEAD
- HEAD = 5ee4d89e (auto-changelog for 2d861f5f). Work commit = 2d861f5f (10:40:53 +0200),
  6 files: contract_86.97, experiment_results_86.97, live_check_86.91, live_check_86.97,
  research_brief_86.97_cycle4, scripts/qa/verify_decision_log_86_97.py.

### B. DETERMINISTIC
- IMMUTABLE CMD `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`
  -> stdout "parses", exit 0. DISCLOSED (correctly) as unable to fail on this class.
- ast.parse on verify_decision_log_86_97.py -> OK.
- LINT GATE (scope derived: work-commit .py files U untracked .py; non-empty asserted):
  `uvx ruff check --select F821,F401,F811 scripts/qa/verify_decision_log_86_97.py`
  -> "All checks passed!", exit 0.
- No frontend/** or backend/** in the diff -> 1b/1c/1d not applicable. No UI claims.
- CONTROL: `python scripts/qa/verify_decision_log_86_97.py` -> ALL GREEN: 48 passed,
  0 failed, exit 0. md5 of the hook IDENTICAL before and after
  (04296c78bce1547b913f4625b737123c).

### A. HARNESS COMPLIANCE (5/5 CLEAN)
1 research-gate: research_brief_86.97_cycle4.md exists (25,683 B), contract cites the
  ENFORCED run wf_aeceef87-d82 with the enforcer's numbers (6>=5 sources, 26>=10 URLs,
  recency ok, brief_status COMPLETE, self_report_disagreed false).
2 contract-before-generate: mtime chain research 10:34:45 < contract 10:36:16 <
  checker 10:37:36 < live_check_86.91 10:39:24 < experiment_results 10:40:17 <
  live_check_86.97 10:40:36. All 7 criteria appear VERBATIM in the contract.
3 experiment_results_86.97.md present (16,238 B).
4 log-last: harness_log has 1 row for 86.97 and it is the PRIOR cycle's PARK
  (Cycle 229 result=FAIL); cycle 4 is NOT logged. masterplan 86.97 status=pending.
5 no-verdict-shopping: evidence CHANGED -- 2d861f5f rewrites the checker (+148/-?) and
  5 artifacts since the cycle-3 verdict. Not a re-spawn on unchanged evidence.

### D. INDEPENDENT MUTATION MATRIX (my own harness; hook NEVER touched on disk)
Method: inject a replace on the checker's HOOK_SRC read, exec in-process. Every cell
asserts the anchor is unique AND that bytes changed. Control validated first.
- CONTROL                     rc=0  ALL GREEN: 48 passed, 0 failed   md5 unchanged
- N-1 delete `bump_type=_flip_magnitude()`  rc=1  41 passed, 7 failed  KILLED (matches)
- N-2 force the :267 .get default           rc=1  42 passed, 6 failed  KILLED (matches)
- N-3 swap flip_created/flip_transitioned   rc=1  46 passed, 2 failed  KILLED (matches)
- N-4 subject-major stops recording reason  rc=1  46 passed, 2 failed  KILLED (matches)
- N-5 N-1 mutant + [3a] deleted             rc=0  34 passed, 0 failed  SURVIVED (matches)
- N-5b MY ADDITION: clean + [3a] deleted    rc=0  34 passed, 0 failed
  => 34/0 identical with and without the mutant, so [3a] removal does not mask the kill
     for an unrelated reason. N-5 IS a real attribution control (attack vector 5 answered).
All five reported cells reproduce EXACTLY, digit for digit. md5 unchanged in every cell.

### MY OWN ADVERSARIAL CELLS
- Q1 hook `return "minor"` -> `return "patch"` (bump magnitude only): **SURVIVED**
  48/0. `bump` is PARSED by DECISION_RE but asserted by NOTHING.
  MITIGATION VERIFIED: the sibling gate verify_changelog_flip_86_91.py KILLS it
  (`[1] magnitude: a created X.0 kickoff is minor, not patch -- got 'patch'`, 41/1),
  and that gate is run in this step's own no-regression block.
- Q2 swap the created_done/transitioned_done RECORDING keys: KILLED 46/2.
- Q3 recursion guard stops matching (criterion-3 bound violated): KILLED 44/5, incl.
  `[3] recursion guard: and writes NO decision line (the BOUND, measured)`.
- Q4 writer drops created_done/transitioned_done from the format string: KILLED 35/5.

### CRITERION-1 SURVIVAL, INDEPENDENTLY REPRODUCED
- 86.91 guard, CONTROL          rc=0  42/0
- 86.91 guard, delete-the-call  rc=0  42/0  -> SURVIVED (control green FIRST). Matches §A.
- 86.97 PRE-FIX checker (2d861f5f~1), CONTROL   rc=0  35/0  (matches the claimed "was 35")
- 86.97 PRE-FIX checker, N-1 mutant             rc=0  35/0  -> SURVIVED. Blocker A reproduced.
- 86.97 PRE-FIX checker, delete-the-call        rc=1  27/5  -> KILLED (the [4] cell worked pre-fix)
- Gap re-derived by ME at execution time: commits=89 lines=45 gap=44 recursion=45.
  Artifact §J states 87/44/43/44; the delta is exactly +2 commits / +1 line / +1 recursion,
  i.e. the work commit + its auto-changelog entry. Fully consistent; the figure MOVES, as
  the artifact says, and the checker asserts the RELATIONSHIP not a number.

### INDEPENDENT SOURCE DERIVATIONS (attack vectors 1, 2, 4)
- The 9 reason states, derived by ME from `grep -n reason` on the hook:
  :160 masterplan_unreadable_at_HEAD, :163 first_commit, :190 no_flip,
  :193/:194/:195 flip_created / flip_transitioned / flip_created_and_transitioned,
  :207 detector_error:<Type>, :216 subject_forced_major, :267 `.get` default `unrecorded`.
  EXACT match with the contract's table, site for site. (attack 1: the table is correct.)
- The four expected (reason, created, transitioned) triples re-derived from branch
  structure at :180-205 and :213-216 -- all four match the shipped SCENARIOS table.
  The "derived BEFORE driving" ordering is a PROCESS claim not checkable from artifacts;
  what IS checkable is that the values are correct by independent derivation AND that the
  table discriminates (4 distinct reasons, 4 distinct bumps; the DISCRIMINATE assertion is
  load-bearing -- it fired under N-1 (2 distinct) and N-2 (1 distinct)).
- Exit enumeration by ME: `grep -n '\bexit\b'` -> 28/33/37 (pre), 228+362+368 (inside the
  43..387 heredoc), 394/396/397 (post). 3 pre + 3 post -- EXACT match with the checker.
  KNOWN-MEMBER RECALL: PASSES.
- Criterion-3: the recursion-guard bound is behaviourally guarded, not assumed -- my Q3
  mutant turns `[3] recursion guard: writes NO decision line` RED.

### CRITERION-5 SWEEP (attack 3), two independent operationalisations
- Known-member recall (members chosen by the masterplan note + the cycle-3 Q/A, NOT by me):
  6/6 present and corrected/bounded IN PLACE --
  live_check_86.91.md:104 (heading REPLACED), experiment_results_86.91.md:444 + :186-188,
  contract_86.91.md:145 + :150, hook :222.
- My own semantic sweep (2 regex families, quoted globs) over *.md/*.sh/*.py/*.json:
  NO residual unbounded carrier. harness_log:35730 and research_brief:55/101/284 are
  QUOTATIONS OF THE DEFECT inside the failure write-up, not carriers.
  masterplan hits are all inside 86.97's own (correctly bounded) defect notes.
  `.claude/agent-memory/researcher/` -> ZERO hits.
- Deliberate non-edits (immutable criterion text at contract_86.91.md:105 + masterplan;
  evaluator_critique_86.91.md) are correctly scoped and correctly disclosed.

### FINDINGS (none is a criterion miss)
W1 [WARN] `bump` is PARSED by DECISION_RE and asserted by NOTHING. Executed proof: the
   hook mutant `return "minor"` -> `return "patch"` at the phase-kickoff branch leaves the
   shipped guard at **ALL GREEN 48/0 -- SURVIVED**. experiment_results:241-243 and
   contract P1:126-129 read "the line is parsed into (bump, reason, created_done,
   transitioned_done) and compared by exact equality", which a reader takes as covering
   bump; §J4 "What is NOT claimed" lists the four undriven reason states but NOT this.
   MITIGATION VERIFIED, so this is disclosure and not a system hole: the sibling
   verify_changelog_flip_86_91.py KILLS the same mutant (41/1, `[1] magnitude: a created
   X.0 kickoff is minor, not patch -- got 'patch'`), and that gate is run in §G.
   The checker's OWN comment (:385) is honest ("expected reason, expected created,
   expected transitioned"); only the prose overreaches.
   FIX: add the expected bump to SCENARIOS (3 lines), or add the bound to §J4.
W2 [WARN] A number that does not reproduce under its own stated command. J2 /
   experiment_results:286 say the `grep -cE "reach(es|ed)? the detector|pre-detector|
   bash exit|recursion guard|86\.97"` over live_check_86.91.md "now returns 5".
   MEASURED: `grep -c` returns **4** (4 matching LINES); 5 is the `grep -o | wc -l`
   MATCH count (line 112 carries two). Verified identical at the worktree, at HEAD and
   at 2d861f5f; the "before" figure of 0 is correct at 2d861f5f~1. The substantive
   property (0 -> non-zero bounding language) reproduces; only the label is wrong.
W3 [NOTE] live_check_86.97.md is +86 lines APPENDED ONLY (`@@ -445,3 +445,89 @@`), so §E
   still prints `ok   [3] the decision line carries a reason` (:171) -- the very assertion
   cycle 4 proved VACUOUS and deleted -- and §G still reports "35 passed" while the guard
   is 48, all under a header (:3-4) asserting "Every block below is verbatim tool output
   from this session, complete and unelided." The correction accompanies rather than
   replaces, one section away from where criterion 5 enforces the opposite.

### OUT OF SCOPE, recorded not charged
Uncommitted edits to backend/api/sovereign_api.py + 5 frontend files are dated
2026-08-14 13:24-13:28, three days BEFORE this cycle (10:34-10:40 on 08-17), are absent
from 2d861f5f, and are unrelated UI work. NOT attributable to 86.97.

### CRITERIA
1 MET   both defects reproduced BY MY OWN EXECUTION (86.91 guard 42/0 -> 42/0 SURVIVED
        with control green first; pre-fix 86.97 checker 35/0 -> 35/0 SURVIVED under N-1);
        gap re-derived live at 89/45/44/45, never the 10-vs-5 figure.
2 MET   rule written down in source, self-tested against a dumber scan; my independent
        enumeration returns the identical 3 pre / 3 post members; each pre-detector member
        classified MUST-LOG or LEGITIMATELY-SILENT with a per-member reason; an
        unclassified member FAILS (proved by [5] and by my Q3).
3 MET   recursion guard judged LEGITIMATELY-SILENT and DRIVEN (exits 0, writes no line);
        stated as a BOUND; the gap tracks the recursion-guard count (44 vs 45 in my run).
4 MET   whole heredoc driven end-to-end in a temp repo; deleting the call turns the guard
        RED (reproduced); detector_source was NOT patched to pretend otherwise.
5 MET   6/6 known members corrected; the last unbounded carrier bounded IN PLACE at
        live_check_86.91.md:104; two independent sweeps find no residual carrier.
6 MET   control observed GREEN (48/0) FIRST and reproduced by me; 4 cells + attribution
        control reproduce digit for digit; buildable() self-tested in BOTH directions;
        every cell asserts mutant rc==0 so a crash cannot score as a kill.
7 MET   2d861f5f touches no masterplan.json, no qa.md/qa-verdict.js, no verdict artifact;
        86.97 still status=pending retry_count=3; earlier commits only ADDED the step
        record and the PARK note -- no status flip, no verdict altered.

### VERDICT REASONING
All 7 criteria MET and independently re-executed; harness compliance 5/5; immutable
command exit 0; lint exit 0; no unintended production change. Capped at CONDITIONAL by
W1 -- a mutation I executed that SURVIVES the shipped guard against an undisclosed bound
the artifacts' wording covers over -- and W2, a figure that does not reproduce under the
command it names, in the cycle whose stated purpose was to close a claim that did not
reproduce. Not FAIL: no criterion is missed and no product defect ships (W1 is covered by
the sibling gate, verified by execution).

