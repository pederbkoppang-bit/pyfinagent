STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.97
COMPLETED: 2026-08-16T20:26:41Z
WRITTEN: 2026-08-16T20:11:33Z

# Q/A write-first record — step 86.97, CYCLE 2

Prior verdict per Main's advisory: CONDITIONAL (wf_3be25861-bde), on criterion 6.
Main's advisory is ADVISORY ONLY (Main is the constrained party). Will gather
qa_wip.py + verdict_history_86_21.py --evidence-only independently.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`
   + git status/diff scope + ruff lint gate
C. LLM judgment vs 7 immutable criteria; mutation-test the NEW guards myself
   (esp. the fixed buildable() oracle — cycle-1's finding was that a non-building
   mutant scored as a KILL)

## Findings log (appended as established)

## A. Harness compliance (5 items)
- research_brief_86.97.md EXISTS (mtime 2026-08-16T21:46:14)
- contract_86.97.md EXISTS (21:48:29) -- AFTER research. ORDER OK.
- experiment_results_86.97.md (22:09:22), live_check_86.97.md (22:10:09),
  scripts/qa/verify_decision_log_86_97.py (22:10:48) -- all AFTER contract. OK.
- log-last: `grep -cE "phase=86\.97" handoff/harness_log.md` = 0 (rc=1). Masterplan
  status for 86.97 = "pending". LOG-LAST OK (not yet logged, not yet flipped).
- no-verdict-shopping: evidence CHANGED. commit 6c801544 touches
  experiment_results_86.97.md (+32-?), live_check_86.97.md (+137), and
  scripts/qa/verify_decision_log_86_97.py (+132). 3 files, 250 ins / 51 del.

## Attempt / sequence evidence
- qa_wip.py 86.97 --spawned-at 2026-08-16T20:11:33Z:
  attempt_number=2, prior_attempts=1, attempt_number_status="ok",
  attempt_number_is_lower_bound=false, source_present=true, records_retained=2
  (gauge, not a counter), records_pruned_known=null.
- verdict_history_86_21.py --step 86.97 --evidence-only: status=no_rows_for_step,
  verdicts=(none).
- CROSS-CHECK: attempt_number (2) > ledger verdict count (0) => THE LEDGER IS STALE.
  sequence: UNRELIABLE from the ledger. Main's advisory says cycle-1 = CONDITIONAL
  (wf_3be25861-bde); that is ADVISORY ONLY.

## B. Deterministic
- IMMUTABLE COMMAND `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`
  -> stdout "parses", EXIT 0. REPRODUCED.
- scripts/qa/verify_decision_log_86_97.py: ALL GREEN 27 passed, 0 failed, exit 0.
  REPRODUCED (Main claimed 27/0).
- Re-derived at MY execution time: commits=55, decision lines=28, gap=27,
  recursion-guard commits=28 -> |27-28|=1 <= 2. NOT the pinned 10-vs-5. GOOD.
- detector heredoc measured at lines 43..387 (matches the contract's "moved to 387").
- ruff on the DERIVED scope `git diff --name-only 52358053..HEAD -- '*.py'` =
  [scripts/qa/verify_decision_log_86_97.py] (non-empty, passed via xargs -0, never
  an unquoted var). F821,F401,F811 -> "All checks passed!" exit 0. DEFAULT ruleset
  -> "All checks passed!" exit 0. Main's "ruff fully clean" REPRODUCES.
- Scope: `git diff --name-only 52358053..HEAD` = 10 files, all in scope
  (.claude/hooks/post-commit-changelog.sh docstring only, masterplan +21 add-only,
  CHANGELOG.md hook-generated, 4x 86.97 artifacts, 2x 86.91 artifacts, the checker).
  NO backend/**, NO frontend/** -> gates 1b/1c/1d N/A. No UI claims in this step.
- masterplan diff 52358053..HEAD: 21 ADDED lines, 0 REMOVED. The only add is the
  new step 86.103. NO status flip, NO criterion edited, NO verdict altered.

## C. MY OWN MUTATION MATRIX (run in-memory via stdin; ZERO repo/scratchpad writes)
Driver: reads the shipped checker, rewrites only REPO/HOOK anchors, execs it against
a mutated hook in a TemporaryDirectory. Note qa-write-guard BLOCKED my attempt to
write a driver file into the scratchpad; I used a stdin heredoc instead (disclosed).

CTRL   unmutated through MY driver ........ ALL GREEN 27/0   (driver proven live)
M-A    neuter compile() leg of buildable() . RED  - ORACLE-NO fails => compile leg IS load-bearing
M-A2   cycle-1 state (no compile leg, no rc
       check) + a SyntaxError mutant ....... the mutant scores "KILLED" => cycle-1
       defect REPRODUCED exactly; the fix is what closes it
M-B    inject a NON-BUILDABLE mutant ....... RED  - "UNSCORABLE ... cannot be scored
       as a kill" => criterion 6's UNSCORABLE arm is WIRED, not dead code
M-C    BUILDABLE but runtime-CRASHING mutant
       (ModuleNotFoundError) ............... RED  - the new rc==0 assert catches it
M-C2   same mutant, rc-check REMOVED ....... ALL GREEN 26/0, scored KILLED =>
       the rc==0 assert is INDEPENDENTLY load-bearing; compile() alone does NOT
       close the crash-as-kill class (it closes only the SyntaxError subset)
M-D    inject `if false; then exit 0; fi` ... RED  - [2] under-match self-test names
       line 18 => criterion 2's recall gate is real
M-E    inject a NEW unclassified pre-detector
       exit ................................ RED  - "UNCLASSIFIED guard ..." + 3/4
M-F    reword the recursion-guard condition . RED  - unclassified AND the DRIVEN
       recursion-guard cell goes red (it logged a line) => behavioural, not a scan
M-G    write the log into a nonexistent
       subdir .............................. RED  - CONTROL dies first and the
       matrix refuses to credit kills; anchor-uniqueness also fired

## C2. INDEPENDENT CROSS-CHECK OF THE 86.91 SURVIVAL (criterion 1b / 4)
Ran scripts/qa/verify_changelog_flip_86_91.py through the same driver:
  CONTROL (unmutated hook) ................ ALL GREEN 42/0   <- control GREEN FIRST
  MUTANT  (delete the production call) .... ALL GREEN 42/0   <- SURVIVES. CONFIRMED.
Extraction re-derived with the SHIPPED extractor (heredoc_python + detector_source
+ NEEDED lifted verbatim out of the 86.91 checker -- NOT re-implemented):
  52358053 (parent) bytes=7597 sha1=f7458a6ab1f5fe96  <- the pinned figure, EXACT
  3894ac71/6c801544/HEAD/worktree bytes=8617 sha1=072056e58af2befa
  call-DELETED at HEAD: byte-identical to HEAD = True
=> the PROPERTY reproduces at HEAD; the PINNED FIGURE (7,597 / f7458a6ab1f5fe96)
   is PRE-CORRECTION and does NOT reproduce against the shipped tree. It was
   invalidated by the criterion-5 docstring edit inside _log_decision, which rode
   in the SAME commit (3894ac71) that states it. FINDING F1 below.

## C3. MORE HARNESS MUTATION
M-H  corrupt the isolation SNAPSHOT ......... all 3 ISOLATION asserts go RED =>
     assert_isolated reads the real file at CHECK time; not a tautology
M-I  force the decision-log input absent .... "[3] the real decision log exists"
     goes RED with "...SKIPPED, and that is reported" => H4 fix works, no vanish

## C4. INDEPENDENT DRIVE OF BOTH SHIPPED MUTANTS + THE RETIRED ONE
baseline ............ rc=0 stderr='' writes changelog-decisions.log (proper line)
retarget (SHIPPED) .. rc=0 stderr='' writes changelog-decisions.log.RETARGETED
                      with byte-identical content; canonical file ABSENT
                      => kills by DESTINATION, no exception, every name defined.
                      H2's stated mechanism is EXACTLY what happens.
os.devnull (RETIRED)  rc=0 stderr="[changelog] decision-log FAILED (NameError:
                      name 'os' is not defined)"; NO files => the cycle-1
                      mis-attribution REPRODUCES exactly.
delete-call (SHIPPED) rc=0 stderr='' no files.
`os` in the heredoc: 0 imports, 0 uses, 0 word occurrences (343 heredoc lines).

## C5. CRITERION 7 -- PROVABLY BEHAVIOUR-PRESERVING
heredoc AST with docstrings stripped, 52358053 vs HEAD .... IDENTICAL
bash half outside the heredoc, 52358053 vs HEAD ........... IDENTICAL
raw file .................................................. differs (docstring only)
R5 rail: qa.md / qa-verdict.js / research-gate.js / researcher.md UNTOUCHED
         (git diff --name-only 52358053..HEAD returns nothing for them; worktree clean)
verify_changelog_flip_86_91.py UNTOUCHED => criterion 4's "must not be patched to
pretend otherwise" is satisfied.
No evaluator_critique file in the diff => no verdict altered.

## C6. NO-REGRESSION CLAIMS -- ALL THREE REPRODUCE
verify_changelog_flip_86_91.py   ALL GREEN 42/0   (reproduced)
verify_workflow_args_boundary.mjs ALL GREEN 96/0  (reproduced, rc=0)
verify_research_gate_workflow.mjs ALL GREEN 124/0 (reproduced, rc=0)

## C7. RESEARCH GATE
brief_status COMPLETE, gate_passed true, 8 read-in-full (floor 5), 23 URLs
(floor 10), recency_scan true, audit_class false, 9 internal files. Contract
cites the researcher. All 7 immutable criteria VERBATIM in contract_86.97.md.
Contract-completeness: all 7 have covering evidence in experiment_results_86.97.md.

## D. CRITERIA
C1 MET  - gap RE-DERIVED at my execution time (55/28/27 vs 28 recursion), NOT the
          10-vs-5; delete-the-call SURVIVES the 86.91 checker with CONTROL 42/0
          GREEN FIRST -- both independently reproduced by me.
C2 MET  - written-down rule in source; recall self-test vs a dumber scan; 3/3
          known members (the masterplan audit_basis names exactly those 3, so the
          known-member set was NOT author-chosen); per-member reason; M-D and M-E
          prove both the recall gate and the classification gate go RED.
C3 MET  - recursion guard DRIVEN (exits 0, writes no line), judged
          LEGITIMATELY-SILENT, stated as a BOUND, tied to the gap by an asserted
          RELATIONSHIP not a pinned number; M-F proves the cell is behavioural.
C4 MET  - whole heredoc driven end-to-end vs a temp repo; deleting the call turns
          the guard RED (mine: KILLED) while the 86.91 checker stays 42/0; the
          86.91 extractor was NOT patched.
C5 MET  - 3 sites REPLACED in place (hook docstring, contract_86.91 P2,
          experiment_results_86.91 heading+claim); immutable text untouched.
          Residual F4 below.
C6 MET  - control-GREEN-first bites (M-G); UNSCORABLE arm WIRED and FAILS (M-B);
          oracle two-sided and load-bearing (M-A); rc==0 assert INDEPENDENTLY
          load-bearing (M-C2 scores the crash a KILL without it). Residual F2.
C7 MET  - behaviour-preserving (above); masterplan +21 add-only (86.103 filed,
          status pending); no flips; no verdict artifact touched.

## E. FINDINGS (none blocking; two WARN cap the verdict at CONDITIONAL)
F1 WARN  "MEASURED ... BYTE-IDENTICAL (7,597 B, sha1 f7458a6ab1f5fe96)" does NOT
         reproduce against the shipped tree (8,617 B / 072056e58af2befa). It is
         EXACT at 52358053 and was invalidated by the criterion-5 docstring edit
         riding in the SAME commit. Sites: verify_decision_log_86_97.py:15
         (docstring, shipped file), contract_86.97.md:50-51,
         experiment_results_86.97.md:24, live_check_86.97.md:43-44 (a block
         headed "Measured:"), research_brief_86.97.md:122-123, AND
         .claude/agent-memory/researcher/project_uncalled_function_86_97.md:17-18
         -- the last is auto-loaded into every future researcher session, i.e. a
         FORWARD-LOOKING consumer. The PROPERTY reproduces; only the figure is
         stale. Fix: state the commit it was measured at, or re-derive.
F2 WARN  Criterion 6 says "every new guard". Sections [1] PRECONDITIONS and [2]
         ENUMERATION/CLASSIFICATION have NO mutation cell in the shipped matrix,
         and unlike the disclosed :214 gap this one is absent from "Scope honesty
         -- what I did NOT do". I mutation-tested them myself: M-D/M-E/M-F/M-H,
         ZERO survivors. Coverage+disclosure gap, NOT a vacuity.
F3 NOTE  Two failure-DETAIL strings mis-name the mechanism -- exactly the class
         this cycle fixed. :422 prints "bash -n rejected the mutant" when the
         compile() leg did the rejecting (observed in M-B). :435 prints "the
         mutant STILL produced a decision line ('')" when the KILLED check failed
         on rc!=0 and the log was in fact empty (observed in M-C). Scoring is
         correct; only the diagnostics mislead a maintainer.
F4 NOTE  experiment_results_86.91.md:186 still reads "An unexplained `none` is no
         longer expressible *(bounded -- see below)*" -- an accompany-form
         pointer surviving 15 lines inside the very section 86.97 rewrote for
         accompanying rather than replacing, while the new text asserts "The
         claim itself now carries its own bound". Harm is low: the section head
         and lead sentence now carry the bound in place, and the pointer still
         resolves (:437). Pre-existing from 86.91 cycle-3, not introduced here.

VERDICT: CONDITIONAL. All 7 criteria MET and independently re-executed; harness
compliance 5/5; no unintended production change. Cap is on evidence integrity
(F1) and scope honesty (F2), both narrow, both cheap, both of the step's own
defect classes. No surviving mutant found anywhere.
