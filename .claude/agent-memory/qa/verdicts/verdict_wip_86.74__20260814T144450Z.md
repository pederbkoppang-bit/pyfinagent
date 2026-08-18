STATUS: INCOMPLETE -- not a verdict
STEP: 86.74
WRITTEN: 2026-08-14T14:44:50Z

# Q/A write-first record -- step 86.74 (risk-judge falsy-zero inversion)

Spawn: Workflow rail, Opus 5 (1M). Commit under review: 9d14291e.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable verification command, lint, diff scope, adjacent suites
C. Re-derive Main's counts (4 -> 0 `or 10.0`; 9->31 tests; 17->51 asserts)
D. Re-run mutation matrix scripts/qa/mutation_matrix_86_74.py
E. Independent mutations (fixture/harness shapes, not just code-side)
F. Criterion-by-criterion MET/NOT MET

## Prior-attempt evidence
- qa_wip.py 86.74 --spawned-at 2026-08-14T14:44:50Z: attempt_number=2,
  prior_attempts=1, attempt_number_status=ok, source_present=true,
  records_retained=2 (gauge, incl. mine).
- prior record verdict_wip_86.74__20260814T143725Z.md is STATUS: INCOMPLETE
  (a DROPPED spawn, ~7 min before me) -> it produced NO verdict. Its content is
  EVIDENCE only; I re-derived everything independently.
- verdict_history_86_21.py --step 86.74 --evidence-only: status=no_rows_for_step,
  verdicts=(none).
- CROSS-CHECK: attempt_number(2) > ledger rows(0) -> LEDGER IS STALE for this
  step. But prior_attempts=1 corresponds to a DROP, not a verdict, so
  "no prior verdict" is consistent. sequence: no prior VERDICTS recorded.
  Not verdict-shopping: there is no prior verdict to shop against.

## A. HARNESS COMPLIANCE (5 items) -- re-derived by me
1. research gate: research_brief_86.74.md 32,922 B, birth==mtime 10:24:44Z. PASS
2. contract-before-generate (UTC via stat -f%B + date -u -r):
   research 10:24:44Z < contract 14:19:46Z (birth==mtime, written once)
   < experiment_results 14:35:15Z < live_check 14:35:46Z < commit 14:36:20Z. PASS
3. artifacts present: contract 12,313 B / experiment_results 12,308 B /
   live_check 3,531 B. evaluator_critique_86.74.md ABSENT = correct pre-verdict.
4. log-last: masterplan 86.74 status="pending". NOT flipped. PASS
5. no-verdict-shopping: N/A (no prior verdict).

## B. DETERMINISTIC (mine)
- IMMUTABLE CMD: 34 passed, EXIT=0 (re-run clean, `echo $?` not PIPESTATUS).
- LINT: scope DERIVED `git diff --name-only 9d14291e^ 9d14291e -- '*.py'` = 6
  files, N=6 asserted non-empty, piped via xargs (NOT unquoted $VAR).
  uvx ruff --select F821,F401,F811 -> "All checks passed!" EXIT=0.
- WORKING TREE: dirty set = session-start snapshot only (.archive-baseline.json,
  sovereign_api.py, 5 frontend, 3 audit jsonl, health.jsonl) + my 2 WIP files.
  NONE touched by 9d14291e. `git diff 9d14291e HEAD -- backend/ scripts/qa/...`
  is EMPTY -> subject files unchanged since the commit. No frontend in the
  commit -> gate 1b N/A. No UI claim -> gate 1c N/A.

## C. RE-DERIVED COUNTS (mine, not the author's)
- `or 10.0` AST BoolOp sites in portfolio_manager.py:
  PRE(9d14291e^) N=4 at :507 :800 :853 :878  -> POST N=0. EXACT MATCH to claim.
  Repo-wide AST sweep of backend/**: 0 hits. Only literal 10.0 post-fix is :995
  (DEFAULT_POSITION_PCT).
- tests: AST FunctionDef test_* PRE=9 POST=31. MATCH. pytest --collect-only=34
  (31 funcs, 3 extra from parametrize). 2 REMOVED names are exactly the two
  defect-encoding tests the author disclosed; 24 added.
- asserts: grep -c 'assert ' PRE=17 POST=51 (the author's STATED rule -> MATCH).
  *** DISCREPANCY (NOTE): ast.Assert count is PRE=17 POST=50. The 51st grep hit
  is line 83, a COMMENT ("They now assert the corrected behaviour and").
  Direction (17->50/51) is unambiguous; a net removal would still be visible.

## D. FINDING -- criterion 3's enumeration is FALSE AS WRITTEN (WARN)
experiment_results_86.74.md:81-83 states `_sizing_pct`'s branches are
"SIZE -> the pct / UNPARSEABLE -> 0.0 / ABSENT -> the default. The default is
reachable from ABSENT and only ABSENT."
I EXECUTED _sizing_pct over a 5x5 state/pct grid + {}. 12 of 26 cells return
DEFAULT, in THREE families, only one of which is ABSENT:
  (state=ABSENT, any pct)  -> 10.0   [legitimate]
  (state=SIZE,  pct=None)  -> 10.0   [NOT absent: state says a size was given]
  (state=<unknown>, any)   -> 10.0   [NOT absent; OVERRIDES an explicit 0.0]
  (state=None,  pct=None)  -> 10.0   [derived ABSENT, legitimate]
The criterion demanded the set be DERIVED from source, not asserted; this
enumeration was asserted in prose and is incomplete.
REACHABILITY (my own grep, exhaustive): `position_pct_state` is written at
EXACTLY ONE site (portfolio_manager.py:409 = _verdict.kind) and read at exactly
one (:1011). _coerce_pct builds SIZE only as PositionVerdict(SIZE, float(raw)),
so SIZE always carries a float. sector_blocked (:521) appends the SAME cand
dicts, so the swap seams (:824/:877/:902) inherit the state key. => both
residual families are UNREACHABLE from production today. Live defect: NO.
Claim defect: YES. Classified WARN (Overgeneralization), not BLOCK.

## E. MUTATION MATRIX -- RE-RUN BY ME, 6/6 KILLED (reproduces)
control GREEN observed FIRST; M1..M6 all KILLED; "restore verified
byte-identical for all 4 subjects: True"; my own pre/post shasum -a 256 of all
4 subjects MATCH exactly, and `git diff` on them is EMPTY after the run.
  pre==post 884840c8.. (pm) 50bcf38b.. (loop) 2f89bdc8.. (attr) 15f1c454.. (rd)
*** HARNESS VACUITY PROBE (mine, the fixture/harness shape qa.md 4c says the
    independent evaluator must attack): mutant runs are `pytest -k <target>`.
    pytest exits **5** when -k selects NOTHING, and the harness scores
    `killed = rc_m != 0` -- so a renamed/typo'd target would score KILLED FOR
    FREE. I verified exit=5 empirically for -k ZZZ_NO_SUCH_TEST. The harness
    guards the mutation-APPLIED side (NOT_APPLIED) but NOT the selection side.
    AS EXECUTED THIS RUN the matrix is NOT vacuous: all 6 targets select
    non-empty sets (7,6,3,6,4,1 tests). Latent harness weakness -> NOTE/WARN.

## F. REGRESSION SWEEP -- whole-tree A/B, my own construction
Scope DERIVED: grep -rln for the 4 changed module names over backend/tests/
= 55 files. Ran the SAME 55 at HEAD and on a FULL pre-86.74 tree extracted via
`git archive 9d14291e^ | tar -x` (pm sha 2ae99e05.. != post 884840c8..).
  HEAD    : 7 failed, 1240 passed
  PRE-FIX : 7 failed, 1214 passed
COUNTS AGREE BUT MEMBERS DIFFER (the qa.md 4b trap). Symmetric difference = 2:
 - HEAD only : test_phase_23_2_6_sector_cap_emit::..._backend_log_has_skipping_buy_evidence
   -> asserts on the LIVE runtime backend.log ("Skipping BUY" count 0>=1). It
      SKIPPED in the extracted tree only because backend.log is gitignored and
      absent there. NOT code-attributable.
 - PREFIX only: test_phase_85_6_anchor_deadlock::test_c2_...
   -> fails with `git show ...: not a git repository` (returncode 128). Pure
      extraction artifact. Passes at HEAD.
=> CODE-ATTRIBUTABLE FAILURE SET IS IDENTICAL PRE AND POST. NO REGRESSION from
   86.74. The author's "both already failed at HEAD" claim REPRODUCES, and I
   verified it more strongly (whole tree, 55 files) than the author did.
*** FINDING (WARN, scope honesty): experiment_results.md section 5 says "Two
    tests in adjacent suites fail". Over the DERIVED affected scope there are
    SEVEN, including TWO MORE IN THE VERY FILE IT CITES
    (test_phase_57_1_reject_binding.py::test_reject_binding_main_path_off_emits_on_blocks
    and ::test_off_identity_prompts_are_verbatim_constants), plus
    test_phase_60_3_data_integrity::test_60_3_flag_defaults_off,
    test_phase_75_prompt_contracts::test_operator_decision_note_exists_with_token,
    test_phase_23_2_6_sector_cap_emit::..._skipping_buy_evidence.
    All 7 are pre-existing (I verified), so the SUBSTANCE is right; the SET was
    hand-narrowed rather than derived. Overgeneralization/under-disclosure.

## G. THE TWO REWRITTEN DEFECT-ENCODING TESTS -- inversion, NOT weakening
Old: `assert abs(b.amount_usd - NAV*0.10) < 0.5` ("10% NAV default (the bug)")
     `assert _buy(orders) is not None`          ("REJECT invisible -> buys")
New: `assert _buy(orders) is None` parametrised over shape_fix [False, True].
The new assertions are STRICTLY STRONGER: they forbid a buy the old ones
REQUIRED. A weakening would have made a buy MORE permissible; this does the
opposite. Old text is quoted in-file (lines 70-82) so the inversion is visible.
NOT a guard-weakening. Verified by reading the diff, not the author's summary.
