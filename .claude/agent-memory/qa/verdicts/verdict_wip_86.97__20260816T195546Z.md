STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.97
WRITTEN: 2026-08-16T19:55:46Z

# Q/A write-first record -- step 86.97

## Prior-attempt / sequence evidence (reported, not aggregated)
- `qa_wip.py 86.97 --spawned-at 2026-08-16T19:55:46Z` -> source_present=true,
  attempt_number=1, attempt_number_status="ok", attempt_number_is_lower_bound=false,
  prior_attempts=0, prior_records=[], records_retained=1 (gauge, includes me),
  records_pruned_known=null.
- `verdict_history_86_21.py --step 86.97 --evidence-only` -> status=no_rows_for_step,
  verdicts=(none). Cross-check: attempt_number 1 vs ledger 0 rows; prior_attempts (0)
  == ledger count (0), so no staleness detectable for this step-id.
- harness_log: `grep -cF 'phase=86.97' handoff/harness_log.md` = 0 (LOG runs after
  EVALUATE; expected).

## A. Harness compliance (5 items) -- CLEAN
1. research-gate-before-contract: research_brief_86.97.md present; envelope
   brief_status=COMPLETE, gate_passed=true, external_sources_read_in_full=8 (>=5),
   urls_collected=23 (>=10), recency_scan_performed=true. Contract cites enforced run
   wf_71bc038d-45a.
2. contract-before-generate (mtime chain): research 21:46:14 < contract 21:48:29 <
   guard script 21:50:24 < live_check 21:53:53 < experiment_results 21:54:34 (local).
3. experiment_results_86.97.md present (7,793 B).
4. log-last: 86.97 NOT in harness_log; masterplan status still `pending`. Correct.
5. no-verdict-shopping: attempt 1, no prior verdict. N/A.

## B. Deterministic
- IMMUTABLE: `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`
  -> `parses`, exit 0.
- Scope derived from the commit (3894ac71), not typed:
  9 files; only ONE .py: scripts/qa/verify_decision_log_86_97.py (NEW).
  Working tree also carries backend/api/sovereign_api.py modified -- PRE-EXISTING at
  session start, NOT touched by this commit.
- Lint gate: `uvx ruff check --select F821,F401,F811 scripts/qa/verify_decision_log_86_97.py`
  -> "All checks passed!" exit 0. (Non-empty file set asserted.)
  Full default ruff: 9 stylistic (EXE001/ISC004 x3/PLW1510 x3/FURB167/FURB105) -- not
  the project gate; NOTE only.
- Guard run: `python scripts/qa/verify_decision_log_86_97.py` -> ALL GREEN 20/0, exit 0.
- Sibling gates reproduce EXACTLY as claimed: 86.91 42/0; args-boundary 96/0;
  research-gate 124/0.
- Hook change is DOCSTRING ONLY -- verified mechanically, not read: heredoc AST with
  docstrings stripped is IDENTICAL parent vs HEAD; the bash half outside the heredoc is
  byte-IDENTICAL; delta is +1,020 B of docstring text.

## C. Independent re-derivations (all reproduce)
- 7,597 B / sha1 f7458a6ab1f5fe96 -- re-derived against the PARENT commit
  (3894ac71^): base=7,597 B sha1 f7458a6ab1f5fe96; call-deleted IDENTICAL. EXACT match.
  (At HEAD the same pair is 8,617 B / 072056e58af2befa, both identical -- the +1,020 B
  docstring, consistent.)
- detector_source confirmed to collect only FunctionDef/Assign/AnnAssign from tree.body
  -> ast.Expr(Call) can never match. Structural-blindness claim CORRECT.
- 86.91 checker run in-memory against a call-deleted hook: control ALL GREEN 42/0,
  mutant ALL GREEN 42/0 -> SURVIVED. Reproduced exactly.
- Gap re-derivation is live, not pinned: my run 53/27/26/27 vs claimed 47/24/23/24
  (21:03) and 51/26/25/26 (21:52). Monotone; |gap-recursion| = 1 at all three points.

## D. Mutation / vacuity testing done BY ME (control observed green first each time)
- CONTROL via runpy monkeypatch of Path.read_text: ALL GREEN 20/0, exit 0.
- CRITERION 4 EXECUTED DIRECTLY (not simulated): delete the production call ->
  the GUARD goes RED, exit 1, FAILED 15 passed/4 failed, headed by
  "[3] a decision line is WRITTEN TO THE FILE". Also fails closed at [4]
  ("CONTROL is dead", "anchor is unique -- found 0 occurrences"). GENUINE.
- M2 fourth UNCLASSIFIED pre-detector exit injected -> RED, 19/2,
  "UNCLASSIFIED guard 'if [ -n "${SOME_NEW_SKIP:-}" ]; then'". Classification NOT vacuous.
- M3 same-line `if [ -n "${Z:-}" ]; then exit 0; fi` (RULE under-matches, DUMB finds) ->
  RED, "unexplained at [32]". The self-test is REAL.
- MUST-LOG classification validated empirically (not required by the criteria, but
  checked): baseline rc=0 line=YES; CHANGELOG deleted rc=0 line=NONE stderr='';
  anchor renamed rc=0 line=NONE stderr=''. Classification is factually right.

## E. FINDINGS

### F1 [WARN] criterion 6: the UNSCORABLE arm is inert for the mutant class used
`buildable()` (`verify_decision_log_86_97.py:305-311`) is `bash -n`. Bash does NOT
parse a quoted heredoc (`<< 'PYEOF'`), so it cannot see Python build errors -- and
BOTH shipped mutants are Python-side mutations inside that heredoc.
MEASURED: mutant `_log_decision(bump_type` (unbalanced paren, -1 B):
  buildable() -> True;  compile() -> SyntaxError "'(' was never closed";
  driven -> rc=1, log empty, stderr shows the SyntaxError;
  the [4] scoring rule `m_log.strip()==""` would score it KILLED.
Criterion 6 requires a non-building mutant to be UNSCORABLE and FAIL. The mechanism is
present and correctly wired, but its oracle is blind to the only build failures these
cells could have -- the same structural-blindness defect this step exists to close,
reproduced in the guard's own scoring. Compounding: [4] never asserts the mutant's
rc==0, so ANY crash-mutant reads as a kill.
NOT falsified: both shipped mutants do compile, so no shipped cell is mis-scored today.
Fix: compile the heredoc body in buildable(), and assert m_rc == 0 beside the empty log.

### F2 [WARN] mutant 2's kill mechanism is MIS-ATTRIBUTED (vacuity shape 11)
`os` appears ZERO times in the heredoc (lines 43-387, `grep -c '\bos\b'` = 0). The
mutant `open(os.devnull, ...)` raises `NameError: name 'os' is not defined`, swallowed
by _log_decision's broad except, stderr:
  `[changelog] decision-log FAILED (NameError: name 'os' is not defined)`
The credited mechanism -- "removing the write itself, so the effect disappears without
the call moving" (script :319; experiment_results :66-68; live_check :170,176-178) --
is not what happened. The cell's CONCLUSION survives weakly (call text present, guard
still red), but the mechanism is wrong and undisclosed.

### F3 [WARN] scope honesty: "both mutants checked BUILDABLE" overstates
commit body, experiment_results and live_check:178 all state the buildability check as
if it protected these cells. Literally true (bash -n ran); the implication is false.
Neither artifact discloses bash -n's heredoc blindness nor the NameError of F2.

### F4 [NOTE] isolation assertion covers only the FIRST drive
`real_after` is read at :268, before the recursion-guard drive (:275) and both mutant
drives (:344). live_check:181-184 states the property more broadly than the code
implements. Low risk (CLAUDE_PROJECT_DIR set per drive); real log verified unchanged.

### F5 [NOTE] the gap re-derivation is inside `if real_before:`
If the real decision log is absent the whole block -- including its check() -- silently
vanishes rather than failing. Zero-assertion-passes-vacuously shape.

### Considered and NOT charged
`verify_changelog_flip_86_91.py:184-185` prints "EVERY 'none' IS EXPLAINED" /
"NO UNEXPLAINED 'none'". Looks like a surviving unbounded form for criterion 5, but
'none' is a value the DETECTOR produces; a pre-detector exit never yields a 'none' at
all, and 86.91's own criterion 4 is scoped to "when the detector decides 'none'".
Correctly scoped by its own vocabulary. Not a finding.

## F. Criteria
1 MET   - both defects reproduced by execution; 7,597 B/f7458a6ab1f5fe96 exact at
          parent; 42/0 survive reproduced; gap re-derived live (53/27/26/27).
2 MET   - rule stated in source; self-test proven real (M3 RED); preconditions
          asserted; classification keyed on condition text and proven non-vacuous
          (M2 RED); known members all found.
3 MET   - recursion guard DRIVEN (rc=0, no line) and recorded as a BOUND; :33/:37
          MUST-LOG and independently confirmed silent.
4 MET   - verified directly: deleting the call turns THE GUARD RED (exit 1, 15/4).
          Extraction not patched.
5 MET   - 3 sites REPLACED in place (heading + sentence rewritten, bound inline);
          my independent semantic sweep found no surviving live unbounded site;
          verbatim criterion/masterplan/critique correctly untouched.
6 PARTIAL - control-first MET and fail-closed; the "does not BUILD -> UNSCORABLE"
          clause is not in force for the mutant class used (F1). Plus F2/F3.
7 MET   - masterplan diff purely additive: 1,397 -> 1,398 steps, ADDED ['86.103'],
          REMOVED [], STATUS CHANGED {} across all steps. 86.97 still pending.
          No verdict altered.

## G. Verdict shape
Worst-of-lenses: correctness ~PASS, reproduce PASS, scope-honesty CONDITIONAL.
=> CONDITIONAL. Not FAIL: every behavioural guard is genuine and independently
verified RED under mutation; the gap is a fixable defect in the guard's own
UNSCORABLE arm plus two undisclosed mechanism/scope claims.

COMPLETED: 2026-08-16T20:04:28Z
