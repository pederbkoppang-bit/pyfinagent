STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.97
WRITTEN: 2026-08-17T08:59:58Z
COMPLETED: 2026-08-17T09:15:08Z

# Q/A write-first record -- step 86.97, cycle 5 (Main's count)

## Plan
A. Harness compliance audit (5 items)
B. Deterministic: immutable command `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`
C. Re-run scripts/qa/verify_decision_log_86_97.py (claimed 52 passed / 0 failed)
D. Mutation matrix independently: N-1..N-7 + my own new cells
E. Attack list from Main: (1) N-7 equivalence proof, (2) 4 pinned bump values correct per rule vs fitted,
   (3) remaining carriers of unbounded criterion-4 claim, (4) parsed-but-never-asserted fields
F. Criterion-by-criterion MET/NOT MET

## Findings (appended as established)

### Prior-attempt evidence (reported, NOT aggregated)
- `qa_wip.py 86.97 --spawned-at 2026-08-17T08:59:58Z` -> source_present=true,
  attempt_number=5, attempt_number_status="ok", attempt_number_is_lower_bound=true,
  prior_attempts=4, records_retained=5 (gauge, includes my own record).
  prior_records: 20260817T084127Z, 20260816T203348Z, 20260816T201133Z, 20260816T195546Z.
- `verdict_history_86_21.py --step 86.97 --evidence-only` -> status=no_rows_for_step,
  verdicts=(none). LEDGER IS STALE: attempt_number (5) > ledger verdict count (0).
  Sequence from the ledger: UNRELIABLE. Main's advisory disclosure (C, C, F, C) is
  ADVISORY ONLY (constrained party).
- harness_log.md:35694 carries `## Cycle 229 -- 2026-08-16 -- phase=86.97 result=FAIL
  (PARKED at the 3-attempt cap)` -- a secondary cross-check consistent with a prior FAIL.

### A. Harness compliance (5 items)
1. RESEARCH-GATE-BEFORE-CONTRACT: PASS. research_brief_86.97.md (brief_status COMPLETE,
   8 sources read in full, recency scan true, gate_passed true) and
   research_brief_86.97_cycle4.md (COMPLETE, 6 sources, 26 URLs, recency true, gate_passed true).
2. CONTRACT-BEFORE-GENERATE: PASS by mtime. brief 2026-08-16T22:29:14 <
   brief_cycle4 2026-08-17T10:34:45 < contract 10:36:16 < guard script 10:56:52 <
   experiment_results/live_check 10:59:30.
3. EXPERIMENT_RESULTS PRESENT: PASS (17,119 B), live_check_86.97.md present (30,133 B).
4. LOG-LAST: PASS for THIS cycle. masterplan 86.97 status = "pending". The only
   harness_log row is the 2026-08-16 PARK (result=FAIL), which is the prior cycle,
   correctly recorded; no row for the in-flight cycle.
5. NO-VERDICT-SHOPPING: PASS. Evidence CHANGED since the cycle-4 verdict --
   commits 2d861f5f (10:40:53) and fee1c51d (10:59:30) touched the guard script and
   both artifacts. Guard assertion count 35 -> 48 -> 52.

### B. Deterministic
- IMMUTABLE COMMAND: `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`
  -> stdout "parses", exit 0. NOTE (Main discloses this too): `bash -n` cannot fail on
  this defect class, and it is additionally blind INSIDE the quoted heredoc. It is
  necessary, nowhere near sufficient; the real gate is the guard script below.
- GUARD RE-RUN by me: `python scripts/qa/verify_decision_log_86_97.py` ->
  `ALL GREEN: 52 passed, 0 failed`. Reproduces the claim exactly.
- Re-derived gap AT MY EXECUTION TIME (window pinned to 2026-08-16T08:23:33Z):
  commits=91, decision lines=46, gap=45, recursion-guard commits=46, |45-46|=1 <= 2.
  This is a DIFFERENT pair of numbers from the 10-vs-5 the step was filed with, which
  is the point of criterion 1 -- the figure is re-derived, not copied.
- GUARD EXIT CODE (clean run): 0.
- RUFF F821,F401,F811 over the git-derived scope (`git diff --name-only 2d861f5f~1 HEAD -- '*.py'`
  = scripts/qa/verify_decision_log_86_97.py, non-empty): "All checks passed!", exit 0.
- 1b/1c/1d N/A: the diff touches no frontend/** and no backend/**; no UI claims. The
  production artifact is a shell hook, and I exercised it end-to-end ~40 times in
  throwaway temp repos (the equivalent runtime smoke).
- NO UNINTENDED PRODUCTION CHANGE: `git show --name-only 2d861f5f fee1c51d` touches only
  handoff/current/{contract,experiment_results,live_check,research_brief}_86.97,
  live_check_86.91.md and scripts/qa/verify_decision_log_86_97.py. Neither commit touches
  .claude/masterplan.json or any evaluator_critique. The peer session's uncommitted edits
  (backend/api/sovereign_api.py, frontend/src/*) are NOT in either commit.
- Production hook byte-identical: `git status` clean on it, mtime 2026-08-16T21:51:22
  (predates both cycle-4/5 commits). My own mirror runs never wrote to the repo.

### C. Independent re-derivations (not taken from the artifacts)

C1. EXIT-PATH ENUMERATION cross-check, criterion 2. I wrote a SECOND rule (split each
    line on ; && || and require `exit` to be the first token of a segment) and compared
    by SYMMETRIC DIFFERENCE, not by count:
      MY exits    : [28, 33, 37, 394, 396, 397]
      GUARD exits : [28, 33, 37, 394, 396, 397]
      SYMMETRIC DIFF: []          <- members agree, not merely cardinality
    3 pre-detector (:28 recursion guard, :33 CHANGELOG absent, :37 Recent-Activity
    missing), 3 post-detector. All 3 pre-detector members are classified with a reason.

C2. THE 86.91 EXTRACTION BLINDNESS, re-measured BY ME at HEAD using the REAL lifted
    `detector_source()` (my first attempt used a re-implementation and gave 8,613 B --
    I re-measured with the shipped function rather than ship a false finding):
      base           : 8617 B  sha1 072056e58af2befa
      CALL DELETED   : 8617 B  sha1 072056e58af2befa   BYTE-IDENTICAL = True
      DEF EDITED     : 8620 B  differs = True (+3 B)
    The artifact's "8,617 B / sha1 072056e58af2befa at HEAD" REPRODUCES EXACTLY.

C3. CRITERION 6's UNSCORABLE ARM, EXECUTED (not read). I added a third [4] cell whose
    mutation is a Python SyntaxError inside the quoted heredoc and ran the guard:
      FAIL [4] QA-PROBE-unbuildable: UNSCORABLE -- the mutant does not build, so it
               cannot be scored as a kill -- the heredoc compile() leg rejected the mutant
      FAILED: 50 passed, 2 failed     rc=1
    The arm fires, names the correct leg, and FAILS the run rather than counting a kill.

### D. My own mutation matrix (mirror differential; repo never written)
Method: copy the SHIPPED guard + a mutated hook into a scratch mirror so
`REPO = parents[2]` resolves there. Control establishes the environmental baseline
(the real decision log is absent in the mirror -> exactly 1 environmental failure).

  CONTROL (unmutated)                         50 passed, 1 failed   <- baseline
  N7  :81 minor->patch (Main: EQUIVALENT)     50 passed, 1 failed   SURVIVED == control
  Q1  :201 major->patch (_flip_magnitude)     50 passed, 1 failed   *** SURVIVED ***
  Q2  :204 minor->patch (_flip_magnitude)     49 passed, 2 failed   KILLED (1 extra)
        killed by exactly: [3a] flip_created ... bump == 'minor'

Q2 is the DISCRIMINATION CONTROL: it proves the bump assertions detect a magnitude
change of exactly this shape, so Q1's survival is a coverage hole, not a dead probe.
Q2 is also Main's cell N-6 (the mutant I raised at cycle 4) -- independently KILLED.

### E. Answers to Main's four attack questions

E1. IS N-7 REALLY EQUIVALENT? YES -- verified two ways, not taken on trust.
    Structural: `bump_type = classify_commit(...)` (:95) is read for the first time at
    :213 and unconditionally overwritten by `_flip_magnitude()` unless it is "major";
    :81 can return only "minor"/"patch", so its value is unobservable.
    EXECUTED: 6 scenarios including `phase-77.0: kickoff no flip` -- the input that
    actually reaches :81 -- with mutant output byte-identical to control on rc, bump,
    reason, created_done, transitioned_done AND the CHANGELOG version headers. Plus the
    mirror run above at control parity. No input path lets the subject classifier's
    minor/patch reach the decision log.

E2. ARE THE FOUR PINNED BUMPS DERIVED OR FITTED? Derived-correct. I re-derived each from
    the documented rule independently of the observed output: no_flip -> none (newly_done
    empty); 99.1 done with sibling 99.2 pending, not an X.0 -> patch; 98.0 created-done
    with sibling 98.1 pending -> minor (X.0 kickoff, phase not emptied); `feat!:` ->
    major on the subject's own authority. All four match.
    BUT the derivation exposes W2 below: the rule's own "major = the flip emptied a whole
    top-level phase" clause is pinned NOWHERE. Scenario 4's `major` comes from :216, a
    path on which `_flip_magnitude()` is NEVER CALLED.

E3. REMAINING CARRIERS OF THE UNBOUNDED CRITERION-4 CLAIM? None in prose. I swept with a
    SECOND operationalization (universal quantifier co-located with a decision/bump token,
    minus bounding tokens, over handoff/current + .claude/hooks + scripts/qa + docs +
    .claude/rules + CLAUDE.md + CHANGELOG.md). All four carriers are bounded IN PLACE --
    `git diff 2d861f5f~1 2d861f5f -- live_check_86.91.md` shows the heading LINE REMOVED
    and replaced, not appended. Post-mortem quotations (night_diagnostics.md:51,
    harness_log.md:35730, research_brief_86.97_cycle4.md) quote the defect and are not
    carriers. Verdict artifacts correctly untouched (criterion 7). One borderline residual
    -> NOTE N2.

E4. ANYTHING ELSE PARSED BUT NEVER ASSERTED? Yes, one: `_observed[_label]["rc"]` (:418).
    The only reads of `_observed` are `o["decision"]` (:442, :453); no [3a] scenario
    asserts rc. Everything DECISION_RE captures is now asserted by exact equality, and I
    proved the bump assertion load-bearing by killing :204. -> NOTE N1.

### F. FINDINGS (all measured, all reproducible)

W1 [CONTRADICTION -- WARN]. `handoff/current/experiment_results_86.97.md:185-187` STILL
   READS: "It is covered incidentally by the end-to-end driver (if it were deleted the
   hook would fail)". I re-measured it MYSELF at HEAD:
     BASE                                        rc=0  bump=none  reason=no_flip
     MUTANT :214 `bump_type = _flip_magnitude()` deleted
                                                 rc=0  bump=minor reason=unrecorded
   The hook does NOT fail; it emits a SPURIOUS minor bump (the 86.68 defect) with an
   unexplained reason (the 86.91 criterion-4 defect). This is the exact sentence the
   masterplan park note and harness_log Cycle 229 name as the FAIL blocker. Cycle 4/5
   ADDED section J1 and cell N-1 and left the false sentence standing, unannotated --
   i.e. the correction ACCOMPANIED rather than REPLACED, which is the discipline section
   F of the same file lectures about ("Why replacement and not a footnote").
   FIX: strike or replace the bullet; state the residual accurately (see N4).

W2 [MISSING_ASSUMPTION -- WARN]. LIVE NON-EQUIVALENT SURVIVING MUTANT: hook :201
   `return "major"      # the whole phase shipped` -> `return "patch"`. Guard stays at
   control parity (D above). NOT equivalent -- behavioural differential measured on a
   flip that empties top-level phase 97:
     CONTROL  bump=major reason=flip_transitioned transitioned_done=97.0,97.1
     MUTANT   bump=patch reason=flip_transitioned transitioned_done=97.0,97.1
   No scenario empties a phase, so `_flip_magnitude()` never returns "major" in any
   drive. J4's "what is NOT claimed" bounds REASON states only; J1's "spanning all four
   bump magnitudes" is true of observed VALUES but reads as branch coverage.
   This is the project's canonical failure mode (fix the instance the reviewer named,
   leave the sibling branch one line above).
   FIX: one scenario -- all steps of one top-level phase -> done, expect bump=major --
   plus one table row. ~5 lines.

W3 [INVALID_PRECONDITION -- WARN]. THE "END-TO-END" DRIVE IS SILENTLY TRUNCATED.
   `CHANGELOG_SEED` (:251-254) uses `|---|---|---|`; the hook requires
   `line.strip().startswith("|------")` (:357). So `insert_idx is None` and the heredoc
   `sys.exit(0)`s at :362. MEASURED, by appending a stderr marker after the final write:
     GUARD_SEED (|---|---|---|)   rc=0 reached_end_of_heredoc=False changelog_modified=False
     REAL-SHAPED (|------|)       rc=0 reached_end_of_heredoc=True  changelog_modified=True
                                        versions=['10.0.0','9.9.9'] new_rows=1
   Production CHANGELOG.md:9 is `|------|--------|--------|`. So heredoc lines 364-386
   (dedup guard, row insert, MAX_ROWS trim, the actual file write) and the hook's bash
   tail :392-397 are executed by ZERO drives, in every cycle of this step.
   Criterion 4's PURPOSE still holds: `_log_decision(bump_type)` is at :278, before the
   cut, and delete-the-call is KILLED (re-verified). But "the WHOLE heredoc end-to-end"
   and the "[3] END-TO-END" label are broader than what runs, and this is the
   fixture-cannot-represent-production shape (vacuity shape #5) inside the guard built to
   close a fixture-blindness defect.
   FIX: one line -- make the seed separator `|------|--------|-------------|`.

N1 [NOTE]. `_observed[...]["rc"]` parsed, asserted nowhere (E4). Low materiality: [3]
   asserts rc==0 on the baseline drive, so only a scenario-specific crash would slip.
N2 [NOTE]. `scripts/qa/verify_changelog_flip_86_91.py:184-185` still prints
   "[2] EVERY 'none' IS EXPLAINED (criterion 4)" / "NO UNEXPLAINED 'none' -- the
   silent-swallow class (criterion 4)", and that file contains ZERO bounding tokens
   (grep -cE "reach(es|ed)? the detector|pre-detector|recursion guard|86\.97" = 0).
   Arguably scoped by the code beneath it (it drives the detector's none-branches), so I
   do NOT charge it -- but it borderline falsifies J2's "This was the last unbounded
   carrier of the claim". Recorded for Main's decision.
N3 [NOTE]. SUPERSEDED markers are at live_check_86.97.md :170, :254, :407; cited as
   :171/:255/:408. Off-by-one; the markers exist and are in place.
N4 [NOTE]. The N-1..N-7 matrix is an ad-hoc, unshipped driver -- quoted but not
   re-runnable. Criterion 6 says "mutation-tested", not "the matrix ships", so this is
   not a miss; I independently reproduced N-6 (killed) and N-7 (survived/equivalent).

### G. CRITERION MAP (all independently re-executed)
1 MET   -- gap re-derived at run time (91/46/45 for me vs 87/44/43 in the artifact vs
           10/5 as filed; the arithmetic between the artifact and me closes exactly:
           +4 commits, +2 lines, +2 recursion). Delete-the-call SURVIVING the 86.91
           guard re-measured by me (C2), control-sensitive.
2 MET   -- rule written down, self-tested against a dumber scan, classification keyed on
           condition text; my independent rule agrees by SYMMETRIC DIFFERENCE (C1);
           unclassified member -> FAIL proven by a shipped [5] cell.
3 MET   -- recursion guard driven (rc=0, NO line) and judged a BOUND; gap arithmetic
           re-derived by me closes at |45-46| = 1.
4 MET on its load-bearing clause (delete-the-call turns the guard RED -- re-executed);
           see W3 for the truncation of the drive.
5 MET   -- four carriers bounded IN PLACE, replacement diff-verified; my independent
           second-operationalization sweep finds no further prose carrier (N2 borderline).
6 MET   -- control GREEN first in both matrices; [1]/[2]/[3]/[3a]/[4] all have cells;
           the UNSCORABLE arm EXECUTED by me and it FAILS the run (C3).
7 MET   -- neither commit touches masterplan.json or any evaluator_critique; step still
           `status: pending`; isolation assertions + my mirror confirm no write to the
           real decision log.

### H. VERDICT REASONING
All 7 immutable criteria MET, harness compliance 5/5, deterministic evidence reproduces
in full, no unintended production change. THREE WARN-level findings, each measured and
each with a named 1-5 line fix: a live FALSE claim in a mandatory artifact (W1), a live
non-equivalent surviving production mutant (W2), and an end-to-end drive that silently
covers 64% of the heredoc (W3). Severity dispatch: WARN -> CONDITIONAL. Not FAIL: no
criterion is unmet and nothing here is a product defect.
