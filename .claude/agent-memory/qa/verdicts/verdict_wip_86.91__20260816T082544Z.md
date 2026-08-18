STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.91
WRITTEN: 2026-08-16T08:25:44Z

# Q/A write-first record -- step 86.91 (changelog flip detector: created-and-closed)

Spawn: Workflow rail. `qa_wip.py 86.91 --spawned-at 2026-08-16T08:25:44Z` ->
source_present=true, attempt_number=1, prior_attempts=0, prior_records=[].
`verdict_history_86_21.py --step 86.91 --evidence-only` -> status=no_rows_for_step,
verdicts=(none). Both sources agree at ZERO priors; sequence is empty, not UNKNOWN.

## A. Harness compliance (5 items) -- CLEAN
1. research gate: research_brief_86.91.md 21,062 B, brief_status COMPLETE,
   gate_passed true, 8 sources read in full (floor 5), 28 URLs (floor 10),
   recency_scan_performed true, audit_class false. Contract cites I3/I5/ISSTA.
2. contract-before-generate (mtime): research 09:58:08 < contract 10:14:17 <
   experiment_results 10:23:14 < live_check 10:24:31. Fix commit 8dc70502 at
   10:23:32 local. Order correct.
3. experiment_results_86.91.md present.
4. log-last: `grep -F 86.91 handoff/harness_log.md` = 0 rows; masterplan 86.91
   status=pending. Not yet logged, not yet flipped. Correct.
5. no verdict-shopping: cycle 1, no prior record, no ledger row.

## B. Deterministic
- IMMUTABLE CMD `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`
  -> `parses`, **exit=0**.
- ruff F821,F401,F811 over the commit-derived py scope
  (scripts/qa/replay_changelog_rule_86_68.py, scripts/qa/verify_changelog_flip_86_91.py;
  derived via `git show --name-only 8dc70502 952ed521 c627a810 | grep '\.py$'`,
  non-empty asserted, xargs not bare $VAR) -> "All checks passed!", exit=0.
- worktree clean for the hook and scripts/qa (peer session's edits are in
  backend/api + frontend/src only; NOT in these three commits).
- `python scripts/qa/verify_changelog_flip_86_91.py` -> ALL GREEN: 24 passed, exit 0.
- `python scripts/qa/replay_changelog_rule_86_68.py` -> exit 0,
  **710 / OLD 252 / SHIPPED 9 / FIXED 11** (Main reported 706 / 250 / 9 / 11).
  Delta explained exactly: 4 commits landed after Main's run (8dc70502, 3b69ddf9,
  952ed521, 2d3e4b68), 2 of them `phase-86.91:` = +2 OLD. SHIPPED/FIXED identical.
- INDEPENDENT re-derivation of criterion 1 with from-scratch code:
  e4f2e844 -> `86.86 before: None -> after: 'done'`, OLD newly_done=[]  (REPRODUCES)
  8b520f6c -> `86.81 before: None -> after: 'done'`, OLD newly_done=[]  (REPRODUCES)
  Both step ids are `status: done` in the masterplan at HEAD; e4f2e844 touched
  backend/services/autonomous_loop.py + a test = real shipped work.
- criterion 8: `git diff --stat c627a810..HEAD -- .claude/masterplan.json` EMPTY.
  86.90/86.91 both `pending`. evaluator_critique_86.86.md diff has **0** deletions
  (pure append of a fresh re-grade). The 9 deletions in the 86.90 artifacts are a
  filename-pointer correction, not a verdict change.
- criterion 5: `git log --since=2026-08-16T00:00:00 -- CHANGELOG.md` = 4 commits,
  all `chore: auto-changelog hook entry`. None of the 3 step commits touch it.
  Decision log on disk (gitignored) holds exactly 2 lines, both
  `bump=none reason=no_flip` -- consistent with no step having flipped.

## C. INDEPENDENT MUTATION MATRIX (mine, in memory; nothing written)
Driver: patch Path.read_text so the SHIPPED checker runs against a mutated hook /
replay source. CONTROL first: unmutated -> exit 0, "ALL GREEN: 24 passed".

| Cell | Mutation | Result |
|---|---|---|
| QA-1 | drop `_FLIP_DECISION["reason"]="masterplan_unreadable_at_HEAD"` | **SURVIVED** (24/24 green) |
| QA-2 | drop the `first_commit` reason | KILLED |
| QA-3 | drop the `detector_error` reason | KILLED |
| QA-4 | collapse the flip-reason ternary to a constant | KILLED |
| QA-5 | sentinel -> `is None` (semantically EQUIVALENT here) | killed **by the anchor-uniqueness check only** -- mis-attributed kill, not a behavioural one |
| QA-6 | `created_done` logs the wrong population | KILLED |
| QA-7 | stop logging populations | KILLED |
| QA-8 | remove the stderr FAILED marker | KILLED |
| QA-9 | move the predicate to an UNEXTRACTED module-level helper (drift) | fails LOUD (checker RED) -- extraction cannot silently diverge |
| QA-10 | FIXTURE: `fake_run` returns the same payload for both refs | checker RED -- the fixture discriminates |
| QA-11 | REPLAY silently ignores `count_created` (literal kept) | **SURVIVED** (24/24 green) |
| QA-12 | REPLAY None-exclusion restored, reworded `not in ("done", None)` | **SURVIVED** (24/24 green) |
| QA-13 | populations computed but `created` dropped from `newly_done` | KILLED |

Extra: fault-injected the NEW `_log_decision` myself (mkdir->PermissionError,
open->OSError): raised=None both times, stderr marker printed, decision log still
2 lines. Never-raises HOLDS for the added logging code -- but the shipped checker
does not cover it (it is not in NEEDED).

## D. Criterion grades
1. MET -- reproduced by execution, real commit e4f2e844, before=None->done, empty
   newly_done; reproduction predates the fix (filing commit c627a810 09:51:57 quotes
   it, fix at 10:23:32). Re-derived independently.
   NOTE: live_check §1's block is a hand-assembled rendering (output placed between
   `EOF` markers, heredoc body elided) -- not a literal terminal capture.
2. MET -- exact predicate stated and matches the shipped diff; no step id in the
   predicate; class proven on 9.99 / 12.7; QA-13 shows the created arm is load-bearing.
3. MET on substance -- three numbers from execution; SHIPPED 9 / FIXED 11 reproduced
   exactly; +2 accounted commit-by-commit and both commits verified to have closed a
   step; PARKED 86.9/86.44 = 0 vs 13 reproduced. FIXED is a superset of SHIPPED by
   construction, so no unreported "lost" set exists.
   ADJUDICATION of the "348-commit corpus": answering on a pinned deterministic
   replacement, with the drift disclosed and the criterion unamended, SATISFIES it.
   The named corpus is provably non-regenerable (git applies a bare --since at the
   current time of day); a criterion cannot demand an impossible act. The operative
   demands -- three executed numbers, an accounted increase, each newly-bumping
   commit shown to have closed a step -- are all met.
   **WARN W1**: experiment_results:130-132 claims "Anyone re-running it gets
   706 / 250 / 9 / 11, today and next month." I re-ran it and got 710 / 252 / 9 / 11.
   `CORPUS_UNTIL = None  # None = HEAD` -- the pin fixes the LOWER bound only; the
   upper bound still floats with HEAD. In a step whose thesis is "a number about a
   clock", an unreproducible reproducibility claim is material.
4. MET, with residuals. Every `return "none"` in `_flip_magnitude` sets a reason
   (verified by reading AND by execution); the `bump_type != "major"` else-branch
   sets `subject_forced_major`; `_log_decision` falls back to `reason=unrecorded`
   (explicit, not silent).
   **WARN W3**: QA-1 SURVIVED -- the assertion literally named "[2] EVERY branch
   that returns 'none' sets a reason -- none is left unrecorded" covers 3 of the 4
   branches; `masterplan_unreadable_at_HEAD` can be deleted with the guard fully green.
   **NOTE N2**: three bash `exit 0` paths run BEFORE the python and write nothing:
   `^chore: (auto-changelog|changelog drift)`, CHANGELOG absent, and no
   "### Recent Activity" section. (a) fires on ~every second commit here (decision
   log has 2 lines against 4 commits since the fix); (b)/(c) are the silent-swallow
   shape exactly. Disclosed nowhere.
5. MET -- no hand edit; every CHANGELOG.md change today is a hook-produced commit.
   No correction to the version line was made, so the conditional's second clause is
   not triggered; treating a retro-bump of released versions as an operator call is
   defensible and is disclosed (contract §7, results §10). The criterion is a
   prohibition, not a mandate.
   Main's live_check §6 NON-CLAIM is CORRECT and verified: 86.90 and 86.91 both
   exist at HEAD~1 (86.91 created in c627a810, three commits back), so the flip
   commit will read `flip_transitioned`. Withholding that claim is right.
6. MET -- 24/24 green with the CONTROL section first; M1/M2/M3 killed; anchors
   uniqueness-checked. Independently corroborated by QA-13, QA-9 and the fixture
   mutation QA-10. Fixtures were fixed, NOT assertions weakened: [0] still asserts
   the exact value `"patch"` and [1] asserts `patch`/`minor`/`major` exactly.
   **WARN W2**: section [5]'s replay guards are pure substring scans and BOTH of my
   replay mutants SURVIVED -- QA-11 (behaviour dropped, literal kept: vacuity shape 3)
   and QA-12 (defect restored, reworded: shape 2). The replay is the instrument that
   produces criterion 3's numbers and its only guard is text matching.
7. MET -- real fault injection into `subprocess.run`; non-propagation, bump=none and
   the stderr marker all asserted, and I killed that section twice (QA-8, QA-3).
   **NOTE N3**: the checker does not cover `_log_decision`, the NEW raise surface this
   step adds. I fault-injected it myself and the property holds -- but that is my
   execution, not the shipped guard's.
8. MET -- masterplan byte-identical since the filing commit; no verdict altered.
   **NOTE N4**: experiment_results §1's file table omits three files riding in
   8dc70502 (evaluator_critique_86.86.md, experiment_results_86.90.md,
   live_check_86.90.md). The first is named in the commit message; the two 86.90
   files are named nowhere.

**NOTE N1**: live_check §4 quotes "(driving the SHIPPED detector, **74** lines
extracted...)" inside a block presented as verbatim. Measured: 109 at 8dc70502, at
952ed521, at HEAD and in the worktree; 76 at 8dc70502~1. 74 reproduces at no
committed state -- a stale/edited capture in a verbatim block. Everything else in
that block reproduces byte-for-byte.

## E. Verdict shape
All 8 criteria substantively MET. Three WARN findings (W1 measurably-false
reproducibility claim; W2 two executed surviving mutants on the criterion-3
instrument; W3 one executed surviving mutant on the criterion-4 guard) with named
fixes, plus four NOTEs. Severity dispatch: WARN -> CONDITIONAL.

COMPLETED: 2026-08-16T08:35:11Z
