STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.91
WRITTEN: 2026-08-16T09:36:30Z

# Q/A cycle 4 evaluation record -- step 86.91

Prior WIP records on disk for 86.91: 20260816T082544Z, 20260816T085153Z,
20260816T091637Z. This spawn = 20260816T093630Z.

## Attempt / sequence evidence (reported, never aggregated)
- `qa_wip.py 86.91 --spawned-at 2026-08-16T09:36:30Z`: source_present=True,
  attempt_number=4 (status ok, is_lower_bound=True), prior_attempts=3,
  records_retained=4 (GAUGE, includes my own record), records_pruned_known=None,
  is_verdict=False.
- `verdict_history_86_21.py --step 86.91 --evidence-only`: status=no_rows_for_step,
  verdicts=(none). CROSS-CHECK: attempt_number(4) > ledger rows(0) -> THE LEDGER IS
  STALE; sequence: UNKNOWN from the authoritative source. Main's advisory [C,C,C] is
  consistent with attempt_number=4 and with three `"verdict": "CONDITIONAL"` blocks
  in evaluator_critique_86.91.md, but is advisory only. I did NOT word-scan bodies.
- `grep -cF "phase=86.91" handoff/harness_log.md` = 0 -- correct at EVALUATE time.

## A. Harness compliance -- all 5 clean
1. Research gate: research_brief_86.91.md, brief_status COMPLETE, gate_passed true,
   external_sources_read_in_full=8 (floor 5), urls_collected=28 (floor 10),
   recency_scan_performed=true, recency section present at :89.
2. Order (mtime, local): research 09:58:08 < contract 10:14:17 < hook 10:14:54 <
   checker 11:33:50 < experiment_results 11:35:04.
3. experiment_results_86.91.md present, with cycle-2/3/4 follow-ups.
4. Log-last respected: 0 harness_log rows; 86.90 and 86.91 both `pending`.
5. Not verdict-shopping: 0ecccafe changed experiment_results, live_check,
   evaluator_critique and the checker. Evidence CHANGED.

## B. Deterministic
- IMMUTABLE `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`
  -> "parses", exit 0.
- HEAD=70db3a45 (auto-changelog for 0ecccafe). Graded commit 0ecccafe.
- `git diff --stat 8dc70502 HEAD -- .claude/hooks/post-commit-changelog.sh` EMPTY.
  Main's self-flag "the hook is unchanged since cycle 1" VERIFIED TRUE.
- masterplan 8dc70502~1..HEAD: 4 ADDED `"status": "pending"` lines only (86.92-86.94);
  0 removed status lines; 0 retry_count changes. No flip.
- No VERDICT_SCHEMA / enforceEscalation / would_auto_fail / max_attempts change in the
  window (grep count 0). Verdict semantics untouched.
- CHANGELOG.md: 5 changes in the window, ALL single-file `chore: auto-changelog hook
  entry for X` commits. No hand edit.
- ruff F821,F401,F811 on the git-derived scope (commit range + uncommitted + untracked
  .py; xargs -0, non-empty asserted): 3 files, "All checks passed!", exit 0.
- Uncommitted backend/api/sovereign_api.py + 5 frontend files: mtime 2026-08-14,
  present in 0 of the step commits -- peer-session work, not this step's diff.
  1b/1c/1d therefore N/A on the step-derived diff (no frontend/**, no backend/**,
  no UI claims).
- `verify_changelog_flip_86_91.py` -> ALL GREEN: 42 passed, 0 failed.
- `replay_changelog_rule_86_68.py` -> 707 in [2026-08-11T00:00:00 .. 8dc70502];
  251 / 9 / 11; +2 = e4f2e844(86.86) + 8b520f6c(86.81); parked 86.9/86.44 -> 0 NEW;
  exit gate control_green=True all_cells_killed=True.
- Criterion 1 independently RE-DERIVED by my own statuses() walk on e4f2e844:
  `86.86 before: None -> after: done`, OLD [] / NEW ['86.86']. 86.81 and 86.86 are
  both `done` at HEAD.
- No step-id literal anywhere in the detector body (hook :98-218 grep for "N.M" -> 0).
- Heredoc parses, 327 lines -- matches experiment_results section 11.

## C. Mutation probes (harness: exec the shipped checker with ONLY the
## `HOOK_SRC = HOOK.read_text(...)` / `REPLAY_SRC = ...` lines repointed at mutated
## copies in the scratchpad; REPO/__file__ preserved. CONTROL first: 42/0.)

- **M-A SURVIVED -- FINDING.** Delete the production call `_log_decision(bump_type)`
  (hook :262); the function body untouched. -> **ALL GREEN 42/0**. Also true by
  construction: `detector_source()` extracts only FunctionDef/Assign/AnnAssign nodes
  named in NEEDED, so a bare `ast.Expr` call site can never enter SHIPPED. `grep -n
  _log_decision` on the checker: :78 (NEEDED), :524 (comment), :532 (drive_log). No
  assertion of the invocation. Production effect is IDENTICAL to the cycle-3 Q1
  mutant; cycle 4 closed the in-function write and left the wiring.
- **M-B2 / M-D SURVIVED -- FINDING.** Hook-side AUTHORABLE whitelist as a post-filter
  (every checker anchor byte-intact): `created_done = [s for s in created_done if s in
  ("86.86","9.99","12.7","77.0","78.1")]` -> **ALL GREEN 42/0**; same with 86.90/86.91
  added. Section [1]'s fixtures use only authored literals; `_RUNTIME_ID` is computed
  at checker :341-343, AFTER section [1], and is referenced only by the replay fixture
  AFTER_R. The cycle-4 remediation of Q4/Q2b landed on the REPLAY half only.
- M-C KILLED: single-id `sid == "86.86"` turns `[1] magnitude: a created X.0 kickoff is
  minor` RED. The 1-id shape is caught; the N-id shape is not.
- **Section [7] is genuinely behavioural (judge_these A).** Second, differently-
  constructed write mutant -- redirect the filename to `elsewhere.log`, syntactically
  valid, builds cleanly -> 5 of the 6 [7] checks go RED (36 passed, 6 failed). So the
  shipped cell's kill is real, not a construction artifact, and the temp-dir
  redirection FAILS CLOSED (a write landing elsewhere yields None -> FAIL).
- **corpus_head (judge_these C):** BOTH stated raise conditions verified by
  construction AND execution -- C1 exec-raises -> RuntimeError("the sliced corpus block
  did not run: division by zero"); C2 start>end (sh line hoisted above CORPUS_SINCE) ->
  RuntimeError("...ran but never called sh()"). RESIDUAL: two `return None` paths
  survive (checker :397-398 anchors-not-found, :419-421 no records). R1 (start anchor
  renamed) and R2 (end anchor reworded) both RETURNED None. At the [6] call site None
  scores DETECTED = a false kill. Latent today: the shipped cell touches neither
  anchor, and the same rename turns [5]'s control RED so the checker still exits 1.
  Reachable for a FUTURE cell whose mutation renames an anchor.
- **Historical counts re-derived by EXECUTION (judge_these D)** -- each cycle's checker
  run against that cycle's own hook + replay: 8dc70502 -> 24 passed / 3 cells;
  98c5b6ab -> 31 / 6; 468c7908 -> 34 / 8; 0ecccafe -> 42 / 10. Every count in
  experiment_results sections 1, 7 and 10 REPRODUCES. No stale figure found.
- live_check section 4 vs a fresh run: byte-identical except the runtime-derived id
  (811.38 captured vs 791.68 today -- by design, HEAD moved) and the fence lines.
  Genuinely regenerated, not spliced.
- Decision log on disk: 5 lines, one per step commit, all `bump=none reason=no_flip`.
  Untouched by my runs (mtime 11:35:36 vs now 11:48). Wiring is LIVE, just unguarded.
- Bash early exits BEFORE the heredoc (hook :27-38): the `^chore: (auto-changelog|
  changelog drift)` recursion guard, CHANGELOG.md-absent, and `### Recent Activity`-
  absent. All three `exit 0` SILENTLY -- no decision line, no stderr marker. Measured:
  10 commits since 8dc70502, 5 decision lines; the 5 missing are exactly the 5
  auto-changelog commits. Undisclosed in contract / experiment_results / live_check
  (grep). Raised at cycles 1-3 per the prior critique; still undisclosed at cycle 4.

## D. Criteria
1 MET (independently re-derived). 2 MET on the product (no id literal; predicate
change stated verbatim) -- guard residual only. 3 MET (251/9/11 reproduce; +2
accounted member by member; both steps `done`; the 348-corpus non-reproducibility
disclosed with its measured cause). 4 MET on the product (closed reason set, 5 live
lines) with two residuals (M-A; bash early exits). 5 MET (no hand edit; the declined
retro-bump is disclosed as an operator call; section 6 PENDING with a correct and
precise prediction). 6 MET (M1 restore-the-None-exclusion KILLED, M6 reworded KILLED,
control observed GREEN first). 7 MET (fault injection into subprocess.run). 8 MET
(nothing flipped, no verdict altered, no schema change).

## E. Verdict issued: CONDITIONAL
Two WARN-level guard findings (M-A; M-B2/M-D with the accompanying claim
contradiction) plus three NOTEs. Product correct on all 8 criteria for the fourth
cycle; the findings are again in the guards and the prose.

COMPLETED: 2026-08-16T09:52:10Z
