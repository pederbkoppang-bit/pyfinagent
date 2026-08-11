STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.36
WRITTEN: 2026-08-11T07:18:15Z
CYCLE: 2 (fix commit 6e8f3169 on top of 5595055c)

## A. HARNESS COMPLIANCE -- CLEAN (5/5)
- research 08:45:31 < contract 08:48:27 < qa.md 08:52:24 < qa_wip.py 08:53:25. ORDER HOLDS.
- research_brief_86.36.md envelope: brief_status COMPLETE, gate_passed true, 10 sources read in
  full, 18 URLs, recency_scan true. contract_86.36.md cites it at lines 17 + 167.
- experiment_results + live_check present and updated in the fix commit.
- log-last: grep -cF 'phase=86.36 ' handoff/harness_log.md = 0; masterplan status=pending,
  retry_count=0. Not yet flipped. Correct order.
- no-verdict-shopping: evidence CHANGED (6e8f3169 = 5 files). Legitimate cycle-2 respawn.
- 3rd-CONDITIONAL: not armed (prior cycle = 1 CONDITIONAL; 0 rows in harness_log).

## B. DETERMINISTIC
- IMMUTABLE COMMAND exit=0, "ALL GREEN -- 204 passed, 0 failed".
  Artifact says 201; I get 204. Explained by the DISCLOSED non-determinism (section [9] emits
  1 PASS per live WIP artifact; 3 more now exist). Mechanism verified. Not a finding.
- ruff --select F821,F401,F811 on the git-DERIVED scope (5595055c^..6e8f3169, 11 .py files,
  non-empty asserted): "All checks passed!" exit=0. B2 FIXED.
  (git diff --name-only HEAD -- '*.py' is EMPTY because the step is committed; used the commit
  range, per feedback_derived_scope_misses_untracked_files.)
- git status --short: no unintended production change. Only untracked WIP records + peer files.
- No frontend/backend touched -> 1b/1c/1d not applicable. No UI claims -> no Playwright gate.

## C. MUTATIONS I RAN MYSELF
- M-B1 (mine, isolated mini-repo): reverted qa-verdict.js STEP 0b to the pre-fix git text ->
  exit=1, RED on exactly two named needles: "carries '__<STAMP>'" and "carries '%Y%m%dT%H%M%SZ'".
  Green control first (193/0 in the mini-repo). AUTHOR'S CLAIM REPRODUCED.
- LOCATOR looseness (the thing I was told to check): "'STEP 0b (binding, phase-86.31" occurs
  EXACTLY ONCE in qa-verdict.js; the bare string "STEP 0b" also occurs exactly once. It cannot
  match a wrong section, and it still locates the PRE-fix heading (which is why my mutation went
  red on the needles, not on "section is locatable"). Correct revision-tolerant behaviour.
- Author's matrix re-run by me: green control, 5/5 KILLED on named assertions, subject sha256
  unchanged (da6db96dddb9b9fc -> da6db96dddb9b9fc). M1 -> "the paths are DISTINCT";
  M2 -> "prune keeps exactly `keep` records". Criterion 6's two named cells satisfied.
- verify_wip_retention_86_36.py re-run by me: 23/23 green (coexistence 965B+110B distinct paths,
  own WRITTEN/COMPLETED; spawn1->cycle1, spawn2->cycle2, STALE, IDENTITY_UNKNOWN; no verdict key
  over 3 reports with a >=3 floor; prune 8->3; keep=0 refused; audit_memory output IDENTICAL).
- reproduce_wip_destruction_86_36.py re-run by me: same path both spawns, 4386 -> 124
  (LOST 4262), spawn 1 unrecoverable. Script states it drives the pre-fix contract (simulation).
- audit_memory.py run by me on the REAL corpus with 12 live records present: 5 pre-existing
  unresolvable-link problems, ZERO mentions of verdict_wip/verdicts. Non-recursive glob holds.
- Guard: git diff EMPTY vs HEAD and across 5595055c^..HEAD. Last touched d23a981e (86.31).

## D. MY OWN EXTRA CELLS (beyond the author's)
- X1-STAMP-VALIDATOR-OFF: _RUN_STAMP_RE -> r".*"  => SURVIVED (both checkers green).
  Production code is CORRECT (I drove it: '../../../backend/main', '86.36; rm -rf /',
  'notastamp' all raise BadStepId) and qa-write-guard.sh is a compensating control, but NO
  assertion pins the new parameter's traversal defence. Real behavioural differential.
  NOTE-level: no immutable criterion depends on it. One-line fix.
- X2-DROP-PRIOR-RECORDS: KILLED on "and lists cycle 1 as a PRIOR, not merged".
- X3-RECORDS-RETAINED-LIE: SURVIVED. records_retained unpinned; it is the number the CLI shows
  a recovering Main. Cosmetic-adjacent NOTE.
- INVERSION PROBE on the loosened section-[6] imperative regex: my mutant (all 7 needles,
  >=900 chars, zero ANTI_DIRECTIVE words, directive gutted to "at your discretion / nothing
  reads the file") passes ALL GREEN under the NEW regex. I HYPOTHESISED A REGRESSION AND THEN
  DISPROVED IT: the OLD regex is defeated by a DIFFERENT inversion (one keeping the
  "create <path>" adjacency literal -- OLD match=True, NEW match=False). Neither dominates;
  lateral move, squarely inside the checker's own disclosed residual R3 ("still a TEXT SCAN").
  Downgraded to NOTE. The artifacts disclose the LOCATOR change but not the REGEX change.
- FOREIGN-ARTIFACT COUPLING: in an isolated mini-repo, ONE malformed verdict_wip_*.md written
  by a different session drives the IMMUTABLE command to exit=1 ("live artifact ... carries a
  valid marker"). Predates 86.36 (section [9] is 86.31's) but 86.36 multiplies the population
  from one-per-step to one-per-run, written by concurrent peers. Currently green. NOTE.
- DEFAULT_KEEP 3->1 survivor (predecessor's): I re-derived the grep MYSELF. prune_wip_records
  and DEFAULT_KEEP are referenced ONLY by verify_wip_retention_86_36.py (passes keep=
  explicitly) and mutation_matrix_86_36.py's anchor string. ZERO production callers. The
  author's reasoning is CORRECT -- pinning it would be a guard without a subject. ACCEPTED.

## E. CRITERIA
1 MET  2 MET  3 MET  4 MET  5 MET  6 MET  -- each by my own execution, not by reading.

## F. VERDICT DIRECTION
PASS. All 6 criteria MET; both cycle-1 blockers fixed and mutation-proven by me; harness
compliance clean; no unintended production change. Findings F1-F4 are NOTE-level and none
violates an immutable criterion. F2 (X1) is worth queueing as a follow-up.

COMPLETED: 2026-08-11T07:31:40Z
