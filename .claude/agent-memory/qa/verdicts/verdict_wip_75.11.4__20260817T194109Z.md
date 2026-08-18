STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 75.11.4
WRITTEN: 2026-08-17T19:41:09Z

# Q/A write-first record -- step 75.11.4 (EVALUATE)

Observer spawned via Workflow rail. Prior WIP records present in verdicts/:
- verdict_wip_75.11.4__20260817T185113Z.md (15710 bytes, mtime 21:02 local)
- verdict_wip_75.11.4__20260817T191121Z.md (15359 bytes, mtime 21:27 local)
- verdict_wip_75.11.4__20260817T193444Z.md (2661 bytes, mtime 21:36 local)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable verification command + lint + scope
C. Mutation re-derivation (do NOT accept the claimed 10/10)
D. Criterion-by-criterion MET/NOT MET

## Findings (appended as established)

### Attempt / sequence evidence
- qa_wip.py --spawned-at 2026-08-17T19:41:09Z: attempt_number=4 (status ok,
  is_lower_bound=true), prior_attempts=3, source_present=TRUE,
  records_retained=4 (gauge, not counter), records_pruned_known=null.
- verdict_history_86_21.py --evidence-only: status=ok, 3 verdicts,
  sequence = FAIL -> FAIL -> NO_VERDICT.
- CROSS-CHECK: prior_attempts (3) vs ledger rows (3) -> EQUAL, ledger is NOT
  stale for this step.

### B. Deterministic
- Immutable command `.venv/bin/python -m pytest
  backend/tests/test_phase_75_11_4_backfill_status_aware.py -q`
  -> 27 passed; BARE EXIT = 0 (re-run without a pipe to avoid masking).
- masterplan 75.11.4: status=pending, retry_count=0, max_retries=3,
  13 success_criteria -- byte-match with the criteria in the spawn prompt.

### Census re-derivation (C10/C12 denominator claim)
Ran classify() from scripts/qa/derive_archive_misattribution_86_29.py over the
whole live archive myself:
  agree=440 mismatch=156 unclassified=222 no_contract=27 TOTAL=845
  -> 845 = 440+156+222+27 REPRODUCES exactly.
  -> judgeable = 440+156 = 596; 156/596 = 26.17% ~ 26.2% REPRODUCES.
  -> unclassifiable = 249; 249/845 = 29.47% ~ 29.5% REPRODUCES.
  -> MISATTRIBUTION_NOTICE.md on disk = 156; mismatch dirs WITHOUT a notice = 0;
     notices on NON-mismatch dirs = 0. Exact 1:1, no over/under-marking.

### My OWN mutations (not the author's matrix) -- shared module handoff_naming.py
The author's matrix mutates the two SCRIPTS' source text. handoff_naming.py is
imported normally, so it is NOT reachable by that harness. I mutated it via
sys.modules injection before an in-process pytest run:
- MUT-A ARCHIVABLE_STATUSES += {"pending","in-progress"} -> 5 failed / 22 passed. KILLED.
- MUT-B resolver loop reduced to PREFIX_RE only -> 15 failed / 12 passed. KILLED.
- MUT-C is_archivable -> `status is not None` -> 5 failed / 22 passed. KILLED.
So the shared resolver+status gate is genuinely covered, including through a
mutation vector the author's own harness cannot reach.

### Independent replication of the cycle-3 matrix (copies + repointed HOUSEKEEPING)
sha256 of the REAL files matches the artifact's stated baselines EXACTLY:
backfill=6c8e0e5ac49c verifier=f07a33170cfe (handoff_naming=2f426db901fe).
  CONTROL_null_mutant            rc=0  27 passed      <- harness measures the subject
  N5a drop live_check half       rc=1  3 failed/24    KILLED (artifact says 3 failed/24) MATCH
  N14 verifier status arm off    rc=1  3 failed/24    KILLED (artifact says 3 failed/24) MATCH
  N5b drop command half          rc=1  4 failed/23    KILLED (artifact says 4 failed/23) MATCH
REAL files byte-unchanged after every cell (sha compare) = True.
Adjacent suites: 75.11.4 + 36.7 + 36.8 = 104 passed -- artifact's "104 passed" MATCH.

### MY OWN NEW CELLS -- TWO SURVIVORS, both proven NON-EQUIVALENT
  Q1 _git_mv returns True but moves nothing   KILLED (13 failed)
  Q2 protected guard deleted                  KILLED (4 failed)
  Q4 verifier re-promotes no-step-id to FAIL  KILLED (3 failed)
  Q3 ROLLING_KEEP_PREFIXES = ()               *** SURVIVED *** 27 passed
  Q5 _safe_target returns dest (clobbers)     *** SURVIVED *** 27 passed
Differentials (hermetic, --execute):
  Q3 SHIPPED keeps evaluator_critique_99.2.json in current/; MUTANT archives it to
     archive/phase-99.2/ -- restoring the phase-81.0 defect the file's own comment
     says "left the verdict gate dark for 13 consecutive step closes".
     54 evaluator_critique_*.json live in handoff/current/ right now.
  Q5 SHIPPED mints research_brief_99.2-v2.md and PRIOR EVIDENCE survives;
     MUTANT overwrites it -- prior archived evidence destroyed.
     This step's diff ADDS the docstring claim "prior evidence is never
     clobbered" (git diff +line 12) and ships no guard that can falsify it.
Neither maps to one of the 13 immutable criteria -> WARN-level, not a criterion miss.

### Cycle-2 NOTE items: 2 of 3 NOT actioned and NOT disclosed
- NOTE(4) "move dest_dir.mkdir below the `if dry_run:` return; remove the 3
  empty archive dirs the dry run created": NEITHER done. `_move` still mkdirs
  at line 222 BEFORE the dry_run guard (read directly). handoff/archive/
  phase-80.5, phase-81.1, phase-82.23 still exist, EMPTY, mtime
  2026-08-17T20:42:23 local (= 18:42:23Z, matching the cycle-2 citation).
  I classified them: all three return ('no_contract', None), so THIS STEP'S OWN
  DRY RUN added 3 dirs to the 845-dir census denominator it reports.
- NOTE(b) "19 files held back measures 20": still says 19 at
  experiment_results:56 and live_check:68. I re-derived it read-only using the
  script's OWN _masterplan_referenced_names + _is_rolling_keep against the live
  current/: 381 protected basenames, 20 held back, the 20th being this step's
  own live_check_75.11.4.md. NOT fixed, NOT annotated, NOT queued.
- NOTE(5) '"SURVIVORS: none" is a global claim under an N-cell matrix': the
  string still appears unscoped at experiment_results:272/375/381 and
  live_check:246/337/345. My two survivors above show why the scope matters.

### Hook cells on a MIRRORED hook (real hook never written)
real hook sha256 = 2278ca9910b0bd15 -- EXACT match to the artifact's stated value.
  CONTROL faithful mirror        rc=0  3 passed        <- green control
  H5 archived file created EMPTY rc=1  1 failed  KILLED (cycle-2's WARN is fixed)
  H6 declaration guard -> true   rc=1  1 failed  KILLED
  H7 live_check dropped from the hook's base list  rc=1  2 failed  KILLED
real hook byte-unchanged after all cells = True.

### Remaining independent cells, all KILLED (match the artifact cell-for-cell)
  M-INV bare run executes   3 failed/24  (artifact: 3 failed/24) MATCH
  M3 no-step-id sweeps      2 failed/25  (artifact: 2 failed/25) MATCH
  M4 unknown-step sweeps    2 failed/25  (artifact: 2 failed/25) MATCH
  N15 verifier PREFIX regex 3 failed/24  (artifact: 3 failed/24) MATCH
  Q6 VARIANT_RE broken      2 failed/25
  Q7 PREFIX_RE broken       1 failed/26
  Q8 HANDOFF_ROOT_KEEP={}   1 failed/26  (phase-36.7 kill-switch guard holds)

### Criterion 13 -- BOTH halves verified independently
  prevention: H6 (hook declaration guard -> always true) KILLED test_c13.
  detection : I planted a foreign contract in a temp phase-99.8 dir ->
              classify() = ('mismatch','12.3'); control ('agree','99.9').

### Other claims re-derived
  455 layout violations: EXACT.  24 PROVENANCE dirs: EXACT.
  census recall controls True, precision controls True (I ran them).
  classify() reads ONLY contract.md / contract_*.md; zero references to
  MISATTRIBUTION anywhere in the census -> the no-circularity claim REPRODUCES.
  All 13 criteria are BYTE-VERBATIM in contract_75.11.4.md (programmatic check).
  Adjacent suites 36.7+36.8 with this one = 104 passed (artifact: 104) MATCH.
  ruff F821,F401,F811 over a DERIVED 7-file scope (git diff HEAD UNION
  ls-files --others, xargs, non-empty guard): All checks passed! exit=0.

### Two more coverage gaps I found
- `quarantine_misattributed_archives.py` (NEW, 174 lines, production) has ZERO
  direct test coverage -- no test in the repo imports or drives it. The C12
  tests assert its RESULT on the live tree, never its behaviour.
- `misc_moved` is assigned 0 at backfill:244 and read at :326 and is NEVER
  incremented, so test_c2's `assert "misc-moved=0" in out` is a TAUTOLOGY
  (vacuity shape 4). It coexists with the genuine
  `assert not list(archive/"misc".iterdir())`, which is what actually kills
  M3/M4 -- so WARN, not sole coverage. Naming the real kill mechanism matters.

### Harness compliance 5/5
1. research gate: brief 20:35:04 precedes contract; envelope brief_status=COMPLETE,
   gate_passed=true, sources_read_in_full=18 (floor 5), urls=72 (floor 10),
   recency=true, coverage.dry=true.
2. contract-before-generate: contract's CURRENT mtime (21:05:54) post-dates
   handoff_naming.py (20:39:28) because cycles 2-3 corrected numbers inside it;
   the ORIGINAL ordering (20:38:30) is recorded independently by the cycle-1 AND
   cycle-2 evaluators. The contract is untracked so git cannot arbitrate --
   stated as a limitation, not papered over.
3. experiment_results + live_check present.
4. log-last: `grep -F "phase=75.11.4" handoff/harness_log.md` -> 0 lines;
   masterplan status=pending, retry_count=0, max_retries=3.
5. no-verdict-shopping: evidence CHANGED (22->27 tests, live_check_99.5.md
   fixture created, 4 new cells, content-equality assert, denominator table) and
   I confirmed the change is REAL by re-running all four cycle-2 survivors in MY
   OWN harness: N5a, N14, N15, H5 are now ALL KILLED.

### Scope / safety of my own work
HEAD 8f281795 at start AND end. `grep -c MUTANT` = 0 in all three housekeeping
scripts and in .claude/hooks/archive-handoff.sh. git status on scripts/housekeeping
shows only the 2 intended M's and 2 intended ??'s; .claude/hooks, .claude/masterplan.json
and scripts/qa are CLEAN. All mutation cells ran on copies under the session
scratchpad; test-module constants repointed via pytest.main(argv, plugins=[...]),
so no plugin or conftest file was written.
OUT-OF-SCOPE TREE MOVEMENT DURING MY EVALUATION: backend/services/autonomous_loop.py
(+17/-1, a phase-86 reports-page summary bugfix), frontend/src/app/reports/page.tsx
and the new frontend/src/components/ReportDetailDrawer.tsx all appeared AFTER my
first git status. They are the peer session's and unrelated to 75.11.4, but the
step's disclosure list names only sovereign_api.py + five frontend files, so it is
stale by three. Gate 1b/1c/1d N/A for THIS step's diff (no frontend/**, no UI
claim, and the only backend/** file is the test module pytest executes).

### VERDICT REASONING (recorded here as evidence; the return value is the verdict)
All 13 immutable criteria MET -- each with at least one guard I made fail MYSELF.
Product correct across three cycles. Capped at CONDITIONAL by WARN-level findings:
the un-actioned/un-disclosed cycle-2 NOTEs (dry-run mkdir + 3 empty dirs on the
live tree that this step's own run created and that its own census counts; the
19-vs-20 stale number; the unscoped "SURVIVORS: none"), two non-equivalent
survivors on adjacent safety properties (Q3, Q5), the untested new quarantine
script, and the misc-moved=0 tautology. None is a criterion miss, so not FAIL.

COMPLETED: 2026-08-17T19:54:14Z
