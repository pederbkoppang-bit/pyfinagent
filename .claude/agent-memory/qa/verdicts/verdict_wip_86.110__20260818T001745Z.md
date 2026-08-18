STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.110
WRITTEN: 2026-08-18T00:17:45Z

# Q/A write-first record -- step 86.110 (heartbeat isolation leak) -- CYCLE 3 spawn

## Attempt / sequence evidence
- qa_wip.py 86.110 --spawned-at 2026-08-18T00:17:45Z -> source_present=true,
  attempt_number=3 (status ok, is_lower_bound=true), prior_attempts=2,
  records_retained=3 (gauge), prior_records = 20260817T231948Z, 20260817T235655Z.
- verdict_history_86_21.py --step 86.110 --evidence-only -> status=ok,
  "2 verdict(s) from the ledger", verdicts: CONDITIONAL -> CONDITIONAL.
- CROSS-CHECK: prior_attempts (2) == ledger rows (2) -> ledger NOT stale.

## HARNESS COMPLIANCE 5/5 CLEAN
1. research_brief_86.110.md envelope: brief_status COMPLETE, sources 7 (>=5),
   urls_collected 25 -- I independently counted 25 unique http(s) URLs, exact
   match -- recency_scan true (:251), coverage.dry true, gate_passed true.
2. mtime order research 2026-08-17T22:32:54 < contract 2026-08-18T00:41:32 <
   code 02:15 < experiment_results/live_check 02:17 (local).
3. experiment_results (8,349B) + live_check (21,573B) present.
4. log-last intact: grep -cF "phase=86.110" harness_log.md = 0; masterplan status
   still "pending".
5. no verdict-shopping: prior CONDITIONAL 2026-08-18T00:15:07Z; all generated
   artifacts stamped AFTER it and BOTH named cycle-2 blockers materially addressed.

## DETERMINISTIC (all reproduced by me)
- IMMUTABLE CMD -> "parses", EXIT=0.
- sweep -> POPULATION 4 / ISOLATED 3 / LEAKING 0 / PROD 1, exit 0; byte-for-byte
  match to live_check S5 including the cross-check block.
- pytest 86_110 file -> 13 passed.
- author matrix -> CONTROL rc=0 collected=13; P1,P2,P3,P5,P6,P7,P8,P9,P10,P11 all
  KILLED (10/10). Restores verified INDEPENDENTLY by sha256.
- ruff F821,F401,F811 over a GIT-DERIVED non-empty 8-file scope -> exit 0.
- runtime smoke: cycle_health / backend.tests.conftest / parse_failure_ledger all
  import; legit(append)=True, legit(longer rewrite)=False, legit(unknown)=False.
- LIVE endpoint GET /api/paper-trading/freshness -> heartbeat cycle_id 3e5afddb,
  age_sec 17086.9, ratio 0.198, band GREEN. The regenerated value is live and safe.
- sha256 recomputation of live_check S2's BEFORE/AFTER capture: both recompute
  EXACTLY -> not a spliced capture.
- FULL SUITE (mine): 20 failed, 3621 passed, 12 skipped, 5 xfailed, 1 xpassed in
  482.31s, 3,659 collected -- EXACT match to the artifact's headline. Symmetric
  difference vs the artifact's ID list = 2, both explained:
    * author's extra test_phase_82_54::test_production_sql_dry_runs_valid -- BQ
      dry-run; PASSES on my rerun.
    * my extra test_phase_23_2_15::test_..._known_pass_scripts_still_pass -- a
      PATH artifact of MY invocation (bare `python` absent); PASSES under the
      activated venv. Retired as mine, not a regression.
  None of the step's 13 tests failed. test_phase_23_2_14_no_reentrant_locks is
  ABSENT from the failures -> the 86.108 lock regression is genuinely repaired.

## CRITERIA -- ALL SIX MET (independently derived)
C1/C3 MET -- my own control/fix differential: guard DISABLED + fix INTACT ->
  test_the_previously_leaking_tests... PASSES; guard DISABLED + fix REVERTED ->
  it FAILS. So the criterion-3 assertion is behavioural and NOT dependent on the
  conftest guard (retires cycle-1's kill-attribution note).
C2 MET -- AST enumeration of the rail-guard file: exactly TWO writer call sites
  (:204, :223), each inside a test that now patches BOTH constants (:194/:202 and
  :219/:222). Per-site cells QA-M6 (site 1 only) and QA-M7 (site 2 only) are BOTH
  KILLED, each naming the correct individual test.
C4 MET -- I re-derived the criterion's OWN literal rule by AST: 8 files patch
  _HISTORY_PATH; exactly 2 do not also patch _HEARTBEAT_PATH
  (test_cycle_heartbeat_alarm.py, scripts/smoketest_stages_5_through_13.py);
  NEITHER reaches any writer. Exact match to the sweep's cross-check. Production
  writer entry points AST-resolved: record_cycle_start :446 and record_cycle_end
  :1979 are BOTH inside run_daily_cycle -> keying the transitive census on it is
  correct AND complete. Second-order path checked: paper_trading reaches it from
  run_now/_run_cycle_background/_scheduled_run and NO test references any of the
  three. Sweep early-skip blind spot: only mutation_86_110.py's own comments.
C5 MET -- heartbeat is field-for-field derived: 3e5afddb has 2 of 174 ledger rows,
  IS the last completed row, updated_at == that row's completed_at exactly; c1 and
  c2 have 0 rows. QA-M5 (point the real file at "c2") -> KILLED.
C6 MET -- masterplan diff vs HEAD EMPTY; 86.108/86.110 still pending, 86.109 done
  from its own step; harness_log 0 rows; verdict_ledger --numstat 2 added/0 removed.

## MY OWN MUTATION CELLS (7; beyond the author's matrix)
QA-M1 length-half-alone  KILLED   (with P11 this completes the sub-expression matrix)
QA-M2 both sites + guard KILLED   (isolated -- see C3)
QA-M3 _isolates Assign   SURVIVED (see N1)
QA-M4 sweep is_test=False KILLED
QA-M5 real heartbeat->c2 KILLED
QA-M6 site 1 only        KILLED (names test_rail_guard_cycle_history_row_carries_flags)
QA-M7 site 2 only        KILLED (names test_cycle_history_row_carries_funnel_counts)

## FINDING I BUILT AND RETIRED (recorded so it is not re-shipped)
The cycle-3 census "TRANSITIVE REACHERS: 8" LOOKED non-reproducing (literal
`run_daily_cycle(` -> 7; AST callers -> 5 + definer; any mention -> 15). It is
NOT: `grep -rlE "run_daily_cycle *\(" --include="*.py" backend scripts tests`
returns EXACTLY those 8 files, same membership. Every per-file mechanism label is
correct (36_12 :289 / 36_17 :271,:503,:566 stub cycle_health.get_log; 85_4 / 85_6
patch _HEARTBEAT_PATH; the two tests/-tree files only regex the source). The
textual 8 is a strict SUPERSET of the AST-executable 5 -- conservative. CENSUS
REPRODUCES. Cycle-2's F2 is genuinely closed.

## CAPPING FINDING (WARN, EVIDENCE-class -- no product defect)
W1. STALE "VERBATIM" SCOPED-SUITE FIGURE, in TWO artifacts.
    experiment_results_86.110.md:36 and live_check_86.110.md:291 both quote
      "68 passed, 3588 deselected"
    for `pytest backend/tests/ -q -k "cycle_health or heartbeat or rail_guard or
    38_2 or 86_38 or 23_2_14 or 86_110"`. I ran that EXACT command twice:
      MEASURED: 71 passed, 3588 deselected  (10.78s / 11.51s)
    68 + 3588 = 3,656, which CONTRADICTS the same artifacts' own "3,659 collected"
    (live_check:151, experiment_results:50). The 3-test gap is exactly the tests
    added to this step's file after the figure was taken (the file contributes 13
    to that selection; 71-13+10 = 68). This is the SAME defect class the step was
    CONDITIONALed for at cycle 1 -- a block labelled as the shipped tree measured
    before the tests existed. The FULL-suite block was regenerated; the sibling
    scoped block was not, in either file.
    Direction understates, and the claim it supports ("this scoped area is green")
    is TRUE -- I measured 71 passed / 0 failed. Fix = re-run and paste, 2 lines.

## NOTES (not capping)
N1. QA-M3 survivor: removing the `_isolates` Assign branch leaves 13/13 green
    while the sweep's cross-check silently degrades from "naive would flag: 2 /
    OVER-reported [test_cycle_heartbeat_alarm.py, scripts/smoketest_stages_5_
    through_13.py]" to "1 / [test_cycle_heartbeat_alarm.py]". Conservative (losing
    an isolation-detection shape can only OVER-report leaks) and the criterion's
    literal rule is the setattr shape, so this is an unguarded ENHANCEMENT beyond
    the criterion. Named fix: one `_isolates` unit case for `ch._HEARTBEAT_PATH = x`.
N2. live_check:181 still asserts the failing set is "byte-identical to the run
    taken before this step's own tests existed" while experiment_results' cycle-3
    note declines to assert byte-identity as an invariant. The earlier run's IDs
    were never recorded on disk, and four independent full runs have now produced
    21 / 20 / 19 / 20 failures with a 2-member symmetric difference in mine. The
    arithmetic is coherent (3,646 -> 3,659 = +13; failed steady at 20) but the
    membership claim is not substantiable from anything on disk. A correction that
    accompanies rather than replaces.
N3. Two broad `except Exception` in new code (conftest.py:161, sweep:108), both
    fail-SAFE (return False / append a note); neither on a risk-guard path. NOTE.
N4. Ordering fact worth recording in the artifact: `record_cycle_start` and
    `record_cycle_end` write the HISTORY row BEFORE `_write_heartbeat`
    (cycle_health.py:421-426 and :487-492), so the `ledger_backed` rule can never
    race a live cycle into a false "leak" verdict. Undisclosed strength, not a defect.
N5. Auto-commit blast radius (carried from cycle 2, still true): the git-derived
    .py scope contains backend/api/sovereign_api.py (mtime 08-17 15:54) and
    backend/services/autonomous_loop.py (08-17 21:42), both PREDATING this step's
    research brief (22:32) -- other work. Main's stated plan to commit with
    EXPLICIT PATHSPECS is the right mitigation.
N6. UI gate 1c NOT triggered (no UI claim in contract/criteria/diff); frontend
    gate 1b NOT triggered (diff touches no frontend file). No restart pending.

## TREE INTEGRITY
Every file I mutated was restored and re-hashed byte-identical to its
pre-verification baseline: conftest 3a69c139, rail_guard 21b2c097, sweep 70461e90,
heartbeat 1c9428d5, ledger dd8d01a2. `git diff --name-only HEAD` over
backend/tests + scripts/qa + the heartbeat is the same 4-file set as at spawn.
No write was blocked; I wrote nothing outside this verdicts/ path.

COMPLETED: 2026-08-18T00:39:42Z
