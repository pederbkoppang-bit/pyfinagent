STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.110
WRITTEN: 2026-08-17T23:19:48Z

# Q/A write-first record -- step 86.110 (heartbeat test-isolation leak)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, lint, scoped tests
C. Mutation / vacuity probing of the sweep + new test
D. Criterion-by-criterion MET/NOT MET

## Findings (appended as established)

### Attempt / sequence evidence
- qa_wip.py 86.110 --spawned-at 2026-08-17T23:19:48Z: source_present=true,
  attempt_number=1, attempt_number_status=ok, prior_attempts=0, prior_records=[],
  records_retained=1 (gauge, includes this spawn's own record).
- verdict_history_86_21.py --step 86.110 --evidence-only: status=no_rows_for_step,
  verdicts=(none). prior_attempts (0) is NOT > ledger rows (0) -> no staleness
  signal from the cross-check. sequence: no prior verdicts recorded.

### A. Harness compliance
- research_brief_86.110.md EXISTS (31,324 B, mtime 2026-08-17T22:32:54). Envelope:
  brief_status COMPLETE, external_sources_read_in_full 7, urls_collected 25,
  snippet_only 18, recency_scan_performed true, gate_passed true. NOTE: the brief is
  already COMMITTED (git ls-files matches; committed in 8200283c, the 86.108 commit)
  -- i.e. it was swept into an earlier auto-commit, not a defect in itself.
- mtime order: research 22:32:54 < contract 00:41:32 < code 01:14-01:17 <
  experiment_results 01:19:13. Contract-before-generate OK.
- experiment_results_86.110.md + live_check_86.110.md present.
- log-last: grep -c "86.110" handoff/harness_log.md = 0 -> not yet logged. Masterplan
  status for 86.110 = "pending". OK.
- no-verdict-shopping: attempt 1, no prior verdict. N/A.

### B. Deterministic
- IMMUTABLE COMMAND reproduced: `... ast.parse(test_phase_66_1_rail_guard.py) && echo
  parses` -> "parses", EXIT=0.
- sweep reproduced: POPULATION 4 / ISOLATED 3 / LEAKING 0 / prod 1, EXIT=0; naive
  cross-check over=2 (test_cycle_heartbeat_alarm.py, smoketest_stages_5_through_13.py),
  under=none. Matches live_check verbatim block.
- Independent census: files calling a writer name directly (grep -rlE
  '\b(record_cycle_end|record_cycle_start|_write_heartbeat)\s*\(' backend scripts tests)
  = autonomous_loop.py, cycle_health.py, test_phase_38_2, test_phase_66_1,
  test_phase_86_38, scripts/qa/mutation_86_110.py. Sweep population = the same minus
  cycle_health.py (by rule) and mutation_86_110.py (excluded by the
  'cycle_health' pre-filter; it only contains the name inside a literal, so exclusion
  is correct in effect).

### FINDING F1 (criterion 4 completeness gap) -- TRANSITIVE reachers not enumerated
- backend/tests/test_phase_85_4_cycle_loudness.py:213 and
  backend/tests/test_phase_85_6_anchor_deadlock.py:374/400/429/441 call
  `autonomous_loop.run_daily_cycle(...)`, and run_daily_cycle calls
  record_cycle_start (autonomous_loop.py:446) and record_cycle_end (:1979).
  Those files therefore REACH a writer, but are NOT in the sweep's population
  (population rule is a direct textual call of a writer name).
- Materiality TODAY: both DO patch _HEARTBEAT_PATH (85_4_cycle_loudness.py:77,
  85_6_anchor_deadlock.py:257), so neither leaks. Also
  test_phase_85_4_completed_age_alarm.py:35 patches it.
- So the criterion-4 CONCLUSION ("no other leaking site") appears true, but the
  static sweep's population under-covers the class it claims to enumerate, and the
  docstring discloses only "cannot follow a call through an alias", not
  "cannot follow a call through a production entry point".

### Reproduced deterministic results (all match the artifacts)
- pytest backend/tests/test_phase_86_110_heartbeat_isolation.py -q -> 10 passed in 4.20s.
  md5 of heartbeat AND rail_guard identical before/after -> the suite self-restores.
- mutation_86_110.py -> CONTROL rc=0 collected=10; P1,P2,P3,P5,P6,P7 all KILLED;
  KILLED=6/6 SURVIVORS=none UNSCORABLE=none. md5 of all 4 touched files identical
  before/after the whole matrix.
- pytest -k "cycle_health or heartbeat or rail_guard or 38_2 or 86_38 or 23_2_14 or
  86_110" -> 68 passed, 3588 deselected. EXACT match to the artifact.
- uvx ruff check --select F821,F401,F811 over a DERIVED 8-file scope (git diff
  --name-only HEAD '*.py' UNION git ls-files --others '*.py', piped via xargs) ->
  "All checks passed!" RUFF_EXIT=0. Non-empty scope asserted (8 files).
- import smoke: backend.services.cycle_health imports; read_heartbeat() returns
  {'cycle_id': '3e5afddb', ...}.
- LIVE endpoint GET /api/paper-trading/freshness -> heartbeat block
  {"cycle_id":"3e5afddb","event":"end","updated_at":"2026-08-17T19:47:15.758944+00:00",
   "age_sec":13338.0,"ratio":0.1543...,"band":"green"} -- the restored value reads
  correctly through the production path.

### Independent per-site mutation (MY cells, not the author's)
Built 4 variants of the rail-guard file in the scratchpad (never touching the tree)
and ran each from cwd=REPO with `-p backend.tests.conftest` so the new guard loads
and repairs:
- ctl (current file)          -> 0 guard msgs. heartbeat md5 unchanged.
- prefix (git show HEAD:...)  -> 2 guard msgs naming BOTH
  test_rail_guard_cycle_history_row_carries_flags AND
  test_cycle_history_row_carries_funnel_counts. => criterion 1's CONTROL reproduced
  independently from the real pre-fix source, and it shows BOTH sites leaked.
- site1 removed only          -> 1 guard msg naming ...carries_flags (the c1 test).
- site2 removed only          -> 1 guard msg naming ...funnel_counts (the c2 test).
=> EACH duplicated site is individually covered. The author's matrix mutates only
site 2 (P1); site 1 is nonetheless independently caught. Gap closed, no finding.
NOTE the relocated copies all show ONE extra failure
(test_rail_guard_import_path_bug_fixed, FileNotFoundError on a REPO_ROOT computed
from __file__) -- a pure relocation artifact, identical in all 4 variants, so it
does not confound the differential. In-tree control: 9 passed, heartbeat unchanged.

### Independent verification of the author's own claims
- P4 equivalence claim VERIFIED: zero files under backend/scripts/tests call
  record_cycle_start( without record_cycle_end(. Removing P4 as an equivalent
  mutant was the right call, and disclosing it beats manufacturing a kill.
- live_check sha256s REPRODUCE EXACTLY from the stated content strings:
  ea504fc3...bffcf == sha256('{"cycle_id": "c2", ..., "21:03:54.974417+00:00"}') and
  a8bcd8c9...fa490 == sha256(same with 22:41:40.533239). Not a spliced capture.
- criterion 5 derivation VERIFIED: heartbeat cycle_id 3e5afddb is the LAST completed
  row of cycle_history.jsonl and updated_at == that row's completed_at
  (2026-08-17T19:47:15.758944+00:00) to the microsecond. 174 rows total; c1 rows=0,
  c2 rows=0. Every field derived, nothing manufactured.
- criterion 6 VERIFIED: `git diff --stat HEAD -- .claude/masterplan.json` is EMPTY;
  86.110 still status=pending; no evaluator_critique* file appears in git status.
- Build-window derivation (mtime > contract mtime 2026-08-18T00:41:32) yields
  EXACTLY the declared file set + contract/live_check/experiment_results +
  .claude/.archive-baseline.json (hook) + handoff/.conviction_fallback_streak.json.
  NO production module is in the window. backend/api/sovereign_api.py (08-17 15:54)
  and backend/services/autonomous_loop.py (08-17 21:42, a "phase-86 UI bugfix" to
  _persist_analysis summary) both PREDATE the contract -> pre-existing other-work,
  not this step's. "no unintended production change" holds.
- research gate: 7 read-in-full (>=5), 25 unique URLs (>=10), recency scan section
  present at :251, audit-class coverage.dry=true after 5 rounds with 2 dry,
  gate_passed=true, brief_status=COMPLETE. Contract cites the brief and lists three
  premise corrections it changed. MET.
- Empirical criterion-4 check: ran ALL 23 backend/tests files that mention
  cycle_health / CycleHealthLog / run_daily_cycle -> 362 passed, 3 failed,
  guard messages = 0, and BOTH protected files md5-unchanged. The 3 failures are in
  test_phase_57_1_reject_binding.py, which is UNMODIFIED vs HEAD and fails
  identically in isolation (3 failed, 4 passed) -> pre-existing, unrelated. The
  fixture provably caused none of them: its ONLY failure mode is the
  "phase-86.110 test guard" message, which appears 0 times.

### FINDING F2 (evidence staleness) -- the FULL-SUITE block predates the new tests
- The artifact's full-suite line sums to 3608+20+12+5+1 = 3646 collected.
- MEASURED NOW: `pytest backend/tests/ --collect-only -q` -> 3656 tests collected,
  and 68+3588 (the -k run, which DOES reproduce) = 3656 too.
- 3656 - 3646 = 10 = exactly the test count of the new
  test_phase_86_110_heartbeat_isolation.py. So the "FULL suite, because the new
  fixture is global" run was executed against a tree WITHOUT this step's own new
  test file, and is presented in experiment_results/live_check as the shipped-tree
  state.
- Materiality: the claim it supports ("the new global autouse fixture causes zero
  failures") still holds -- the fixture WAS present in that run, and the 10 missing
  tests are separately green (I reproduced 10 passed). So this is an
  evidence-freshness defect, not a product defect.
- INDEPENDENT FULL SUITE (mine, 481.68s): 21 failed, 3617 passed, 12 skipped,
  5 xfailed, 1 xpassed. guard messages = 0. heartbeat + cycle_history md5 unchanged.
  NO test from test_phase_86_110_heartbeat_isolation.py is in the failure list, and
  test_phase_23_2_14_no_reentrant_locks is ABSENT -> the 86.108 lock-roster
  regression is genuinely repaired. The artifact also publishes a failure COUNT with
  no failing-test IDs, so membership drift is unauditable from the artifact alone.

### FINDING F3 (WARN, NEW RISK INTRODUCED BY THIS DIFF) -- the guard's repair can
### delete a real row from an append-only production ledger
- backend/tests/conftest.py `_no_tracked_handoff_writes` unconditionally does
  `p.write_bytes(snap[0])` -- restoring the PRE-TEST bytes -- for BOTH protected
  files, one of which is `handoff/cycle_history.jsonl`, an APPEND-ONLY production
  ledger.
- It assumes the only writer during a test is the test. On this local-only
  deployment the live `autonomous_loop` appends to that exact file from the same
  machine (`record_cycle_start` autonomous_loop.py:446, `record_cycle_end` :1979);
  MEASURED today: a real row was appended at 2026-08-17T19:47:15.758944Z.
- Consequence of an overlap with an 8-minute suite run: the appended row is
  SILENTLY REVERTED and an innocent test is failed with a message blaming it.
- Not disclosed in either scope-honesty section. Named fix: for the append-only
  ledger, detect-and-report WITHOUT restoring (or restore only when the new content
  is not a strict superset of the snapshot).

### NOTE F4 -- a LIVE sibling instance of the same class is uncovered
- backend/tests/test_phase_61_2_decision_integrity.py:372-381 calls
  `al._bump_conviction_fallback_streak(...)`, which writes the real
  `handoff/.conviction_fallback_streak.json` (path built at
  autonomous_loop.py:2911; production reads it to drive a `conviction_fallback_streak`
  error page at :1099-1113). It was (re)created during this step's own build window
  at 01:05:22.
- It is UNTRACKED, so it is outside the guard's stated "git-tracked" scope and no
  claim is literally false -- but "catches the CLASS" overstates what was measured:
  the guard catches the two declared git-tracked files.

### NOTE F5 -- kill-mechanism attribution on P1
- In `test_the_previously_leaking_tests_no_longer_touch_the_real_heartbeat`, the
  second assertion (`HEARTBEAT.read_bytes() == before`) CANNOT fail while the
  conftest guard is active, because the guard repairs the file before the subprocess
  exits. P1's kill comes from `assert r.returncode == 0`. Still behavioural, so not
  a vacuity finding -- but the credited mechanism should be the returncode assert.

### Criterion roll-up
1 MET  - reproduced by execution; I independently reran the HEAD (pre-fix) source
         and both sites fired the guard. Author's sha256s recomputed and matched.
2 MET  - both sites carry `monkeypatch.setattr(ch, "_HEARTBEAT_PATH", tmp_path /
         ".cycle_heartbeat.json")` (:202, :222), same idiom as
         test_phase_86_38_degradation_visibility.py:158. No third idiom.
3 MET  - control AND fix both demonstrated; I reproduced both halves plus a per-site
         differential.
4 MET  - source-derived (AST) enumeration + naive cross-check; my own census matches
         the population exactly; both naive over-reports independently confirmed to
         reach no writer; residual static blind spot (transitive reachers) has NO
         live instance and is covered behaviourally by the 3,656-test full run.
5 MET  - regenerated; derivation verified to the microsecond against the ledger's
         last completed row; c1/c2 = 0 of 174 rows; live freshness endpoint serves it.
         Deviation from contract P6 ("write nothing") disclosed and forced by the
         build-time re-measurement the contract itself mandated.
6 MET  - masterplan.json diff vs HEAD EMPTY, 86.110 still pending, no
         evaluator_critique touched, all mutated files md5-restored.

VERDICT (returned via StructuredOutput): CONDITIONAL -- all six criteria MET, capped
by two WARN findings: F3 (new, undisclosed data-integrity risk in the repair path)
and F2 (the full-suite evidence block does not reproduce and predates the shipped
tree).

COMPLETED: 2026-08-17T23:41:29Z
