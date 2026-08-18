STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.110
WRITTEN: 2026-08-17T23:56:55Z

# Q/A write-first record -- step 86.110 (heartbeat test-leak isolation)

Spawn: Workflow rail, Opus 5 (1M). Cycle: TBD (prior-attempt evidence gathered below).

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git scope, lint, scoped tests, runtime smoke
C. LLM judgment vs 6 immutable criteria + guard-vacuity mutation of the NEW guard
D. Residuals (a)/(b) raised last cycle -- judge block vs queue

## Findings (appended as established)

### Prior-attempt / prior-verdict evidence
- `qa_wip.py 86.110 --spawned-at 2026-08-17T23:56:55Z`: source_present=true,
  attempt_number=2, prior_attempts=1, records_retained=2 (gauge),
  attempt_number_status=ok, identity_checked=true.
- `verdict_history_86_21.py --step 86.110 --evidence-only`: status=ok,
  detail="1 verdict(s) from the ledger", verdicts = CONDITIONAL.
- CROSS-CHECK: prior_attempts (1) == ledger rows (1). Ledger is NOT stale for
  this step. sequence: [CONDITIONAL].

### B1. Immutable verification command
`bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/tests/test_phase_66_1_rail_guard.py\").read())" && echo parses'`
-> `parses`, EXIT=0. REPRODUCES.

### B2. Criterion-4 sweep, re-run by me
`python scripts/qa/heartbeat_leak_sweep_86_110.py` -> exit 0.
POPULATION 4, ISOLATED 3, LEAKING 0, prod writers 1. Byte-for-byte the same
table as live_check section 5. Naive over-reports = 2, under = none. REPRODUCES.

### B3. Mutation matrix, re-run by me (with my OWN restore check)
`python scripts/qa/mutation_86_110.py` -> KILLED=9/9, SURVIVORS=none,
UNSCORABLE=none, exit 0. CONTROL rc=0 collected=13 ran FIRST.
My independent shasum of the 3 mutated files + heartbeat + cycle ledger BEFORE
and AFTER the matrix: all five byte-identical. The script's own restore claim is
corroborated, not merely asserted.

### B4. INDEPENDENT control/fix reproduction (my own, clean-room)
Ran the REAL pre-fix test functions fetched from `git show HEAD:` in a scratch
dir with no conftest, PYTHONPATH=repo, and `cycle_health._HEARTBEAT_PATH`
pre-pointed at a scratch SENTINEL via a `pytest_configure` hook. This drives
the actual test code (not a re-implementation) and touches ZERO production
state.
  PRE-FIX  (HEAD)     : 2 passed, rc=0, SENTINEL CHANGED = True,
                        content -> {"cycle_id": "c2", "event": "end", ...}
  POST-FIX (worktree) : 2 passed, rc=0, SENTINEL CHANGED = False,
                        content -> SENTINEL_UNTOUCHED
=> criterion 1 (leak reproduced by execution) and criterion 3 (control AND fix
   both demonstrated) independently established. A PASSING test wrote the
   heartbeat pre-fix; the same test does not post-fix.

### B5. live_check section 2 SHA-256 values recomputed
BEFORE claimed ea504fc3...bffcf -> recomputed from stated content (no trailing
newline) = ea504fc3...bffcf MATCH.
AFTER claimed a8bcd8c9...fa490 -> recomputed = a8bcd8c9...fa490 MATCH.
The capture is not spliced/edited.

### B6. Criterion-4 completeness: my own census vs the author's
(A) direct writer callers, NO cycle_health prefilter, roots
    backend/scripts/tests/frontend/.claude:
      autonomous_loop.py, cycle_health.py(defn), test_phase_38_2, test_phase_66_1,
      test_phase_86_38, scripts/qa/mutation_86_110.py(this step's own harness)
    => symmetric difference vs the sweep's population = EMPTY (after excluding
       the definition module and this step's own matrix script). Sweep is
       complete for DIRECT callers.
(B) criterion-4's LITERAL rule (every setattr(...)_HISTORY_PATH site) = 7 files:
      test_cycle_heartbeat_alarm, test_phase_38_2, test_phase_66_1,
      test_phase_85_4_completed_age_alarm, test_phase_85_4_cycle_loudness,
      test_phase_85_6_anchor_deadlock, test_phase_86_38.
    The sweep's naive-survey line prints only 2 because it excludes files that
    patch BOTH constants. Verified: the three 85_x files each carry a
    _HEARTBEAT_PATH reference. So the literal rule adds no uncovered member.
(C) TRANSITIVE reachers via run_daily_cycle: SIX callers, not the TWO the
    author's residual (a) names -- 85_4_cycle_loudness, 85_6_anchor_deadlock,
    36_12_kill_switch_trading_path_block, 36_17_halt_stop_loss_enforcement, plus
    tests/services/test_autonomous_loop_async.py and tests/verify_phase_25_B3.py.
    ADJUDICATED: none leaks. 85_4/85_6 patch _HEARTBEAT_PATH directly; 36_12 and
    36_17 stub `cycle_health.get_log` (36_12:289/369/433, 36_17:271/503/566) and
    36_12's own docstring at :242 documents this exact hazard; the two root
    `tests/` files only regex the SOURCE of run_daily_cycle and never execute it.
    => the residual's SUBSTANTIVE claim ("no live instance leaks") HOLDS; its
       ENUMERATION ("all three") under-counts the population at 2-of-4. WARN-level
       scope-honesty imprecision, not a defect.

### B7. MY OWN mutation cells (targeting shapes the author's matrix cannot reach)
Control GREEN first (rc=0). Each cell restored + sha-verified; heartbeat and
cycle ledger restored to their exact pre-run digests (1c9428d5..., dd8d01a2...).

  QA-M1  KILLED    revert ONLY SITE 1 (flags test, ~:194), site 2 intact
                   -> killed by test_the_previously_leaking_tests_no_longer_
                      touch_the_real_heartbeat. BOTH sites are INDIVIDUALLY
                      covered. (This retires the plausible-but-wrong finding
                      that the structural test's first-match `_isolates` leaves
                      one site unguarded -- it does, but the behavioural test
                      does not.)
  QA-M2  *** SURVIVED ***   append_only rule, drop the ANDed prefix half:
                   `return after.startswith(before) and len(after) > len(before)`
                   ->  `return len(after) > len(before)`
                   ENTIRE 13-test suite stays GREEN. rc=0, zero failures.
  QA-M3  KILLED    ledger_backed keyed form -> bare `cid in ledger`
                   -> killed by 2 tests. The keyed form IS load-bearing.

**QA-M2 is a REAL SURVIVING MUTANT in the code cycle 2 added.** Why the
author's suite cannot see it: `test_a_real_APPEND_to_the_cycle_ledger_is_not_
reverted` supplies `rewritten = b'{"cycle_id": "zzz"}\n'` (20 bytes) against a
42-byte snapshot, so its "a REWRITE is not an append" case varies TWO properties
at once (not-a-prefix AND shorter) and is satisfied by the length half alone.
Truncation (b"") and no-growth (before) are likewise length-only. So NO fixture
in the suite can distinguish the prefix half. Cell P9 mutates the whole return
to `True`, which the length-only cases do catch -- sub-expression granularity is
the gap.
CONSEQUENCE if the prefix half were ever weakened: any LONGER rewrite of
handoff/cycle_history.jsonl by a test is classified "legitimate concurrent
append", so the guard neither restores nor fails -- blind to that leak class.
Named fix: add a longer-than-snapshot non-prefix fixture, e.g.
  rewritten_longer = b'{"cycle_id":"z"}\n{"cycle_id":"y"}\n{"cycle_id":"x"}\n'
  assert legit("handoff/cycle_history.jsonl", before, rewritten_longer) is False
This makes the experiment_results claim "ALL THREE arms are mutated ... All
KILLED" true at rule granularity but NOT at sub-expression granularity, and the
anti-blindness argument it supports is therefore incompletely evidenced.
SEVERITY: WARN (illusory-guard #17 -- a genuine behavioural guard coexists, and
the conftest guard is voluntary hardening that NO immutable criterion requires).
NOTE the length clause is REDUNDANT given startswith (equal content cannot reach
the check -- the sha comparison short-circuits it), so the suite exercises only
the redundant half and never the load-bearing one.

### B13. Full-suite F2 claim, re-run by ME (independent)
`pytest backend/tests/ -q -p no:cacheprovider --no-header`
MINE  : 19 failed, 3622 passed, 12 skipped, 5 xfailed, 1 xpassed in 547.19s
THEIRS: 20 failed, 3621 passed, 12 skipped, 5 xfailed, 1 xpassed in 511.28s
Collected reconciles to 3659 in BOTH (19+3622+12+5+1 = 20+3621+12+5+1 = 3659),
which corroborates "3,646 before this step's 13 tests + 13 = 3,659".
Failing-ID symmetric difference = EXACTLY ONE member:
  test_phase_82_54_cost_budget_columns::test_production_sql_dry_runs_valid
  -- failed for the author, PASSED for me. It is a BigQuery dry-run, i.e.
  network/environment dependent. So the F2 claim reproduces 19/20, and the
  claim that the set is "byte-identical" is NOT a stable invariant: one member
  of it is flaky by construction. NOTE-level.
`grep -c "phase-86.110 test guard"` over MY full run -> **0**. The author's
zero-guard-firings claim reproduces. 0 of this step's own 13 tests failed.
CAVEAT ON MY OWN RUN, disclosed: my QA-M1/M2/M3 cells ran concurrently ~2 min
into this suite, briefly editing conftest.py and the rail-guard file. No 86.110
test failed and no guard message appeared, so no contamination materialised, but
the 547s vs 511s wall-clock gap is probably mine.

### B14. Tree integrity after ALL my verification
shasum -a 256 of the 3 mutated files + heartbeat + cycle ledger is byte-identical
to my pre-verification snapshot (21b2c097 / 70461e90 / 3a69c139 / 1c9428d5 /
dd8d01a2). My evaluation left no residue.

### B15. Residual (b), verified accurate
test_phase_61_2_decision_integrity.py:372-375 drives
`al._bump_conviction_fallback_streak`, which writes
handoff/.conviction_fallback_streak.json (autonomous_loop.py:2911), a file
production READS (autonomous_loop.py:1099, :1113). `git ls-files --error-unmatch`
-> "did not match any file(s) known to git", i.e. UNTRACKED, so genuinely
outside the guard's declared git-tracked scope. The author's characterisation is
precise and no claim is literally false. QUEUE, do not block: no immutable
criterion reaches it.

### B16. NOTE -- auto-commit blast radius (session-level, not 86.110's fault)
`git diff --name-only HEAD -- '*.py'` also returns backend/api/sovereign_api.py
(mtime 15:54) and backend/services/autonomous_loop.py (mtime 21:42) -- both
PREDATE this step's research brief (22:32) and both are other steps' uncommitted
work (autonomous_loop's diff is a "phase-86 UI bugfix" to _persist_analysis's
summary field). experiment_results' "the only production file touched is none"
is TRUE for THIS step's authorship. But `auto-commit-and-push.sh` does
`git add -A`, so a status flip here ships those under 86.110's name, and
autonomous_loop.py is a production module the running backend has already
imported -- the "no restart pending from this step" line is correct as scoped
and misleading if read as a statement about the tree.

## VERDICT DECIDED: CONDITIONAL
worst-of-lenses: correctness=PASS, does-it-reproduce=PASS,
scope-honesty=CONDITIONAL -> min = CONDITIONAL.
All 6 immutable criteria MET and independently re-derived by execution; harness
compliance 5/5 clean; masterplan and verdict ledger untouched. Capped by ONE
executed surviving mutant (QA-M2) in code this cycle shipped, plus a coverage
claim that mutation falsifies, plus two enumeration/stability NOTEs.

COMPLETED: 2026-08-18T00:12:56Z

### B8. Criterion-by-criterion (independent evidence)
C1 MET  - B4 PRE-FIX: 2 passed, sentinel CHANGED, wrote cycle_id "c2".
C2 MET  - diff adds `monkeypatch.setattr(ch,"_HEARTBEAT_PATH", tmp_path/...)` at
          BOTH sites (:195 area and :220 area). test_phase_86_38's `health`
          fixture uses the identical idiom (verified :158
          `monkeypatch.setattr(ch,"_HEARTBEAT_PATH", tmp_path/"hb.json")`), so
          this is the SAME idiom, not a third one. QA-M1 proves per-site cover.
C3 MET  - B4 POST-FIX: 2 passed, sentinel UNCHANGED. Control AND fix both shown,
          by me, independently of the author's capture.
C4 MET  - sweep re-run exit 0; my independent census (A) has EMPTY symmetric
          difference with the sweep's population; the criterion's literal
          setattr-rule census (B) adds no uncovered member; transitive reachers
          (C) all adjudicated non-leaking.
C5 MET  - heartbeat = {"cycle_id":"3e5afddb","event":"end","updated_at":
          "2026-08-17T19:47:15.758944+00:00"}. Ledger: 174 rows; "3e5afddb"
          appears 2x; "c2" 0x; "c1" 0x. LAST ledger row is 3e5afddb
          status=completed completed_at=2026-08-17T19:47:15.758944+00:00 --
          byte-identical to the heartbeat's updated_at. Payload SHAPE matches
          production `_write_heartbeat` exactly (cycle_health.py:555-558,
          event "end" from :492). Derived, not manufactured. Disposition stated.
C6 MET  - .claude/masterplan.json diff is EMPTY; 86.110 still status=pending.
          handoff/verdict_ledger.jsonl --numstat = 1 added / 0 removed (append
          only, no prior verdict altered). harness_log has 0 rows for 86.110
          (log-last respected).

### B9. Deterministic gates
- Lint: DERIVED scope via `git diff --name-only HEAD -- '*.py'` UNION
  `git ls-files --others --exclude-standard -- '*.py'` = 8 files (non-empty
  asserted), piped through xargs. `uvx ruff check --select F821,F401,F811
  --no-cache` -> "All checks passed!" RUFF_EXIT=0.
- Runtime smoke: `import backend.services.cycle_health` OK;
  `import backend.tests.conftest` OK (protected map loads with both rules).
  Backend live: GET /api/health -> 200.
- Frontend gate 1b: NOT triggered -- this step's diff touches no frontend file.
- UI gate 1c: NOT triggered -- no immutable criterion makes a UI claim and the
  diff touches no frontend file. The "dashboard reads this file" line is
  motivation; verified at source level instead (consumers:
  cycle_health.read_heartbeat / cycle_heartbeat_alarm, freshness_cron.py:36-38).

### B10. Harness compliance (5 items)
1. research-gate-before-contract: research_brief_86.110.md envelope
   brief_status COMPLETE, external_sources_read_in_full 7 (>=5),
   urls_collected 25 (>=10), recency_scan_performed true, gate_passed true.
   mtime 22:32:54 < contract 00:41:32 < artifacts 01:44-01:56. ORDER OK.
2. contract-before-generate: contract 00:41:32 precedes every generated
   artifact (test file 01:44:06, sweep 01:45:37, matrix 01:45:17,
   rail_guard 01:52:29, conftest 01:46:07). OK.
3. experiment_results present (6,891 bytes) + live_check (18,072 bytes). OK.
4. log-last: harness_log has 0 `phase=86.110` rows; masterplan unflipped. OK.
5. no-verdict-shopping: prior verdict CONDITIONAL (ledger cycle 1,
   wf_e7115d07-ae1, 23:43:23Z). Evidence CHANGED since: experiment_results and
   live_check both mtime 01:56:20 (> 23:43:23), conftest 01:46:07, matrix
   01:45:17, test file 01:44:06 -- all AFTER the prior verdict. The two named
   blockers were materially addressed (concurrency rule + 3 new cells + 3 new
   tests; full-suite re-measured with the ID list). LEGITIMATE cycle-2 respawn.

### B11. Contract-vs-execution divergence (checked, then retired)
The contract's P6 says "criterion 5, disposition: WRITE NOTHING" and Scope
honesty says "It writes nothing to handoff/.cycle_heartbeat.json". The step
REGENERATED it instead. I first read this as an undisclosed plan deviation --
it is NOT: the contract's own P6 continues "**Recorded as a measurement at
evaluation time, not as a claim carried from the gate** -- if the value has
moved again, the artifact says so", and the value HAD moved back to `c2`. The
reversal is disclosed prominently in live_check section 2 and experiment_results
"Three things this step found". Criterion 5 explicitly permits either option
with a stated reason. NOTE only.

### B12. Code-review heuristics (5 dimensions evaluated)
No BLOCK fired. Nothing security-relevant (no secrets, no subprocess with
non-literal args beyond the venv python + literal paths, no LLM->execution
path). No trading-domain invariant touched (kill switch, stop-loss,
perf_metrics all untouched). No tautological assertion found in the new suite --
it carries explicit anti-vacuity negatives (test file :90-91 comment/docstring
must NOT satisfy `_isolates`; :172 guard must pass a clean test). One WARN:
illusory-guard #17 sub-shape (fixture cannot represent the failure) at
conftest.py `_is_legitimate_concurrent_write` append_only, evidenced by QA-M2.
