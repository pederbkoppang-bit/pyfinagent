# live_check -- phase-86.24

**Code commit:** `d5180e27`. **Measurement tree:** `70e646b7`.
**Measured:** 2026-08-10, 09:45-10:20 CEST.

## A. Criterion 4 -- both named modules, post-midnight boundary AND mid-day

The "post-midnight boundary" is simulated by putting the LOCAL calendar day one
behind UTC, which is exactly the 00:00-02:00 CEST window in which these tests
used to fail.

```
$ python -m pytest backend/tests/test_phase_82_0_macro_ingestion.py \
                   backend/tests/test_phase_86_2_replay_poison_row.py -q
24 passed in 1.16s                    <- system clock (local == UTC)

$ TZ=Pacific/Midway python -m pytest <same two modules> -q
24 passed                             <- local 2026-08-09, UTC 2026-08-10

BEFORE the fix, same command, same tree:
1 failed, 23 passed                   <- system clock
3 failed, 21 passed                   <- TZ=Pacific/Midway
```

This is also asserted IN the suite, so it cannot rot:
`test_the_two_repaired_modules_PASS_AT_A_SHIFTED_CLOCK` re-runs both modules in a
subprocess under the shifted TZ, and carries a **positive control** that fails if
the shift did not actually move the local date -- without it the test would pass
by simply not shifting anything.

## B. Criterion 1 -- recall validation, run BEFORE the method was trusted

```
TZ=Europe/Oslo         local 2026-08-10 == UTC   ->  1 of 3 known positives red
TZ=Pacific/Kiritimati  local 2026-08-10 == UTC   ->  1 of 3
TZ=Pacific/Midway      local 2026-08-09 != UTC   ->  3 of 3   <-- ALL THREE
```

The static alternative was REJECTED rather than adjusted, per the criterion: an
own-clock AST scan (49 of 457 files) MISSES `test_phase_86_2_replay_poison_row.py`
entirely, because that file contains zero clock calls -- the clock read is in
production.

## C. The population, before and after -- full suite, frozen tree

```
PRE-FIX    base    16 failed / 3351 passed        shifted 19 failed / 3348 passed   delta 3
CYCLE 1    base    15 failed / 3360 passed (375.86s)   shifted 15 / 3360 (368.58s)
CYCLE 2    base    15 failed / 3362 passed (376.94s)
           shifted 15 failed / 3362 passed (373.19s)
           DELTA   EMPTY -- no test changes verdict with the clock
```

The counts reconcile end to end: the poison-row test flipped red->green
(-1 failed, +1 passed), cycle 1 added 8 tests (3351 -> 3360) and cycle 2 added 2
more (3362) -- the uncovered-band test and the recompute property test.

## D. Criterion 3 -- the adjudication, measured not argued

**CORRECTED IN CYCLE 2. The cycle-1 text here claimed the trailing leg covers the
overnight window. That is FALSE IN A BAND, and the claim is replaced rather than
softened.** Full sweep, `_AUDIT_PATH` redirected to tmp:

```
anchor    nav   armed  stale  daily  trailing  ANY
TODAY     99.0  True   False  False  False     False
TODAY     95.0  True   False  True   False     True
TODAY     92.0  True   False  True   False     True
TODAY     89.0  True   False  True   True      True
TODAY     80.0  True   False  True   True      True
STALE     99.0  False  True   False  False     False
STALE     95.0  False  True   False  False     False   <-- 5% loss: NOTHING fires
STALE     92.0  False  True   False  False     False   <-- 8% loss: NOTHING fires
STALE     89.0  False  True   False  True      True
STALE     80.0  False  True   False  True      True
```

Between the daily limit (4%) and the trailing limit (10%), a stale anchor leaves
nothing firing. The cycle-1 guard exercised only `nav=80.0` and so could not see
the gap it claimed to close.

**The adjudication is unchanged -- there is no live defect -- because the
ENFORCEMENT PATH NEVER EVALUATES AGAINST A STALE ANCHOR:**

```
paper_trader.check_and_enforce_kill_switch
  :1413   if sod_anchor_needs_reroll(snap, today):   <-- re-anchor FIRST
  :1460   breach = evaluate_breach(...)              <-- then evaluate
  :1468   if breach["any_breached"] and not state.is_paused():   <-- keys on
                                                        any_breached, NEVER armed
  :1372   pre_armed = pre.get("baselines_present", ...)  <-- order gate, not armed
```

The band above is reachable only by a READ-ONLY caller (the badge endpoint). It
is now pinned by `test_a_stale_anchor_leaves_the_band_between_the_two_limits_UNCOVERED`,
which asserts the uncomfortable fact together with a fresh-anchor control showing
the same navs DO breach when the anchor is current -- so the result is
attributable to staleness rather than to an inert threshold.

Staleness is also asserted for anchors 1, 2, 7 and 365 days old, with a same-day
control proving a kill switch that never armed could not pass by default.

**No assertion was weakened.** `daily_loss_breached is True` in the poison-row
test is byte-unchanged; only the fixture's day became relative.

## E. Criterion 5 -- no global time freeze

None introduced. `test_no_global_time_freezing_fixture_is_introduced` sweeps
EVERY `conftest.py` in the repo (excluding `.venv`, `node_modules`) for
`freeze_time`, `freezegun`, `time_machine`, `time-machine`, `libfaketime`,
`FrozenDateTimeFactory`, `travel(` and fails on any hit. Measured: no time
library is installed at all -- `freezegun`, `time-machine`, `pytest-freezegun`
and `libfaketime` are all absent, and neither `faketime` nor `datefudge` is on
PATH.

## F. Criterion 6 -- mutation matrix

`python scripts/qa/mutation_matrix_86_24.py`

```
M1 KILLED  control rc=0 mutant rc=1   revert the macro tests to the LOCAL clock domain
M2 KILLED  control rc=0 mutant rc=1   re-pin the poison-row fixture to the day it was written
M6 KILLED  control rc=0 mutant rc=1   SNAPSHOT the fixture date at import (cycle-2 finding 2)
M7 KILLED  control rc=0 mutant rc=1   point the band test OUTSIDE the band (cycle-2 finding 1)
M3 KILLED  control rc=0 mutant rc=1   give the STALE-anchor test a FRESH anchor
M4 KILLED  control rc=0 mutant rc=1   remove the clock shift from the differential test
M5 KILLED  control rc=0 mutant rc=1   point the how-stale sweep at a FRESH anchor

tracked sources UNCHANGED: True
  test_phase_82_0_macro_ingestion.py     566a607e91365c67
  test_phase_86_2_replay_poison_row.py   5c1ce1116769d118
  test_phase_86_24_clock_dependence.py   36f469402a7e8333
stray mutant files left behind: none
All 7 mutants killed.
```

Every mutant is a COPY written under `backend/tests/` with a temporary name and
removed in a `finally`; the tracked files are never opened for writing, and that
is proven by digest rather than asserted. A copy inside the test tree (not
`/tmp`) is deliberate -- pytest's rootdir, the repo-root conftest egress guards
and the `backend.*` import path all have to apply exactly as they do for the real
module, or the mutant would run under different rules than its subject. The
harness returns exit 2 rather than a clean result if a stray file survives or a
digest moves.

## G. The book was never touched

`handoff/kill_switch_audit.jsonl` = `ea78508bee73887c82df2346da408c72...`,
64 lines, byte-identical across every run in this step including both full-suite
runs and the mutation matrix. The new module redirects `ks._AUDIT_PATH` to
`tmp_path` in a fixture and never writes the operator's journal.
