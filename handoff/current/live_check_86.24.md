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
POST-FIX   base    15 failed / 3360 passed (375.86s)
           shifted 15 failed / 3360 passed (368.58s)
           DELTA   EMPTY -- no test changes verdict with the clock
```

16 -> 15 failures and 3351 -> 3360 passes reconciles exactly: the poison-row test
flipped red->green (-1 failed, +1 passed) and this step added 8 tests.

## D. Criterion 3 -- the adjudication, measured not argued

```
armed=False  daily_baseline_stale=True  daily_loss_breached=False
trailing_dd_breached=True  any_breached=True
```
A stale daily anchor disarms the DAILY leg only. The trailing drawdown limit is a
high-water mark, not date-scoped, and still fires -- so the overnight window is
not naked. Asserted every day now, including for anchors 1, 2, 7 and 365 days
stale, and with a same-day control proving a kill switch that never armed could
not pass the staleness test by default.

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
M3 KILLED  control rc=0 mutant rc=1   give the STALE-anchor test a FRESH anchor
M4 KILLED  control rc=0 mutant rc=1   remove the clock shift from the differential test
M5 KILLED  control rc=0 mutant rc=1   point the how-stale sweep at a FRESH anchor

tracked sources UNCHANGED: True
  test_phase_82_0_macro_ingestion.py     566a607e91365c67
  test_phase_86_2_replay_poison_row.py   5cf5073d39707e6d
  test_phase_86_24_clock_dependence.py   03ad07ced183b80d
stray mutant files left behind: none
All 5 mutants killed.
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
