# live_check -- phase-86.24

**Code commit:** `d5180e27`. **Measurement tree:** `70e646b7`.
**[phase-86.34]** Those two fields were two commits behind by the time cycle 3
landed. The tree that section F's digest is now measured against is `37e0543f`.
Sections above this line are the cycle-1/2 record and are NOT re-measured;
only section F was regenerated.
**Measured:** 2026-08-10, 09:45-10:20 CEST.

## A. Criterion 4 -- both named modules, post-midnight boundary AND mid-day

**[phase-86.34 CORRECTION -- the sentence that stood here was DIRECTIONALLY
INVERTED, and it survived one round of remediation.]** It read: *"The
'post-midnight boundary' is simulated by putting the LOCAL calendar day one
behind UTC, which is exactly the 00:00-02:00 CEST window in which these tests
used to fail."* The second clause is false. MEASURED with `zoneinfo`:

```
00:30 CEST -> local 2026-08-10 / UTC 2026-08-09   = local AHEAD of UTC
01:30 CEST -> local 2026-08-10 / UTC 2026-08-09   = local AHEAD of UTC
TZ=Pacific/Midway (UTC-11)                        = local BEHIND UTC
```

So the fixture is the **MIRROR** of the window it claims to reproduce, not that
window. What is true, and is the operative property, is that both put the LOCAL
calendar day on a DIFFERENT date from UTC -- which is what the tests are
sensitive to, so no result in this file changes.

The correction is recorded rather than the sentence silently deleted: a claim
that is quietly removed teaches the next reader nothing about how it got there.

**How this survived**: the phase-86.34 cycle-1 remediation corrected the test
docstring and offered `grep -cF "one day behind" <this file>` = 0 as proof this
file was clean. That oracle is VACUOUS -- the literal never appeared here (the
wording is "calendar day one\nbehind UTC": no "day" between "one" and "behind",
and line-wrapped), so it returned 0 at every commit whether or not the claim was
present. The Q/A caught it and returned FAIL.

**The honest count after this edit, and it is NOT zero.** A substring oracle
cannot distinguish an assertion from a quotation of it, and this correction
quotes the retired sentence deliberately:

```
$ grep -cF "which is exactly the 00:00-02:00 CEST window" handoff/current/live_check_86.24.md
2          # before this edit: 1
```

Both occurrences are INSIDE this correction block -- one in the quoted original,
one naming the oracle. **Zero occurrences remain outside it**, which is the
property that matters and the one the checker below actually tests. Reporting
"0" here would have been a third vacuous claim in the same paragraph that
complains about the first two.

Re-runnable, and it fails if the assertion returns anywhere else in the file:

```
$ python scripts/qa/verify_86_24_direction_claim.py
```

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
overnight window. That is FALSE IN A BAND.**

**Where the claim was withdrawn, stated as a location list rather than as a
completeness claim** -- cycle 2's first attempt said "replaced rather than
softened", which is a claim about the whole tree, and the cycle-2 Q/A showed it
did not survive a recall test: the withdrawn sentence was still standing in live
source two lines above an edit I had just made. Derived over the seven files this
step owns or edited:

| location | state |
|---|---|
| `experiment_results_86.24.md` §1 | struck through + replaced (cycle 2) |
| `live_check_86.24.md` §D (this section) | replaced (cycle 2) |
| `test_phase_86_24_clock_dependence.py` | test renamed; the false claim quoted only as history (cycle 2) |
| `test_phase_86_2_replay_poison_row.py:55-58` | **MISSED in cycle 2, rewritten in cycle 3** -- live source, inside the comment block cycle 2 edited |
| `contract_86.24.md` §2 | annotated in cycle 3 (dated artifact -- annotated, not rewritten) |
| `research_brief_86.24.md` `:24/:101/:143/:356/:441` | annotated at the head in cycle 3 (dated artifact) |

**Out of scope, and deliberately so:** a words-based sweep of the whole tree
returns 161 hits for "trailing leg still fires" / "per-leg" / "still fires", and
almost all are OTHER steps' claims about other guards. Prior work also stated the
BOUNDED form correctly -- `experiment_results_85.6.md:244` says "bounding
exposure to `[daily_limit, trailing_limit)`", which is exactly right. It was this
step's cycle-1 wording that dropped the bound; other steps' dated artifacts are
not rewritten here.

Full sweep, `_AUDIT_PATH` redirected to tmp:

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

**[phase-86.34 -- REGENERATED IN FULL, cycle 2.]** The Q/A (`wf_839de1e6-c3c`)
found this block still recorded `test_phase_86_24_clock_dependence.py =
36f469402a7e8333` while the real value was `9b5cb2e44e6ba8a4`. **This step made
it stale** (`36f469402a7e8333` -> `55e24bb26a93f131` at `a37f9da5` ->
`9b5cb2e44e6ba8a4` at `73ce11ba`), and cycle 1 refreshed only the poison-row
digest sitting beside it. So the whole block is replaced with fresh output rather
than the one number edited -- criterion 4's rule, applied to the defect
criterion 4 was written about.

Producing command, re-run at tree `a9707993`:

```
$ python scripts/qa/mutation_matrix_86_24.py
id   verdict   probe                                          mutation
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
M1   KILLED    control rc=0 mutant rc=1                       revert the macro tests to the LOCAL clock domain
M2   KILLED    control rc=0 mutant rc=1                       re-pin the poison-row fixture to the day it was written
M6   KILLED    control rc=0 mutant rc=1                       SNAPSHOT the fixture date at import instead of recomputing per call
M7   KILLED    control rc=0 mutant rc=1                       point the band test OUTSIDE the band -- does it discriminate?
M3   KILLED    control rc=0 mutant rc=1                       give the STALE-anchor test a FRESH anchor -- does it discriminate?
M4   KILLED    control rc=0 mutant rc=1                       remove the clock shift from the differential test -- its positive control must fire rather than the test passing for free
M5   KILLED    control rc=0 mutant rc=1                       point the how-stale sweep at a FRESH anchor
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tracked sources UNCHANGED: True  [('test_phase_82_0_macro_ingestion.py', '566a607e91365c67'), ('test_phase_86_2_replay_poison_row.py', 'fb97b52ecf7fb5be'), ('test_phase_86_24_clock_dependence.py', '9b5cb2e44e6ba8a4')]
stray mutant files left behind: none

All 7 mutants killed.
```

All three digests above are emitted by that command in the same run, so they
cannot drift from each other again the way the hand-maintained list did.


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
