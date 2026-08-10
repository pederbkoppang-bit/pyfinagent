# live_check -- step 86.34

**Captured**: 2026-08-10, **19:20-19:26 UTC / 21:20-21:26 CEST**, after book cycle
`a5654ab9` completed (21:15:34 CEST) and the 19:30 freeze lifted.
**Ambient state is recorded deliberately** -- this step exists because a result
that depends on the wall clock is not a result until you say which clock.

Tree: `git rev-parse HEAD` = see the commit that carries this file. Producing
commands are inline next to every number.

---

## A. N1 -- direction, MEASURED with zoneinfo (criterion 1)

```
$ python3 -c "<zoneinfo sweep, inline below>"
  now UTC = 2026-08-10 19:20
  Pacific/Midway         local=2026-08-10 utc=2026-08-10  -> SAME
  Pacific/Kiritimati     local=2026-08-11 utc=2026-08-10  -> AHEAD
  Europe/Oslo            local=2026-08-10 utc=2026-08-10  -> SAME

  the real 00:30 / 01:30 CEST window (what 86.24 claimed to simulate):
    00:30 CEST -> local 2026-08-10 / UTC 2026-08-09  = local AHEAD
    01:30 CEST -> local 2026-08-10 / UTC 2026-08-09  = local AHEAD
```

**The direction the test actually simulates, measured at capture time:**

```
$ python3 -c "from test_phase_86_24_clock_dependence import _date_shifting_tz; ..."
  selector -> Pacific/Kiritimati   local=2026-08-11 utc=2026-08-10  shifted=True
```

So at this hour the suite simulates local **AHEAD** -- which is the same
direction as the real 00:30/01:30 CEST window. Earlier in the UTC day the
selector returns Midway and simulates **BEHIND**. Both satisfy the operative
property (local calendar day != UTC calendar day); the step's point is that the
zone is now *chosen* rather than *assumed*.

### The grep -- WITHDRAWN. It was a vacuous oracle and it cost a FAIL.

**[cycle 3] This section previously offered the two greps below as criterion-1
proof. The second one is worthless and the cycle-1 Q/A (`wf_839de1e6-c3c`)
returned FAIL on exactly that.**

```
$ grep -cF "one day behind" handoff/current/live_check_86.24.md
0          # <- offered as proof the file was clean. IT WAS NOT.
```

That literal has **never** existed in that file, at any commit -- the evaluator
checked `cefe7515`, `da9263d6`, `9424939c` and `HEAD` and got 0 at every one. The
file's wording is `calendar day one\nbehind UTC`: no "day" between "one" and
"behind", **and a line break in the middle**, so a line-oriented `grep` could
never match it. The 0 was structural. The inverted claim was standing, asserted,
at `live_check_86.24.md:12-13` the entire time.

**And the number is now stale as well as vacuous**: it reads **1** at `HEAD`,
because the cycle-2 correction block introduced the literal when it quoted the
dead grep. Both defects were found by evaluators, neither by me.

**The replacement** is `scripts/qa/verify_86_24_direction_claim.py`, which checks
the property the criterion is actually about -- *is the claim ASSERTED anywhere
outside the correction block* -- and is proven by execution rather than by
reading:

```
$ python scripts/qa/verify_86_24_direction_claim.py
total occurrences        : 2
occurrences OUTSIDE block: 0
  positive controls: C1 ok, C2 ok, C3 ok (each verified able to fail)
OK -- the claim appears only inside the phase-86.34 correction block
exit=0
```

Mutation-proven at tree `b4054ff5` (exit codes captured without a pipeline, since
`$?` after a pipe reads `tail`, not the checker):

| cell | exit | result |
|---|---|---|
| clean tree | 0 | OK |
| **the cycle-2 Q/A's V3** -- re-assert the claim in the tail of section A, after the correction block | **1** | `ASSERTED at line(s) [95]` -- **this returned 0 before the sentinel; it is the gap the evaluator found** |
| delete the `[END phase-86.34 CORRECTION]` sentinel | **1** | fails CLOSED rather than silently restoring the over-wide scope |
| restored | 0 | OK |

The cycle-1 Q/A independently replayed the **pre-fix** file
(`git show 551d5188:handoff/current/live_check_86.24.md`) through the checker and
got `exit 1, ASSERTED at line(s) [13]` -- the same input on which the old grep
returned 0. That is the difference between the two oracles, measured.

**The count is 2, not 0, and that is stated deliberately.** A substring oracle
cannot distinguish an assertion from a quotation, and the correction quotes the
retired sentence on purpose. Reporting 0 would have been a third vacuous claim.

### The test-file occurrence

```
$ grep -cF "one day behind" backend/tests/test_phase_86_24_clock_dependence.py
1
```

That one is the old claim being *quoted and refuted*, at `:291-296`:

```
    This used to hardcode `TZ=Pacific/Midway` and claim it "puts the LOCAL date
    one day behind UTC, which is exactly the 00:00-02:00 CEST window in which the
    two macro tests used to fail". Both halves were wrong:

      * DIRECTION -- at 00:30/01:30 CEST the local date is one day AHEAD of UTC.
        Midway (UTC-11) is BEHIND. The fixture was the MIRROR of the window it
```

A naive `grep -c` reads that as criterion 1 unmet. The assertion is gone; the
citation is deliberate, because a correction that deletes the wrong sentence
without recording it teaches the next reader nothing.

## B. N2 -- swept population, before and after (criterion 2)

```
  total conftest.py in tree : 70
  OLD rule ('.venv' in parts) kept : 34  of which vendored under .venv* : 32
  NEW rule ('.venv*' prefix)  keeps : 2
  NEW-rule survivors:
      backend/tests/conftest.py
      conftest.py
```

Both survivors are first-party. The guard prints its own population at run time
(`[86.34] conftest sweep population: ...`) and asserts **both** that it is
non-empty and that no swept path contains a `.venv*` element -- the second
assertion is the one that matters, because "non-empty" is a proxy and
first-party is the property.

## C. Mutation -- the cells criterion 3 asked for (criterion 3)

The three 86.34 guards previously existed only as prose in
`experiment_results_86.34.md`; there was **no executable matrix**, which does not
satisfy "in `scripts/qa/mutation_matrix_86_24.py` or a successor". New file
`scripts/qa/mutation_matrix_86_34.py`.

**Criterion 3 has TWO halves and the first was initially missing.** It requires
(a) *"inject a conftest containing a global time-freezing fixture into a fake
repo root and require the guard to go RED"* and (b) *"mutate the exclusion rule
itself ... and require a NAMED assertion to fire"*. My first matrix covered only
(b). Half (a) was unprovable because the repo contains no time-freezing conftest
by construction, so a **test-only seam** was added --
`PYFINAGENT_86_34_SWEEP_ROOT`, defaulting to the real `REPO`, mirroring the
existing `PYFINAGENT_86_24_PROW_PATH` seam at `:386` of the same file.

```
$ python scripts/qa/mutation_matrix_86_34.py
repo   : /Users/ford/.openclaw/workspace/pyfinagent
utc    : 2026-08-10 19:29 -> non-shifting zone chosen for N1: Pacific/Midway

  N1-HARDCODE-NONSHIFTING-TZ  KILLED       replace the runtime zone selector with a zone that does NOT shift the date now (Pacific/Midway) -- the positive control must fire
  N2-REVERT-EXCLUSION         KILLED       revert the sweep to the EXACT-element rule -- it re-admits the vendored .venv.py313.bak corpus, so the first-party assertion must fire
  N2-EMPTY-POPULATION         KILLED       make the sweep match NOTHING -- the non-vacuity assertion must fire (the guard is broken at its SUBJECT, not deleted)
  N2-POISONED-CONFTEST        KILLED       inject a conftest declaring a session-scoped freeze_time into a fake repo root -- the named offenders assertion must fire

OK -- all 4 cell(s) KILLED
exit=0
```

The RED that the poison cell produces, verified to come from the **named**
assertion rather than any convenient error:

```
E       assert not ['sub/conftest.py: freeze_time', 'sub/conftest.py: freezegun']
[86.34] conftest sweep population: 1 first-party file(s): ['sub/conftest.py']
```

**The N1 cell is deliberately NOT "hardcode Pacific/Midway".** That mutant is
correct for 11 hours of the day and would survive during 00:00-10:59 UTC --
reporting the wall clock instead of the guard. The cell instead hardcodes
whichever candidate zone does **not** shift the date at run time (printed
above), so it kills at every hour.

**A third probe defect, mine, recorded because it nearly became a false
survivor.** The poison cell first reported `MIS-ATTRIBUTED` -- it checks that the
RED carries the named assertion text, but `run()` returned only the last 600
characters of output, which held pip's `RequestsDependencyWarning` and not the
assertion. The guard was correct; the discriminator was reading a truncated
window. `run()` now returns the full output and callers truncate for display.
That is the third time in this step that the probe, not the subject, was the
broken thing.

## D. Two real defects this capture found -- both caused by 86.34's own fix

Neither was in the step text. Both are recorded because the matrix found them,
not because I predicted them.

**D1 -- M4's anchor went stale.** `mutation_matrix_86_24.py` cell M4 bound the
literal `env = {**os.environ, "TZ": "Pacific/Midway"}`. 86.34 replaced that with
`_date_shifting_tz()`, so the anchor matched **0 times**:

```
  M4    test_phase_86_24_clock_dependence.py   *** NO ***      (anchor did not bind)
```

The harness fails loudly on this (`if n != 1: ... survived.append(mid)`), so it
surfaced as a survivor rather than a silent pass -- the anchor-uniqueness check
in that file's own docstring doing its job. **Re-anchored to the new expression;
the cell's intent is unchanged.**

**D2 -- M1 SURVIVED, for exactly this step's reason.** M1 hardcoded
`tz="Pacific/Midway"`. At 19:23 UTC Midway does not shift the date, so the
mutant ("revert the macro tests to the LOCAL clock domain") is behaviourally
identical to the original:

```
  M1   SURVIVED  control rc=0 mutant rc=0   revert the macro tests to the LOCAL clock domain
```

It was KILLED earlier the same day at ~10:5x UTC, inside Midway's window. **The
cell was reporting the wall clock, not the guard** -- the same disease as the
fixture 86.34 was filed to repair, one layer up in the tooling. Fixed by giving
the matrix its own `_date_shifting_tz()` and setting `tz=` from it.

**After both repairs:**

```
$ python scripts/qa/mutation_matrix_86_24.py
M1 KILLED   M2 KILLED   M6 KILLED   M7 KILLED   M3 KILLED   M4 KILLED   M5 KILLED
tracked sources UNCHANGED: True  [('test_phase_82_0_macro_ingestion.py', '566a607e91365c67'),
   ('test_phase_86_2_replay_poison_row.py', 'fb97b52ecf7fb5be'),
   ('test_phase_86_24_clock_dependence.py', '9b5cb2e44e6ba8a4')]
stray mutant files left behind: none
All 7 mutants killed.
```

The clock-dependence module's digest is `9b5cb2e44e6ba8a4` in this final run
(`55e24bb26a93f131` in the earlier run in this same capture) because the
`PYFINAGENT_86_34_SWEEP_ROOT` seam was added between them. Both runs are
reported rather than only the last, so the change is visible instead of
silently overwritten.

## E. N3 -- the regenerated digest (criterion 4)

Producing command, stated next to the number so it is reproducible:

```
$ python -c "import hashlib;print(hashlib.sha256(open('backend/tests/test_phase_86_2_replay_poison_row.py','rb').read()).hexdigest()[:16])"
fb97b52ecf7fb5be
```

Independently corroborated by the matrix run in section D, which digests the
same file and prints `fb97b52ecf7fb5be`. The stale value was
`5c1ce1116769d118`. **Regenerated by re-running the producer, not by editing the
number in place.**

## F. 86.24 is NOT re-opened (criterion 6)

```
  HEAD     sha256=ac991bbed30c9c73 status=done
  worktree sha256=ac991bbed30c9c73 status=done
  byte-identical: True   status still done: True
```

`.claude/masterplan.json`'s 86.24 `verification` block is byte-identical to
`HEAD` and its status is unchanged.

## G. The immutable verification command

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_phase_86_24_clock_dependence.py -q'
..........                                                               [100%]
10 passed in 8.30s
exit=0

  ambient at run time: UTC 2026-08-10 19:20 / local 21:20 CEST
```

**19:20 UTC is outside Midway's 00:00-10:59 window** -- precisely the 13 hours in
which the pre-86.34 hardcoded fixture was RED. The suite passing here is the
demonstration, not a coincidence.

## H. What this capture does NOT establish

- **The Q/A has not run at capture time.** No verdict is claimed here.
- **`.venv.py313.bak` is a fact about this machine.** A fresh clone has 2
  conftests under both rules, so the 34-vs-2 delta is not reproducible
  elsewhere. The *property* asserted (no `.venv*` element among swept paths) is.
- **The two other live sites sharing the `.venv` scoping bug** that the research
  gate found are still unfixed; out of scope here and not silently absorbed.
- **D1 and D2 are changes to a CLOSED step's tooling** (`mutation_matrix_86_24.py`).
  They repair breakage that 86.34's own edit caused and do not touch 86.24's
  verdict, its immutable criteria, or its status -- proven in section F.
- **Full-default `ruff` is not clean here and never was.** The project has no
  `[tool.ruff]` section; the step's own convention is the `F821,F401,F811`
  selection, which passes on all three touched files. Under ruff's *default*
  rules the UNTOUCHED `mutation_matrix_86_24.py` at `HEAD~1` already reports 10
  findings and the new matrix reports 6 -- so this is a pre-existing repo-wide
  style gap, not a regression introduced here. Stated rather than presented as
  "ruff passed".
- **The seam is a new surface.** `PYFINAGENT_86_34_SWEEP_ROOT` lets any caller
  point the conftest sweep at another tree. It defaults to the real `REPO`, is
  read inside one test, and mirrors an existing accepted seam in the same file --
  but it is an escape hatch and a future reader should know it exists.
