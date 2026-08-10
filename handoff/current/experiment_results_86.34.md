# Experiment results -- step 86.34

**Step**: `86.34` (phase-86, P3) | **Phase**: GENERATE | **Date**: 2026-08-10

> **UPDATED 2026-08-10 22:0x CEST, cycle 2/3.** The Q/A (`wf_839de1e6-c3c`)
> returned **FAIL** and separately flagged THIS FILE as stale -- written 19:02,
> never refreshed after commits `1b7e4601` (21:26) and `73ce11ba` (21:31), so it
> asserted criterion 3 MET on evidence that did not yet exist. Both are fixed
> below and the rows say which cycle actually closed them. The verdict is
> recorded verbatim in `evaluator_critique_86.34.md`.

## 0. N1 was not a wording defect -- the suite was RED

The step filed N1 as "a directionally inverted claim". Measuring it turned up
something larger, and the immutable verification command **was failing when this
step started**:

```
$ python -m pytest backend/tests/test_phase_86_24_clock_dependence.py -q
E   assert '2026-08-10' != '2026-08-10'
...:261: AssertionError
FAILED ...::test_the_two_repaired_modules_PASS_AT_A_SHIFTED_CLOCK
1 failed, 9 passed
```

`:261` is a **positive control** written during 86.24 -- *"the TZ shift did not
move the local date; this test would have passed without testing anything"*. It
worked exactly as designed. The defect is one level down: **the fixture cannot
guarantee its own precondition.**

| zone | offset | shifts the date | window |
|---|---|---|---|
| `Pacific/Midway` (was hardcoded) | UTC-11 | **11/24 hours** | 00:00-10:59 UTC |
| `Pacific/Kiritimati` | UTC+14 | 14/24 | 10:00-23:59 UTC |
| both | -- | **24/24** | whole day |

General law (from the gate): for a fixed offset `o`, the shift holds on exactly
`|o|` of 24 hours. **No fixed-offset zone works all day**, so a TZ-only fixture
is structurally hour-dependent. Reproduce:
`python scripts/qa/measure_tz_fixture_coverage_86_34.py` (committed `9424939c`).

**AND I MUST STATE THE UNCOMFORTABLE PART.** 86.24 closed on a PASS earlier today
at roughly **10:5x UTC -- about five minutes inside Midway's window**. The
evaluator's `34 passed` was real and honestly obtained. The same command that
evening is red. **A step I closed today rested on a suite that is red for 13
hours out of 24.** It does not invalidate 86.24's substance (both evaluators
verified there is no live defect and `kill_switch.py` is byte-unchanged), but
anyone reading "86.24 = PASS" deserves to know the suite was not a standing gate.

## 1. Files changed

| File | Change |
|---|---|
| `backend/tests/test_phase_86_24_clock_dependence.py` | `_date_shifting_tz()` picks a zone that provably shifts the date NOW; docstring corrected; conftest sweep scoped to first-party + population asserted and printed |
| `handoff/current/live_check_86.24.md` | section F digest **regenerated**, header staleness noted |
| `docs/runbooks/per-step-protocol.md` | §4 records that contract-before-generate can be UNPROVABLE |
| `scripts/qa/measure_tz_fixture_coverage_86_34.py` | new (committed `9424939c`) |
| `scripts/qa/mutation_matrix_86_34.py` | **new, cycle 2** (`1b7e4601`, `73ce11ba`) -- carries criterion 3's cells. Omitted from this list in cycle 1, which the Q/A flagged: the file holding a criterion's whole evidence was missing from the file list. |
| `scripts/qa/mutation_matrix_86_24.py` | **cycle 2** -- M4 re-anchored, M1 made hour-independent |
| `scripts/qa/verify_86_24_direction_claim.py` | **new, cycle 3** -- the non-vacuous criterion-1 oracle |
| `handoff/current/live_check_86.24.md` | **cycle 3** -- the inverted claim CORRECTED (it survived cycle 1); section F regenerated in full |

**`.claude/masterplan.json` 86.24 verification block byte-identical: True;
status still `done`.** (criterion 6)

## 2. Criterion-by-criterion

| # | Criterion (abridged) | Evidence | Status |
|---|---|---|---|
| 1 | N1 corrected in BOTH locations, MEASURED, old sentence gone from source AND live_check | **cycle 1 claimed MET and the Q/A returned FAIL on it.** Only the test docstring was corrected; the inverted claim still stood at `live_check_86.24.md:12-13`, and the grep offered as proof was a VACUOUS ORACLE (the literal never existed in that file at any commit). Corrected in cycle 3 with `scripts/qa/verify_86_24_direction_claim.py`, which is mutation-proven to fail | MET (cycle 3) |
| 2 | sweep population DERIVED, excludes `.venv*`, count asserted non-zero AND printed | **34 kept / 32 vendored -> 2, both first-party**; `assert swept` + print | MET |
| 3 | the N2 guard gets its mutation cell; reverting the rule fires a NAMED assertion | **cycle 1 claimed MET on 3 cells and that was an OVERCLAIM** -- criterion 3 has TWO halves and the poisoned-conftest-into-a-fake-repo half was not done. Closed in cycle 2 with a 4th cell + the `PYFINAGENT_86_34_SWEEP_ROOT` seam; **4/4 KILLED** | MET (cycle 2) |
| 4 | N3 fixed by REGENERATING, not editing the number; command stated | digest regenerated to `fb97b52ecf7fb5be`, producing command inline | MET |
| 5 | the contract-before-generate blindness recorded where a future Q/A reads it | runbook §4 | MET |
| 6 | 86.24 not re-opened; its verification block byte-identical | verified above | MET |

## 3. Mutation -- and two cells SURVIVED first

| cell | first run | after the fix |
|---|---|---|
| **N1-REVERT-DYNAMIC-TZ** (hardcode Midway) | KILLED | KILLED |
| **N2-REVERT-EXCLUSION** (exact-element `.venv`) | ***SURVIVED*** | **KILLED** |
| **N2-EMPTY-POPULATION** | ***SURVIVED*** | **KILLED** |

Two different problems, and only one was a real gap:

- **N2-REVERT-EXCLUSION survived because I had asserted the wrong property.** I
  asserted the population was *non-empty*; reverting the rule re-admits 32
  vendored conftests which happen to contain no suspect token today, so the
  suite stayed green — *green by the luck of the vendored corpus*, exactly what
  the step text predicted. **Fixed** by asserting the property the rule exists to
  establish: no swept path may contain a `.venv*` element.
- **N2-EMPTY-POPULATION survived because MY PROBE was wrong.** I mutated the
  assertion away (`swept = []; assert True`) instead of breaking its subject. A
  mutation that deletes the guard tests nothing. **Redone** by making
  `_first_party()` reject everything, leaving the assertion in place — it now
  fires.

**A CAVEAT ON N1-REVERT-DYNAMIC-TZ THAT I AM NOT HIDING:** that cell killed
*because it is currently 17:0x UTC, outside Midway's window*. Run the same cell
at 09:00 UTC and it would **survive** — the mutant would be green because Midway
works at that hour. **The mutation cell is itself clock-dependent**, which is the
very disease this step treats. It is honest evidence at this hour and not a
standing guarantee; a durable version needs the mutant driven at a pinned hour.
Recorded rather than presented as a clean kill.

## 4. Verbatim

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_phase_86_24_clock_dependence.py -q'
10 passed in 2.47s        exit=0          (was: 1 failed, 9 passed)

selector at 17:00 UTC -> Pacific/Kiritimati, local 2026-08-11 != UTC 2026-08-10
   (Midway would have given 2026-08-10 -> no shift)

conftest sweep: total 70 | OLD rule kept 34 (32 vendored) | NEW rule keeps 2
digest: fb97b52ecf7fb5be (regenerated; was 5c1ce1116769d118, stale via da9263d6)
ruff F821/F401/F811: All checks passed!  exit=0
```

## 5. Scope and what I cannot verify

- **The Q/A has NOT run.** It is being held until after the 20:00 CEST book
  cycle: the immutable command runs pytest over `backend/tests`, and the
  standing rule bars that near the cycle. The step is NOT claiming a verdict.
- **The N1 mutation cell's kill is hour-dependent** (above).
- **Not fixed here**: the two additional live sites sharing the `.venv` scoping
  bug that the gate found, and the `time-machine` operator ask.
- **86.24 is not re-opened.**

---

## 6. DISPOSITION -- PARKED after 2 Q/A cycles, remediated but UNGRADED

**Status: `pending`. Not closed. No verdict is claimed.**

| cycle | Q/A | verdict | outcome |
|---|---|---|---|
| 1 | `wf_839de1e6-c3c` | **FAIL** | vacuous criterion-1 oracle; claim still asserted in the second file |
| 2 | `wf_6c44bae0-a83` | **CONDITIONAL** | both blockers confirmed fixed; 3 new findings, all mine |
| 3 | -- | **not run** | all 3 findings fixed in `82e3e55d`; **nobody has graded that** |

Parked under the standing rule *"park any step that will not close after 2 Q/A
cycles, with a disposition"*, and because the day's subagent spend reached
**8,101,029 tokens across 48 workflow runs** against a hard 50%-of-weekly-Max
ceiling whose remaining headroom I cannot read. Two Q/A cycles on this step cost
**374,538 tokens**. Spending a third tonight to grade a remediation the operator
has not yet seen is the wrong trade.

**What the next session inherits, precisely:**

- Every criterion is MET as of `82e3e55d` *by my own measurement*: checker exit 0,
  suite `10 passed`, `mutation_matrix_86_34.py` 4/4 KILLED,
  `mutation_matrix_86_24.py` 7/7 KILLED, ruff clean on 4 files, 86.24's masterplan
  block `ac991bbed30c9c73` and `status: done` unchanged.
- **That sentence is exactly the kind of self-report this step exists to
  distrust.** Cycle 1 also believed its criteria were met. Run ONE fresh Q/A on
  the current tree; do not re-do the work.
- The three cycle-2 findings and their fixes are in the commit message of
  `82e3e55d` and in `evaluator_critique_86.34.md`, which carries both verdicts
  verbatim.

**The one thing worth carrying beyond this step:** the criterion-1 failure was
not a missed edit, it was a *measurement instrument that could not produce the
finding it was searching for*, published as proof. Three sibling instances landed
the same day (a `fail|error` grep against code that logs `subprocess timeout`, a
`cc_rail:%` pattern needing absent punctuation, a discriminator reading a
truncated tail). `scripts/qa/verify_86_24_direction_claim.py` is the shape of the
answer: scope the property, carry positive controls that must fire, and prove the
guard fails on a real broken subject rather than on a synthetic one.
