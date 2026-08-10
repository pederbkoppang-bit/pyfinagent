# Contract -- step 86.34

**Step**: `86.34` (phase-86, P3, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-10 | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written BEFORE any code.** `git diff` on the target files is empty at this
moment; the mtime ordering is the evidence.

---

## 1. Research gate

**PASSED** -- `wf_41a60c51-cc7`, tier `simple`, brief
`handoff/current/research_brief_86.34.md` (33,369 chars). Enforced: **8 sources
read in full** (floor 5), **32 URLs** (floor 10), recency scan performed, all 8
claimed URLs present in the brief, `urls_collected_corroborated: 32 <= 32`,
`brief_status_in_brief: COMPLETE`.

*(This is also the second live run of the phase-86.37 born-inert marker, and it
behaved: `rail_dropped: null`, marker read and reported.)*

### The gate escalated finding (a), and it is now the headline

**N1 IS NOT A WORDING DEFECT. THE SUITE IS RED RIGHT NOW.**

```
$ python -m pytest backend/tests/test_phase_86_24_clock_dependence.py -q
E   assert '2026-08-10' != '2026-08-10'
backend/tests/test_phase_86_24_clock_dependence.py:261: AssertionError
FAILED ...::test_the_two_repaired_modules_PASS_AT_A_SHIFTED_CLOCK
1 failed, 9 passed
```

The assertion at `:261` is a **positive control I wrote this morning** -- *"the
TZ shift did not move the local date; this test would have passed without
testing anything"*. It is working exactly as designed. The defect is one level
down: **the fixture cannot guarantee its own precondition.**

Measured independently by me
(`scripts/qa/measure_tz_fixture_coverage_86_34.py`, committed `9424939c`) and
confirmed by the gate:

| zone | offset | hours it shifts the date | window |
|---|---|---|---|
| `Pacific/Midway` (in use) | UTC-11 | **11 / 24** | 00:00-10:59 UTC |
| `Pacific/Kiritimati` | UTC+14 | 14 / 24 | 10:00-23:59 UTC |
| both together | -- | **24 / 24** | the whole day |

The gate states the general law: **for a fixed offset `o`, a date shift holds on
exactly `|o|` of 24 hours.** No fixed-offset zone gives a constant non-zero
delta, so a TZ-only fixture is *structurally* hour-dependent.

**AND THE TIMING IS UNCOMFORTABLE, SO I AM STATING IT PLAINLY.** 86.24 closed on
a PASS earlier today at roughly 10:5x UTC -- inside Midway's 00:00-10:59 window,
with about five minutes to spare. The evaluator's `34 passed` was real and
honestly obtained; the same command hours later is red. **A step I closed today
rests on a suite that is red for 13 hours out of 24.** That does not invalidate
86.24's substance -- both evaluators verified there is no live defect and
`kill_switch.py` is byte-unchanged -- but the suite is not usable as a standing
gate until this is fixed, and anyone reading `86.24 = PASS` deserves to know
that.

### Findings (b) and (c), both confirmed and (b) is worse than recorded

**(b)** The `".venv" in cf.parts` exact-element filter admits `.venv.py313.bak`.
Measured: **32 of 34** swept conftests are vendored (94%); the gate widened it
to **22,131 of 23,183 `.py` files repo-wide (95.5%)**. Git already knows
(`.gitignore:16` = `.venv*/`), `git ls-files` reports the true first-party
population as **2**. The gate found **two more live sites sharing the bug**, and
an in-repo fix pattern at `lint_limits_usage.py:79`. **No mutation cell covers
this guard.**

**(c)** `live_check_86.24.md:156` records `5c1ce1116769d118`; the file now
hashes `fb97b52ecf7fb5be`. Stale via a **legitimate** commit (`da9263d6`), and
the header's commit/tree fields at `:3` are two commits behind.

## 2. Hypothesis

A fixture that can silently fail to establish its precondition is the same
vacuity class as a guard that cannot fail -- except this one fails *loudly*,
which is better, and turns the suite red for a reason unrelated to any code
change, which is worse. The fix is to make the fixture **choose a zone that
provably shifts the date at the moment of the run**, and keep the positive
control so the choice is verified rather than trusted.

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. N1 is corrected in BOTH locations and the correction is MEASURED, not reasoned: show, with zoneinfo output, the local-vs-UTC date at 00:30 and 01:30 CEST (local AHEAD) and under TZ=Pacific/Midway (local BEHIND), and state which direction the test actually simulates. A grep proving the old sentence is gone from live source AND from handoff/current/live_check_86.24.md is required -- a claim withdrawn in prose while surviving in source is the phase-86.31 failure repeated.
2. N2's swept population is DERIVED and REPORTED as a number by the guard itself: after the fix the sweep must exclude every path element matching `.venv*` and node_modules, and the test must assert the swept count is non-zero AND print it. Show the before/after counts on this machine (measured before the fix: 70 total, 34 kept, 32 of them vendored under .venv.py313.bak, 2 project files).
3. N2's guard gets the mutation cell it never had, in scripts/qa/mutation_matrix_86_24.py or a successor: inject a conftest containing a global time-freezing fixture into a fake repo root and require the guard to go RED; and mutate the exclusion rule itself (revert to the exact-element match) and require a NAMED assertion to fire. A guard that has not been observed failing does not count.
4. N3 is fixed by REGENERATING the affected capture block, never by editing the number in place: re-run the producing command, paste its output, and update the file header's commit/tree fields to the tree actually measured. State the command next to the digest so a reader can reproduce it.
5. The contract-before-generate blindness is recorded permanently somewhere a future Q/A will read (the runbook section 4 harness-compliance audit is the natural home): when a step's contract and its generated artifacts land in ONE commit, the mtime/commit ordering check is UNPROVABLE and must be reported as such, never as a green tick.
6. 86.24's verdict is NOT re-opened and its immutable criteria are NOT touched: prove with a diff that `.claude/masterplan.json`'s 86.24 verification block is byte-identical and its status remains done.

**Verification command** (immutable):
`bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_phase_86_24_clock_dependence.py -q'`

**Note the command is currently RED.** That is the point: criterion 1's fix must
make it green *at any hour*, not at a lucky one.

## 4. Plan

**P1 -- make the fixture choose a zone that provably works NOW.** Select
`Pacific/Midway` or `Pacific/Kiritimati` by whichever actually shifts the date at
the current UTC hour (they cover 24/24 between them). **Keep the `:261` positive
control unchanged** -- it is what proves the choice took, and removing it to make
the suite green would be the exact anti-pattern this project keeps catching.

**P2 -- correct the direction claim** at `test_...:235-237` and
`live_check_86.24.md:8-10`, and say which direction is being simulated *and why
the zone is now chosen dynamically*.

**P3 -- scope the conftest sweep to first-party files.** Exclude any path element
matching `.venv*`; assert the swept count is non-zero and PRINT it. Prefer
`git ls-files` as the authority where practical (the gate's recommendation, and
`lint_limits_usage.py:79` is the in-repo precedent).

**P4 -- mutation cells** for the N2 guard: a poisoned conftest in a fake repo
root must turn it red, and reverting the exclusion rule must fire a NAMED
assertion.

**P5 -- regenerate** `live_check_86.24.md` section F rather than editing the
number, and update the stale header fields.

**P6 -- record the contract-before-generate blindness** in
`docs/runbooks/per-step-protocol.md` §4.

### Explicitly NOT doing

- **Not** deleting or weakening the `:261` positive control to get green.
- **Not** re-opening 86.24's verdict or touching its immutable criteria.
- **Not** introducing a global time-freezing fixture (86.24 criterion 5 forbids
  it, and this step must not violate a criterion of the step it is repairing).
- **Not** adding `time-machine` -- it is an operator ask, still open.

## 5. References

- `handoff/current/research_brief_86.34.md` (gate PASSED, `wf_41a60c51-cc7`)
- `scripts/qa/measure_tz_fixture_coverage_86_34.py` (committed `9424939c`)
- IANA tz theory; pytest `norecursedirs`; ruff `respect-gitignore`;
  `git ls-files`; SLSA provenance; GitLab date/time FE guide
