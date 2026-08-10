# Contract -- phase-86.24

**Step:** 86.24 (P2) -- the test suite changes colour with the wall clock, so it
cannot be a gate.
**Date:** 2026-08-10
**Research:** `handoff/current/research_brief_86.24.md` (43,147 chars)

---

## 1. Research gate -- PASSED

| | |
|---|---|
| launch | Workflow rail, `.claude/workflows/research-gate.js`, run `wf_fd810665-56e` |
| tier | `moderate` · audit-class YES (dry after 8 rounds, K=2) |
| sources read in full | **14** (floor 5) · snippet-only 30 |
| URLs collected | **44** (floor 10) |
| recency scan | performed |
| `gate_passed` | **true**, RECOMPUTED by the script; all 14 claimed URLs found in the brief |

## 2. THE ADJUDICATION -- and it is the whole step

The step says the kill-switch case must be **adjudicated, not patched**, because
"the daily safety anchor goes stale at midnight" would be a real defect in a
long-running backend.

**Verdict: the production behaviour is CORRECT BY DESIGN. There is no
production staleness defect. The TEST hard-codes an ageing date.**

The evidence, measured rather than argued:

- **The rule is PER-LEG.** `kill_switch.py:865-867` computes
  `daily_baseline_stale` and disarms only the **daily** leg. Measured:
  `evaluate_breach` returns `any_breached: True` with `armed: False` -- the
  date-independent **trailing** leg still fires. The book is not uncovered.
- **It was installed against a MEASURED live incident** (`kill_switch.py:857-861`):
  on 2026-07-26 the badge served `sod_date=2026-07-24` with `armed: true`, and a
  **two-day** move was being reported as a same-day loss -- losing same-day
  coverage and biasing toward a spurious flatten at the same time.
- **The order gate does not read `armed`.** It reads `baselines_present`
  (`kill_switch.py:868-881`), so the disarmed daily leg does not gate trading.
- **phase-85.6 gave it an out-of-band exit** (`paper_trader.py:1220-1300`).
- **phase-86.12** separately established that the daily leg fires on a drawdown
  present at cycle time, because `mark_to_market` (Step 5) precedes enforcement
  (Step 5.5) in the same cycle.

**So the assertion is not to be weakened.** `test_c1_c2_a_poison_row_first_...`
asserts `daily_loss_breached is True`, which requires an ARMED daily leg, which
requires `sod_date == today (UTC)`. Its fixture pins `2026-08-09`, and it passed
on the day it was written. The fixture must become relative to the clock; the
assertion stays exactly as it is.

## 3. THE SECOND FINDING -- three tests, TWO mechanisms, and I did not know this

The two macro tests are **not** the same bug as the kill-switch one:

> production reads **UTC** (`data_ingestion.py:344, :375`); the tests assert
> against **local** `date.today()`. On CEST the two disagree for exactly the
> window **00:00-02:00 nightly**, then agree again.

That is why they "healed themselves 45 minutes later" -- not a time bomb, a
**timezone-domain mismatch** with a recurring two-hour window. The three tests
have flip instants **two hours apart**, which is itself the evidence that one
mechanism could not explain all three.

**No static method can find this class**: both sides call the clock, the literals
are irrelevant, and nothing in the syntax says which side is wrong. Only
executing both reveals it.

## 4. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. The date-dependence population is DERIVED by a stated, re-runnable method, and that method's recall is validated against the three named tests BEFORE the report is used -- a method that misses any of them is rejected, not adjusted.
2. For EACH member, state which case it is: (a) test artifact or (b) real production staleness. Show the evidence for the classification; an unclassified member is an open finding, not a pass.
3. The kill_switch `sod_date` anchor is specifically adjudicated: either demonstrate the staleness cannot occur in the running backend, or file it as a live defect. Do not weaken the assertion.
4. Both named modules pass at a simulated post-midnight boundary AND at a normal mid-day time; show both runs.
5. No global time-freezing fixture is introduced.
6. Mutation-test every new guard, including reverting each fix individually; a guard whose mutant survives does not count.

**Verification command (immutable):**
```
bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_0_macro_ingestion.py backend/tests/test_phase_86_2_replay_poison_row.py -q'
```
Current state on this tree: **1 failed, 23 passed** (the kill-switch case).

## 5. Plan

**S1 -- the derivation, and why the obvious method is REJECTED.** The research
measured four candidates over 457 test files. **Method A (own-clock AST scan,
49 files) MISSES known positive #3** -- `test_phase_86_2_replay_poison_row.py`
contains *zero* clock calls, because the clock read is in production
(`kill_switch.py:986`). Per criterion 1 a method that misses a known positive is
**rejected, not adjusted**, so A is out. Method C (date literal AND an AST import
of one of the 124 clock-reading production modules; **90 files**) catches both
files and is the best static CANDIDATE set -- but static analysis cannot prove
its own completeness.

**The recall-validatable method is a DIFFERENTIAL RUN at a shifted clock**, and
its recall is directly observable rather than argued. Prior art is real and
industrial: Debian reproducible-builds varies the date as one of ~22 systematic
axes ("398 days difference"; "future builds run 6h and 23min ahead"), and
ChaosAPI (OOPSLA 2026) is the academic form, measured to find more flaky tests
than rerunning. **Rerunning provably cannot find this class.**

**MEASURED ALREADY, before writing this contract** (recall validation first, per
criterion 1):

```
TZ=Europe/Oslo       local 2026-08-10 == UTC  ->  1 of 3 known positives red
TZ=Pacific/Kiritimati local 2026-08-10 == UTC  ->  1 of 3
TZ=Pacific/Midway    local 2026-08-09 != UTC  ->  3 of 3   <-- ALL THREE
full suite, base     16 failed / 3351 passed
full suite, Midway   19 failed / 3348 passed
```
The population is `(base failures) ∪ (differential delta)`. The delta is
**3 tests**: the two macro tests, and -- unexpectedly -- one of my own from
phase-86.27, which is a genuine catch and is dealt with in S5.

**S2 -- state the blind spot rather than let it be discovered.** A TZ shift moves
`date.today()`/`datetime.now()` (local) and does **NOT** move
`datetime.now(timezone.utc).date()`. So it covers the timezone-domain class
completely and does **not** cover the "pinned fixture date ages past UTC today"
class for a test that is currently green. Covering that axis needs a real clock
offset (`+1 day`), which on macOS needs `time-machine` -- **not installed, and I
will not add a dependency unilaterally**. Raised as a numbered operator ask; the
static Method-C candidate set (90 files) is reported as the interim cover, and
labelled a candidate set, not a population.

**S3 -- fix the kill-switch test without touching the assertion.** Derive the
fixture's timestamps and `date=` from **today's UTC date**, so the test
exercises the ARMED path deterministically on any day. Add a **sibling test that
pins an explicitly-past date and asserts the DISARMED path** -- so the staleness
rule that 36.9 installed gains the coverage it currently lacks. Net: one test
becomes clock-independent, and the production rule gains a test. Nothing is
relaxed.

**S4 -- fix the two macro tests at the domain mismatch.** The assertion must
compare **like with like**: production resolves the macro end date in UTC, so
the test must assert against the UTC date, not `date.today()`. That is a
correction of the test's *question*, not of its strictness.

**S5 -- my own phase-86.27 test, caught by this step's own method.** The
differential surfaced `test_a_spelling_absent_from_the_entire_REPO_is_still_refused`
as red. Diagnosed: **not clock-dependent at all** -- it went red because the Q/A
independently derived the same three spellings, and Main transcribed the verdict
verbatim into `evaluator_critique_86.27.md`, so all three candidates became
"present in the repo". Already fixed in `ebeb03da` by drawing from an
**unbounded** family (arbitrary leading zeros; measured 16 of 16 widths resolve),
which keeps the absence requirement at its strictest. **That fix has not been
Q/A-graded**, and this step's artifacts will say so rather than let the green
imply otherwise. It is reported here because this step's method found it.

**S6 -- criterion 5 is satisfied by construction, and asserted.** No autouse
fixture, no global freeze. The differential is an explicitly-invoked
`TZ=... pytest` run, not a fixture; a test will assert that no autouse
time-freezing fixture exists in any conftest.

**S7 -- criterion 6.** Mutation-test each fix individually: revert the relative
fixture date (the kill-switch test must go red on any day that is not the day it
was written), revert the UTC comparison in the macro tests (must go red under the
shifted TZ), and confirm each mutant is killed by the assertion named.

## 6. Explicitly NOT in scope

- **Installing `time-machine`, `freezegun` or `libfaketime`.** All three are
  absent; `libfaketime` is Unix-`LD_PRELOAD`-only and does not fit macOS anyway.
  A dependency addition is an operator decision.
- **Any change to `kill_switch.py`.** The adjudication says production is
  correct; a P2 test-hygiene step must not edit a live safety module.
- **The 13 other pre-existing failures.** They are not clock-dependent (measured:
  they fail identically in both runs) and belong to their own steps.
- **`.json`/`.jsonl`/`.csv`/`.sql` fixture dates.** The static sweep covered
  `*.py` only; stated as a gap, not silently omitted.

## 7. References

- `handoff/current/research_brief_86.24.md` -- 14 sources incl. time-machine vs
  freezegun vs libfaketime, Debian reproducible-builds' date-variation axis,
  ChaosAPI (OOPSLA 2026), Google's flaky-test post, Fowler on non-determinism.
- `handoff/current/captures_86.24/base_run_failure_set.txt` -- the 16-member base
  set and the recall validation.
- `handoff/current/captures_86.24/tz_midway_full_suite.txt` -- the 19-member
  shifted-clock set.
