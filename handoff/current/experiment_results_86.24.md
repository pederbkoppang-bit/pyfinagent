# phase-86.24 -- GENERATE

**Step:** 86.24 (P2) -- the test suite changes colour with the wall clock.
**Contract:** `handoff/current/contract_86.24.md`
**Research:** `handoff/current/research_brief_86.24.md` (gate PASSED, recomputed;
14 sources, 44 URLs, audit-class dry after 8 rounds)
**Code commit:** `d5180e27`. **Measurement tree:** `70e646b7` (the changelog
commit on top; the four touched files are byte-identical between them).

---

## 1. The adjudication, which is what the step actually turns on

The step required the kill-switch `sod_date` case to be **adjudicated rather
than patched**, because "the daily safety anchor goes stale at midnight" would
be a genuine defect in a long-running backend.

**Verdict: production is CORRECT BY DESIGN. There is no production staleness
defect. The TEST hard-coded an ageing date.** Four measured supports:

| | evidence |
|---|---|
| the rule is **per-LEG** | `evaluate_breach` returns `any_breached: True` with `armed: False`. The daily leg disarms; the date-independent **trailing** leg keeps firing. Now asserted every day by `test_a_YESTERDAY_anchor_DISARMS_the_daily_leg_but_the_trailing_leg_still_fires`. |
| it was installed against a **measured live incident** | `kill_switch.py:857-861`: on 2026-07-26 the badge served `sod_date=2026-07-24` with `armed: true`, and a **two-day** move was being reported as a same-day loss -- losing same-day coverage AND biasing toward a spurious flatten at once. |
| the order gate does not read `armed` | it reads `baselines_present` (`kill_switch.py:868-881`), so a disarmed daily leg does not gate trading. |
| the fire path was separately proven | phase-86.12: `mark_to_market` (Step 5) precedes enforcement (Step 5.5) in the same cycle, so the leg fires on a drawdown present at cycle time. |

**So the assertion was not weakened.** `test_c1_c2_a_poison_row_first_...` still
asserts `daily_loss_breached is True`; only the fixture's day moved.

## 2. The finding I did not go in expecting: THREE tests, TWO mechanisms

The two macro tests are **not** the same bug:

> production resolves in **UTC** (`data_ingestion.py:344` and the vintage stamp
> at `:375`, both `datetime.now(timezone.utc).date()`); the tests asserted
> against **local** `date.today()`.

On CEST those disagree for exactly **00:00-02:00 nightly**, then agree again --
which is why they "healed themselves 45 minutes later". A **timezone-domain
mismatch**, not a time bomb. The three tests' flip instants are **two hours
apart**, and that gap is itself the evidence that one explanation could not have
covered all three. Treating the kill-switch diagnosis as the answer for all
three would have been this project's recurring error in a new costume.

## 3. Criterion 1 -- the derivation, and why the obvious method is REJECTED

Measured by the research over **457 test files**:

| method | size | catches macro | catches poison-row | verdict |
|---|---|---|---|---|
| A -- own-clock AST scan | 49 files | YES | **NO** | **REJECTED** |
| B -- date-literal regex `\b20\d{2}-\d{2}-\d{2}\b` | 129 | YES | YES | recall OK, and the trailing `\b` **misses 40 files** whose only literals are ISO *datetimes* |
| C -- literal AND an AST import of one of the 124 clock-reading production modules | 90 | YES | YES | best static CANDIDATE set |

**A fails structurally, not by tuning.** `test_phase_86_2_replay_poison_row.py`
contains **zero** clock calls -- the clock read is in production. Criterion 1
says a method that misses a known positive is **rejected, not adjusted**, so A is
out. And **no static method finds the timezone-domain class at all**: both sides
call the clock and nothing in the syntax says which is wrong.

**The recall-validatable method is a DIFFERENTIAL RUN at a shifted clock**, whose
recall is observable rather than argued. Validated BEFORE use:

```
TZ=Europe/Oslo        local 2026-08-10 == UTC   ->  1 of 3 known positives red
TZ=Pacific/Kiritimati local 2026-08-10 == UTC   ->  1 of 3
TZ=Pacific/Midway     local 2026-08-09 != UTC   ->  3 of 3   <-- ALL THREE
```

Prior art is industrial, not invented here: Debian reproducible-builds varies the
date as one of ~22 systematic axes ("398 days difference"; "future builds run 6h
and 23min ahead"), and ChaosAPI (OOPSLA 2026) is the academic form, measured to
find more flaky tests than rerunning. **Rerunning provably cannot find this
class** -- it is deterministic given the clock.

## 4. Criterion 2 -- every member, classified

Population = (tests red in the base run) ∪ (tests whose verdict changes under the
shift). Full-suite measurement, pre-fix:

```
base    16 failed / 3351 passed
shifted 19 failed / 3348 passed      delta = 3
```

| member | class | evidence |
|---|---|---|
| `test_phase_82_0_...::test_macro_end_date_is_severed_from_backtest_end_date` | **(a) test artifact** -- timezone-domain mismatch | asserted `date.today()` (local) against a production value resolved in UTC |
| `test_phase_82_0_...::test_ingested_rows_carry_a_vintage` | **(a) test artifact** -- same | same, on the `realtime_start` vintage stamp |
| `test_phase_86_2_...::test_c1_c2_a_poison_row_first_no_longer_strands_the_replay` | **(a) test artifact** -- pinned fixture date judged against now | production rule ADJUDICATED CORRECT in §1 |
| `test_phase_86_27_...::test_a_spelling_absent_from_the_entire_REPO_is_still_refused` | **NOT clock-dependent** -- a false positive of the differential, diagnosed | see §6 |

**No member is class (b) -- real production staleness.** The one candidate for
(b) was the kill-switch anchor, and §1 is the demonstration that it cannot occur
as a defect in the running backend.

Also classified, and NOT changed: the other three `date.today()` uses in
`test_phase_82_0_macro_ingestion.py` (`:215`, `:241`, `:275`) build fixture rows
at offsets of 1, 2, 60, 65, 70, 200 and 3000 days against SLA bounds of 5 and 225
days. A one-day local/UTC discrepancy cannot flip any of them, and they are
**empirically inert**: they passed under the shifted clock. Classified by
measurement, not by reading the margins.

## 5. Criteria 3-6 -- results

| criterion | result |
|---|---|
| 3 kill-switch adjudicated, assertion not weakened | §1; `daily_loss_breached is True` unchanged, and the STALE path GAINED coverage |
| 4 both modules pass post-midnight AND mid-day | `24 passed` under the system clock and under `TZ=Pacific/Midway` (the local!=UTC window); asserted in-suite by `test_the_two_repaired_modules_PASS_AT_A_SHIFTED_CLOCK`, which carries a positive control proving the shift took effect |
| 5 no global time-freezing fixture | none introduced; `test_no_global_time_freezing_fixture_is_introduced` sweeps EVERY conftest in the repo for `freeze_time`/`freezegun`/`time_machine`/`libfaketime`/`travel(` |
| 6 mutation, each fix reverted individually | **5/5 killed**, tracked sources digested unchanged, zero stray files |

```
M1 KILLED  revert the macro tests to the LOCAL clock domain
M2 KILLED  re-pin the poison-row fixture to the day it was written
M3 KILLED  give the STALE-anchor test a FRESH anchor -- does it discriminate?
M4 KILLED  remove the clock shift from the differential test
M5 KILLED  point the how-stale sweep at a FRESH anchor
tracked sources UNCHANGED: True    stray mutant files: none
```

**The immutable verification command went from `1 failed, 23 passed` to
`24 passed`.**

**The population is now empty, measured on a frozen tree (`70e646b7`):**

```
BASE    (local == UTC)  15 failed, 3360 passed, 12 skipped, 5 xfailed, 1 xpassed  375.86s
SHIFTED (local != UTC)  15 failed, 3360 passed, 12 skipped, 5 xfailed, 1 xpassed  368.58s
DELTA   EMPTY -- no test changes verdict with the clock
```

The base count moved 16 -> 15 and passes 3351 -> 3360; that reconciles exactly:
the poison-row test flipped red->green (-1 failure, +1 pass) and this step added
8 tests.

## 6. My own phase-86.27 test, surfaced by THIS step's method

The pre-fix differential flagged
`test_phase_86_27_live_origin_class.py::test_a_spelling_absent_from_the_entire_REPO_is_still_refused`.

**Diagnosed: not clock-dependent at all.** It went red because phase-86.27's Q/A
independently derived the same three novel spellings, put them in its verdict,
and Main transcribed that verdict verbatim into `evaluator_critique_86.27.md` --
so all three candidates became "present in the repo" and the `>= 3` floor could
never be met again. **The test was self-defeating by construction: any honest
audit trail that named the spellings killed it.**

Fixed in `ebeb03da` by drawing from an **unbounded** family (`inet_aton` accepts
arbitrary leading zeros; measured 16 of 16 widths resolve to the same address),
which keeps the absence requirement at its strictest -- absent from the whole
tracked tree, records included -- rather than relaxing it. **That fix has not
been graded by a Q/A**, and it is reported here rather than left to look like a
clean green.

## 7. Not claimed

- **The population is proven empty only along the axis the method covers.** A TZ
  shift moves `date.today()`/`datetime.now()` (local) and does **NOT** move
  `datetime.now(timezone.utc).date()`. The "pinned fixture date ages past UTC
  today" axis needs a real clock offset (`+1 day`), which on macOS requires
  `time-machine` -- **not installed, and I did not add a dependency
  unilaterally.** Operator ask. The static Method-C set (90 files) is the interim
  cover and is a **candidate set, not a population**.
- **`.json`/`.jsonl`/`.csv`/`.sql` fixture dates were not swept.** The static
  work covered `*.py` only. Stated as a gap.
- **`kill_switch.py` is byte-unchanged.** The adjudication says production is
  correct; a P2 test-hygiene step does not edit a live safety module.
- **The 15 remaining failures are not clock-dependent** -- identical sets under
  both clocks -- and belong to their own steps.
- **`handoff/kill_switch_audit.jsonl` byte-identical throughout**
  (`ea78508bee73887c...`, 64 lines); the new module redirects `_AUDIT_PATH` to
  `tmp_path` and never touches the operator's journal.
