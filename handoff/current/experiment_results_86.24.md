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
defect. The TEST hard-coded an ageing date.**

**The verdict is unchanged from cycle 1; the REASONING is not.** Cycle 1 led with
a support that the Q/A measured to be false in a band, and it is struck through
below rather than quietly removed -- a withdrawn claim is more useful to the next
reader than a tidy table. Four supports stand, and the first is the decisive one:

| | evidence |
|---|---|
| **the enforcement path never sees a stale anchor** -- THE decisive support | `paper_trader.check_and_enforce_kill_switch` re-anchors at `:1413` (`if sod_anchor_needs_reroll(snap, today)`) **before** `evaluate_breach` at `:1460`, and the flatten branch at `:1468` keys on `breach["any_breached"]`, never on `armed`. So the code that decides whether to flatten is never handed a stale anchor at all. |
| ~~the rule is per-LEG, so the trailing leg keeps firing~~ | **WITHDRAWN IN CYCLE 2 -- THIS WAS FALSE IN A BAND, and it was the headline support.** Measured: a STALE anchor with `sod=100, peak=100` at limits 4%/10% gives `any_breached=False` at nav 95 (5% loss) and 92 (8%), while the same navs against a FRESH anchor breach. Between the two limits a stale anchor leaves **nothing** firing. The cycle-1 guard exercised only `nav=80.0` -- a 20% drop, above the trailing limit -- so it could not detect the gap it claimed to close. Found by the cycle-1 Q/A, reproduced by me, and now pinned as `test_a_stale_anchor_leaves_the_band_between_the_two_limits_UNCOVERED` with a fresh-anchor control. |
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
| 3 kill-switch adjudicated, assertion not weakened | §1 -- **the conclusion is unchanged (no live defect) but the RATIONALE was corrected in cycle 2**: it rests on the roll-before-evaluate ordering, not on the trailing leg. `daily_loss_breached is True` byte-unchanged, and the STALE path GAINED coverage including the uncovered band |
| 4 both modules pass post-midnight AND mid-day | `24 passed` under the system clock and under `TZ=Pacific/Midway` (the local!=UTC window); asserted in-suite by `test_the_two_repaired_modules_PASS_AT_A_SHIFTED_CLOCK`, which carries a positive control proving the shift took effect |
| 5 no global time-freezing fixture | none introduced; `test_no_global_time_freezing_fixture_is_introduced` sweeps EVERY conftest in the repo for `freeze_time`/`freezegun`/`time_machine`/`libfaketime`/`travel(` |
| 6 mutation, each fix reverted individually | **7/7 killed**, tracked sources digested unchanged, zero stray files |

```
M1 KILLED  revert the macro tests to the LOCAL clock domain
M2 KILLED  re-pin the poison-row fixture to the day it was written
M6 KILLED  SNAPSHOT the fixture date at import instead of recomputing per call
M7 KILLED  point the band test OUTSIDE the band -- does it discriminate?
M3 KILLED  give the STALE-anchor test a FRESH anchor -- does it discriminate?
M4 KILLED  remove the clock shift from the differential test
M5 KILLED  point the how-stale sweep at a FRESH anchor
tracked sources UNCHANGED: True    stray mutant files: none
```

M6 and M7 are cycle-2 additions, one per Q/A finding. **M6 needed a structural
change to the harness**: it mutates the poison-row module but must RUN the module
that carries the killing assertion, so a cell can now name a different `run`
target and pass the mutant's path through an env seam. Its control runs that same
target unmutated, so a cell can never score a kill against an already-red file.

**The immutable verification command went from `1 failed, 23 passed` to
`24 passed`.**

**The population is empty along the covered axis, re-measured on the CYCLE-2
frozen tree (`7eb85983`):**

```
BASE    (local == UTC)  15 failed, 3362 passed, 12 skipped, 5 xfailed, 1 xpassed  376.94s
SHIFTED (local != UTC)  15 failed, 3362 passed, 12 skipped, 5 xfailed, 1 xpassed  373.19s
DELTA   EMPTY -- no test changes verdict with the clock
```

Cycle 1 measured the same shape at `70e646b7` (15 / 3360 both ways). The counts
reconcile end to end: pre-fix base was 16 failed / 3351 passed; the poison-row
test flipped red->green (-1 failed, +1 passed), cycle 1 added 8 tests (3360) and
cycle 2 added 2 more (3362) -- the uncovered-band test and the recompute
property test.

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

- **A member of the blind class was introduced BY THIS STEP and is now fixed.**
  The cycle-1 repair set `_UTC_TODAY` **once at module import** while
  `kill_switch.py:986` recomputes at call time -- the masterplan's own case (a)
  verbatim ("once-computes a date while the assertion recomputes it"). If UTC
  midnight fell between collection and execution the test would go red. The
  evidence was already in my own mutation matrix and I misread it: cell M2 pins
  that date and is scored KILLED, which is a proof of the failure mode, not of
  the guard. Found by the cycle-1 Q/A. Fixed in cycle 2 (recomputed per call),
  and the property is now asserted directly by
  `test_the_poison_row_fixture_date_is_RECOMPUTED_not_snapshotted` -- necessary
  because an ordinary run **cannot** kill a re-snapshot mutant, which only
  misfires across midnight.
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

---

# CYCLE 2 -- 2026-08-10. Both Q/A findings fixed; the second was mine, in the file this step repaired.

The cycle-1 verdict was **CONDITIONAL**, `ok:false`, with all six criteria MET on
substance and every deterministic check reproduced independently. Both violations
were rationale/disclosure defects, and both were real.

## Finding 1 -- my headline support was FALSE IN A BAND

Cycle 1 justified "a stale daily anchor is harmless" with *"the trailing leg
still fires, so the overnight window is not naked"*. **Reproduced by me, and it
does not hold between the two limits:**

```
anchor    nav   armed  stale  daily  trailing  ANY
TODAY     95.0  True   False  True   False     True
STALE     99.0  False  True   False  False     False
STALE     95.0  False  True   False  False     False    <-- 5% loss, nothing fires
STALE     92.0  False  True   False  False     False    <-- 8% loss, nothing fires
STALE     89.0  False  True   False  True      True
STALE     80.0  False  True   False  True      True
```
(measured with `_AUDIT_PATH` redirected to tmp; live journal byte-identical.)

**My guard exercised only `nav=80.0`** -- a 20% drop, above the trailing limit --
so it was structurally incapable of seeing the gap it claimed to close. That is
asserting a general property from a single point, on a money-path safety module,
with the wrong reason then written into a test docstring where a future
maintainer would rely on it.

**The conclusion survives; the reason changes.** Verified independently by
reading the enforcement path: `paper_trader.check_and_enforce_kill_switch`
re-anchors at `:1413` (`if sod_anchor_needs_reroll(snap, today)`) **before**
`evaluate_breach` at `:1460`, and the flatten branch at `:1468` keys on
`breach["any_breached"]`, never on `armed`. **The band is reachable only by a
read-only caller (the badge endpoint), never by the code that decides whether to
flatten.**

Fixed: the test is renamed to what it actually proves, the false claim is struck
through in §1 rather than deleted, and the uncomfortable measurement is pinned as
`test_a_stale_anchor_leaves_the_band_between_the_two_limits_UNCOVERED` -- with a
fresh-anchor control, so the result is attributable to staleness and not to an
inert threshold, and with a comment naming the ordering it depends on.

## Finding 2 -- I introduced a member of the class this step exists to remove

The cycle-1 repair set `_UTC_TODAY = datetime.now(timezone.utc).date()` **once at
module import**, while `kill_switch.py:986` recomputes at call time. That is the
masterplan's own case (a), verbatim: *"a fixture that hard-codes or ONCE-COMPUTES
a date while the assertion recomputes it"* -- introduced by the repairing commit,
inside the repaired file.

**The evidence was already in my own mutation matrix and I misread it.** Cell M2
pins that date to a past day and is scored **KILLED** -- which is a proof that the
module goes red whenever the import-time snapshot is a day behind evaluation
time. I read it as the guard working. The sibling module written in the same
commit does not have the shape; it recomputes inside `_day()`.

The window was minutes (collection -> execution, ~377s on a full run) rather than
24 hours, but **a narrower instance of a defect is still an instance of it**, and
it was in no artifact.

Fixed: `_day()`/`_ts()` recompute per call. And because an ordinary run **cannot**
kill a re-snapshot mutant -- it only misfires across midnight, so it is an
equivalent mutant under normal conditions -- the property is asserted directly by
`test_the_poison_row_fixture_date_is_RECOMPUTED_not_snapshotted`, which injects a
clock that advances a day between two calls. Cell **M6** now drives that
assertion and is KILLED.

## Verification on the cycle-2 tree (`7eb85983`)

| check | result |
|---|---|
| immutable command | **24 passed** |
| new module | **10 passed** (8 + the two cycle-2 additions) |
| both modules + new module under `TZ=Pacific/Midway` | **34 passed** |
| ruff F821/F401/F811 over the changed scope | exit 0 |
| mutation matrix | **7/7 killed**, tracked digests unchanged, no strays |
| full-suite differential | 15 failed / 3362 passed **both ways**, DELTA EMPTY |
| `handoff/kill_switch_audit.jsonl` | `ea78508bee73887c...`, 64 lines, byte-identical |

## What cycle 2 did NOT change

- `kill_switch.py` is still byte-unchanged. The adjudication's **conclusion** is
  unchanged -- there is no live defect -- and no assertion was weakened;
  `daily_loss_breached is True` in the poison-row test remains byte-identical.
- The blind-spot disclosure stands: a TZ shift does not move UTC, so the
  "pinned fixture ages past UTC today" axis is still uncovered and still needs
  `time-machine`, which I have not installed.
- **A new, small test seam is disclosed**: the recompute test honours
  `PYFINAGENT_86_24_PROW_PATH` so the mutation matrix can point it at a mutant
  copy. Unset in every normal run; it exists because the test reads its subject
  by path and a copy elsewhere would otherwise never be exercised.

---

# CYCLE 3 -- the cycle-2 finding fixed, and the step is PARKED (not closed)

The cycle-2 verdict was **CONDITIONAL** with ONE violation, and it was the
sharpest of the three findings across this step:

> the support I had *myself withdrawn as false* survived verbatim and unannotated
> in **live source**, at `test_phase_86_2_replay_poison_row.py:55-58` -- inside
> the very comment block cycle 2 edited two lines below -- while
> `live_check_86.24.md` §D asserted the claim "is replaced rather than softened".

**That is a completeness claim, and it did not survive a recall test.** It is the
same shape as the two findings before it and the same shape as this step's own
subject: I fixed the occurrences I was looking at and asserted the class was
closed.

## What was actually wrong with my first attempt at the fix

My first sweep for the withdrawn proposition was a **words-based grep** for
"trailing leg still fires" / "per-leg" / "still fires", and it returned **161
hits tree-wide**. Almost none were the proposition: `36.12`'s per-leg
independence, `82.11`'s "still fires", dozens of archived critiques. That is the
same proxy error as the "no pinned calendar dates" tripwire earlier in this step
-- a check that flags legitimate cases is not a check.

Two facts that narrowed it correctly:

1. **The population is the files THIS step owns or edited** -- seven of them.
   Other steps' dated artifacts are not mine to rewrite, and the cycle-2 Q/A said
   so explicitly.
2. **Prior work already had it right.** `experiment_results_85.6.md:244` says
   *"trailing leg still fires, bounding exposure to `[daily_limit,
   trailing_limit)`"* -- the bounded form, which is exactly correct and matches
   my measurement. **It was this step's cycle-1 wording that dropped the bound.**
   So the defect is mine and local, not a tree-wide myth.

## Fixed

| location | action |
|---|---|
| `test_phase_86_2_replay_poison_row.py:55-58` | **rewritten** -- states the roll-before-evaluate ordering, gives the measured band, and cross-references the band test in the other module (with a note on why the stale case cannot live in this one) |
| `contract_86.24.md` §2 | **annotated**, struck through with the measurement and the corrected support -- a dated artifact, so annotated rather than rewritten |
| `research_brief_86.24.md` `:24/:101/:143/:356/:441` | **annotated at the head** -- dated artifact; notes that the brief's own `:143` already reached the decisive support |
| `live_check_86.24.md` §D | the completeness claim is **replaced by a location table**, which is auditable in a way "replaced rather than softened" was not |

Verified after: immutable command **24 passed**; all three modules **34 passed**;
under `TZ=Pacific/Midway` **34 passed**; `handoff/kill_switch_audit.jsonl`
`ea78508bee73887c...` byte-identical.

## DISPOSITION -- PARKED, and why not a third Q/A

The operator's standing rule is **park any step that will not close after 2 Q/A
cycles, with a written disposition**. This step has had two, both CONDITIONAL.
A third Q/A would also arm the 3rd-CONDITIONAL escalation, which converts an
honest CONDITIONAL into a FAIL -- and every finding so far has been a
rationale/disclosure defect on top of substance that both evaluators measured as
correct. Turning that into a FAIL on a counter would be the harness logging
instead of correcting.

**Nothing here is unsafe to leave as it stands, and everything shipped is a
tightening:**

- The immutable command went from **1 failed / 23 passed** to **24 passed**.
- The clock-dependence population is empty along the covered axis: **15 failed /
  3362 passed under BOTH clocks, delta empty**.
- `kill_switch.py` is byte-unchanged and **no assertion was weakened** --
  `daily_loss_breached is True` is byte-identical across `d5180e27^..HEAD`.
- The staleness rule GAINED coverage it never had: the stale path, the
  uncovered band, four staleness offsets, and the recompute property.
- Both evaluators independently verified the adjudication's conclusion: **there
  is no live defect.** The cycle-2 Q/A enumerated all six `evaluate_breach`
  callers to confirm it.

**What a fresh session needs to do to close it:** one Q/A pass on the cycle-3
tree. There is no known outstanding remedy -- the cycle-2 path-to-pass is fully
executed. If a third CONDITIONAL does arrive, the escalation makes it a FAIL, so
the next session should read this disposition first and decide deliberately
rather than spawn reflexively.

**Open and disclosed, unchanged:** the TZ-vs-UTC blind spot (needs
`time-machine`, not installed, an operator ask); the `.json`/`.csv` fixture-date
gap; the disclosed `PYFINAGENT_86_24_PROW_PATH` test seam; and phase-86.27's
`ebeb03da` fix, which no Q/A has graded.
