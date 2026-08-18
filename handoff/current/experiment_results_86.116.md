# experiment_results -- step 86.116

**Step:** 38% of `financial_reports.historical_prices` rows are duplicate
`(ticker,date)` keys and NOTHING under `backend/` de-duplicates them, so every
backtest lookback is positionally compressed. **P1, money path.**

**Immutable verification command:**

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/backtest/cache.py\").read()); print(\"parses\")"'
parses
```

## What changed

| file | change |
|---|---|
| `backend/backtest/cache.py` | **NEW** `_dedupe_index()`; called from **both** read paths -- `preload_prices` and `cached_prices` |
| `backend/tests/test_phase_86_116_price_dedup.py` | **NEW** -- **13 tests** |
| `scripts/qa/verify_86_116.py` | **NEW** -- criteria 1, 2, 3, 5, 6; **43 invariants** (29 under `--offline`) |
| `scripts/qa/mutation_86_116.py` | **NEW** -- criterion 7; **14 cells across 2 targets** + 1 declared equivalent |
| `backend/tests/test_phase_82_12_string_column_guards.py` | line pins re-derived (658->700, 718->760) -- my insertion moved them |

**Both read paths, deliberately.** Step 86.59 spent three evaluation cycles
learning that guarding one of two structurally identical call sites is not
guarding. `test_both_read_paths_are_covered` asserts it rather than trusting it.

## Results, by criterion

Verbatim output: **`live_check_86.116.md`**.

**Criterion 1 -- census RE-MEASURED by this step.** 1,859,482 rows over
1,152,607 distinct keys; **706,875 duplicated keys = 61.33% OF KEYS**;
**706,875 excess rows = 38.01% OF ROWS**; max multiplicity 2; **336 of 513
tickers**. Per-year: 2017 90.5%, 2018-2025 62-65%, **2026 0.1%**.

*The normalisation rule is stated beside every share because the two are not
interchangeable*: share-of-keys divides by distinct `(ticker,date)` pairs,
share-of-rows by total rows. Quoting one as the other misstates the defect by
~23 points.

*A correction to my own prose:* an earlier revision claimed the script's
`excess_rows == total_rows - keys` assertion stops the two shares "drifting
apart silently". It cannot -- over that grouping `SUM(n) == total_rows` and
`COUNT(*) == keys` by construction, so it is an **algebraic identity**. It is
renamed `census_sql_is_internally_consistent` and now claims only what it does.
Criterion 1 is met independently, by the printed normalisation rule and the
`% OF KEYS` / `% OF ROWS` label on every share.

**Criterion 2 -- the harm, DRIVEN through real code.** Loaded `AKAM` 2025 via
the **real** `preload_prices`, factors from the **real** `screener` functions:

| | as-loaded (pre-fix) | de-duplicated |
|---|---|---|
| rows | 390 | 250 |
| `mom_1m` | 0.83 | **−0.52** |
| `mom_3m` | −1.60 | **+15.04** |
| `rsi_14` | 23.7 | **54.5** |
| `vol_ann` | 0.3343 | **0.4182** |
| sessions in a "21-period" lookback | **12** | 22 |

**Both momentum terms SIGN-FLIP** -- the pre-fix values do not merely differ in
magnitude, they point the other way. RSI 23.7 sits beside the `< 20` band where
`rank_candidates` applies `score *= 0.8`, so this is not cosmetic.

**Criterion 3 -- proven, with the positive control the criterion demands.**
Against the pre-fix tree: `drop_duplicates` 0 files, `index.duplicated` 0,
`is_unique` 0. **Positive control:** the same probe finds `set_index` in 2
files, and finds this step's own fix in the post-fix tree. A grep returning
nothing proves nothing until it is shown able to return something.

**Criterion 4 -- de-duplicate ON READ, keyed on the INDEX. Nothing deleted.**
No `DELETE`, no table rewrite, no write-side change. The one-time repair is
**ASK-1** below.

*The method is the finding.* `drop_duplicates()` would have been the obvious
choice and is **wrong here**: pandas ignores the index in that call, so it
compares values. Measured through the real loader, `AVB` 2026 returns **159 rows
for 155 distinct dates** under `drop_duplicates()` -- four duplicate dates left
behind -- against exactly 155 under `~index.duplicated()`. It happens to suffice
only where the duplicate rows are byte-identical (`AKAM` 2025), which is not
something a reader can rely on. Cell **M3** pins this.

**Criterion 5 -- parity against an ORACLE.** 12 frame shapes (n = 0, 1, 2, 5,
50, 260 × duplicated/not). On every unique-index frame the fix returns **the
same object** -- not an equal copy -- so it is provably inert; on every
duplicated frame it returns exactly `index.nunique()` rows. The oracle asserts
**both branches were exercised**, so a fix that never fires and a fix that
always fires would each be caught.

**Criterion 6 -- the gate path, corrected TWICE.** This section has now been
wrong twice and the artifact records both, because the second error is one I
have on file as a recurring habit of mine.

*First correction (research gate):* this is **not** a Sharpe-formula bug -- the
engine's NAV is a per-day dict, so duplication does not double-count NAV points.

*Second correction (cycle-1 Q/A):* my replacement mechanism was **not wired**. I
credited `barriers = daily_vol × vol_barrier_multiplier`. That key has **zero
readers**: it is written into `engine._strategy_params` and read by nothing, and
`rotation_runner.py::_DEAD_KEYS` lists it by name under *"NO engine reader
(reverted in 9fbd9cd6)"*. The formula I quoted exists only as a **comment**, and
`_compute_triple_barrier_label` sets `tp_price`/`sl_price` from fixed
`tp_pct`/`sl_pct` plus a cost term -- **no volatility term at all**.

**The mechanism that IS wired** is inverse-volatility **position sizing**:

```
historical_data   features["annualized_volatility"]
  -> backtest_engine   volatility = fv.get("annualized_volatility")
  -> signal dict
  -> backtest_trader   size_position(probability, volatility, nav)
  -> backtest_trader   vol_scale = min(target_vol / stock_vol, 3.0)
```

Measured volatility ratio (pre/post) on AKAM **0.7995**, giving **position-size
inflation 1.2508×** against a 3.0 cap; the bound under *full* duplication is
`√2` = **1.4142×** (since returns become `[r, 0, r, 0, ...]` and the std falls by
`1/√2` = 0.7071).

**The direction is counter-intuitive and the artifact says so explicitly:**
`stock_vol` is in the **denominator**, so an *understated* volatility makes
positions **larger**, not smaller. The backtest was taking more risk than its own
vol-targeting believed. So the original conclusion was right *and understated*.

Two tripwires keep this honest: one asserts `vol_barrier_multiplier` is still in
`_DEAD_KEYS`, the other that the barrier label still has no volatility term. If
either changes, the section must be re-derived rather than trusted.

**No threshold is adjusted**: `min_dsr=0.95` / `max_pbo=0.20` untouched.

**Criterion 7 -- 14 cells, 14 KILLED, 0 SURVIVED, 0 UNSCORABLE across TWO
targets**, control GREEN first at 13 collected, run against the **real file and the real suite**. Scoring is strict by
design: a non-zero exit is not a kill (pytest exits **5** on "no tests
collected"), the mutant must exit **1**, must **collect the same 13 tests** as
the control, and the **named** test must be among the failures. Restore verified
by SHA-256; SIGTERM/SIGINT/SIGHUP restore the file and the matrix refuses to
start from a target already containing a `MUTANT` marker.

One cell is declared **EQUIVALENT-BY-DESIGN up front with its evidence**
(`keep="first"` → `keep="last"`): which of two same-date rows survives is
immaterial, measured at 0.0% gap at both p50 and p99 with a 0.93% maximum. It is
scored as neither a kill nor a survivor. Omitting it silently would have hidden
a real property of the fix.

## Regression: measured, not assumed

Full suite: **20 failed, 3633 passed** in 8m24s. Attribution measured by
re-running the same 20 node ids against the pre-change file (restored via `git
show HEAD:`, restored back with a verified matching sha256):

- **18 already failed before this change** → filed as step **86.118**;
- **1 was mine** -- `test_phase_82_12` pins `cache.py` line numbers within ±6
  and my insertion shifted them. Fixed by **re-deriving from source**, which is
  what that table's own comment demands;
- **1 is an ordering artifact** -- passes when run alone.

After the pin fix: **0 failures attributable to this change.**

## Two mistakes of mine, recorded rather than smoothed over

**1. A SIGTERM stranded the pre-fix file on disk.** The first comparison ran past
the 10-minute command ceiling and was killed *while the pre-fix `cache.py` was
swapped in*, leaving the fix absent. Caught by checking `grep -c _dedupe_index`
immediately after and restored from a pre-swap backup with sha256 confirmed
identical. The mutation matrix installs signal handlers for exactly this; an
ad-hoc shell command had none.

**2. My first failure-set diff was meaningless.** It reported all 20 failures as
newly introduced. I had stripped the `FAILED ` prefix from one file and not the
other, so `comm` compared two formats that could never match -- a diff that
cannot produce agreement will always report total disagreement.

## Numbered operator asks

- **ASK-1.** Repair the table itself? A one-time de-duplication of
  `historical_prices` (**706,875 excess rows**) would remove the need for
  read-side filtering. **This step deliberately did not do it** -- criterion 4
  forbids a `DELETE` or rewrite here, and the read-side fix makes the repair
  non-urgent. It is *bounded and terminal*: 2026 is already at 0.1%, so the
  write side is not reproducing the problem.
- **ASK-2.** Re-run the DSR/PBO gates on de-duplicated data? Every historical
  gate number was computed on compressed horizons and depressed volatility. This
  step reports the **mechanism and its scale** but does not re-run the gates,
  because doing so is a strategy-validity exercise, not a data fix.
- **ASK-3.** `backend/agents/mcp_servers/data_server.py:99` raises
  `KeyError('date')` per the research gate -- a separate proven defect, recorded
  here rather than absorbed. File as its own step?

## Scope honesty -- what this step does NOT do

- **It deletes nothing** and changes no write path.
- **It does not claim a Sharpe-formula bug** -- the gate refuted that reading and
  the corrected mechanism is stated above.
- **It does not re-run or move any gate.** `min_dsr` / `max_pbo` untouched.
- **It does not fix `data_server.py:99`**, the 18 pre-existing red tests
  (**86.118**), the picker score (**86.59**, parked), or the entry path
  (**86.60**, blocked by a peer session).
- **It does not unblock 86.117 by fiat** -- that step re-measures for itself.
- **No flag promoted, no `.env` written.**

**RESTART WAS PENDING, and my earlier claim that it was not was FALSE.**
Measured, not reasoned: uvicorn **pid 41635** started **2026-08-17T15:57:16Z**
(etime 16h14m at the time of writing, no `--reload`), while this fix landed
2026-08-18T07:55. `backend/backtest/backtest_engine.py:25` does a **module-level**
`from backend.backtest import cache`, so that process holds the **pre-fix**
module in `sys.modules`, and `backend/api/backtest.py` runs `run_backtest`
in-process. **Every API-triggered backtest still reads duplicated frames.**

My justification -- *"the backtest loads it per run"* -- was wrong in the way
this project has a standing rule about: `preload_prices` is indeed called per
run, but **from the already-imported module object**. Committed is not in force.

The fix **is** live for every *fresh* process (harness, CLI, scripts); I
confirmed a fresh import carries `_dedupe_index`. So this is a **disclosure**
defect, not a broken fix.

**RESOLVED 2026-08-18T08:26:53Z -- the operator authorised the restart and it
is DONE.** `launchctl kickstart -k gui/<uid>/com.pyfinagent.backend`; pid
**41635 -> 89340**, verified by asserting the pid CHANGED rather than by a
health code alone. The pid launchd reports and the pid holding `:8000` are the
same (89340), so it is not an orphan; `/api/health` 200; 86.108's
`/api/settings/flags` and `/api/observability/parse-failures` both answer 200.

`kickstart -k` was chosen over `bootout`+`bootstrap` deliberately: no plist
`EnvironmentVariables` changed, and CLAUDE.md records a bootout/bootstrap race
that left the backend down ~4 minutes on 2026-08-09. A fresh interpreter
re-imports every module from disk, which is what this fix needed.

**Now in force on pid 89340:** 86.116's `_dedupe_index` (3 references on disk,
verified before the restart), 86.108's two read-only routes, 86.109's
`emit_alarm=False`.

---

## Cycle 2 -- response to the CONDITIONAL (`wf_6c5d3dfc-43a`)

**All three findings accepted. None disputed. No production code changed** --
every fix was to evidence, exactly as the evaluator characterised them.

**1. I credited a dead key.** Detailed above. This is a repeat of a failure I
have on record: *a correct observation can credit the wrong mechanism*. The
evaluator did not merely doubt the mechanism -- it enumerated all seven non-86.116
references to `vol_barrier_multiplier` and showed not one is a read, found the
`_DEAD_KEYS` entry naming it, and noted that a comment in the same file points at
`_compute_vol_target_scale`, **a function that does not exist in the repo**. Then
it supplied the mechanism that IS wired.

**2. My first re-runnable command aborted.** `--base-rev` defaulted to `HEAD`,
which was the pre-fix tree only while the fix was uncommitted. The moment
`539f16eb` landed, `HEAD` became the post-fix tree, the criterion-3 probe found
this step's own `_dedupe_index`, and the script died before printing anything --
so the *first* command the live_check advertises produced no evidence at all. Now
pinned to `539f16eb~1`. **A default that silently expires the moment the work is
committed is a defect, not a convenience.**

**3. The read-path fixtures could not see the method swap.** `_fake_rows`
duplicated each row byte-identically, and byte-identical rows *are* value
duplicates -- so replacing `~index.duplicated()` with `drop_duplicates()` left
both read-path tests green. The step's own headline calls the method choice "THE
METHOD IS THE FINDING", and it was pinned only at helper level. The fixture now
differs by `1e-9` (the shape 394,719 real duplicated keys have), cells **M3b**
and **M3c** confirm the mutant now dies on **both** read paths, and a new test
asserts the fixture precondition directly so a future edit restoring identical
twins fails loudly instead of silently disarming two tests.

**A brittle first attempt, recorded.** My initial dead-key tripwire was a
grep-with-filters and it fired on `quant_optimizer.py:715` -- `if
"vol_barrier_multiplier" in params:`, the *setter's* own guard, not a reader. A
probe that cannot tell a write from a read is not a probe, so it was replaced
with the repo's authoritative statement (`_DEAD_KEYS` membership) plus a direct
check that the barrier label has no volatility term.

**Evidence after cycle 2:** 31 invariants, 13 tests, mutation **8/8 KILLED**
with control GREEN first, collected-count parity and SHA-256 restore.

---

## Cycle 3 -- response to the second CONDITIONAL (`wf_10d2c895-28e`)

**One finding, accepted: the tripwire I added in cycle 2 to close the previous
finding was itself vacuous.** The evaluator's phrasing is exact and worth
keeping: *the remedy for brittleness introduced the vacuity.*

The guard read `("vol" not in <slice>) or ("self.tp_pct" in <slice>)`. The second
clause is **true on the unmutated tree**, so it short-circuited the first. I
reproduced the evaluator's result before touching anything -- `MUTANT vol added:
A=False B=True GUARD=True <- SURVIVES` -- and confirmed the window overshoot
(function **1753** chars, slice 2600, so 847 chars into `_compute_sample_weights`;
the evaluator measured 1,788/812 and the direction is the same).

**v3 has no escape clause and reads the AST**, collecting every identifier the
function *references* and rejecting any containing `vol`. Against the evaluator's
own mutants: `daily_vol/vol_mult` **KILLED**, `self.daily_volatility` **KILLED**,
and a comment-only mention correctly does **not** fire -- precise, not merely
sensitive. The slice is bounded by `ast`, not a character count.

**Both tripwires are now mutation-tested, which required a second matrix
target.** The matrix ran only against `cache.py` -- exactly how an untested guard
shipped. `verify_86_116.py` is now a second target, exercised through `--offline`
so no cell depends on BigQuery, with its own control and SHA-256 restore.

**The first version of those cells SURVIVED, and that is the more useful
result.** Disarming `_ok(name, EXPR)` to `True` cannot be caught by that
assertion -- the identical outcome step 86.59 produced. The rules are therefore
named predicates (`_declares_dead_key`, `_volatility_identifiers`) backed by a
fixture of known-bad inputs they must reject; and an **incomplete** fixture is
now reported as a fixture failure rather than raising `KeyError`, which had made
T3 UNSCORABLE instead of a kill.

**Evidence after cycle 3:** 33 invariants, 13 tests, **11/11 KILLED across two
targets**. `backend/backtest/cache.py` is byte-identical to what all three cycles
evaluated (`9f5f1d67...`) -- **no production code has changed since the original
fix**, and `git diff --name-only -- backend/backtest/cache.py` is empty.


## Cycle-3 and cycle-4 follow-up

*(The cycle-4 Q/A noted that this GENERATE artifact still carried the pre-fix
counts while the whole follow-up lived only in `evaluator_critique_86.116.md` --
so the EVALUATE artifact was carrying the GENERATE evidence. Corrected here, and
the numbers above are now DERIVED from the live_check capture rather than typed.)*

**Cycle 3 capped on a guard that could not fire**, and its replacement had two
further problems, both found by DRIVING it:

- a **dead `and`-clause** no fixture could falsify (`scale_pre < 3.0 and
  scale_post < 3.0`, where the first implies the second under the ordering guard
  above it) -- replaced with `max(...) < 3.0`;
- **unreachability offline**, which is how the vacuous version shipped in the
  first place. `gate_effect()` takes a plain dict, so the tripwires now drive
  the real function with synthetic volatilities.

Paired negative AND positive: `gate_guard_rejects_saturating_inputs` and
`gate_guard_accepts_unsaturated_inputs`. Offline invariants **19 -> 29**; full
run **43 invariants**. Cells **T4/T5/T6** added; matrix **14/14**.

**Cycle 4 then found two MORE guards that cannot fail**, which is the honest
state of this step and is recorded rather than glossed:

- `size_inflation_is_above_one` is implied by `pre_fix_volatility_is_the_lower_one`
  three lines above (`size_inflation = 1.0/vol_ratio`, and `vol_ratio < 1.0` is
  already asserted). The judge swept 256 input pairs and it was never the
  failing invariant. It is redundant rather than wrong -- unlike the cycle-3
  one, its failure could not corrupt a reported number -- but it has no cell.
- `_declares_dead_key` infers MEMBERSHIP from two independent token searches, so
  a re-wired `_DEAD_KEYS` tuple that still mentions the key classifies True.

Both are queued rather than fixed here: **86.116's attempt budget is exhausted
at 5/5**, so any further change to it is unevaluated by construction.
