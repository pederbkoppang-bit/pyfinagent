# Experiment Results -- phase-82.51

**Step:** 82.51 (P1) -- publication-lag look-ahead on every fundamentals read.
**Date:** 2026-08-06. **Cycle:** 2 (cycle-1 Q/A returned CONDITIONAL on three WARNs; all three fixed -- see §11).
**Contract:** `handoff/current/contract_82.51.md`
**Research brief:** `handoff/current/research_brief_82.51.md` (`gate_passed: true`,
audit-class, dry after 6 rounds / 2 dry, 8 sources read in full, 37 URLs, 12 files)

---

## 1. What changed

| File | Change | Lines |
|------|--------|-------|
| `backend/backtest/cache.py` | `_embargoed_cutoff()` seam; both read paths share one rule; `apply_embargo` param | +47 / -1 |
| `backend/backtest/fundamentals_coverage.py` | `FUNDAMENTALS_EMBARGO_DAYS = 60`, derived `effective_coverage_start()` | +47 / -0 |
| `backend/backtest/backtest_engine.py` | refusal judged against the effective start; availability record extended | +30 / -9 |
| `backend/backtest/quant_optimizer.py` | **cycle 2** -- the consumer left on the raw start (§11) | +10 / -1 |
| `backend/agents/mcp_servers/data_server.py` | **cycle 2** -- the LIVE path opts out of the embargo (§11) | +7 / -1 |
| `backend/tests/test_phase_82_12_string_column_guards.py` | two classified line numbers re-derived (see §9) | +14 / -3 |
| `backend/tests/test_phase_82_51_fundamentals_embargo.py` | new -- 18 tests | 485 (new) |
| `scripts/backtest/run_82_51_embargo_ab.py` | new -- the criterion-4 A/B runner | 100 (new) |
| `backend/backtest/experiments/mda_cache.json` | **side effect of the required backtests**, not an intentional edit | +30 / -30 |

Figures from `git diff --numstat` and `wc -l`, run as the last action before
writing this file.

## 2. Verbatim output of the immutable verification command

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_51_fundamentals_embargo.py -q
warnings.warn(
..................                                                       [100%]
18 passed in 1.76s
```

## 3. Criterion 4 -- the real before/after backtest

Two runs, the embargo the **only** variable. Commands verbatim:

```
$ source .venv/bin/activate && FUNDAMENTALS_EMBARGO_DAYS=0  python scripts/backtest/run_82_51_embargo_ab.py
$ source .venv/bin/activate && FUNDAMENTALS_EMBARGO_DAYS=60 python scripts/backtest/run_82_51_embargo_ab.py
```

`strategy=qarp`, `2025-06-30 .. 2026-02-28`, `train_window_months=6`,
`test_window_months=2`.

| metric | embargo 0 (before) | embargo 60 (after) | delta |
|--------|--------------------|--------------------|-------|
| **sharpe** | **4.4201** | **3.7449** | **-0.6752 (-15.3%)** |
| deflated_sharpe | 0.9377 | 0.8959 | -0.0418 |
| **n_trades** | **40** | **40** | **0** |
| total_return_pct | 7.4197 | 5.3583 | -2.0614 |
| max_drawdown | -1.6710 | -1.3994 | +0.2716 |
| training samples | 40 | 36 | -4 |
| elapsed | 116s | 113s | -- |

`data_availability` from the after-run, showing the extended record:

```
'fundamentals': True, 'fundamentals_coverage_start': '2024-06-30',
'fundamentals_embargo_days': 60,
'fundamentals_effective_coverage_start': '2024-08-29',
'fundamentals_window_start': '2025-06-30', 'fundamentals_label_dependent': ['qarp']
```

**The pre-registered expectation held.** The contract committed in advance to
"Sharpe DOWN or flat, `n_trades` DOWN or flat, and **a delta of exactly 0.0000
is an ALARM**". Sharpe fell 0.6752, trades were flat, and the delta is not zero
-- so the embargo was genuinely applied rather than wired through the §4 decoy.

**Read plainly: about 15% of this strategy's measured Sharpe was look-ahead.**
It was reading fundamentals that had not been published as of the cutoff.

**Honest limits of this measurement, stated rather than buried:**
- The window yields **one** walk-forward window and 40 training samples. That is
  a small sample; the direction is meaningful, the magnitude is one observation,
  not an estimate with an error bar.
- `n_trades` is flat because `max_positions` / `top_n_candidates` bound the
  trade count here, not label availability. The step anticipated a trade-count
  drop; on this window there is none, and I am not presenting the flat number as
  a confirmation of anything.
- The window starts 2025-06-30 deliberately: the gate measured that at a
  2025-03-31 cutoff a 60-day embargo takes 5-quarter feature coverage from 41.7%
  of the universe to 0.0%, so an earlier start would have measured the coverage
  hole rather than the embargo.

## 4. The step's central premise is REFUTED, and the refutation is a trap

The step offers "(b) filtering on a real filing date where one exists".
**`filing_date` exists and is worthless.** Measured by me, then independently by
the gate:

```
n_rows 4798 | n_filing_missing 0 | n_filing_unparseable 0
lag (filing_date - report_date): mean 0.0  p50 0  p90 0  p99 0  min 0  max 0
```

The producer, found by the gate at `backend/backtest/data_ingestion.py:278`:

```python
"filing_date": report_date,  # Approximation; true filing date not available from yfinance
```

Switching the filter to `filing_date` would produce a **byte-identical result
set and a byte-identical backtest** while looking like the correct fix, and would
report a Sharpe delta of exactly `0.0000` -- which reads as "the leakage was
immaterial". `test_filing_date_is_still_a_decoy_and_must_not_be_filtered_on`
turns that latent trap into a tripwire.

`ingested_at` is also unusable: all 4798 rows were bulk-backfilled over two weeks
in 2026-03/04 (3374 on a single day) against `report_date`s spanning
2024-06-30..2026-02-28. It records when *we* fetched, not when the market saw it.

**Also refuted: the step's "measured lag on the live table: mean 66 / median 60 /
p90 90".** It cannot have been measured on this table, where the lag is
identically 0. The gate traced it to a tier-4 vendor blog with a disclosed
commercial COI, measuring SEC EDGAR across 5,194 companies -- **whose own
large-cap subsample says 43d mean / 61d max.** Our universe is 503 S&P 500
tickers, all large accelerated filers, so 66d was the wrong calibration target.

## 5. Criterion 5 -- the decision, and why 60

**Fixed 60-calendar-day embargo on `report_date`.** Option (b) is unavailable
(§4), so this is not a preference between two live options -- it is the only
implementable choice, and that is recorded as the reason.

**Why 60 and not 45:** the largest cohort in the table is the fiscal-year-end
quarter -- **744 rows / 443 tickers** -- governed by the **10-K**, whose
large-accelerated deadline is **60 days**. The other quarters are 10-Qs at 40. A
45-day embargo covers the 10-Q deadline but **under-covers the 10-K by 15 days**,
leaking on precisely the biggest cohort. Measured cost on this table: 45 -> 60
costs 5.5pp of visible row-days but **zero** 5-quarter tickers at either the
2025-06-30 or 2025-12-31 cutoff; 60 -> 90 costs another 8.7pp for no additional
legal coverage on a large-cap universe.

**Recorded as an approximation, not a correctness proof.** A fixed lag is wrong
at both tails -- OpenSourceAP's analogous rule still has >50,000 violating
observations (open issue). The correct fix is a real filing date from **82.50**.
The constant's docstring says exactly this, so the next reader inherits the
caveat and not just the number.

## 6. This fix would otherwise have RE-CREATED the defect 82.21 closed

Without the refusal-site change, a window starting 2024-07-01 would pass
`window_is_covered()`, be recorded `fundamentals: True`, and yet have **every**
`cached_fundamentals()` call return `[]` -- because nothing is visible until
2024-08-29. **43 business days in the measured grid have zero visible rows at
N=60.** That is "records coverage it does not have", reintroduced by this step's
own fix.

`FUNDAMENTALS_COVERAGE_START` is left at `2024-06-30` (it is the raw measurement,
and `_fundamentals_coverage.json` was NOT edited); `effective_coverage_start()`
is **derived**, never a second literal. The embargo is applied at the refusal
site rather than inside `window_is_covered`, which is what keeps 82.21's
semantics intact and its boundary test green.

## 7. Mutation matrix -- 9 mutants, all killed

Production code only; restores write back captured bytes rather than
`git checkout`, so a concurrent session cannot be clobbered.

```
baseline: rc=0 GREEN

M_A embargo becomes a no-op                          DIED  (the whole fix reverts to report_date <= cutoff)
M_B embargo in branch 1 ONLY (one seam short)        DIED  (a fix covering the preload path but not the SQL path)
M_C refusal judged against RAW start (kwarg deleted) DIED  (the 82.21 false-pass this fix would re-create)
M_C2 refusal start= raw VALUE  [cycle 2]             DIED  (the Q/A's defeat of my original guard)
M_G optimizer left on the raw start  [cycle 2]       DIED  (the consumer cycle 1 missed)
M_H live data server embargoes after all  [cycle 2]  DIED  (the undisclosed live-path change)
M_D embargo 60 -> 45                                 DIED  (the recorded decision is pinned, not decorative)
M_E availability record drops the embargo key        DIED  (82.21's record is genuinely extended)
M_F effective start hardcoded not derived            DIED  (a literal that would drift silently)

=== 9 died, 0 survived ===
```

Licenses exactly "these 9 mutants died", not "no survivors". **M_B is the one
that matters**: it applies the embargo to the preload path only, leaving the SQL
fallback on the raw cutoff -- the stop-one-seam-short shape that has been my
recurring failure. Its killers, named rather than counted (kill counts are
construction-dependent): `test_both_read_paths_agree_on_the_same_fixture` and
`test_the_sql_path_binds_the_embargoed_cutoff_as_its_parameter`.

**M_B silently went SKIP on the first cycle-2 run** -- its anchor no longer
matched after the `apply_embargo` edit, and the harness reported
`SKIP -- anchor not found` rather than a kill. A skipped mutant is not a killed
one, and had the harness printed nothing I would have carried "9 died" while my
most important mutant never ran. Re-anchored and re-run; it dies.

## 8. Regression, lint, and one failure that is NOT mine

```
$ python -m pytest backend/tests/ -q -k "backtest or fundamental or coverage or cache or macro or string_column"
1 failed, 274 passed, 2548 deselected, 1 warning in 17.11s
```

The failure is
`test_dod4_tier1_coverage_investment.py::test_paper_trader_execute_buy_average_up_recomputes_avg_entry`.
**Proven pre-existing rather than asserted:** I ran it in a `git worktree` at
HEAD. The first attempt failed for an unrelated environment reason (no `.env` ->
pydantic ValidationError), which would have been a false confirmation, so I
symlinked the env and re-ran. At HEAD, with the same environment, it fails with
the **identical** assertion `assert None is not None`. It exercises
`paper_trader`, which none of my changed modules import.

**Lint:** `All checks passed!` on a derived file scope (`ruff --select F,E9`). The scope is derived as `git diff --name-only HEAD -- '*.py'` union the untracked `*.py` set, and asserted non-empty; the cycle-1 artifact called it "5-file" by typing rather than counting -- it is 8 after the cycle-2 edits.
One `F541` was introduced by me in the new A/B script and is fixed.

## 9. Guards that broke, and were fixed deliberately

The gate predicted `test_phase_82_12_string_column_guards.py` would break because
it pins classified `file:line` claims, and it did. **Fixing the first entry
revealed a second** -- inserting the seam shifted every subsequent `cache.py`
entry by +35 -- so rather than chase failures one at a time I re-derived **all
four** classified lines from source the same way the test does:

```
OK    backend/services/outcome_tracker.py:  100 analysis_date  actual=[101]
OK    backend/backtest/cache.py:  647 report_date    actual=[647]
STALE backend/backtest/cache.py:  672 date           actual=[707]
OK    backend/tools/sec_insider.py:  226 date           actual=[226, 240]
```

Both `cache.py` entries updated with the reason recorded inline. That table's own
docstring says *"file:line claims rot. Re-derive rather than trusting the
table."* -- so this is its designed maintenance path, not a weakened guard. The
`cache.py:date` read is in `cached_macro`; its semantics are untouched and only
its position moved.

`_fundamentals_coverage.json` was deliberately **not** edited -- it is a raw
measurement and `snapshot_drift()` depends on it.

## 10. What I did NOT do

- **No new data source.** 82.50 owns SEC EDGAR; this step only makes its absence
  loud.
- **No change to `window_is_covered` semantics**, and no change to
  `data_ingestion.py:278`. Fixing the producer requires a real filing date, which
  is 82.50's job.
- **No second backtest strategy.** The gate noted `triple_barrier` would also
  move (every strategy is feature-dependent via `_NUMERIC_FEATURES`), which
  contradicts the step's claim that non-fundamentals strategies would show a zero
  delta. I ran only `qarp`, whose delta is directly interpretable. A
  `triple_barrier` run would be evidence about the whole engine and is worth
  doing, but it is not what criterion 4 asks for.
- **No live positions touched.** These are historical backtests.

## 11. Cycle 2 -- three findings from the Q/A, all real, all mine

The cycle-1 Q/A returned **CONDITIONAL** with all five criteria MET and three
WARN findings *outside* the criteria. It proved criteria 1/2/3 by running its own
7-mutant matrix -- including the branch-1-only mutant I asked it to attack, and
one I had not written (`embargo = 100000d`, to check criterion 2 cannot pass by
excluding everything). All three findings are fixed.

### 11.1 A consumer I changed the meaning of, and never grepped

`backtest_engine.py` now judges coverage against the **effective** start
(2024-08-29), but `quant_optimizer.py:176` still called
`window_is_covered(window_start)` -- the **raw** start (2024-06-30). For any
window beginning in that 60-day gap the optimizer would keep `qarp` in the
selectable pool as COVERED, and the engine would then raise
`backtest REFUSED`. **Pre-82.51 the two agreed; my diff made them disagree** --
and that function's own docstring promises it uses "the same 82.21 predicate the
engine uses", which my change had quietly falsified.

This is the stop-one-seam-short shape again, one level up: not a helper vs a call
site, but **one call site of a shared predicate vs the others.** Fixed, and the
new `test_every_window_is_covered_consumer_judges_against_the_effective_start`
derives its own consumer file set, asserts it non-empty, and checks the bound
VALUE at every site -- so the next person to change this predicate's meaning
cannot miss a consumer the way I did.

### 11.2 A guard that asserted a keyword's NAME, not its VALUE

My `test_the_refusal_site_judges_against_the_effective_start` asserted only that
a `start=` kwarg was present. The Q/A defeated it by substituting
`start=FUNDAMENTALS_COVERAGE_START` -- **semantically the exact bug, and the
guard stayed GREEN.** It ran my own AST predicate against the mutated text to
prove it. So my "M_C DIED" was true only for the kwarg-*deletion* construction I
happened to pick.

Replaced with a **behavioural** guard: build an engine at `window_start =
2024-07-01` (inside the gap) and assert `ValueError: REFUSED`. A behaviour cannot
be reworded around. M_C2 -- the Q/A's own mutant -- now dies.

### 11.3 A live consumer, under a section claiming no live change

`cached_fundamentals` has a consumer outside the backtest tree:
`backend/agents/mcp_servers/data_server.py:149`, calling it with
`cutoff = date.today()`. My change would have hidden the most recently reported
quarter from the **live agent pipeline** for 60 days -- while §10 of this artifact
said "No live positions touched. These are historical backtests."

**Decision, recorded:** the embargo is **wrong** on that path and is now scoped
out of it. The embargo reconstructs *"what could I have known at a past cutoff"*;
a live as-of-today query asks *"what is true now"*, and every row already in the
table has been published -- the ingester only fetches reported figures. Embargoing
there would suppress real data, not prevent look-ahead.

Implemented as `cached_fundamentals(..., apply_embargo: bool = True)`, with the
live call site passing `False` and a comment saying why. **The default is True**
so a future caller who does not think about leakage gets the protected path;
opting out has to be deliberate. Two guards pin both halves.

### 11.4 The measurement was re-verified, not assumed

These fixes changed `cached_fundamentals`' signature after the criterion-4 runs,
so the numbers were re-measured rather than carried forward. The embargo=60 arm
re-run verbatim:

```
embargo_days=60
sharpe=3.7449
deflated_sharpe=0.8959
n_trades=40
total_return_pct=5.3583
```

Identical to §3. The cycle-2 changes did not move the result -- as expected,
since the backtest path (`historical_data.py:61`) uses the default
`apply_embargo=True`.

### 11.5 Two NOTE-level corrections the Q/A raised

- §8 said lint ran on a "derived 5-file scope". The derivation yields **6** files
  (8 after cycle 2). I typed the number instead of counting it; the outcome
  (`All checks passed!`, exit 0) does reproduce. Corrected in §8.
- §7 said M_B "dies on three separate tests"; the Q/A measured 2 under its own
  construction. Kill counts are construction-dependent, so the killers are now
  **named** rather than counted.

## 12. Cycle 3 -- the same defect recurred INSIDE the fix for it

The cycle-2 Q/A returned CONDITIONAL again. All five criteria stayed MET and all
three cycle-2 WARNs were confirmed cured (it proved WARN-2 by running its own
`start=FUNDAMENTALS_COVERAGE_START` substitution and watching it die; it endorsed
WARN-3's reasoning explicitly: *"you did not silence my finding, you scoped it
correctly"*). It blocked on **two red tests my cycle-2 edits introduced**.

### 12.1 The finding that matters: I repeated WARN-1 while fixing WARN-1

Adding `apply_embargo=` to `cached_fundamentals` broke
`test_phase_75_mcp_truth.py`, whose `_FakeCache` double takes the OLD call shape.
**That is the identical unswept-consumer defect the cycle-1 Q/A raised -- recurring
inside the fix for it.** I swept the consumers of `window_is_covered` because I
was told to, and did not sweep the consumers of the function whose signature I
was changing in the same edit.

Aggravating, and worth more than the test fix: `data_server.get_fundamentals`
wraps that call in a broad `except`, so the `TypeError` never surfaced -- it was
swallowed into a logged error dict. **On the live MCP path a call-shape mismatch
degrades silently.** The double now accepts the real shape and records it.

### 12.2 The line numbers went stale a second time, for the same reason

§9 said all four classified lines were "re-derived from source". They were --
**at cycle-1 state.** My cycle-2 `apply_embargo` docstring added 11 lines above
them, so 647/707 became 658/718 and the guard went red again. Fixed by
re-deriving **programmatically as the last action on `cache.py`**, with the
script asserting each rewrite is not a no-op. Not by adding 11.

### 12.3 One real regression, found only by running the WHOLE suite

A `-k` subset is what let both failures through, so this cycle ran
`pytest backend/tests/` entire: **32 failures**. Rather than guess which were
mine, I diffed against a `git worktree` at HEAD -- 29 also failed there.

Of the remaining 3, **exactly one is a real consequence of this change**:
`test_phase_82_46_trial_pool_composition::test_unrunnable_members_are_excluded_by_a_DERIVED_rule`
asserted `selectable_strategies_for_window(FUNDAMENTALS_COVERAGE_START) ==
covered`. Post-embargo the raw start is *inside* the gap, so the dependent
strategies correctly drop out there. The assertion's stated intent -- "the
boundary is the MEASURED coverage start, not a typed date" -- is preserved; the
boundary moved to `effective_coverage_start()`, and the raw start is now pinned
on the other side so this cannot silently revert.

**The other two are environment, not code, and my own baseline nearly fooled me.**
`test_phase_23_2_10_watchdog...` and `test_phase_23_2_6_sector_cap_emit` read
`handoff/logs/*.log` with a 24h freshness window. They "passed" at HEAD only
because `handoff/logs` is gitignored and the worktree had **0 log files vs 46**
-- they pass vacuously when the logs are absent. In the real tree they fail
because the watchdog log is 46.5h stale and no `Skipping BUY` line exists. A
worktree diff is only valid when the environment matches, which is the same trap
that produced a false confirmation in §8.

**Final state: 31 failures = 29 pre-existing + 2 environment-dependent. Zero code
regressions from this diff.**

### 12.4 Also strengthened: the consumer sweep the Q/A blinded twice

It defeated my sweep two ways: a variable merely *named* `effective_start`
holding the raw constant, and aliasing the import so the file dropped out of a
text-derived file set. Both fixed -- the file set is now derived from **imports**
of the symbol under any alias, and the bound value is **resolved through local
assignments** rather than substring-matched. Its `>= 2 consumers` floor asserts
the set is not under-derived.

More importantly, the sweep is no longer sole coverage for that consumer:
`test_the_optimizer_drops_a_dependent_strategy_inside_the_embargo_gap` now drives
`selectable_strategies_for_window` **behaviourally**, which is the Q/A's named
fix and cannot be reworded around.

### 12.5 Corrected figures

`git ls-files -m -o --exclude-standard -- '*.py'` yields **10** files; lint over
all of them: `All checks passed!`. The §8 regression capture (`1 failed, 274
passed`) was from the cycle-1 tree and no longer reproduces -- §12.3's whole-suite
numbers supersede it.
