# experiment_results -- step 86.59

**Step:** the stock picker analyses the SAME 4-6 names every day because its
score is built only from slow trailing returns. **P1, money path.**

**Immutable verification command:**

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/tools/screener.py\").read()); print(\"parses\")"'
parses
```

## What this step SHIPS

**A measurement, and no behaviour change.** That is the deliverable the
criteria describe: criterion 1 says the stability must be *measured*, criterion
4 says the three existing dark flags must be evaluated *before new code is
written*, and criterion 5 forbids promoting a flag. Two new files, both under
`scripts/qa/`:

| file | what it is |
|---|---|
| `scripts/qa/rank_stability_86_59.py` | the measurement -- drives the REAL `screen_universe` + `rank_candidates` with stored BigQuery prices swapped in for the yfinance network call |
| `scripts/qa/mutation_86_59.py` | criterion 7 -- 14 cells, control-green-first, SHA-256-verified restore |

**No production file is modified.** `git status --short -- backend/` shows
nothing from this step; no `.env` write, no flag promotion, no gate touched,
no restart pending.

## Why it measures the system rather than a reimplementation

The script monkeypatches **only** `screener.yf.download`. Every factor
(`momentum_1m/3m/6m`, `rsi_14`, `volatility_ann`, `sma_50_distance_pct`,
`pct_to_52w_high`), every filter, every overlay and the sort are computed by
production code. Two details that a rewrite would have gotten wrong:

- `momentum_6m = _pct_change(close, len(close) - 1)` is anchored to the **first
  bar of the fetched window**, not to a fixed 126-day lookback, so the window
  length is a *factor definition* and is recorded beside every figure;
- `rank_candidates` only attaches a sector when the caller supplies
  `sector_lookup`, so the real `build_sector_map` is used (502/513 tickers)
  -- without it criterion 3 would report `UNKNOWN` for 100% of slots and
  measure the harness.

## Results, by criterion

Full evidence with verbatim command output: **`live_check_86.59.md`**.
Normalisation for every figure: 20 **consecutive sessions** ending 2026-08-17,
trailing window 126 sessions, 513 US tickers, one-sided turnover.

**Criterion 1 -- MEASURED, and it partially refutes the step's premise.**
Spearman rho **0.9622** mean / 0.9319 min over the full screened cross-section;
one-sided top-10 turnover **15.8%/day**; **3 of 19** adjacent sessions had zero
top-10 turnover. So the composite is *highly persistent but not frozen*. The
step is filed on *"the ranking is effectively frozen"*; over 20 sessions that
is too strong. Criterion 1 explicitly permits this outcome and requires it be
reported either way.

**Criterion 2 -- vacuously satisfied, and that is stated rather than
exploited.** No new or reweighted term is introduced, so there is nothing to
justify out-of-sample. The reason is not laziness: criterion 2 *also* demands
DSR/PBO for any such term, and those gates are computed on backtests that read
`financial_reports.historical_prices` -- which returns **duplicated rows**
(§ below). Reporting a gate number off that data would be a figure I could not
stand behind. The reweighting fix is therefore filed as **86.117**, explicitly
BLOCKED-BY **86.116**.

**Criterion 3 -- measured distribution, N = 20 stated.** 12 distinct names
across 100 analysed slots (12.0% of ceiling); **Information Technology 72.0%**
of slots. The one-sector finding reproduces. The live "before" is taken from
the system, not the replay: **18 distinct tickers analysed live** over the same
20 sessions, against the step text's *"only 8 distinct across 8 cycles"*.

**Criterion 4 -- all three existing dark flags MOVE the slate.**

| arm | turnover/day | distinct | top sector |
|---|---|---|---|
| baseline | 15.8% | 12 | Information Technology 72% |
| `sector_neutral` | 28.4% | 22 | Industrials 20% |
| `soft_diversity_w0.30` | 22.1% | 17 | Information Technology 40% |
| `min_k_sectors=3` | 17.9% | 14 | Information Technology 60% |

**This is the step's most consequential result.** The diversity mitigation is
not missing -- it is built, sitting in the tree, switched off. Writing a fourth
one is precisely the duplicate work criterion 4 exists to prevent, which is why
this step ships no new scoring code.

**Criterion 5 -- nothing promoted, nothing written.** Operator asks below.

**Criterion 6 -- parity, honestly characterised.** With no new behaviour there
is nothing to disable, so a "flag-OFF parity" claim would be vacuous. What is
actually demonstrated is stronger and checkable: the step modifies **no**
production file, so the live candidate list is unchanged by construction, and
the measurement additionally shows `rank_candidates(top_n=10)` agreeing with a
slice of the full ranking on every cycle (an independent call, cell M3).

**Criterion 7 -- 14 cells, 14 KILLED, 0 SURVIVED, 0 UNSCORABLE**, control
GREEN first on all three modes, SHA-256-verified restore.

## Finding (a): the declared weights are not the effective weights

Declared **40/35/25**; measured effective **22.6 / 37.0 / 40.4**. The term with
the *smallest* declared weight has the *largest* effective influence, because
influence scales with weight x cross-sectional sigma and the 6m horizon carries
~3.0x the dispersion of the 1m.

**No existing flag fixes it, and the near-miss is worth naming**: reading the
source suggests `multidim_momentum_enabled` does, because it calls `_zscore`.
It does not -- it z-scores the *finished composite* as one scalar, so it cannot
reweight the horizons inside it. Measured: 50 of 10,139 ranked positions move
(0.493%), and every displacement is a `round(..., 4)` tie. Filed as **86.117**.

## Three defects this step found in ITS OWN deliverable

Recorded because the last session's evaluations blocked on guard vacuity, and
two of these are exactly that class.

1. **A tautological guard.** `slate_is_a_prefix_of_the_full_ranking` compared
   `ranked` to its own defining expression on the line above. It could not fail
   on any input. Found by trying to mutate it. Replaced with an independent
   `rank_candidates(top_n=10)` call.
2. **A guard that survived being weakened.** The tie-explanation rule was
   inline; mutating it to `len(moved) >= 0` left the suite green -- an
   assertion cannot detect its own weakening from inside. Fixed with a paired
   negative fixture of known-bad inputs (`_TIE_FIXTURE`); M12 now KILLS.
3. **A stranded mutant.** A 2-minute timeout SIGTERMed the matrix mid-cell and
   left `return moved >= 0  # MUTANT` on disk, where it then failed the fixture
   it was written to protect. `try/finally` does not run on SIGTERM. The matrix
   now installs signal handlers that restore, and **refuses to start** from a
   target already containing a `MUTANT` marker.

**A prediction that failed, kept in the record.** The multidim analysis began
as a prediction of *exact list equality* and that failed (1 of 6 cycles). The
structural claim survived; the test did not match it. Both the failure and the
corrected test are in `live_check_86.59.md` § 6.

## Two defects found in the SYSTEM, filed rather than absorbed

**86.116 (P1) -- 38.0% of `financial_reports.historical_prices` rows are
duplicate `(ticker,date)` keys and nothing under `backend/` de-duplicates
them.** Found when the replay crashed on `InvalidIndexError`. Table-wide:
706,875 duplicated keys of 1,152,607; 336 of 513 tickers; **62-64% of keys in
every year 2017-2025** and 0.1% in 2026 -- which is why it was never noticed.
Not a price-accuracy defect (differing closes disagree by 0.0% at p50 *and*
p99, max 0.93%); the harm is **positional**, because `_pct_change` indexes by
row, so a "21-session" lookback spans ~12-13 real sessions, and duplicated bars
inject 0% returns that depress `volatility_ann`. `preload_prices` and
`cached_prices` both `set_index('date').sort_index()` with no
`drop_duplicates`, and `drop_duplicates` appears **nowhere** under `backend/`.
Consumers are production backtest paths. **This is why criterion 2's DSR/PBO
requirement cannot be honestly met today.**

**86.117 (P2) -- the declared/effective weight gap above**, BLOCKED-BY 86.116.

## Numbered operator asks (criterion 5)

- **ASK-1.** Promote `paper_min_k_sectors_analyzed = 3`? Measured: top-sector
  share 72% → 60%, distinct names +2, at the smallest turnover cost of the
  three arms (+2.1pp/day). The brief prefers it because it changes only which
  names reach the deep-analyse slice and does **not** mutate `composite_score`,
  so it leaves the DSR/PBO gates uncontaminated.
- **ASK-2.** Promote `paper_soft_sector_diversity_enabled` with `w = 0.30`?
  Measured: top-sector share 72% → 40%, distinct +5, turnover +6.3pp/day. A
  larger diversity gain for a larger churn cost, and it *does* overwrite
  `composite_score`.
- **ASK-3.** `sector_neutral_momentum_enabled` is measured here for
  completeness (72% → 20%, distinct +10) but is **not recommended**: this
  project's own 2026-06-01 replay measured **-0.166** long-only Sharpe for hard
  sector-neutralisation.
- **ASK-4.** Widen `paper_analyze_top_n` (currently 5)? The step does **not**
  recommend this and the research gate warns against treating slate width as a
  defect. Raised only because the criteria require operator-gated options to be
  named rather than silently declined.

## Scope honesty

This step did **not** widen the slate, promote any flag, write `.env`, add
hysteresis or an incumbent bonus, implement residual momentum, or claim the
trade drought as its consequence (86.47 owns that and is PARKED). It does not
claim 86.60's entry-path fix. Everything it discovered outside its own scope
was filed as its own step rather than absorbed.
