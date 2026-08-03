# Research Brief — step 82.2: three new label methods for overpriced-market lenses

Tier: **complex**. Audit-class: **no**. Status: **IN PROGRESS** (write-first; appended as sources are read).

Scope: (a) valuation/crowding-stretch REGIME GATE, (b) QARP defensive tilt (long-only),
(c) mean-reversion on overextension + the `mean_reversion`-is-degenerate claim, (d) a
hedged/cash-timing overlay folded onto one of the three. Deliverable is a design brief,
**no code changes**.

---

## 1. Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/backtest/backtest_engine.py` | 1308 | `STRATEGY_REGISTRY` (:32), `_NUMERIC_FEATURES` (:42), `_compute_label` dispatcher (:1116), all 4 label methods | LIVE |
| `backend/backtest/historical_data.py` | 476 | `build_feature_vector()` (:53) — the ONLY feature source a label method has | LIVE |
| `backend/backtest/candidate_selector.py` | ~230 | `screen_at_date()` pre-filters the universe to `top_n_candidates` BEFORE any label is computed | LIVE — **gates label diversity** |
| `backend/backtest/cache.py` | — | `cached_prices` / `cached_fundamentals` / `cached_macro(cutoff_date)` (:471) memoised per-date | LIVE |
| `backend/backtest/quant_optimizer.py` | — | `mr_holding_days` range `(5, 30)` (:75); `strategy` is a categorical knob | LIVE |
| `backend/backtest/historical_data.py:281` | 40 | `compute_turbulence_index(cutoff_date, universe_tickers, lookback=252)` — Mahalanobis cross-sectional stress index | **DEAD CODE — zero callers repo-wide** |

### 1.1 The label-method contract (answers Q1)

**Signature** — every method in `STRATEGY_REGISTRY` is a bound method on `BacktestEngine`:

```python
def _compute_<name>_label(self, ticker: str, entry_date: str) -> int | None
```

**Dispatch** — `backend/backtest/backtest_engine.py:1116-1120`:

```python
def _compute_label(self, ticker: str, entry_date: str) -> int | None:
    method_name = STRATEGY_REGISTRY.get(self.strategy, "_compute_triple_barrier_label")
    method = getattr(self, method_name)
    return method(ticker, entry_date)
```

So adding a strategy = (1) add a `"name": "_compute_x_label"` row at `:32-38`, (2) define the
method on the class. `self.strategy` is coerced to a registry key at `:199`
(`self.strategy = strategy if strategy in STRATEGY_REGISTRY else "triple_barrier"`) — an
unregistered name silently falls back to `triple_barrier`, so the registry row is mandatory.

**Legal return values** — `{+1, 0, -1, None}`. Verified across all four existing methods
(`:742`, `:1124`, `:1145`, `:1208`). Semantics:
- `+1` / `-1` / `0` become the multiclass target `y` for `GradientBoostingClassifier`
  (`_train_model`, `:812`).
- `None` means "**drop this sample entirely**" — `_build_training_data:674-676` does
  `label = self._compute_label(...); if label is None: continue`. `None` is NOT a class;
  it is a row filter. Both the feature row and the label are discarded together.
- No other values are legal. Nothing normalises or clips the return.

**Where labels are consumed**: `_build_training_data` (`:618-741`) → `labels = np.array(labels_list)`
(`:731`) → `_train_model(X, y, sample_weights)` (`:812`). Also `_predict_and_trade` calls
`self._compute_label(ticker, test_start)` at `:937` to record the realised label for
test-window diagnostics. **There is no class-balance check anywhere** — a degenerate label
set trains a degenerate classifier silently (the model just predicts the majority class);
nothing raises, nothing warns. This is why 82.2's ">95% single-class" fixture gate has to
be a NEW test, not an existing guard.

**Sampling cadence** — `_build_training_data:641-646`: biweekly sample dates across the
train window (`pd.DateOffset(weeks=2)`), crossed with the candidate tickers. Purge
(AFML Ch.7) drops sample dates whose `1.5 * holding_days` label span overlaps the test
window (`:658-662`).

### 1.2 `_compute_mean_reversion_label` — the degeneracy verdict (answers Q2)

Source, `backend/backtest/backtest_engine.py:1145-1206`. Stage 1 gate at `:1174-1175`:

```python
is_oversold   = sma_dist < -0.05 and rsi < 35
is_overbought = sma_dist >  0.10 and rsi > 70
if not is_oversold and not is_overbought:
    return 0
```

**Units check first (this is the trap that would have made the diagnosis wrong).**
`sma_50_distance` is a **fraction, not a percent**:
`historical_data.py:105` → `features["sma_50_distance"] = (current_price - sma_50) / sma_50`.
So `-0.05` really is −5% and `0.10` really is +10%. (Note the near-miss: the *screener*
and *candidate_selector* use a DIFFERENT key, `sma_50_distance_pct`, which IS ×100 —
`candidate_selector.py:80` `sma_distance = (current_price - sma_50) / sma_50 * 100`. Two
keys, two unit conventions, one repo. A future label method that reaches for the wrong one
gets a 100× threshold error.) So the thresholds are *sane*; the raw-percent-vs-fraction
hypothesis is **DISPROVEN**. Likewise `reversion_target = entry_price * (1 + abs(sma_dist)/2)`
(`:1195`) is entry×1.04 for an 8% gap — an achievable 4% move, not a 500% one.

**Verdict: the ~all-neutral claim is PLAUSIBLE and STRUCTURALLY EXPLAINED, but the cause is
NOT the threshold magnitude. There are four compounding funnels, and the dominant one is
upstream of the label method entirely.**

1. **Universe pre-selection is momentum-ranked (dominant cause).**
   `_run_window:399-404` screens candidates with `candidate_selector.screen_at_date(...)`,
   whose `_rank_candidates` (`candidate_selector.py:180-215`) scores
   `mom_score*0.4 + rsi_score*0.2 + vol_score*0.2 + sma_score*0.2` where
   `sma_score = min(1, max(-1, sma_val/10))` — **positive SMA distance is rewarded**, and
   `rsi_score = 1 - abs(rsi-50)/50` **penalises RSI extremes in both directions**. The label
   method only ever sees the top-`top_n_candidates` names by that score. A name with
   `rsi < 35` is scored `1 - 15/50 = 0.70` on the RSI leg and gets a *negative* SMA leg — it
   is systematically ranked out. **The oversold branch (`+1`) is starved of candidates by
   construction**, and the overbought branch needs `rsi > 70`, also RSI-penalised. This is
   the same monosector-momentum funnel 82.1 identified in the live book, reproduced in the
   backtest universe.
2. **Conjunctive stage-1 gate.** Both legs must fire simultaneously. `sma_dist < -0.05`
   AND `rsi < 35` is a joint tail event; on a broad equity panel it is single-digit-percent
   of stock-days, and the momentum pre-filter above shrinks it further.
3. **Asymmetric thresholds bias the survivor toward `0`.** Oversold needs −5%; overbought
   needs +10% — **twice as strict**. In a rising tape (2023-2026 sample window) the +10%/RSI>70
   combination fires more than the oversold one, but its `-1` still requires a *subsequent
   fall*, which trend-following tape rarely delivers inside 15 trading days.
4. **Stage 2 fails open to `0`, not to `None`.** `:1206` `return 0  # Signal present but
   reversion didn't materialize`. So every signalled-but-unrealised row *adds to the neutral
   class* instead of being dropped. Combined with the `return 0` at `:1177` for no-signal,
   **`0` is the return value on two of three paths and is the only value reachable without a
   forward price move**.

The upshot: `mean_reversion` is not broken by a typo — it is a *correct* mean-reversion
labeller pointed at a *momentum-selected* universe with a conjunctive tail gate and a
fail-to-neutral stage 2. **Do not "fix" it by loosening the constants**; that treats a
selection problem as a threshold problem. The design fix (see §4.3) is to express the
overextension in **volatility units** and to make the neutral class a *drop* (`None`)
rather than a *class* on the no-signal path — or to widen the candidate pool for this
strategy. This is a claim to TEST in 82.3 (count the class histogram on the fixture), not
to assert; the measurement is one `collections.Counter` over the fixture's labels.

### 1.3 Features actually available to a label method (answers Q3)

A label method gets exactly what `self.data_provider.build_feature_vector(ticker, entry_date)`
returns (`historical_data.py:53-276`), plus anything it fetches itself via `cache.cached_prices`.
`_NUMERIC_FEATURES` (`backtest_engine.py:42-52`) is the 38-name subset that reaches the ML
matrix — **it is a filter applied at `:723` (`[c for c in _NUMERIC_FEATURES if c in df.columns]`),
not a limit on what a label method can read.** Several features are *built but not in the
list* and are therefore invisible to the model though usable by a labeller.

**VALUATION — exists:**
| Feature | Built at | Caveat |
|---|---|---|
| `pe_ratio` | `historical_data.py:157` | Only when `net_income > 0` — **negative-earnings names have NO P/E** (silently absent, not NaN) |
| `pb_ratio` | `:169` | Needs `total_equity > 0` and `shares > 0` |
| `fcf_yield` | `:178` | **capex is approximated as 0** (`fcf = ocf * 4`) — overstates FCF yield |
| `dividend_yield` | `:183` | Only if `dividends_per_share` present |
| `market_cap` | `:155` | In `_NON_STATIONARY` — fracdiff'd in the matrix, raw in the fv |
| `price_at_analysis` | `:73` | raw close |

**NOT AVAILABLE — do not design against these:** EV/EBITDA, EV/Sales, P/S, CAPE/Shiller,
earnings yield spread vs bonds, forward estimates, analyst targets, book-to-market as a
distinct field, any index-level P/E. Confirmed by reading the whole builder — the only
valuation primitives are the six above.

**QUALITY — exists:** `roe` (`:165`), `profit_margin` (`:167`), `debt_equity` (`:163`),
`revenue_growth_yoy` (`:209`), `quality_score` (`:255`, the Asness-QMJ 4-dimension composite:
profitability + growth + safety + payout, each normalised to [0,1] then averaged), plus
`total_debt`/`total_equity`/`total_assets`/`net_income`/`total_revenue` raw.
`annualized_volatility` (`:92`) doubles as the QMJ safety leg.
**Not available:** gross-profits-over-assets (Novy-Marx GPOA) — there is no `gross_profit`
or COGS field; accruals; ROIC; interest coverage.

**MACRO / REGIME — exists, per-date, six series only** (`historical_data.py:268-274`):
`fed_funds_rate` (FEDFUNDS), `cpi_yoy` (CPIAUCSL), `unemployment_rate` (UNRATE),
`yield_curve_spread` (T10Y2Y), `consumer_sentiment` (UMCSENT), `treasury_10y` (DGS10).
**Defect to note (out of 82.2 scope, worth its own step): `cpi_yoy` is assigned the raw
CPIAUCSL *index level* (~310), not a year-over-year rate.** `historical_data.py:269` is
`features["cpi_yoy"] = macro.get("CPIAUCSL", {}).get("value")` — no differencing. The name
lies. Any regime rule that thresholds `cpi_yoy` as a percentage is wrong today.
**Not available:** VIX, credit spreads (BAA-AAA), any breadth series, any index price level,
any equity-risk-premium input. `treasury_10y` + `yield_curve_spread` are the only two
usable "price of money" regime inputs, and there is **no equity index series** in the macro
join, so an index-level CAPE or ERP gate is **NOT implementable without new ingestion**.

**PRICE/FLOW — exists and under-used:** `momentum_1m/3m/6m/12m` (`:76-79`),
`momentum_12_1` (`:85` — **built but NOT in `_NUMERIC_FEATURES`**), `rsi_14` (`:88`),
`annualized_volatility` (`:92`), `daily_volatility` (`:96` — **built, NOT in the list**),
`sma_50_distance` / `sma_200_distance` (`:105/:108`), `bb_upper_distance` / `bb_lower_distance`
/ `bb_pct_b` (`:117-121` — **all three built, NONE in `_NUMERIC_FEATURES`**),
`volume_ratio_20d` (`:127`), `amihud_illiquidity` (`:136`), `anomaly_count` (`:133`),
`var_95_6m`/`var_99_6m`/`expected_shortfall_6m`/`prob_positive_6m` (`:131` via
`_compute_monte_carlo_var`), `sector` / `industry` (`:264-265`, categorical, excluded from
the matrix).

`bb_pct_b` and `daily_volatility` are the two most valuable finds here: **a volatility-unit
overextension measure is already computed and free**, which is exactly what §4.3 needs.

### 1.4 Per-row vs cross-sectional (answers Q4) — plainly

**The label-method interface is strictly PER-ROW: `(ticker, entry_date) -> int | None`.
There is no cross-sectional argument, and no per-date dataframe is passed in.**

But that is an interface fact, not a capability limit. Three things make a date-level
computation implementable *without changing the signature*:

1. **The outer loop is already per-date.** `_build_training_data:657-666`:
   `for sample_date in sample_dates:` → `for ticker in tickers:`. Every ticker at a given
   date is processed consecutively, so a per-date statistic computed on first touch and
   memoised on `self` (keyed by `entry_date`) is computed once and reused for the whole
   cross-section — no O(N²) blow-up.
2. **`cache.cached_prices` and `cache.cached_macro` are memoised**, so a label method that
   pulls the same date's macro or a benchmark series pays the cost once
   (`cache.py:471-488` — `_macro_cache[cutoff_date]`, plus a `_macro_full` fast path).
3. **A ready-made cross-sectional regime measure already exists and is dead:**
   `HistoricalDataProvider.compute_turbulence_index(cutoff_date, universe_tickers, lookback=252)`
   at `historical_data.py:281-320` — Mahalanobis distance of the current cross-asset return
   vector from its 252-day mean, i.e. a *co-movement/stress* index. **Zero callers repo-wide**
   (grepped). It is exactly the shape a crowding/stress gate needs.

**The one real gap:** the universe list is NOT on `self`. `_run_window` receives
`universe_tickers` as a parameter (`:388`) and passes candidate tickers down; `__init__`
(`:140-250`) stores `market`, `data_provider`, `candidate_selector`, `trader`, params — but
never a universe or the current candidate list. So a cross-sectional label method needs
**one small plumbing change: stash the window's candidate/universe list on `self` in
`_run_window`** (e.g. `self._current_universe = universe_tickers` right after `:388`).
That is a 1-line addition and is the *only* structural blocker. Say it explicitly in the
contract; do not design a cross-sectional rank and discover this in GENERATE.

**Conclusion:** a REGIME gate (date-level scalar) is **implementable today** via memoised
per-date computation + the 1-line universe stash. A true cross-sectional *rank* (percentile
of this ticker vs its peers at this date) is **also implementable** by the same route, but
costs a full feature-vector pass over the candidate set per date unless it is restricted to
price-only inputs (which `cached_prices` makes cheap). Prefer price-only cross-sectional
stats.

### 1.5 How the feature vector is built; is macro joined per-date? (answers Q5)

`build_feature_vector(ticker, cutoff_date)` (`historical_data.py:53-276`) makes exactly three
data calls, all point-in-time by construction:
- `get_point_in_time_prices(ticker, cutoff_date, lookback_days=504)` (`:30`) →
  `cache.cached_prices(ticker, start, cutoff_date)` — window is `504*1.5` calendar days back.
- `get_point_in_time_fundamentals(ticker, cutoff_date)` (`:38`) → up to 5 most recent
  quarterlies, `report_date DESC`, index 0 = latest as-of cutoff.
- `get_point_in_time_macro(cutoff_date)` (`:46`) → `cache.cached_macro(cutoff_date)`.

Early-exit guard at `:66-67`: `if prices.empty or len(prices) < 20: return features` — the
returned dict then holds only `ticker` + `date`, which is why every label method starts with
`if not fv or fv.get("price_at_analysis") is None: return None`.

**Macro IS joined per-date.** `cache.cached_macro(cutoff_date)` (`cache.py:471-511`) returns,
for each `series_id`, the **most recent observation with `date <= cutoff_date`** — the fast
path walks the DESC-sorted preloaded entries and breaks on the first `entry["date"] <= cutoff`
(`:483-486`); the BQ fallback does the same with `ROW_NUMBER() OVER (PARTITION BY series_id
ORDER BY date DESC) ... WHERE date <= @cutoff` and a 30s timeout (`:493-508`). Result is
memoised in `_macro_cache[cutoff_date]`, so it is O(1) for the rest of that date's tickers.

**Vintage caveat that matters for 82.0's un-freeze.** The join is on the **observation date**,
not on `realtime_start`. So the backtest sees a macro value on the day the observation is
*dated*, not the day it was *published* — a look-ahead of days-to-weeks for monthly series
(CPI/UNRATE/UMCSENT are dated to the month START; DGS10/FEDFUNDS are daily and unaffected).
82.0 added a `realtime_start` vintage column to `historical_macro`; **`cached_macro` does not
read it** (the query at `:493` selects only `series_id, value, date`). Any regime gate built
on the monthly series inherits this optimistic bias. Two mitigations: prefer the *daily*
series (`DGS10`, `FEDFUNDS`, and `T10Y2Y` which is also daily) for gating, or file a
follow-up step to filter on `realtime_start <= cutoff`. **Recommend the daily-series route
for 82.2** — it sidesteps the vintage issue entirely and needs no new work.

### 1.6 A structural finding that changes how 82.2 should be scoped

**Two of the four existing "label" methods are not labels — they are feature transforms.**
`_compute_quality_momentum_label` (`:1124-1143`) and `_compute_factor_label` (`:1208-1300`)
read `build_feature_vector(ticker, entry_date)` and map **today's features** to a class with
**zero forward information**. `_compute_triple_barrier_label` (`:742`) and
`_compute_mean_reversion_label` (`:1145`) both walk forward through `cached_prices`;
`quality_momentum` and `factor_model` never fetch a future price.

Consequence: for those two strategies the GradientBoosting stage is being asked to predict a
*deterministic function of its own inputs* (`momentum_6m > 5 AND quality_score > 0.3`, etc.).
The model learns the rule, in-sample accuracy is trivially high, and the ML layer adds no
information the rule did not already contain — the only edge is whatever edge the hand-written
rule has. This is not a 82.2 blocker, but it means "add three strategies like the existing ones"
would reproduce the defect. **All three 82.2 candidates below are forward-looking.** (Worth its
own masterplan step: audit whether `quality_momentum` / `factor_model` backtest results have
ever been interpreted as ML performance.)

---

## 2. External research

### Search-query variants run (three-variant discipline)

| # | Query | Variant type |
|---|---|---|
| 1 | `factor crowding measurement crowding-aware position sizing MSCI` | year-less canonical |
| 2 | `quality minus junk Asness Frazzini Pedersen quality at reasonable price construction` | year-less canonical |
| 3 | `CAPE valuation out-of-sample predictive power forward returns negative results Goyal Welch` | year-less canonical (negative-results framing) |
| 4 | `factor crowding 2026 momentum crowded trade AI semiconductor concentration risk` | current-year frontier |
| 5 | `short-horizon mean reversion overreaction labeling volatility-scaled thresholds machine learning class imbalance financial labels` | mechanism / year-less |

Honest note: I did not run a separate 2025-only query. The current-year query plus the
year-less queries surfaced the two 2025-26 papers that carry the recency scan (§2.3), so the
window is covered, but the query log above is what was actually run.

### 2.1 Read in full (6; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://arxiv.org/html/2512.11913v1 — Lee (KAIST), *Not All Factors Crowd Equally* | 2026-08-03 | preprint (q-fin.PM, 11 Dec 2025) | curl + tag-strip, full body | **[ADVERSARIAL]** "Crowding-based factor selection fails to generate alpha (Sharpe: 0.22 vs. 0.39 factor momentum benchmark)"; "crowding predicts crashes, not means—useful for risk management, not alpha generation". Crowded **reversal** factors: 1.7–1.8× crash probability. Crowded **momentum**: **0.38× crash risk, p=0.006** — momentum crowding is *benign*. |
| 2 | https://tevgeniou.github.io/EquityRiskFactors/bibliography/QualityMinusJunk.pdf — Asness, Frazzini & Pedersen, *Quality Minus Junk* | 2026-08-03 | peer-reviewed WP (AQR) | WebFetch, PDF text extracted (1.1 MB) | Quality = safe, profitable, growing, well-managed, across four z-scored dimensions averaged. QMJ risk-adjusted return **0.66%/month US, 0.45% global**, t > 2.5, Sharpe ≈0.6–0.8. **Profitability and safety carry the largest premium; payout the weakest.** QARP = rank on quality *and* P/B jointly. |
| 3 | https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr513.pdf — NY Fed Staff Report 513, *Decomposing Short-Term Return Reversal* | 2026-08-03 | official (Fed staff report, 45 pp) | curl + pdfplumber, full text | Reversal profit is **first-month only**: 1.57%/mo (t=9.48) in month 1 → 0.40% (t=2.51) month 2 → "essentially zero" beyond. Loser-side reversal loads on **lagged Amihud illiquidity and realized volatility** (liquidity provision); winner-side loads on sentiment. Costs ≈ **80.5 bp/month**; 3-factor alpha 1.34%/mo → **net ≈0.54%/mo (t=3.9)**. Plain reversal is far weaker: monthly Sharpe **0.14 vs 0.52** for the decomposed version. |
| 4 | https://www.nber.org/system/files/working_papers/w11468/w11468.pdf — Campbell & Thompson, *Predicting the Equity Premium Out of Sample* | 2026-08-03 | peer-reviewed WP (NBER, 29 pp) | curl + pdfplumber, full text | Unrestricted: "only two out of four valuation ratios, three out of seven interest-rate variables … deliver positive out-of-sample R²". OOS R² are "positive, but **very small**". Sign + positivity restrictions lift 3/4 valuation ratios. **Book-to-market yields a negative OOS R² in all four panels.** |
| 5 | https://www.msci.com/documents/1296102/10203728/MSCIFactorCrowdingModel-factsheet.pdf/47de6db3-4fa9-88d5-21ee-db098e2a818f — MSCI Factor Crowding Model | 2026-08-03 | industry (vendor doc, 2 pp, complete) | WebFetch → pdfplumber, full text | Four crowding dimensions, verbatim: **VALUATIONS** ("Prices Bid Up"), **PAIRWISE CORRELATION & VOLATILITY** ("Stocks of a Factor moving together with wider swings"), **FACTOR REVERSAL** ("Strong recent performance promotes performance-chasing"), **SHORT INTEREST SPREAD** ("Bottom quintile heavily shorted relative to top"). Combined into one standardized score. |
| 6 | https://theideafarm.com/wp-content/uploads/2026/01/20260112CAPE.pdf — Ma, Marshall, Nguyen & Visaltanachoti, *CAPE Ratios and Long-Term Returns* (v. 12 Jan 2026) | 2026-08-03 | preprint (40 pp) | curl + pdfplumber, full text | Component (cap-weighted stock-level) CAPE: **average OOS R² = 56% for 10-year returns**, stronger in the post-1995 half, robust to a 1-year lag and to Bonferroni/BH data-mining adjustment. **Horizon is 10 years throughout** — the paper makes no short-horizon claim. |

### 2.2 Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312432 | paper landing | duplicate of source #2 |
| https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk | industry | duplicate of #2 |
| https://www.aqr.com/Insights/Datasets/Quality-Minus-Junk-Factors-Monthly | dataset | data, not readable prose |
| https://www.msci.com/research-and-insights/insights-gallery/which-factors-may-be-crowded | industry | gated |
| https://www.msci.com/data-and-analytics/factor-investing/crowding-solutions | industry | marketing page |
| https://www.msci.com/downloads/web/msci-com/data-and-analytics/crowding-solutions/MSCI-Crowding-Solutions-brochure.pdf | industry | superseded by #5 |
| https://alphaarchitect.com/cape-ratios/ | blog | **403 on both WebFetch and curl** |
| https://www.nber.org/system/files/working_papers/w10483/w10483.pdf — Goyal & Welch (2004/2008) | paper | the primary negative result; read via #4's and #6's extensive direct quotation rather than separately |
| https://link.springer.com/article/10.1007/s11156-006-7213-0 — *Mean Reversion of Short-Horizon Stock Returns: Asymmetry* | paper | paywalled |
| https://doi.org/10.3390/math13233889 — *Label-Driven Optimization of Trading Models* | paper | evaluates 8 look-ahead labelling windows (3–10 days) × 6 model types — directly relevant, not fetched (budget) |
| https://arxiv.org/pdf/2105.13727 — *Slow Momentum with Fast Reversion* | preprint | budget |
| https://pubmed.ncbi.nlm.nih.gov/41755202/ — OU-process mean-reversion + ML | paper | budget |
| https://www.benzinga.com/markets/market-summary/26/06/60001409/ — BofA Global FMS, Jun 2026 | press | **80% of fund managers crowded into global semis (record); "long Mag-7" a distant 12%** — the empirical anchor for 82.1's crowding framing |
| https://hedgeco.net/news/06/2026/hedge-funds-may-face-the-ai-crowding-risk.html | press | corroborates the above |
| https://arxiv.org/abs/2512.11913 | landing | same as #1 |

**Unique URLs collected: 21.**

### 2.3 Recency scan (last 2 years, 2024–2026)

Searched explicitly for 2024-2026 material on factor crowding, CAPE OOS predictability, and
crowded-momentum concentration. **Result: 3 new findings that materially change the design,
plus 1 that supersedes an older citation.**

1. **Lee (Dec 2025, arXiv:2512.11913) supersedes the naive "crowding gate" premise.** Crowding
   is real and measurable but **efficiently priced in the mean**; it is only informative about
   the **tail**. And the sign is factor-specific: crowded momentum shows *lower* crash risk
   (0.38×, p=0.006), crowded reversal *higher* (1.7–1.8×). Also documents that crowding
   **accelerated post-2015**, correlated with factor-ETF volume growth (ρ = −0.63).
2. **Ma et al. (Jan 2026) rehabilitates CAPE — but only at a 10-year horizon** (OOS R² 56%,
   stronger post-1995). Nothing in it supports valuation-based timing at 15–90 days.
3. **BofA Global Fund Manager Survey, June 2026**: semiconductor concentration at a record
   80% of managers — the highest single-trade crowding reading in the survey's history.
4. **Goyal, Welch & Zafirov (2024)**, cited in #6, re-examines the predictor zoo with new
   variables and reaches "qualitatively similar" negative conclusions for monthly/annual
   horizons — the 2008 negative result still stands where this engine operates.

### 2.4 Consensus vs debate

**Consensus:** (i) short-horizon reversal is real, concentrated in the first month, and
liquidity-driven on the loser side (#3); (ii) a quality composite earns a premium, driven
mainly by profitability and safety (#2); (iii) valuation ratios predict returns at *long*
horizons and barely at all at short ones (#4, #6).

**Live debate:** whether crowding is *tradeable*. MSCI (#5) markets it as actionable ("ride the
crowding wave", "alert to potential bubbles"); Lee (#1) tests exactly that and **fails** — and
Barroso, Edelen & Karehnke (2022), quoted in #1, go further: "crowding does not generate tail
risk when arbitrageurs rationally condition on feedback". Lee splits the difference: crowding
is tail information, not mean information, and its sign flips by factor family.

**Debate on Goyal-Welch:** #4 is the direct rebuttal — a real-time investor would impose sign
and positivity restrictions, and doing so turns most negative OOS R² positive. But even the
rebuttal concedes the R² are "very small".

### 2.5 Pitfalls from the literature, mapped to this repo

1. **Do not gate momentum off because momentum looks crowded.** #1 measures crowded momentum
   as having *0.38×* the crash probability. A gate that flattens the book when the momentum
   funnel is crowded is contradicted by the best 2025 evidence. If a crowding overlay is built,
   key it on **co-movement / turbulence**, which #3 independently links to the compensation for
   liquidity provision, not on "momentum has run".
2. **Do not build a CAPE-style valuation timer at this engine's horizon.** #6's 56% OOS R² is a
   10-year number. `holding_days` defaults to 90 and `mr_holding_days` to 15. #4's honest
   summary for short horizons is "positive, but very small", and **book-to-market is negative
   OOS in every panel** — which is the closest analogue to the `pb_ratio` this repo has.
3. **Cost-adjust the reversion target or the label lies.** #3: costs are 80.5 bp/month against
   a 134 bp/month gross alpha. `_compute_triple_barrier_label:762-765` already shifts barriers
   by `2 × transaction_cost_pct`; **`_compute_mean_reversion_label` does not** (`:1195`, `:1199`
   use raw price targets). Any new reversion label must carry the cost shift.
4. **Raw-percent thresholds are the classic degeneracy cause.** Fixed % gates fire on a
   high-vol name every week and on a low-vol name never, so the class mix swings with the
   cross-section's volatility rather than with the signal. Express every threshold in **σ
   units** — this repo already computes `daily_volatility` (`historical_data.py:96`) and
   `bb_pct_b` (`:121`), so σ-scaling is free.
5. **Payout is the weakest QMJ leg** (#2). `quality_score` weights all four dimensions equally
   (`historical_data.py:255`), so 25% of the score sits on the weakest input — and this repo's
   `fcf_yield` is *biased upward* because capex is approximated as 0 (`:180`). A QARP label
   should re-weight toward profitability + safety rather than reuse the flat composite.

---

## 3. Application to pyfinagent — three implementable candidates

All three are **forward-looking** (§1.6), **long-only-safe** (`-1` means "do not buy / exit",
never "short"), and use **σ-scaled** thresholds so the class mix cannot collapse.
Shared helper, computable per row from existing features:

```
sigma_h = fv["daily_volatility"] * sqrt(H)          # historical_data.py:96 ; H = horizon in days
rt_cost = 2 * self.trader.transaction_cost_pct / 100 # mirrors backtest_engine.py:762-763
```

`daily_volatility` is present in the feature vector but **absent from `_NUMERIC_FEATURES`**
(`backtest_engine.py:42-52`), so a label method may read it while the model cannot — no leakage
concern either way, but adding it to the list is a separate decision.

### 3.1 Candidate A — `stretch_regime` (valuation/crowding-stretch REGIME GATE + the (d) overlay)

**Rule.** A σ-scaled triple barrier whose barrier *width* is modulated by a **date-level stretch
state**, so the same forward move earns a different label in a calm vs a stretched tape.

```
# per-date scalar, computed once per entry_date and memoised on self._stretch_cache[entry_date]
turb  = self.data_provider.compute_turbulence_index(entry_date, self._current_universe)
                                           # historical_data.py:281 — currently DEAD CODE
stretch = percentile_rank(turb, trailing 252 daily values)     # in [0,1]

# per-row
z_up = 1.0 + 0.6 * stretch        # stretched tape ⇒ demand a bigger up-move to call it +1
z_dn = 1.0 - 0.3 * stretch        # …and call a smaller down-move -1
walk forward H = self.holding_days days:
    +1 if price >= entry * (1 + z_up * sigma_h + rt_cost)
    -1 if price <= entry * (1 - z_dn * sigma_h + rt_cost)
     0 on time expiry
```

**Features used — all confirmed to exist:** `daily_volatility` (`historical_data.py:96`),
`price_at_analysis` (`:73`), `cache.cached_prices`, and
`compute_turbulence_index` (`:281`). Optional secondary regime inputs, all confirmed present in
the macro join (`:268-274`): `yield_curve_spread` (T10Y2Y) and `treasury_10y` (DGS10) — **use
only these two**, because they are FRED *daily* series and therefore immune to the
`realtime_start` vintage look-ahead described in §1.5. **Do not** use `cpi_yoy` (it is a raw
index level, §1.3) and **do not** design against CAPE/ERP/VIX/breadth — none exist here.

**The (d) hedged / cash-timing overlay lives here.** The engine is long-only with no cash or
hedge instrument in the label space, so "go to cash" has to be expressed *through* the label:
as `stretch → 1`, `z_up` rises, fewer rows earn `+1`, the classifier emits fewer BUY signals,
and `BacktestTrader` holds cash by default. That is the overlay, and it needs no new
instrument. **Requires the 1-line `self._current_universe = universe_tickers` stash in
`_run_window` (§1.4).**

**Non-degeneracy risk and mitigation.** Risk: at extreme `stretch`, `z_up = 1.6σ` could starve
the `+1` class on quiet names. Mitigation: `z_up ∈ [1.0, 1.6]` and `z_dn ∈ [0.7, 1.0]` are
hard-bounded by construction, and a ±1σ barrier over H days is roughly a 30/40/30 split by
definition of σ. Assert the fixture histogram; if any class exceeds 95%, the σ estimate is the
suspect (check `len(daily_returns) > 20` at `historical_data.py:95` — below that,
`daily_volatility` is **absent**, and the label must return `None`, not fall back to a raw %).

**Citations.** Barrier-in-σ-units: López de Prado AFML Ch.3 (already the repo's stated basis at
`:750`). Regime keyed on co-movement/turbulence rather than on "momentum is crowded":
Lee 2025 (arXiv:2512.11913) — crowded momentum has **0.38×** crash risk (p=0.006) while crowded
reversal has 1.7–1.8×, so a momentum-crowding gate would push the wrong way; and NY Fed SR513,
which finds reversal profits load on **realized volatility and Amihud illiquidity**, i.e. the
turbulence channel is the one carrying compensation. MSCI's "pairwise correlation & volatility"
dimension is precisely what the Mahalanobis turbulence index measures.

### 3.2 Candidate B — `qarp` (quality at a reasonable price, defensive tilt, long-only)

**Rule.** A σ-scaled forward barrier whose **asymmetry** is set by a QARP score — high QARP
makes `+1` easier to earn and `-1` harder. Every row gets a label, so the tilt cannot starve a
class.

```
# quality leg — re-weighted toward the dimensions that carry the premium (Asness et al.)
q = 0.5 * profitability + 0.3 * safety + 0.2 * growth      # payout dropped, see below
     profitability from fv["roe"] (historical_data.py:165) and fv["profit_margin"] (:167)
     safety        from fv["debt_equity"] (:163) and fv["annualized_volatility"] (:92)
     growth        from fv["revenue_growth_yoy"] (:209)
# value leg — sigmoid on P/B centred at 3.0, mirroring the existing idiom at :1233
v = 1 / (1 + exp((fv["pb_ratio"] - 3.0) / 1.5))            # fallback: pe_ratio centred 20 (:1238)
qarp = 0.5 * q + 0.5 * v                                    # in [0,1]

k_up = 1.25 - 0.5 * qarp        # high QARP ⇒ lower up-barrier
k_dn = 0.75 + 0.5 * qarp        # high QARP ⇒ deeper down-barrier before we call it -1
walk forward H = self.holding_days:
    +1 if price >= entry * (1 + k_up * sigma_h + rt_cost)
    -1 if price <= entry * (1 - k_dn * sigma_h + rt_cost)
     0 on time expiry
```

**Features used — all confirmed:** `roe`, `profit_margin`, `debt_equity`,
`annualized_volatility`, `revenue_growth_yoy`, `pb_ratio`, `pe_ratio`, `daily_volatility`,
`price_at_analysis`. **Deliberately NOT reusing `quality_score`** (`historical_data.py:255`):
it weights all four QMJ dimensions equally, and per Asness et al. payout carries the *weakest*
premium while this repo's payout leg is built on an `fcf_yield` that assumes capex = 0
(`:178-180`) — an upward-biased input on the weakest dimension. Recomputing the three strong
legs in the label method is ~15 lines and avoids inheriting that bias. **Do not** design
against gross-profits-over-assets (Novy-Marx): there is no `gross_profit`/COGS field.

**Non-degeneracy risk and mitigation.** Risk 1: `pb_ratio` is missing whenever
`total_equity <= 0` or shares are absent (`:169`) — for a book-negative name the `v` leg
vanishes. Mitigation: the P/E fallback already used at `:1238`, then a neutral `v = 0.5`; only
return `None` when *both* are absent **and** `daily_volatility` is absent. Risk 2: `k_up` and
`k_dn` are bounded to `[0.75, 1.25]`, so the barrier is never further than 1.25σ — the class
mix stays near the σ-implied 30/40/30 regardless of the QARP distribution. Risk 3 (real):
`roe`/`profit_margin`/`debt_equity` all depend on quarterly fundamentals being present; on the
fixture, count how many rows return `None` before judging the class balance.

**Citations.** Asness, Frazzini & Pedersen, *Quality Minus Junk*: QMJ earns 0.66%/mo (US) and
0.45%/mo (global), t > 2.5; profitability and safety carry the largest premium, payout the
weakest; "AQR ranks on both quality and P/B to generate quality at a reasonable price". The
value leg is deliberately a *tilt*, not a gate, because Campbell & Thompson find book-to-market
delivers a **negative OOS R² in all four panels** — a hard cheapness screen is the one
valuation construct the OOS literature specifically does not support.

### 3.3 Candidate C — `reversion_sigma` (mean reversion on overextension — the repaired version)

**Rule.** Three changes vs the incumbent: σ-units instead of raw %, a cost-adjusted reversion
target, and `None` (drop) instead of `0` on the no-signal path.

```
z = fv["bb_pct_b"]              # historical_data.py:121 — (P - lower)/(upper - lower), ±2σ_20 bands
                                # z > 1 ⇒ above the upper band ; z < 0 ⇒ below the lower band
H = self.mr_holding_days        # default 15, optimizer range (5, 30) — keep it
sigma_h = fv["daily_volatility"] * sqrt(H)

if z is None or fv["daily_volatility"] is None:      return None
if 0.05 <= z <= 0.95:                                return None   # NO SIGNAL ⇒ DROP, not class 0
overextended_up   = z > 0.95      # ≈ at/through the upper 2σ band
overextended_down = z < 0.05

walk forward up to H trading days:
    if overextended_down and price >= entry * (1 + 0.75 * sigma_h + rt_cost):  return  1
    if overextended_up   and price <= entry * (1 - 0.75 * sigma_h + rt_cost):  return -1
return 0        # signalled but did NOT revert — an informative class, not a filler
```

**Features used — all confirmed:** `bb_pct_b` (`historical_data.py:121`, **built today and
unused by any consumer**), `daily_volatility` (`:96`), `price_at_analysis` (`:73`),
`cache.cached_prices`, `self.mr_holding_days`, `self.trader.transaction_cost_pct`. Optional
per SR513's liquidity-provision result: gate the *long/oversold* branch on
`amihud_illiquidity` (`:136`) being above its trailing median — the paper finds the loser-side
profit loads positively and significantly on lagged Amihud.

**Non-degeneracy risk and mitigation.** Dropping the no-signal rows converts the problem from
"95% class 0" into "too few rows". A ±2σ Bollinger band is breached on roughly 5% of stock-days
unconditionally, and §1.2 shows the momentum-ranked candidate screen makes the *oversold* side
rarer still. Three mitigations, in order of preference: (i) widen the signal band to
`z > 0.85 / z < 0.15` (≈1.5σ) and measure; (ii) raise `top_n_candidates` for this strategy so
the screen is less selective; (iii) if the fixture still yields too few rows, keep the
no-signal rows as class `0` but **report the histogram honestly** rather than tuning until it
looks balanced. The `>95%` fixture gate should be applied to the *retained* rows and paired
with a **minimum-row-count** assertion, or a 3-row label set trivially passes the balance test.

**Citations.** NY Fed Staff Report 513 (*Decomposing Short-Term Return Reversal*): the profit
"accrues mainly during the first month after portfolio formation" — 1.57%/mo (t=9.48) in month
1, 0.40% (t=2.51) in month 2, "essentially zero" thereafter, which **validates
`mr_holding_days ∈ [5, 30]` and forbids extending the horizon**; costs of 80.5 bp/mo against a
1.34%/mo alpha leave 0.54%/mo (t=3.9), which is why the cost shift is mandatory; loser-side
profits load on lagged Amihud illiquidity and realized volatility. Note the same paper's
warning that *plain* price reversal is much weaker than the decomposed version (monthly Sharpe
0.14 vs 0.52) — this candidate is the plain kind, so expect the modest end of the range.

### 3.4 Mean-reversion verdict (the claim the caller asked me to prove or disprove)

**Verdict: the "~all-neutral, 0 trades" claim is CREDIBLE and the mechanism is identified — but
the commonly assumed cause (thresholds too tight / wrong units) is DISPROVEN.** `sma_50_distance`
is a fraction (`historical_data.py:105`), so `-0.05` and `0.10` mean −5% and +10% as intended,
and the reversion target `entry × (1 + |sma_dist|/2)` (`:1195`) is a ~4% move, not an absurd
one. The real drivers, in order of weight: **(1) the momentum-ranked candidate screen
structurally excludes oversold names** (`candidate_selector.py:180-215` rewards positive SMA
distance and penalises RSI extremes in both directions); (2) the conjunctive stage-1 gate needs
two tail conditions at once; (3) the overbought threshold is 2× stricter than the oversold one;
(4) **stage 2 fails to class `0`, not to `None`** (`:1206`), so every signalled-but-unrealised
row inflates the neutral class. Two of the three return paths yield `0`, and `0` is the only
value reachable without a forward price move.

This remains a **claim to measure, not to assert** — the proof is one `collections.Counter`
over the fixture's labels plus a count of rows surviving each stage. 82.3 should record the
per-stage funnel (rows in → rows passing stage 1 → rows returning ±1), not just the final
histogram; only the funnel distinguishes cause (1) from cause (2).

---

## 4. Research Gate Checklist

Hard blockers:
- [x] ≥5 authoritative external sources READ IN FULL — **6** (1 preprint-adversarial, 1 Fed
      staff report, 1 NBER WP, 1 AQR WP, 1 vendor doc, 1 finance preprint)
- [x] 10+ unique URLs total — **21**
- [x] Recency scan (2024–2026) performed and reported — §2.3, 4 findings
- [x] Full papers/pages read, not abstracts — pdfplumber/curl full-text on all 6
- [x] file:line anchors for every internal claim — §1

Soft checks:
- [x] Internal exploration covered `backtest_engine.py`, `historical_data.py`,
      `candidate_selector.py`, `cache.py`, `quant_optimizer.py`
- [x] Contradictions noted — §2.4 (MSCI vs Lee; Goyal-Welch vs Campbell-Thompson)
- [x] Per-claim citation
- [ ] **Gap**: `alphaarchitect.com/cape-ratios/` returned 403 on both WebFetch and curl; the
      "simple adjustment improves CAPE predictability" claim rests on source #6 alone.
- [ ] **Gap**: Goyal & Welch (2008) itself was read through #4's and #6's direct quotation, not
      fetched separately.

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 15,
  "urls_collected": 21,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "summary": "Label methods are (self, ticker, entry_date) -> {+1,0,-1,None}; None DROPS the row (backtest_engine.py:674-676) and no class-balance guard exists anywhere. The interface is per-row, but the outer loop is already per-date (:657-666) and caches are per-date memoised, so a date-level regime scalar is implementable; the ONE blocker is that the universe list is not on self (needs a 1-line stash in _run_window:388). A dead Mahalanobis turbulence index already exists at historical_data.py:281. Available: pe/pb/fcf_yield/dividend_yield valuation; roe/profit_margin/debt_equity/revenue_growth/quality_score quality; six FRED macro series joined per-date. NOT available: CAPE, ERP, VIX, breadth, gross profits, short interest; cpi_yoy is a raw index level, not a rate. mean_reversion degeneracy CONFIRMED-plausible but NOT a units bug (sma_50_distance is a fraction) -- causes are the momentum-ranked candidate screen, a conjunctive tail gate, asymmetric thresholds, and stage 2 failing to class 0 rather than None. Three forward-looking sigma-scaled candidates designed (stretch_regime with the cash-timing overlay folded in, qarp, reversion_sigma). Key adversarial finding: crowded MOMENTUM has 0.38x crash risk (Lee 2025), so a momentum-crowding gate pushes the wrong way -- key the regime on turbulence instead.",
  "brief_path": "handoff/current/research_brief_82.2.md",
  "gate_passed": true
}
```

