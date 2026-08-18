# Research Brief -- step 86.116

**Tier:** moderate (caller-specified). **Audit-class:** NO (coverage reported for information only).

**Topic:** duplicate-row contamination in stored financial time series -- detection,
de-duplication-on-read vs table repair, and the effect of positional (iloc-based)
lookbacks and interleaved zero-returns on momentum and realized-volatility
estimates in backtests.

---

## ENVELOPE (born inert -- phase-86.37; updated in place as sources land)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 19,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "gate_passed": true
}
```

Gate logic: `external_sources_read_in_full = 6 >= 5` AND `recency_scan_performed == true`
AND all hard-blocker checklist items satisfied AND `coverage.audit_class == false`
(so `coverage.dry` is informational and not required) -> `gate_passed: true`.
The two unchecked items in the checklist are **soft** checks, recorded as gaps.

---

## Status log (write-first, append-only)

- [t0] Brief created; envelope born inert. Read `.claude/agents/researcher.md` +
  `.claude/rules/research-gate.md` in full.
- [t1] Internal: `cache.py` (777L) + `screener.py` (759L) read in full; targeted reads of
  `historical_data.py`, `candidate_selector.py`, `data_server.py`, `data_ingestion.py`.
  HEADLINE: **zero** `drop_duplicates`/`duplicated` calls exist anywhere in `backend/`.
- [t2] Sources 1-6 fetched in full; year-scoped recency variants run.
- [t3] Local pandas/numpy experiment quantified the bias.
- [t4] **Live BigQuery census run: the contamination is REAL and already in the table.**
- [t5] Duplicate multiplicity / value-divergence / per-year profile measured.

---

## Search queries run (three-variant discipline)

| # | Query | Variant |
|---|---|---|
| 1 | duplicate rows financial time series database detection deduplication data quality | year-less canonical |
| 2 | pandas drop_duplicates duplicate index financial data pitfalls backtest | year-less canonical |
| 3 | realized volatility estimation spurious zero returns stale prices bias downward | year-less canonical |
| 4 | price staleness zero returns volatility estimation bias 2026 | current-year frontier |
| 5 | duplicate records data quality impact machine learning time series 2025 2024 study | last-2-year window |

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://arxiv.org/html/2410.07607 | 2026-08-18 | paper (preprint; JASA 2026) | WebFetch, arXiv native HTML | Staleness -> zero returns biases co-volatility **downward** by `phi(x,y)=(1-x)(1-y)/(1-xy)`, which "lies strictly between zero and one"; correction is inverse-probability weighting `V*=V/phi`. Without correction "integrated volatility errors roughly triple". |
| 2 | https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html | 2026-08-18 | official doc | WebFetch | Verbatim: **"Indexes, including time indexes are ignored."** `drop_duplicates()` is VALUE-keyed, not date-keyed. |
| 3 | https://pandas.pydata.org/docs/user_guide/duplicates.html | 2026-08-18 | official doc | WebFetch | `.loc` on a duplicated label returns a **Series, not a scalar** -- dimensionality silently changes. `Index.is_unique` / `Index.duplicated()` are the detection primitives; `allows_duplicate_labels` is experimental and "many methods fail to propagate" it. |
| 4 | https://arxiv.org/html/2511.10964 | 2026-08-18 | paper (preprint, 2025) | WebFetch | **[ADVERSARIAL]** Injecting 30%/50% duplicate *records* **improved** most credit-risk models (LDA F1 0.9675, "an improvement of 17% wrt the model trained on the original, uncorrupted dataset"); attributed to "a regularizing effect". |
| 5 | https://arxiv.org/html/2504.00638v1 | 2026-08-18 | paper (preprint, 2025) | WebFetch | **[ADVERSARIAL, qualified]** Standard CIFAR-10 models *gain* from duplication (70.72% -> 75.11% at 60%), but adversarially-trained models **lose** 41.16% -> 20.90%. Test/train duplication is a distinct problem: "there is a 9%-14% drop in accuracy when they removed repeated images between test set and train set." |
| 6 | https://www.federalreserve.gov/pubs/ifdp/2007/905/ifdp905.htm | 2026-08-18 | official (Federal Reserve IFDP) | WebFetch | **[ADVERSARIAL on direction]** "A large fraction of the observed high-frequency returns in both markets under study is equal to zero" (~90% FX / ~92% Treasury at 1s), yet microstructure noise is "generally believed to **elevate** estimates of integrated volatility" -- the opposite sign to staleness. |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.tandfonline.com/doi/full/10.1080/01621459.2026.2634436 | journal (JASA 2026) | Paywalled; same paper as read-in-full #1 (preprint used instead) |
| https://papers.ssrn.com/sol3/Delivery.cfm/5024285.pdf?abstractid=5024285 | preprint mirror | Duplicate of #1 |
| https://pubsonline.informs.org/doi/10.1287/mnsc.2019.3527 | journal (Mgmt Sci, "Zeros") | Paywalled |
| https://www.tandfonline.com/doi/abs/10.1080/07350015.2021.1999821 | journal (JBES) | Paywalled abstract only |
| https://academic.oup.com/jrsssc/advance-article/doi/10.1093/jrsssc/qlag026/8687456 | journal (JRSS-C) | Adjacent (zero-inflated SV for inflation), not price series |
| https://www.sciencedirect.com/science/article/pii/S0306437925000341 | journal (2025 tabular DQ) | Paywalled |
| https://www.sciencedirect.com/science/article/abs/pii/S0304407612000127 | journal (jump-robust vol) | Paywalled abstract |
| https://www.sciencedirect.com/science/article/abs/pii/S0378426611000860 | journal (idiosyncratic vol) | Paywalled abstract |
| https://public.econ.duke.edu/~get/browse/courses/201/spr11/DOWNLOADS/VolatilityMeasures/SpecificlPapers/hansen_lunde_forecasting_rv_11.pdf | paper (PDF) | Binary PDF; budget spent on higher-relevance sources |
| https://arxiv.org/pdf/2108.05935 | paper (Data Quality Toolkit) | PDF form; superseded by #4 for the duplicate-injection question |
| https://arxiv.org/pdf/2310.19992 | paper (realized correlation) | Tangential (betas, not duplicates) |
| https://arxiv.org/html/2506.07928v1 | paper (RV forecasting) | Forecasting, not contamination |
| https://repub.eur.nl/pub/7582/ei2006-10.pdf | paper (realized range) | Tangential estimator design |
| https://arxiv.org/pdf/2011.06909 | paper (multivariate SV) | Tangential |
| https://ceur-ws.org/Vol-2038/paper3.pdf | workshop paper (dup detection) | Record-linkage focus, not time series |
| https://docs.cloud.google.com/bigquery/docs/table-constraints | official vendor doc | **FETCH FAILED both ways** -- WebFetch 301s to `docs.cloud.google.com`, and `curl -sL` + tag-strip yielded only **229 chars** (JS-rendered). See "New failure mode" below. |
| https://saturncloud.io/blog/how-to-drop-duplicated-index-in-a-pandas-dataframe-a-complete-guide/ | blog | Community tier; superseded by official doc #3 |
| https://ioflood.com/blog/pandas-drop-duplicates/ | blog | Community tier |
| https://dagshub.com/blog/mastering-duplicate-data-management-in-machine-learning-for-optimal-model-performance/ | blog | Community tier |

**URL accounting:** 25 unique URLs (6 read in full + 19 snippet-only). Re-derived by
enumerating the two tables above, not carried from search-result counts; duplicate
routes to the same paper (arXiv `abs`/`pdf` forms of 2410.07607) were collapsed to one
entry rather than counted separately.

---

## Recency scan (last 2 years, 2024-2026)

**Performed.** Queries 4 and 5 above were scoped to 2026 and to 2025/2024 respectively.
**Result: 3 new findings that complement (and one that partially contradicts) the older
canonical sources.**

1. The staleness-bias literature **advanced in the window and is now peer-reviewed**:
   arXiv:2410.07607 (Oct 2024) was published in *JASA* in **2026**
   (`10.1080/01621459.2026.2634436`). It supersedes the older simultaneous-zeros
   assumption -- prior work assumed "zero (or near-zero) returns occur simultaneously
   across all assets at each time stamp", which this relaxes via a staleness *factor*
   model with per-asset probabilities. This matters here because duplicate rows produce
   **asset-specific, non-simultaneous** zero returns -- exactly the case the older models
   excluded and the new one covers.
2. **2025 ML-duplication work runs against the intuition** that duplicates always harm
   (sources #4, #5). Both find duplicate *records* often act as a regularizer and can
   *raise* test accuracy. See "Consensus vs debate" -- this does **not** transfer to
   pyfinagent, and the reason is load-bearing.
3. No new work in the window changes the **pandas** semantics: `drop_duplicates()`
   ignoring the index is unchanged through pandas 3.0.x (verified against the installed
   pandas **3.0.1** in `.venv`).

---

## Key findings

1. **The contamination is not hypothetical -- it is already in the live table.** Measured
   2026-08-18 against `sunny-might-477607-p8.financial_reports.historical_prices`
   (`location=us-central1`):
   - `total_rows = 1,859,482`; `distinct (ticker,date,market) = 1,152,607`
   - **`DUPLICATE ROWS = 706,875` = 38.01% of all rows**
   - **61.3% of distinct keys (706,875 / 1,152,607) carry a duplicate**
   - `max_multiplicity = 2` (exactly two copies, never three)
   - **336 of 513 tickers affected (65.5%)**; span `2017-01-03 .. 2026-07-02`
2. **It is LEGACY, not an ongoing leak.** Per-year share of keys duplicated:
   2017 **90.5%**, 2018-2025 steady at **62-64.5%**, **2026 = 0.1% (68 keys)**. The
   write-side probe added in phase-75.9 (`data_ingestion.py:99-112`) is holding; the
   damage predates it. This is the single most design-deciding fact: a **one-time table
   repair is bounded and terminal**, whereas read-side dedup is the durable guard.
3. **`drop_duplicates()` is the WRONG tool here, and the data proves it.** 394,719 of the
   706,875 duplicated keys (55.8%) have a `close` that differs between the two copies --
   so a value-keyed dedup would silently leave them. The differences are float noise, not
   re-adjustment: `p50 = p90 = p99 = 0.0%` gap, `max = 0.93%`. Combined with the pandas
   doc's **"Indexes, including time indexes are ignored"** (source #2), the correct
   primitive is date-keyed: `df[~df.index.duplicated(keep=...)]` (source #3).
4. **Measured effect of duplication on the exact helpers in scope** (local experiment,
   pandas 3.0.1, 252 synthetic bdays, seed 7):

   | series | rows | zero-return share | ann. vol | momentum_1m | momentum_3m | rsi_14 |
   |---|---|---|---|---|---|---|
   | clean | 252 | 0.00% | **0.2194** | -9.885 | -14.870 | 17.11 |
   | every bar duplicated | 504 | 50.10% | **0.1557** | -10.598 | -16.106 | 13.00 |
   | only last 40 bars duplicated | 292 | 13.75% | **0.2040** | -10.598 | -16.106 | 13.00 |

   Volatility ratio full-dup/clean = **0.7093**, against the closed-form prediction
   `sigma_obs/sigma_true -> sqrt(n/(n+m)) = 1/sqrt(2) = 0.7071` -- i.e. **volatility is
   understated by ~29%** under full duplication. This is the same *direction* the
   staleness literature predicts (source #1).
5. **Positional lookbacks and distributional statistics fail differently, and this is the
   most actionable asymmetry.** In row 3 above, duplicating only the most recent 40 bars
   corrupts `momentum_1m`, `momentum_3m` and `rsi_14` to **exactly the same values** as
   duplicating the entire history, while annualized vol is barely moved (0.2040 vs
   0.1557). Because `_pct_change` uses `series.iloc[-periods-1]`, an `iloc` lookback is
   corrupted by **recent** duplicates only, in proportion to the local duplicate density;
   `std()`-type statistics are corrupted in proportion to the **global** duplicate rate.
   A repair that fixes only recent data would restore momentum and leave vol wrong; a
   sampling-based audit that looks only at the tail would find momentum broken and
   declare vol fine.
6. **`.loc` slicing silently inflates row counts.** Measured: a one-month `.loc` slice
   returned **21 rows clean vs 42 duplicated**. This is `cache.py:560` exactly. Per source
   #3, `.loc` on a duplicated label also changes the *return type* (Series, not scalar).

---

## New failure mode worth recording (tooling)

The researcher memory note `feedback_gcloud_docs_fetch.md` says cloud.google.com is
JS-rendered and prescribes `curl -sL` + tag-strip as the workaround. **That workaround no
longer works on the new `docs.cloud.google.com` host**: `curl -sL` on
`/bigquery/docs/table-constraints` returned **229 characters** of extractable text. The
BigQuery "primary keys are not enforced" fact is therefore recorded here as **NOT
independently verified from the vendor doc in this session** -- it is instead evidenced
by the repo's own behaviour (the table demonstrably holds 706,875 duplicate keys), which
is stronger evidence than the prose anyway.

---

## Internal code inventory (file:line anchors)

| File:line | Role | Status |
|---|---|---|
| `backend/backtest/cache.py:254-257` | `preload_prices`: `group.drop(columns=["ticker"]).set_index("date").sort_index()` | **NO DEDUP.** Duplicate BQ rows pass verbatim into `_prices_full` |
| `backend/backtest/cache.py:560` | `cached_prices` fast path `full.loc[start_date:end_date]` | Returns all duplicate labels; measured 42 rows where clean gives 21 |
| `backend/backtest/cache.py:592-595` | `cached_prices` BQ-fallback `set_index("date").sort_index()` | **NO DEDUP** -- contamination on *both* read paths, so a one-sided fix is one seam short |
| `backend/backtest/cache.py:758` | `cached_macro` fallback `ROW_NUMBER() OVER (PARTITION BY series_id ORDER BY date DESC) ... rn=1` | The only dedup-shaped SQL on any cache read path; **macro only**. Prior art for the SQL idiom |
| `backend/backtest/data_ingestion.py:108-112` | `_get_existing_price_dates`: `SELECT DISTINCT ticker, date` | Sole duplicate defence; **write-side + advisory** (BQ enforces no PK). Working as of 2026 (0.1% dup rate) |
| `backend/backtest/data_ingestion.py:99-106` | phase-75.9 docstring: *"...producing duplicate (ticker,date) bars that distort features/MTM/Sharpe downstream"* | **The risk was already named in-repo**; only the fail-closed re-raise shipped -- no read-side guard, and no backfill of pre-existing damage |
| `backend/tools/screener.py:626-632` | `_pct_change` -> `series.iloc[-periods-1]` | **POSITIONAL lookback.** 21 *rows* != 21 trading days under duplication |
| `backend/tools/screener.py:195-197` | `momentum_1m/3m/6m` | `momentum_6m = _pct_change(close, len(close)-1)` is endpoint-to-endpoint and therefore **duplicate-invariant**; 1m/3m are not |
| `backend/tools/screener.py:203-204` | `close.pct_change().dropna()` -> `std()*sqrt(252)` | `dropna()` does not remove zeros; measured -29% vol under full duplication |
| `backend/tools/screener.py:635-647` | `_compute_rsi`: `series.diff()` + 14-row rolling mean | Zero deltas dilute `avg_gain`/`avg_loss`; measured 17.11 -> 13.00 |
| `backend/tools/screener.py:161-165` | `validate_ohlcv(...)` phase-50.5 quality gate | **Confirmed does NOT dedupe** (`inspect.getsource`: no `duplicated`/`drop_duplicates`/`is_unique`; 84 lines). It is the natural insertion point for the screener path |
| `backend/tools/screener.py:169` | `close = ticker_data["Close"]` from **yfinance**, not BQ | The screener's duplicate exposure is a *separate source* from the backtest's BQ exposure -- do not assume one fix covers both |
| `backend/backtest/historical_data.py:53` | `get_point_in_time_prices` -> `cache.cached_prices(...)` returned unmodified | Pure pass-through; inherits contamination |
| `backend/backtest/historical_data.py:108-111` | `momentum_1m/3m/6m/12m = _pct_change(close, 21/63/126/252)` | Same positional class, on the **feature vector that trains the model** |
| `backend/backtest/historical_data.py:124-132` | `annualized_volatility`, `daily_volatility` | `daily_volatility` sets **triple-barrier width** (AFML Ch.3): understated vol -> tighter barriers -> shifted label distribution |
| `backend/backtest/historical_data.py:385-389` | `compute_turbulence_index`: per-ticker `pct_change()` -> `pd.DataFrame(returns_dict).dropna()` | Cross-sectional align across tickers with **non-unique index labels** |
| `backend/backtest/historical_data.py:565` | `(daily_ret.loc[common] / dollar_vol.loc[common]).mean()` (Amihud illiquidity) | Label-based division on a possibly-duplicated index |
| `backend/backtest/candidate_selector.py:52-73` | `screen_at_date` re-implements the same momentum/RSI/vol block | **Duplicate logic** of `screener.py` -- a fix must land in BOTH or they drift |
| `backend/backtest/backtest_trader.py:188-201` | `mark_to_market(date, prices: dict[str,float])` | **NOT directly affected** -- NAV is built from a dict, one entry per business day, so portfolio Sharpe is not diluted by row count. Contamination reaches the gate through the *features/labels*, not the Sharpe formula |
| `backend/agents/mcp_servers/data_server.py:99-115` | `get_prices`: `for _, row in df.iterrows(): row["date"]` | **LATENT BUG, PROVEN, separate from duplicates:** `cached_prices` returns `date` as the **index**, so `row["date"]` raises. Reproduced: `KeyError "'date'"` -> swallowed at `:119` -> the tool returns `{"prices": [], "error": "'date'"}` for **every** non-empty frame |
| repo-wide grep | `drop_duplicates` / `.duplicated(` in `backend/` | **ZERO occurrences.** All 3 hits repo-wide are in `scripts/qa/` (86.59 rank-stability tooling + its mutation matrix) |
| `backend/config/settings.py:59` | `bq_dataset_reports = "financial_reports"` | `historical_prices` lives in `financial_reports` (**us-central1**), NOT `pyfinagent_data` -- a census query against the wrong dataset 404s |

---

## Consensus vs debate (external)

**Consensus.** Stale/repeated prices produce excess zero returns, and excess zero returns
bias volatility **downward** (#1). The magnitude is a multiplicative factor strictly
inside (0,1) that vanishes as staleness -> 0 and is unrecoverable as staleness -> 1.
pandas' own docs are unambiguous that `drop_duplicates()` ignores the index (#2) and that
duplicate labels change `.loc`'s return dimensionality (#3).

**Genuine debate 1 -- sign of the bias.** The Fed IFDP paper (#6) says microstructure
noise *elevates* integrated-volatility estimates at high frequency, the opposite sign to
#1's staleness effect. Both are right: they are two competing forces at intraday
frequencies. **This resolves cleanly for pyfinagent and in our favour as a diagnosis**:
duplicate rows in a *daily* bar series carry no bid-ask bounce, so there is no offsetting
upward noise term -- only the downward zero-return term applies. The measured 0.7093
ratio matching `1/sqrt(2)` to 3 decimals is direct evidence that the downward term is
acting alone.

**Genuine debate 2 -- do duplicates even hurt?** Sources #4 and #5 both find duplicate
*records* frequently *improve* ML test accuracy (regularization). **This does not transfer
here, and the reason is the crux of the step.** In #4/#5 duplication changes the
*composition of the sample* while each row's feature values stay correct. In pyfinagent
duplication corrupts the *feature values themselves* -- `momentum_1m`, `rsi_14`,
`annualized_volatility` are computed **from** the duplicated series, so this is
measurement error in X, not resampling. #5 also supplies the qualifier that matters:
robustness-trained models *degraded* sharply (41.16% -> 20.90%), and train/test
duplication caused a 9-14% accuracy drop -- and a walk-forward backtest with duplicated
bars in both the train window and the test window is precisely that leakage geometry.

---

## Pitfalls (from literature + measurement)

1. **Reaching for `drop_duplicates()`.** Value-keyed, index-ignoring (#2); would miss the
   394,719 keys whose two copies differ by float noise. Use `~index.duplicated()`.
2. **Assuming duplicates are exact.** 55.8% are not (though the divergence is <=0.93%).
   Which copy you keep (`keep="first"` vs `"last"`) is therefore a real choice, not a
   formality.
3. **Assuming a whole-table audit generalises.** Positional lookbacks break on *recent*
   duplicates; distributional statistics break on the *global* rate. A tail-only or
   head-only sample will mis-diagnose one of the two (finding 5).
4. **Fixing one read path.** `cache.py` has two (`:254` preload and `:592` fallback), and
   the screener has a third, yfinance-fed one.
5. **Treating this as a Sharpe-formula bug.** It is not: NAV comes from a per-business-day
   dict (`backtest_trader.py:188`). It reaches the DSR/PBO gate through corrupted
   features and mis-scaled triple-barrier labels.
6. **Repairing the table without a read-side guard** (or vice versa). The table repair is
   terminal for the 2017-2025 damage; the read-side guard is what makes recurrence
   non-fatal. They are complements.
7. **Trusting a `SELECT DISTINCT` write-side probe as an invariant.** BigQuery enforces no
   uniqueness constraint; the probe is advisory and, as the 2017 90.5% figure shows,
   was absent or bypassed historically.

---

## Application to pyfinagent

- **Detection** is a one-liner and should become a standing check:
  `df.index.has_duplicates` / `df.index.duplicated().sum()` at `cache.py:256` and
  `cache.py:595`, plus the BQ census SQL used above (`COUNT(*)` vs
  `COUNT(DISTINCT ticker|date|market)`).
- **De-dup-on-read vs table repair is not either/or.** The measured profile argues for
  **both, in this order**: (a) read-side `~index.duplicated(keep="last")` at the two
  `cache.py` seams -- cheap, immediately correct, and protects every consumer including
  `historical_data.py`, `candidate_selector.py` and `data_server.py`; then (b) a one-time
  table repair, which is bounded (multiplicity is exactly 2, the window is closed at
  2026-07-02) and removes the 38% storage/scan tax.
- **Any re-baselining claim must be quantified, not assumed.** Every historical backtest
  and every optimizer trial in `optimizer_best.json` / `quant_results.tsv` was computed on
  features derived from a series where ~61% of keys were doubled. Volatility features were
  understated (up to ~29%), momentum features were computed over compressed windows, and
  triple-barrier widths were consequently too tight. Whether the resulting Sharpe/DSR is
  biased **up or down is not determined by this brief** -- it depends on how the
  classifier used the distorted features -- and should be *measured* by an A/B replay
  (deduped vs raw), not asserted.
- **`data_server.py:99-115` is an independent, proven defect** that this step's internal
  scope surfaced: the MCP `get_prices` tool cannot return a single price. It should be
  queued as its own masterplan step rather than folded into this one.
- **Do not fix `screener.py` and `candidate_selector.py` separately by hand** -- they are
  duplicate implementations of the same block, which is how they will drift.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**6**: 2 official
      pandas docs, 1 Federal Reserve IFDP, 3 arXiv preprints incl. one JASA-2026-published)
- [x] 10+ unique URLs total incl. snippet-only (**25**)
- [x] Recency scan (2024-2026) performed + reported (3 findings; section above)
- [x] Full pages read, not abstracts, for the read-in-full set (arXiv native HTML per the
      `/html/` chain; no `arxiv.org/pdf/` fetched)
- [x] file:line anchors for every internal claim (inventory table above)

Soft checks:
- [x] Internal exploration covered every module in the caller's scope, plus
      `data_ingestion.py`, `price_quality.py`, `backtest_trader.py`, `settings.py`
- [x] Contradictions / consensus noted (two genuine debates, both resolved with reasons)
- [x] Claims cited per-claim, with measured values distinguished from cited values
- [ ] **Gap:** the BigQuery "constraints are not enforced" vendor doc could not be fetched
      by either route (see "New failure mode"); the claim rests on observed table state
- [ ] **Gap:** no A/B replay of a backtest with vs without dedup was run -- out of scope
      for RESEARCH (that is a GENERATE/measurement task), but it is the decisive experiment
</content>
