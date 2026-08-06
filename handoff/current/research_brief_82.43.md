# Research Brief -- step 82.43 (macro feature-drop silent degradation)

> **SUPERSEDED 2026-08-06 by Main, during the 82.43 GENERATE phase.** This
> brief reports the macro-present/absent feature widths as `35 / 29`. Main
> accepted that pair without deriving it, and the 82.43 Q/A refuted it.
> Re-derived independently by both Main and the Q/A on the step's own
> fixtures: **macro-present 18, macro-absent 12** (`len(_NUMERIC_FEATURES)`
> is 37). The DELTA of 6 -- exactly the six macro features -- reproduced in
> every configuration, so this brief's substantive finding is sound and only
> the absolute pair was wrong. Annotated, not rewritten: a research brief is
> a dated record of what was believed at gate time.


TIER: moderate | AUDIT_CLASS: false | 2026-08-06 | STATUS: internal half COMPLETE, external half in progress

Objective: `historical_data.py` guards the macro block with a bare `if macro:`. When
`cache.cached_macro(cutoff)` returns `{}` the builder silently drops all six macro
features and returns a shorter vector. 82.13 fixed the layer above (refusal detection +
`data_availability`); 82.21 shipped the analogous fundamentals fix one field over.

## Internal code inventory (all line numbers RE-DERIVED live 2026-08-06)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/backtest/historical_data.py` | 299-305 | the `if macro:` guard + the six macro keys | DEFECT SITE. Confirmed at **:299** (step text said "around 269"; 82.21 shifted it) |
| `backend/backtest/historical_data.py` | 62 | `macro = self.get_point_in_time_macro(cutoff_date)` | source of the dict |
| `backend/backtest/historical_data.py` | 46-48 | `get_point_in_time_macro` -> `cache.cached_macro` | thin passthrough, no guard |
| `backend/backtest/historical_data.py` | 64-74 | 82.21 seeds `fundamentals_available: False` at CONSTRUCTION | THE TEMPLATE (covers the early return at :77-78) |
| `backend/backtest/historical_data.py` | 150-169 | 82.21 rationale + the "do NOT add to `_NUMERIC_FEATURES`" ban | THE TEMPLATE |
| `backend/backtest/cache.py` | 651-731 | `cached_macro` -- 4 return paths | Q2 subject |
| `backend/backtest/cache.py` | 671-681 | fast-path per-series loop + PIT vintage predicate (:674-679) | partial-resolution site |
| `backend/backtest/cache.py` | 343-345 | `_MACRO_OUTCOMES` = warm/empty/refused_unparseable/refused_stale/loaded/unknown | |
| `backend/backtest/cache.py` | 346 | `_REFUSAL_OUTCOMES` = **only** `{refused_unparseable, refused_stale}` | **`"empty"` is NOT a refusal** -- see Q2 cause E4 |
| `backend/backtest/cache.py` | 394-543 | `preload_macro`, 5 return paths | 82.13 subject |
| `backend/backtest/cache.py` | 99-107 | `MACRO_PUBLICATION_LAG_DAYS` (DGS10/T10Y2Y=1d ... CPIAUCSL=50d, GDP=125d) | drives partial resolution |
| `backend/backtest/cache.py` | 128-156 | `_effective_vintage` = MIN(realtime_start, obs+lag); returns None on unparseable | fail-closed |
| `backend/backtest/backtest_engine.py` | 124-134 | `_NUMERIC_FEATURES` -- **37 entries**, the six macro names at :132-133 | |
| `backend/backtest/backtest_engine.py` | 366-411 | `_preload_macro_and_record` (82.13 template) | records `macro`/`macro_outcome`/`macro_detail`/`macro_rows` |
| `backend/backtest/backtest_engine.py` | 413-474 | `_preload_fundamentals_and_record` (82.21 template) | REFUSE-vs-RECORD split |
| `backend/backtest/backtest_engine.py` | 212-213 | `BacktestResult.data_availability` default `{"macro": True, "fundamentals": True}` | |
| `backend/backtest/backtest_engine.py` | 518-530 | the real call site; `data_availability=dict(_availability)` | the seam to extend |
| `backend/backtest/backtest_engine.py` | 923-925 | `df = pd.DataFrame(features_list)`; `feature_cols = [c for c in _NUMERIC_FEATURES if c in df.columns]` | **the silent-drop filter** |
| `backend/backtest/backtest_engine.py` | 943-954 | `train_medians = X.median()`; `.fillna(train_medians)`; `.fillna(0)` | the imputation |
| `backend/backtest/backtest_engine.py` | 950-952 | `_train_feature_medians` filters `pd.notna(v)` | all-NaN col NEVER persisted |
| `backend/backtest/backtest_engine.py` | 805-837 | `_build_predict_features`; `row = {f: fv.get(f, np.nan)}` (:826) | PREDICT asymmetry |
| `backend/backtest/analytics.py` | 847-849 | `report["data_availability"] = _availability` | the surfacing seam |
| `backend/backtest/fundamentals_coverage.py` | 1-327 | 82.21's derived-rule module (AST, snapshot, `window_is_covered`) | shape reference |
| `backend/tests/test_phase_82_13_preload_refusal_handling.py` | whole | 82.13 guard template | |
| `backend/tests/test_phase_82_21_fundamentals_coverage.py` | 195, 221, 230 | the THIRD leg + every-return-path + not-a-model-input guards | THE guard template |

## MEASUREMENTS (driven through the real production functions, not reasoned)

Script: `scratchpad/measure_82_43.py` -- monkeypatches only the BQ accessors
(`cache.cached_prices/cached_fundamentals/cached_macro`) and calls the REAL
`HistoricalDataProvider.build_feature_vector` and the REAL
`BacktestEngine._build_training_data`.

### M1 -- Q1: the two feature counts (criterion 2)

| macro state | `build_feature_vector` total keys | of `_NUMERIC_FEATURES` (37) present | real `_build_training_data` X.shape |
|---|---|---|---|
| present (6 series) | 45 | **35** | **(72, 35)** |
| empty `{}` | 39 | **29** | **(72, 29)** |
| MIXED (macro on half the rows) | -- | -- | **(72, 35)**, `n_nan=0` after imputation |
| all-null values | 45 | 35 | (72, 35), every macro col == **0.0** |

**The shorter vector DOES reach the fit. Nothing re-pads it.** `X` is what
`GradientBoostingClassifier.fit` receives. 35 vs 29 is a **6-column** difference
(the 37-entry list minus 2 features absent for unrelated fixture reasons -- the
delta of interest is exactly the 6 macro names).

The asymmetry criterion 2 is really asking about:
- absent for **EVERY** row -> the key never appears in any dict -> the column is
  never created by `pd.DataFrame(features_list)` -> `feature_cols` (:924) drops it
  -> **35 -> 29**. Visible only if someone measures the width.
- absent for **SOME** rows -> the column exists (from the rows that have it) ->
  NaN on the missing rows -> survives :924 -> **median-imputed at :953**. Completely
  invisible; the model is fed a fabricated macro value for those rows.

### M2 -- Q6: what the current code does with an all-NaN column (measured)

```
X.median()                     -> {'momentum_1m': 0.2, 'fed_funds_rate': nan}
persisted _train_feature_medians -> {'momentum_1m': 0.2}       # pd.notna filter :951 drops it
after X.fillna(train_medians)  -> fed_funds_rate = [nan, nan, nan]   # fillna(NaN) is a NO-OP
after X.fillna(0)      (:954)  -> fed_funds_rate = [0, 0, 0]
```

An all-NaN macro column **survives the `feature_cols` filter and becomes a constant
`0.0`** -- it is NOT median-imputed (the median of an all-NaN column is NaN, and
`fillna(NaN)` does nothing), it falls through to the `fillna(0)` backstop at :954.
This is the measurement the step demanded before recommending option (a). See Q6.

### M3 -- Q5/Q2: **partial macro is real and live** (the finding that reframes the step)

`cached_macro` returns a **per-series dict**, and the fast-path loop (:671-681)
resolves each series independently. With PIT enabled (default TRUE, :114) and
per-series publication lags of 1d (DGS10, T10Y2Y) to 125d (GDP), early cutoffs
resolve only the daily series. Measured live against the real BigQuery table with
the real `preload_macro` + `cached_macro`:

```
cutoff       n_series  six_resolved  missing_of_the_six
2018-01-05    2       2/6         ['fed_funds_rate','cpi_yoy','unemployment_rate','consumer_sentiment']
2018-01-15    2       2/6         (same)
2018-02-01    2       2/6         (same)
2018-02-20    6       6/6         []
2018-06-01    7       6/6         []
...
biweekly sample grid 2018-01-01..2019-12-31 (n=53): {'empty': 1, 'partial': 3, 'full': 49}
```

So on the REAL first two years of the walk-forward window: **1 cutoff empty, 3
cutoffs partial, 49 full**. `if macro:` is **TRUE** on the 3 partial cutoffs, so the
six keys ARE assigned -- four of them to `None` (via `.get("FEDFUNDS", {}).get("value")`
on an absent series). Those rows are the minority, so the columns survive :924 and
the four missing values are **median-imputed with the 2018-2019 median macro level**.
This is happening today, silently, and no flag anywhere records it.

**Consequence for the fix shape: a boolean `macro_available` would be WRONG.** It
would read `True` on exactly the cutoffs that are degraded.

### M4 -- live table state (BigQuery `financial_reports.historical_macro`, 4734 rows)

| series | rows | NULL values | min date | max date | NULL realtime_start |
|---|---|---|---|---|---|
| CPIAUCSL | 101 | 0 | 2018-01-01 | 2026-06-01 | 0 |
| DGS10 | 2146 | 0 | 2018-01-02 | 2026-08-03 | 0 |
| FEDFUNDS | 103 | 0 | 2018-01-01 | 2026-07-01 | 0 |
| GDP | 34 | 0 | 2018-01-01 | 2026-04-01 | 0 |
| T10Y2Y | 2147 | 0 | 2018-01-02 | 2026-08-04 | 0 |
| UMCSENT | 102 | 0 | 2018-01-01 | 2026-06-01 | 0 |
| UNRATE | 101 | 0 | 2018-01-01 | 2026-06-01 | 0 |

Schema: `value FLOAT **NULLABLE**`, `date STRING REQUIRED`, `realtime_start DATE NULLABLE`.
So a present-but-null value is **structurally possible but does not occur today** (0/4734).
Note GDP is ingested but is NOT one of the six features -- it is loaded and never read.

### M5 -- Q4/Q6: sklearn behaviour (measured, sklearn 1.8.0)

```
GradientBoostingClassifier      : accepts NaN -> NO  (ValueError: Input X contains NaN)
HistGradientBoostingClassifier  : accepts NaN -> YES
constant-column feature_importances_ -> [1., 0.]   # a constant column gets exactly 0 importance
```

The project uses `GradientBoostingClassifier` (`.claude/rules/backend-backtest.md`), which
**cannot** take NaN -- so "just pass NaN through" is not an option without a model swap.
A constant-0 column is ignored by the ensemble (importance 0.0), which is why option (a)
would be *silent*: it neither errors nor moves a metric.

### M6 -- Q6: PREDICT-path asymmetry (`_build_predict_features` :826-837)

```
train HAD macro, predict row LACKS it -> {'momentum_1m': 0.5, 'fed_funds_rate': 5.33}
```
The macro value is **fabricated from the train median**, not zeroed. The reverse
direction (train lacked macro, predict row has it) silently drops it via the
`feature_names` projection. Both directions are currently unrecorded.

## Q2 -- Enumerated causes of an empty macro dict (derived from every return path)

Derivation rule: enumerate every `return` statement in `cache.cached_macro` and every
branch that can leave `result` empty; then every path in the callees it depends on
(`_pit_enabled`, `_effective_vintage`, `preload_macro` via `_macro_full`/`_macro_unavailable`).

| # | Path (file:line) | Trigger | Empty is... |
|---|---|---|---|
| E1 | `cache.py:653-655` | memoized `{}` for that cutoff returned forever | **derivative** -- caches whatever E2-E5 produced; never an independent cause, but makes the state sticky within a run |
| E2 | `cache.py:658-683` fast path, `str(entry["date"]) > cutoff_date` (:672) for every entry of every series | cutoff earlier than any observation date | **EXPECTED** (legitimate early cutoff) |
| E3 | `cache.py:674-679` PIT vintage predicate rejects every entry | cutoff before the publication date of every row (measured: 2018-01-01) | **EXPECTED** at the window edge, but see M3 -- the *partial* form of this is the harmful one |
| E3b | `cache.py:678` `vintage is None` (from `_effective_vintage` :149-150, unparseable `date` STRING) | corrupt/misformatted date string | **DEFECTIVE** (data quality, fails closed and silently) |
| E4 | `cache.py:723-727` BQ fallback `except Exception: rows = []` | 30s timeout / auth error / query error | **DEFECTIVE** -- swallowed, logged at WARNING only, then memoized by E1. Reachable whenever `_macro_full` is empty and `_macro_unavailable` is False -- which is exactly the `"empty"` outcome, because `_REFUSAL_OUTCOMES` (:346) contains only the two `refused_*` values, so a 0-row table does NOT arm 82.13's suppression |
| E5 | `cache.py:689-691` `_macro_unavailable` short-circuit | 82.13 macro-free mode after a REFUSAL | **EXPECTED and already recorded** by `data_availability.macro=False`. This is the only cause 82.13 covers |
| E6 | `cache.py:728` BQ fallback returns 0 rows legitimately | cutoff before any row / PIT excludes all | **EXPECTED** (the SQL mirror of E2/E3) |

**Recall test (a path my method must EXCLUDE, and does).** `cache.py:722`
`assert _bq_client is not None, "Cache not initialized"` -- an uninitialised cache
does NOT return an empty dict, it **raises**. The raise propagates to
`_build_training_data`'s `except Exception: continue` (:916-917), which drops the
entire training sample rather than producing a macro-free one. My rule (enumerate
paths that *return* while `result` is empty) correctly excludes it: it is a
sample-loss defect, not an empty-macro cause. Same exclusion applies to
`preload_macro:406`. This is a real separate defect class (silent sample loss) but
it is out of scope for 82.43 and should be queued separately if not already.

Second recall test: `preload_macro`'s `"warm"` path (:409-413) returns a POSITIVE int
having loaded nothing -- it is an ambiguous-return defect (82.13's subject) but it
cannot produce an empty `cached_macro`, because `_macro_full` is non-empty by
construction on that branch. Correctly excluded.

## External research

### Search queries run (three-variant discipline)
1. Current/next-year frontier: `imputation of structurally absent features distribution shift gradient boosting 2026`
2. Last-2-year window: `missing values gradient boosting native NaN handling vs imputation 2025 2024`
3. Year-less canonical: `consistency of supervised learning with missing values Josse missingness incorporated in attributes`
4. Domain cross-check (year-less): `walk-forward backtest feature set changes between windows model comparability quant`

### Read in full (6; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://ar5iv.labs.arxiv.org/html/1902.06931 | 2026-08-06 | paper (Josse, Chen, Prost, Varoquaux, Scornet 2019/2024, *Statistical Papers* 65) | WebFetch, ar5iv full HTML | "the widely-used method of imputing with a constant, such as the mean prior to learning is consistent when missing values are not informative." AND the binding caveat: "If the imputed value changes from train set to test set...the learning algorithm may fail, since the imputed data distribution would differ between train and test sets." Recommends MIA for trees. |
| https://scikit-learn.org/stable/modules/impute.html | 2026-08-06 | official docs (scikit-learn 1.9.0, §8.4) | WebFetch full page | §8.4.5 "Keeping the number of features constant": "By default, the scikit-learn imputers will **drop fully empty features, i.e. columns containing only missing values**" (`keep_empty_features=True` opts out). §8.4.6: `MissingIndicator` / `add_indicator` -- "When using imputation, preserving the information about which values had been missing can be informative." §8.4.7 lists `HistGradientBoostingClassifier` as NaN-native; plain `GradientBoostingClassifier` is NOT on the list. |
| https://ar5iv.labs.arxiv.org/html/2202.10580 | 2026-08-06 | paper (Perez-Lebel, Varoquaux, Le Morvan, Abraham, Josse -- *GigaScience* 2022; 13 tasks, 4 health DBs, ~520,000 CPU-h) | WebFetch, ar5iv full HTML | "For imputation-based pipelines, prediction significantly improves with the missingness mask added as input features." "Native support for missing values in supervised machine learning predicts better than state-of-the-art imputation with much less computational cost." "Conditional imputation using Iterative or KNN imputers does not perform consistently better than constant imputation." "For all health databases studied, either the covariates are Missing Not At Random (MNAR) or the outcome to predict depends on the missingness." |
| https://arxiv.org/html/2407.19804 | 2026-08-06 | paper (Le Morvan & Varoquaux, Inria, 2024) | WebFetch native HTML | "Gains in prediction R2 are 10% or less of the gains in imputation R2"; "imputation matters for prediction, but only marginally"; "Good imputations matter less for more expressive predictors." Their protocol: "Train, validation and test sets are imputed using the same imputation model trained on the train set." |
| https://arxiv.org/html/2602.06713 | 2026-08-06 | preprint (Shannon, Liu, Reluga 2026) | WebFetch native HTML | "when the probability of missingness depends on the data, many state-of-the-art methods fail to account for the resulting distribution shift between the observed data used for training and the full data distribution used for evaluation." Existing methods "do not minimise mean-squared error on the full data distribution". Importance-weighted correction buys "average reductions of 3% in RMSE and 7% in Wasserstein distance". |
| https://arxiv.org/html/2512.06356 | 2026-08-06 | preprint (Song et al. 2026) | WebFetch native HTML | Names the exact dichotomy this step faces: "**Uniform missing**: each feature dimension of each node is missing randomly" vs "**Structural missing**: either we observe all features for a node, or we observe none." Structural absence causes "feature distribution shift when generalizing to unseen graph structures". CAVEAT: graph-learning domain; the taxonomy transfers, the numbers do not. |

### Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/abs/2507.21807 (MIBoost) | preprint | variable selection after multiple imputation -- adjacent, not the question |
| https://link.springer.com/article/10.1007/s42979-026-05081-7 | journal 2026 | paywall-risk; superseded by Perez-Lebel for the same claim |
| https://openreview.net/forum?id=fIz8K4DJ7w | preprint | diffusion imputation; far from a 37-column tabular matrix |
| https://arxiv.org/html/2607.07725v1 (SHIFT) | preprint 2026 | genomics; cited only for "88.8% of features structurally absent" scale |
| https://machinelearningmastery.com/navigating-missing-data-challenges-with-xgboost/ | blog | community tier; sklearn docs cover it authoritatively |
| https://www.geeksforgeeks.org/machine-learning/classifiers-in-scikit-learn-that-handle-nannull/ | community | lowest tier; measured directly instead (M5) |
| https://inria.hal.science/hal-02024202/ , https://hal.science/hal-02024202 , https://arxiv.org/abs/1902.06931 , http://juliejosse.com/wp-content/uploads/2019/03/On_the_consistency_of_supervised_learning_with_missing_values-12.pdf , https://ideas.repec.org/a/spr/stpapr/v65y2024i9d10.1007_s00362-024-01550-4.html | mirrors of the Josse paper | de-duplicated to the ar5iv full read |
| https://blog.quantinsti.com/walk-forward-optimization-introduction/ , https://www.susanpotter.net/quant/walk-forward-optimization/ | industry/practitioner | walk-forward mechanics; neither addresses feature-set drift between windows -- see the negative finding in Q4 |
| https://archive.linux.duke.edu/cran/web/packages/mixgb/vignettes/Using-mixgb.html | package vignette | R/mixgb; no transfer |
| https://coder-wang-uspsa.medium.com/... , https://medium.com/@xwang222/... , https://www.dexlabanalytics.com/blog/... , https://doaj.org/article/e701caae716c40cd825fa39121790b5a | blog/community | XGBoost/LightGBM native NaN; not the estimator this repo uses |

**URLs collected (unique): 37.** Read in full: **6**.

### Recency scan (2024-2026) -- performed

Searched the 2024-2026 window explicitly (queries 1 and 2 above). **Found 3 new sources that
COMPLEMENT rather than supersede the canonical Josse 2019 / Perez-Lebel 2022 pair:**

1. **Le Morvan & Varoquaux 2024** (arXiv:2407.19804) -- *tempers* the urgency: imputation
   sophistication buys almost nothing downstream ("gains in prediction R2 are 10% or less of the
   gains in imputation R2"), especially for expressive learners. This argues AGAINST spending
   effort on a clever macro imputer and FOR simply recording the state.
2. **Shannon, Liu & Reluga 2026** (arXiv:2602.06713) -- *adds* a mechanism absent from the older
   work: imputation itself induces a covariate shift between the observed subpopulation and the
   full distribution, which standard pipelines never correct. Directly relevant to M3 (early-window
   rows imputed with a median drawn from the later, macro-complete population).
3. **Song et al. 2026** (arXiv:2512.06356) -- *names* the uniform-vs-structural missingness
   dichotomy that this step is really about.

The 2024-2026 window did **not** change the operative recommendation (record the missingness;
prefer native handling / MIA; constant imputation is acceptable). No source was found that
contradicts the "add a missingness indicator" consensus.

## Consensus vs debate (external)

**Consensus.** (a) Constant imputation is a defensible baseline and is *consistent* when
missingness is uninformative (Josse 2019; Perez-Lebel 2022; Le Morvan 2024). (b) The missingness
**mask** should be recorded -- "prediction significantly improves with the missingness mask added
as input features" (Perez-Lebel 2022) -- and sklearn ships `add_indicator` for exactly this.
(c) Real-world missingness is usually informative/MNAR, which is what makes (b) pay.

**Debate / tension.** The literature says *feed* the mask to the model; **this repo's 82.21
precedent bans that** (`historical_data.py:164-168`) because a coverage flag is a perfect proxy for
a date threshold and the classifier would learn the coverage boundary instead of the economics.
That is a genuine, argued deviation from the literature, not an oversight -- see Q3.

## Pitfalls (from literature)

- Josse's consistency result **requires the same imputation at train and test**; M6 shows this
  repo violates it across the train-has-macro / predict-lacks-macro seam.
- sklearn's own imputers **drop fully-empty columns by default** -- the same shape as
  `feature_cols` at :924. The framework treats "all missing" as "not a feature", which is evidence
  that the current behaviour is a recognised idiom, not a novel bug; the defect is that it is
  **unrecorded**, not that it happens.
- Imputing with a training median drawn from a *different* subpopulation is the exact covariate
  shift Shannon et al. 2026 formalise.

## Q1 -- Does the shorter vector reach the fit? (criterion 2) -- MEASURED

**Yes. Nothing re-pads it.** `X` returned by the real `_build_training_data` is what
`GradientBoostingClassifier.fit` receives.

- **macro present: 35 features** (`X.shape = (72, 35)`)
- **macro absent for every row: 29 features** (`X.shape = (72, 29)`)

Code path: `build_feature_vector` skips the six assignments at `historical_data.py:300-305` ->
`pd.DataFrame(features_list)` (`backtest_engine.py:923`) never creates the columns ->
`feature_cols = [c for c in _NUMERIC_FEATURES if c in df.columns]` (:924) silently drops them.
(35 rather than 37 because two non-macro features are absent for unrelated fixture reasons; the
macro delta is exactly 6.)

**This is a training defect, not merely an observability one -- but the harmful form is the
partial one.** Absent-for-SOME-rows keeps the column and **median-imputes** it (:953), so the model
is fed fabricated macro values with no trace. Measured live: 3 of 53 early cutoffs are partial
(M3). The 29-column whole-run form additionally requires macro to be empty at *every* cutoff, which
in the current live config means a REFUSAL (already labelled by 82.13) or a total BQ outage.

## Q2 -- see the enumeration table above (E1-E6) + both recall tests.

## Q3 -- The right SHAPE

**Reject option (a) (emit explicit nulls to stabilise the vector shape).** MEASURED (M2): an
all-NaN column **survives** `feature_cols`, `X.median()` is NaN, `fillna(train_medians)` is a
no-op, and `fillna(0)` at :954 turns it into a **constant 0.0**. That converts a visible
`35 -> 29` width change into an invisible fabricated constant -- and `0.0` is an in-range,
semantically meaningful value for `yield_curve_spread` (a flat curve) and `cpi_yoy` (zero
inflation). Strictly worse, exactly as the step feared. Confirmed by the measurement, not reasoned.

**Recommend option (b): record coverage on the result, alongside `data_availability`** -- the
82.21 shape.

**But NOT as a boolean.** M3 shows macro has a **partial** state that fundamentals does not:
`cached_macro` returns a per-series dict, `if macro:` is truthy when even one series resolves, and
`macro.get("CPIAUCSL", {}).get("value")` yields `None` for the rest. A boolean `macro_available`
would read `True` on precisely the degraded cutoffs. Use a **count (0-6)**.

**Exact seams (three, all re-derived):**
1. `historical_data.py:70-74` -- seed `"macro_series_available": 0` in the `features` dict at
   construction (same reason 82.21 seeded its flag there: the early return at :77-78), then assign
   `len([s for s in _MACRO_SERIES if s in macro])` unconditionally, replacing the bare `if macro:`
   at :299 with an unconditional six-key assignment plus the count.
2. `backtest_engine.py:919-925` -- after `df` is built, measure what the MATRIX actually got
   (which of the six macro names are in `feature_cols`, and per-column NaN counts). This is the
   honest measurement: what the cache promised is not what the model received.
3. `backtest_engine.py:518` / `:530` + `analytics.py:847-849` -- merge into
   `_availability` so `data_availability` carries it and the report surfaces it. `BacktestResult`'s
   default at :212-213 must gain the new key too, or the "absent" third state returns.

**Does 82.21's `_NUMERIC_FEATURES` trap apply? YES, and slightly harder.** Measured coverage is a
step function: partial/empty only before ~2018-02-20, full for every cutoff after. So a macro
availability flag is a near-perfect proxy for `date >= 2018-02-20` -- the same regime-dummy leak
82.21 banned, on a boundary that is *even more* degenerate (constant for 49 of 53 sampled cutoffs
and every later one). **Keep it out of `_NUMERIC_FEATURES` and carry over the guard**
(`test_flag_is_not_a_model_input`, 82.21 tests :230). This is the deliberate deviation from
Perez-Lebel's "add the mask as an input feature": here the mask's informativeness *is* the leak.

## Q4 -- What the ML literature says about a silently varying feature set

**Correctness:** not a correctness failure for a tree ensemble in the asymptotic sense. Constant
imputation is *consistent* under uninformative missingness (Josse 2019), and imputation quality
barely moves downstream prediction (Le Morvan 2024, "10% or less"). A dropped column is also
sklearn's own default behaviour (§8.4.5).

**Three ways it nonetheless bites here:**
1. **Train/test imputation must match.** Josse: "If the imputed value changes from train set to
   test set...the learning algorithm may fail." M6 measures exactly that violation (train has real
   macro, predict fabricates the train median).
2. **The missingness is informative, not MCAR.** It is deterministic in the date. Perez-Lebel:
   "either the covariates are MNAR or the outcome to predict depends on the missingness" -- and a
   macro regime *is* correlated with forward returns, so the imputed early-window rows are
   imputed with a median drawn from a different regime (the Shannon 2026 covariate shift).
3. **Comparability.** A 29-feature model and a 35-feature model are different models. Per-window
   metrics aggregated across windows (`run_backtest`) then average non-comparable models. Notably,
   the walk-forward practitioner literature (QuantInsti, Potter) is **silent** on feature-set drift
   between windows -- a negative finding worth stating: no external source was found that
   sanctions or forbids it, so this is a local design call.

**How strongly should the artifact state the consequence?** Moderately. Honest framing: *a real
silent-fabrication and comparability defect with a small measured blast radius today (3-4 of ~53
early cutoffs partial, 1 empty), which becomes a whole-run 6-column drop only under a refusal or
outage.* Do NOT claim the backtest is invalid.

## Q5 -- The THIRD leg

82.21's third leg was "covered-but-genuinely-null" (`pe_ratio is None` while
`fundamentals_available is True`). **The macro analogue exists, but it is NOT a null VALUE -- it is
PARTIAL RESOLUTION**, and it is stronger evidence because it happens in production.

- **Present-but-null value:** structurally possible (`value FLOAT NULLABLE`) but **0 of 4734 rows
  are NULL** (M4). Do not build the guard on this -- it would be a fixture representing a state
  that does not occur.
- **Partial resolution (the real third leg):** `cached_macro` resolves per series, so at
  2018-01-05 only `DGS10` + `T10Y2Y` resolve (M3). `if macro:` is TRUE, four of the six features
  are `None`. Fixture: have `cached_macro` return `{"DGS10": {...}, "T10Y2Y": {...}}` only, and
  assert `macro_series_available == 2` AND that the four absent features are distinguishable from
  the empty-dict case. **A two-way empty/full guard passes against a fix that merely renames; the
  partial leg is the mutation-resistant one**, because a boolean flag satisfies two-way and FAILS
  three-way.
- **Fixture hazard (would make a guard vacuous):** `cached_macro` memoises at
  `cache.py:653-655`/`:682`/`:690`/`:730`. A test that calls it twice for the same cutoff gets the
  cached dict and never re-exercises the branch. Call `cache.clear_cache()` (:197-200 clears
  `_macro_full` and `_macro_cache`) and `cache.reset_macro_status()` (:382-386) between legs, and
  assert the precondition took effect.

## Q6 -- Pitfalls specific to THIS fix

1. **(MEASURED, the big one)** Explicit nulls make the all-NaN column survive `feature_cols` and
   become a constant `0.0` -- fabricated values instead of a visible shorter vector. See Q3.
2. `0.0` is in-range for `yield_curve_spread` and `cpi_yoy`, so it is indistinguishable from data.
3. **Cannot pass NaN through.** MEASURED (M5, sklearn 1.8.0): `GradientBoostingClassifier` raises
   `ValueError: Input X contains NaN`; only `HistGradientBoostingClassifier` accepts it. Switching
   estimators is out of scope and would invalidate every historical metric.
4. **MEASURED:** a constant column gets `feature_importances_ == 0.0`. The MDA cache
   (`_MDA_CACHE_PATH`, `backtest_engine.py:142`) and the `FEATURE_TO_AGENT` bridge would then report
   macro at zero importance -- reading as "macro doesn't matter" rather than "macro wasn't there".
5. `_train_feature_medians` drops all-NaN columns via `pd.notna` (:951), so the PREDICT path
   zero-fills them. Consistent with train, but only by accident; document it or it will be
   "fixed" into an inconsistency later.
6. Adding the flag to `_NUMERIC_FEATURES` would make it a silent 38th feature and a date proxy.
7. Any change to the feature count changes the model -- `optimizer_best.json` comparability. Keep
   the observable OUT of the matrix so the fix is metric-neutral by construction, and say so.
8. `BacktestResult.data_availability`'s default (:212-213) must gain the new key, else "absent"
   returns as a third state -- the exact ambiguity 82.21's construction-time seeding removed.

## Application to pyfinagent (summary)

Mirror 82.21, with one deliberate divergence: **count, not boolean**, because macro has a partial
state fundamentals lacks. Seed at construction (`historical_data.py:70-74`), assign
unconditionally (replacing `if macro:` at :299), measure the matrix truth at
`backtest_engine.py:919-925`, thread into `data_availability` at `:518`/`:530` + defaults at
`:212-213`, surfaced by `analytics.py:847-849`. Do NOT emit explicit nulls into the six macro
features. Do NOT add the flag to `_NUMERIC_FEATURES`. Guard with three legs (empty / partial /
full) and clear the memo between them.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6**
- [x] 10+ unique URLs total -- **37**
- [x] Recency scan (last 2 years) performed + reported -- 3 new complementary findings
- [x] Full papers / pages read (not abstracts) -- the one abstract-only fetch (arXiv:2202.10580)
      was upgraded to a full ar5iv read before counting
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in scope (historical_data, cache, backtest_engine,
      analytics, fundamentals_coverage, both test templates)
- [x] Contradictions noted -- Perez-Lebel "add the mask as a feature" vs 82.21's ban (Q3)
- [x] Claims cited per-claim
- GAP (soft, disclosed): the walk-forward practitioner literature is silent on feature-set drift
  between windows; Q4's comparability argument rests on first principles + the Josse train/test
  condition, not on a source that addresses walk-forward feature drift directly.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 20,
  "urls_collected": 37,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_82.43.md",
  "gate_passed": true
}
```
