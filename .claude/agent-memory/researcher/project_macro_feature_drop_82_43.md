---
name: macro-feature-drop-82-43
description: 82.43 macro `if macro:` silent drop -- the real defect is PARTIAL macro (live, 3/53 cutoffs), not the empty dict; explicit nulls make it WORSE (constant 0.0); a boolean flag is wrong because macro has a partial state fundamentals lacks
metadata:
  type: project
---

Step 82.43: `historical_data.py:299` (`if macro:`) drops six macro features when
`cache.cached_macro` returns `{}`. Measured 2026-08-06 through the REAL production
functions, not reasoned.

**The step's framing under-describes the defect.** The empty-dict case is the *visible*
one; the harmful one is **PARTIAL** macro.

- `cached_macro` returns a **per-series dict** and resolves each series independently
  (`cache.py:671-681`), so `if macro:` is TRUE while individual series are absent and
  `macro.get("CPIAUCSL", {}).get("value")` returns `None`.
- MEASURED live on the real biweekly grid 2018-01-01..2019-12-31 (n=53):
  **1 empty, 3 partial (2/6 series), 49 full.** Caused by per-series publication lags
  (DGS10/T10Y2Y = 1d vs CPIAUCSL = 50d, GDP = 125d) under the 82.15 PIT predicate.
- Partial rows are a MINORITY, so the six columns survive and the missing values are
  **median-imputed** -- fabrication with zero trace. The empty case is the loud one; the
  partial case is the silent one.

**Consequences a boolean flag would get wrong.** `macro_available: bool` reads `True` on
exactly the degraded cutoffs. Use a **count (0-6)**. This is where the 82.21
fundamentals template must DIVERGE, not be copied -- fundamentals is one row (all-or-
nothing), macro is six independent series.

**Explicit nulls are WORSE than the shorter vector (measured).** `X.median()` of an
all-NaN column is NaN -> `fillna(train_medians)` is a NO-OP -> `fillna(0)`
(`backtest_engine.py:954`) makes it a **constant 0.0**. A visible 35->29 width change
becomes an invisible fabricated constant, and 0.0 is in-range for `yield_curve_spread`
(flat curve) and `cpi_yoy`. Always measure what the imputation chain does to an all-NaN
column BEFORE proposing "just emit nulls for shape stability".

**Feature counts (criterion-2 answer): 35 (macro present) vs 29 (macro absent),** via the
real `_build_training_data`. The drop happens at
`feature_cols = [c for c in _NUMERIC_FEATURES if c in df.columns]` (:924) -- a column
that no row carries is never created by `pd.DataFrame(features_list)`. Absent-for-SOME-
rows keeps the column and imputes; absent-for-EVERY-row drops it. That asymmetry is the
whole story.

**82.13's suppression has a hole.** `_REFUSAL_OUTCOMES` (`cache.py:346`) is ONLY
`{refused_unparseable, refused_stale}` -- `"empty"` is not in it. So a 0-row macro table
does NOT arm `set_macro_unavailable`, and every cutoff falls through to the per-cutoff
30s BQ query whose `except Exception: rows = []` (:723-727) swallows a timeout into `{}`.

**Recall-test exclusion that works:** `cache.py:722` `assert _bq_client is not None`
RAISES, it does not return `{}`; the raise is eaten by `_build_training_data`'s
`except Exception: continue` (:916), dropping the whole training sample. That is a
separate silent-sample-loss defect class, correctly outside an "empty macro dict"
enumeration.

**Fixture hazard that makes a guard vacuous:** `cached_macro` memoises at :653/:682/:690/
:730. A guard that calls it twice for one cutoff never re-exercises the branch. Call
`cache.clear_cache()` (:197-200) + `cache.reset_macro_status()` (:382-386) between legs.

**Literature tension worth remembering:** Perez-Lebel 2022 (GigaScience) says "prediction
significantly improves with the missingness mask added as input features", but 82.21
BANS the coverage flag from `_NUMERIC_FEATURES` because it is a perfect date proxy.
Measured: macro coverage is a step function at ~2018-02-20, so the macro flag is an even
MORE degenerate regime dummy. The ban carries over -- record out-of-band, never as a
model input. That is a deliberate, argued deviation from the literature.

Other measured facts: `GradientBoostingClassifier` REJECTS NaN (sklearn 1.8.0); only
`HistGradientBoostingClassifier` accepts it. A constant column gets
`feature_importances_ == 0.0`, so a zero-filled macro column would read as "macro doesn't
matter" in the MDA cache. `historical_macro` has **0 NULL values in 4734 rows** across 7
series, so a present-but-null VALUE fixture would represent a state that does not occur.
GDP is ingested but is not one of the six features -- loaded and never read.

See [[project_fundamentals_coverage_82_21]] for the template this diverges from and
[[project_macro_preload_refusal_82_13]] for the layer above.
