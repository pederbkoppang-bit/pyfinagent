# Contract -- phase-82.43

> **SUPERSEDED 2026-08-06 (during GENERATE, cycle 2).** The pair `35 / 29`
> below was carried from the research brief and NEVER DERIVED. Measured on
> this tree through the production `_NUMERIC_FEATURES` filter (len 37):
> **macro-present 18, macro-absent 12, macro-partial 18**. The DELTA (6, the
> six macro features) was always correct, so the argument below stands; the
> pair did not. This is a dated planning snapshot, so it is ANNOTATED rather
> than rewritten. See `experiment_results_82.43.md` sections 3 and 11.


**Step:** 82.43 (P1) -- a bare `if macro:` in the feature builder silently drops
all six macro features, with no signal to the caller.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.43.md`,
`gate_passed: true`, 6 external sources read in full, 37 URLs, recency scan
performed, 9 internal files inspected.

---

## 1. Research-gate summary, and how it RESHAPES the step

The gate did not merely confirm the step; it found the step under-describes the
defect. Main re-derived the two decisive measurements independently.

1. **Line number re-derived: the bare guard is at `historical_data.py:299`**,
   not "around 269". 82.21 shipped earlier today and inserted lines above it.
2. **Criterion 2, MEASURED through the real `_build_training_data`:**
   macro-present `X.shape=(72,35)`, macro-absent `(72,29)`. **The shorter vector
   DOES reach `GradientBoostingClassifier.fit`** -- nothing re-pads it. The drop
   site is the `feature_cols` filter (`backtest_engine.py:924`): a key absent
   from EVERY row is never created by `pd.DataFrame(features_list)` at `:923`.
   So this is a training defect, not only an observability one.
3. **THE REFRAMING FINDING -- partial macro is live, and it is worse than the
   empty case the step describes.** `cached_macro` resolves each series
   INDEPENDENTLY (`cache.py:671-681`), so `if macro:` is **True** while
   individual series are missing. Measured on the real biweekly grid
   2018-01-01..2019-12-31 (n=53 cutoffs): **1 empty, 3 partial (2 of 6 series),
   49 full** -- driven by publication lags (DGS10/T10Y2Y ~1 day vs CPIAUCSL ~50
   days) under the 82.15 point-in-time predicate. Because partial rows are a
   minority, the columns SURVIVE and the missing values are **median-imputed**.
   That is silent fabrication, and the step's own framing would have missed it.
4. **Therefore a boolean flag is the WRONG shape, and this is where 82.21's
   template must DIVERGE rather than be copied.** A boolean `macro_available`
   reads **True** on exactly the degraded (partial) cutoffs. Fundamentals is one
   row -- all-or-nothing; macro is six independent series. Use a **0-6 count**.
5. **Option (a) from the criterion -- "emit explicit nulls so the vector shape
   is stable" -- is MEASURABLY WORSE, exactly as the step feared.** Main
   reproduced it: an all-NaN column SURVIVES `feature_cols`, `X.median()` is
   `NaN`, `fillna(train_medians)` is a no-op, and `fillna(0)`
   (`backtest_engine.py:954`) makes it a **constant 0.0** -- which is in-range
   for `yield_curve_spread` (a flat curve) and `cpi_yoy`. It converts a visible
   35->29 width change into an invisible fabricated constant.

   ```
   ABSENT macro keys (today):        cols: 2   macro cols present: []
   ALL-NaN macro columns:            cols: 4   cpi_yoy median: nan  ->  [0, 0, 0]
   PARTIAL macro (the live case):    cpi_yoy median: 2.5  ->  imputed 2.5 (no trace)
   ```

   So this contract takes the criterion's **second** branch -- *record the
   absence explicitly* -- and the artifact must say why, because choosing the
   first branch would have been a regression dressed as a fix.
6. **A hole in 82.13 found in passing:** `_REFUSAL_OUTCOMES` (`cache.py:346`) is
   only `{refused_unparseable, refused_stale}`. `"empty"` is **not** a refusal,
   so a 0-row macro table never arms `set_macro_unavailable`, and every cutoff
   falls through to the per-cutoff 30s BQ query whose `except Exception: rows =
   []` swallows a timeout into `{}`. Queued, not fixed here.
7. **Fixture hazard that would make a guard vacuous:** `cached_macro` memoises
   (`cache.py:653/:682/:690/:730`), so a test that calls it twice for one cutoff
   never re-exercises the branch. Guards must call `cache.clear_cache()` and
   `cache.reset_macro_status()` between legs.
8. **`GradientBoostingClassifier` REJECTS NaN** (sklearn 1.8.0 `ValueError`);
   only `HistGradientBoostingClassifier` accepts it. "Pass NaN through" is not
   available without a model swap, which is far outside this step.
9. **Literature, and it sets how strongly to state the consequence.** Constant
   imputation is consistent under uninformative missingness (Josse et al.), and
   imputation quality barely moves prediction (Le Morvan & Varoquaux 2024:
   "gains in prediction R2 are 10% or less"). BUT Josse's own condition -- *"If
   the imputed value changes from train set to test set... the learning
   algorithm may fail"* -- **is violated here**: train has macro, predict
   fabricates the train median. So the honest statement is **moderate**: a real
   silent-fabrication and comparability defect with a small measured blast
   radius, NOT an invalid backtest. The artifact will say exactly that.
10. **A literature-vs-repo tension to disclose rather than resolve silently.**
    Perez-Lebel 2022 finds prediction improves when the missingness mask is
    added as an input feature. 82.21 nonetheless BANNED its coverage flag from
    `_NUMERIC_FEATURES` as a date proxy. Measured here: macro coverage is a step
    function at ~2018-02-20, making a macro count an **even more** degenerate
    regime dummy. The ban carries over -- a deliberate, argued deviation from the
    cited literature.

---

## 2. Hypothesis

`if macro:` conflates three states that must be distinguishable: all six series
present, some present, none present. Replacing the implicit signal with an
explicit **count** (0-6), seeded at construction so it exists on every return
path, and surfacing per-window macro coverage on the result, makes both the
empty and the partial case visible without changing what reaches the model.

Falsifiable predictions:
- A macro-complete fixture yields `macro_series_count == 6` and 6 macro keys.
- An empty fixture yields `0` and no macro keys, and the matrix is 6 columns
  narrower.
- A **partial** fixture (2 of 6) yields `2`, keeps 2 keys, and is distinguishable
  from both -- which a boolean cannot do.

## 3. Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. "a fixture in which cached_macro returns an empty dict produces a feature
   vector whose shape matches the macro-complete case, or records the absence
   explicitly, asserted by a test that fails against the current bare
   `if macro:` behaviour"
2. "the feature-count reaching the model is measured for both the macro-present
   and macro-absent fixtures and the two numbers are recorded in the step
   artifact"
3. "a macro-present fixture is unaffected, so the fix cannot pass by always
   emitting sentinels"
4. "the distinct causes of an empty macro dict (refusal, early cutoff, vintage
   miss, BQ timeout) are enumerated from the source with file:line and each
   classified as expected or defective"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_43_macro_feature_absence.py -q`

Criterion 1 is a disjunction. This contract takes **"records the absence
explicitly"** and NOT "shape matches", for the measured reason in §1.5 -- the
shape-matching branch is a regression. That choice is stated, not silent.

### Seam-and-mutant map

| # | Seam | Guard drives | Mutant that must kill it |
|---|---|---|---|
| 1 | `historical_data.py:299` and the seed at `:70-74` | `build_feature_vector`, three-way (empty / partial / full) | restore the bare `if macro:`; hardcode the count; drop the seed |
| 2 | `backtest_engine._build_training_data` | the REAL matrix builder, asserting both widths | make the recorded width a literal instead of `X.shape[1]` |
| 3 | the macro-complete path | the full fixture, asserting real values not just key presence | emit sentinels unconditionally |
| 4 | `cached_macro`'s return paths | an enumeration derived from the source, asserted non-empty | -- (documentation criterion; guarded by an assertion that the enumeration is non-empty and each entry classified) |

## 4. Plan

**D1 (criteria 1 + 3) -- an explicit COUNT, not a boolean.**
`historical_data.py`: seed `features["macro_series_count"] = 0` at construction
(so it survives the short-price early return, the same reason 82.21 seeded its
flag), then assign `len(...)` of the resolved series unconditionally, replacing
the bare guard. The six feature keys stay conditional -- they are NOT emitted as
explicit nulls, per §1.5.

**Keep `macro_series_count` OUT of `_NUMERIC_FEATURES`** (§1.10), guarded.

**D2 (criterion 2) -- measure what the MATRIX actually got.**
Record the realised feature count from `X.shape[1]` at the matrix seam and
surface per-window macro coverage on the result via `data_availability`, mirroring
82.13/82.21's double-write into `report["analytics"]`.

**D3 (criterion 4) -- enumerate the causes from the source**, each with a
re-derived `file:line` and an expected/defective classification, asserted
non-empty by a guard so the enumeration cannot silently empty out.

**Queued, not fixed here:** the `_REFUSAL_OUTCOMES` hole (§1.6), and the
separate silent-sample-loss defect the gate found while recall-testing
(`cache.py:722` raises, and `_build_training_data`'s `except Exception:
continue` at `:916` drops the whole training sample).

## 5. Non-scope

No model swap (NaN-native `HistGradientBoosting` is out of scope). No change to
the imputation policy itself -- this step makes the degradation VISIBLE; deciding
whether median-imputing a structurally absent feature is acceptable is 82.53's
job. No live positions touched; paper trading untouched.

## 6. References

- `handoff/current/research_brief_82.43.md`
- Josse et al., *On the consistency of supervised learning with missing values*
- Le Morvan & Varoquaux (2024), imputation quality vs prediction
- Perez-Lebel et al. (2022), GigaScience -- missingness mask as input
- scikit-learn imputation guide §8.4.5
- Internal: `backend/backtest/historical_data.py:70-78,299-306`,
  `backend/backtest/cache.py:346,651-730`,
  `backend/backtest/backtest_engine.py:212-213,518,530,918-925,950-956`,
  `backend/backtest/analytics.py:847-849`,
  `backend/tests/test_phase_82_13_preload_refusal_handling.py`,
  `backend/tests/test_phase_82_21_fundamentals_coverage.py`
