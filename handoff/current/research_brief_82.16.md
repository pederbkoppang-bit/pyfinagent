# Research Brief -- masterplan step 82.16

**Topic:** Non-forward-looking labels in `STRATEGY_REGISTRY` (`quality_momentum`,
`factor_model`) -- degenerate supervised targets computed from contemporaneous
features, producing tautological in-sample accuracy and meaningless Sharpe/DSR.

**Tier:** complex | **Audit-class:** false | **Started:** 2026-08-05
**Status:** COMPLETE -- gate_passed: true

---

## 0. Session log (write-first trace)

- [x] Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full
- [x] Internal audit: 7 label methods
- [x] Internal audit: STRATEGY_REGISTRY mapping + dispatch site
- [x] Internal audit: quality_score fundamentals dependency
- [x] Internal audit: removal blast radius
- [x] Internal audit: existing test precedent
- [x] External: 9 sources read in full
- [x] Recency scan
- [x] JSON envelope

---

_(sections appended below as work completes)_

## 1. INTERNAL AUDIT -- the seven label methods (all anchors RE-DERIVED 2026-08-05)

Measurement command:
```
grep -n "STRATEGY_REGISTRY\|_compute_.*_label\|def _run_window\|cached_prices" \
  backend/backtest/backtest_engine.py
```
File total: `wc -l backend/backtest/backtest_engine.py` -> **1604** lines.

### 1a. Forward-lookingness verdict table (MEASURED, not asserted)

| # | Method | file:line (def) | Fetches data strictly AFTER entry_date? | Exact call | Verdict |
|---|--------|-----------------|------------------------------------------|-----------|---------|
| 1 | `_compute_triple_barrier_label` | backtest_engine.py:808 | YES | `end_date = entry_date + holding_days*1.5` (:819); `cache.cached_prices(ticker, entry_date, end_date)` (:820); walks `idx in range(1, len(prices))` (:835) | **FORWARD** |
| 2 | `_compute_quality_momentum_label` | :1190 | **NO** | only `self.data_provider.build_feature_vector(ticker, entry_date)` (:1195). Zero `cached_prices` call in the whole body (:1190-1209) | **NOT FORWARD -- CONFIRMED BROKEN** |
| 3 | `_compute_mean_reversion_label` | :1211 | YES *on the signal branch only* | `end_date = entry_date + mr_holding_days*2` (:1246); `cache.cached_prices(...)` (:1247); walks `idx in range(1, len(prices))` (:1254) | **FORWARD (conditionally -- see 1b)** |
| 4 | `_compute_factor_label` | :1274 | **NO** | only `build_feature_vector(ticker, entry_date)` (:1296). Composite of `pb/pe/momentum_12m/momentum_1m/annualized_volatility/quality_score/dividend_yield`, thresholded at :1347/:1349. No `cached_prices` anywhere in :1274-1351 | **NOT FORWARD -- CONFIRMED BROKEN** |
| 5 | `_compute_stretch_regime_label` | :1457 | YES | `self._walk_barriers(...)` (:1487) -> `cache.cached_prices(ticker, entry_date, end_date)` (:1439) with `end_date = entry + horizon*1.6` (:1438) | **FORWARD** |
| 6 | `_compute_qarp_label` | :1489 | YES | `self._walk_barriers(...)` (:1534) | **FORWARD** |
| 7 | `_compute_reversion_sigma_label` | :1536 | YES | `self._walk_barriers(...)` twice (:1574, :1579) | **FORWARD** |

**The step description is CONFIRMED on the two named strategies and no third
method is broken.** `quality_momentum` and `factor_model` are the only two of
the seven whose bodies contain no post-`entry_date` data access. Verified by
reading each body in full, not by grep alone.

### 1b. The trap that will break a naive test: EARLY RETURNS THAT NEVER REACH THE FORWARD FETCH

Four of the five FORWARD methods have gate branches that return **before** the
forward price fetch. For those inputs the label is *provably* invariant to
post-entry prices -- so a fixture row that lands on a gate branch will make a
correct, forward-looking method look broken:

| Method | Pre-fetch early return | line |
|--------|------------------------|------|
| `_compute_mean_reversion_label` | `if not is_oversold and not is_overbought: return 0` | :1242-1243 |
| `_compute_stretch_regime_label` | `bars is None` -> None (:1474-1475); `stretch is None` -> None (:1479-1480) | :1473-1480 |
| `_compute_qarp_label` | `if not (cheap and quality and low_debt and profitable): return None` | :1523-1524 |
| `_compute_reversion_sigma_label` | `return None  # no overextension` | :1581 |

Criterion 1 says "asserting the returned label changes for **at least one**
fixture row" -- that wording is exactly right and must be honoured: the fixture
must be **constructed to pass each method's gate**, not sampled at random.
A per-strategy fixture spec is unavoidable (see recommendations).

### 1c. `meta_label` -- NOT a false verdict either way

`STRATEGY_REGISTRY` (:32-47) maps 8 names to 7 methods; `"meta_label":
"_compute_triple_barrier_label"` (:37). The meta-labelling second stage is
applied in `_run_window` at **:518-527** (`if self.strategy == "meta_label" and
len(train_features) >= 50: meta_model = self._train_meta_label_model(...)`),
i.e. AFTER labels exist -- it is a *sizing* model, not a labelling method.
Therefore a registry-enumerating test that iterates `meta_label` invokes
`_compute_triple_barrier_label` and correctly PASSES. No special-casing needed,
and none should be added (a special case would be a hand-written exemption list,
which criterion 2 forbids in spirit).

### 1d. Dispatch seam (what a test must stand up)

- Dispatcher: `_compute_label(self, ticker, entry_date)` at **:1182-1186** --
  `method = getattr(self, STRATEGY_REGISTRY.get(self.strategy, "_compute_triple_barrier_label"))`.
  Note the `.get(..., default)` **silent fallback**: an unknown strategy silently
  becomes triple_barrier. A test that mutates a *copy* of the registry must
  therefore also set `engine.strategy`, or go through `getattr(engine, method_name)`
  directly (preferred -- it removes the fallback from the equation).
- Only production call site of `_compute_label`: `_build_training_data` at **:739**.
- Constructor clamp: `self.strategy = strategy if strategy in STRATEGY_REGISTRY else "triple_barrier"` (:211).

## 2. INTERNAL AUDIT -- the MEASUREMENT (this is the load-bearing evidence)

I re-ran the 82.2 mutation experiment across **all 8 registry names** (82.2's own
test covers only the 3 new ones). Script:
`/private/tmp/.../scratchpad/measure_82_16.py`; run as
`source .venv/bin/activate && PYTHONPATH=<repo> python measure_82_16.py`.
Fixture: `backend/tests/fixtures/phase_82_2_label_fixture.py`
(10 tickers x 88 entry dates = **880 rows**). Mutation = under
`patch.object(BE.cache, "cached_prices", ...)`, collapse every bar after the
entry bar to `0.5 x entry_price` (SPY exempt, mirroring the 82.2 test).

```
strategy          method                            non-None  changed  label distribution
triple_barrier    _compute_triple_barrier_label          880      375  {0: 85, -1: 505, 1: 290}
quality_momentum  _compute_quality_momentum_label        880        0  {0: 543, 1: 337}
mean_reversion    _compute_mean_reversion_label          880      119  {0: 836, 1: 35, -1: 9}
factor_model      _compute_factor_label                    0        0  {None: 880}
meta_label        _compute_triple_barrier_label          880      375  {0: 85, -1: 505, 1: 290}
stretch_regime    _compute_stretch_regime_label          880      565  {0: 304, 1: 261, -1: 315}
qarp              _compute_qarp_label                    528      462  {0: 244, 1: 218, -1: 66, None: 352}
reversion_sigma   _compute_reversion_sigma_label         409      321  {None: 471, 1: 126, -1: 87, 0: 196}
```

### 2a. THE BIGGEST FINDING: `factor_model`'s "0/880 changed" is VACUOUS

`factor_model` returns **None on all 880 rows** of the committed fixture. Cause:
`_compute_factor_label` requires `momentum_12m` (`if mom_12m is None: return None`,
backtest_engine.py:1315-1316) and the fixture's `build_feature_vector`
(`phase_82_2_label_fixture.py:128-136`) supplies only
`price_at_analysis / sma_50_distance / rsi_14 / annualized_volatility /
momentum_6m / quality_score` + 4 fundamentals -- **no `momentum_12m`, no
`momentum_1m`, no `pb_ratio`, no `dividend_yield`**.

So the archived claim `handoff/archive/phase-82.2/experiment_results.md:149`
("quality_momentum and factor_model changed **0/880**") is **arithmetically true
but evidentially empty for factor_model**: `None != None` is False for every row,
so the counter would read 0 even for a perfectly forward-looking method.
`quality_momentum`'s 0/880 is REAL (880 non-None labels, none moved).

The source reading still convicts `factor_model` (:1274-1351 contains no
`cached_prices` call at all), so the VERDICT is unchanged -- but **the new test
must assert non-None coverage per strategy or it will inherit the same blind
spot**, and it will emit a misleading failure message after someone fixes
`factor_model` if the fixture is not extended first. This is a mutation-guard
precondition in the sense of `feedback_a_green_suite_can_be_blind`.

### 2b. `mean_reversion` changes on only 119/880 (13.5%)

Because `if not is_oversold and not is_overbought: return 0` (:1242-1243) fires
before the forward fetch on most rows. It passes "at least one row changes" with
room to spare, but a test that demanded e.g. ">=25% of rows change" would FAIL a
correct method. Keep criterion 1's "at least one fixture row" wording.

### 2c. The existing engine builder CANNOT invoke triple_barrier (measured)

`_engine()` at `backend/tests/test_phase_82_2_candidate_strategies.py:35-41` sets
only `data_provider`, `trader`, `holding_days`, `mr_holding_days`. Measured:

```
_engine() has tp_pct: False        _engine() has sl_pct: False
_engine() has strategy: False      _engine() has _strategy_params: False
_compute_triple_barrier_label under patched cache
  -> AttributeError : 'BacktestEngine' object has no attribute 'tp_pct'
```

`_compute_triple_barrier_label` reads `self.tp_pct` (:830) and `self.sl_pct`
(:831). A registry-enumerating test that reuses `_engine()` verbatim will
**AttributeError on `triple_barrier` AND `meta_label`** (2 of 8 names). The
builder must be extended (add `tp_pct`, `sl_pct`), not the test narrowed.
Un-patched, the first failure is instead
`AssertionError: Cache not initialized -- call init_cache() first` from
`backend/backtest/cache.py`, which masks the AttributeError -- so the monkeypatch
must be in place before the attribute gap is even visible.

### 2d. Monkeypatch seam (exact names)

- `backend/backtest/backtest_engine.py:25` -> `from backend.backtest import cache`.
  The module object is bound as `backend.backtest.backtest_engine.cache`, so the
  established patch is `patch.object(BE.cache, "cached_prices", fake)`
  (`test_phase_82_2_candidate_strategies.py:49`). **Module-level import, NOT
  function-local** -- verified by reading the import block :21-25. `_walk_barriers`
  (:1439), `_market_stretch` (:1421), `_compute_triple_barrier_label` (:820) and
  `_compute_mean_reversion_label` (:1247) all call `cache.cached_prices` through
  that same module binding, so ONE patch covers every forward read.
- The feature seam is NOT patched at module level: each label method calls
  `self.data_provider.build_feature_vector(...)`, so the fake is injected by
  assigning `eng.data_provider` (the `_Provider` shim,
  `test_phase_82_2_candidate_strategies.py:26-28`). No monkeypatch of
  `HistoricalDataProvider` is needed.
- `self.trader.transaction_cost_pct` is read by `_compute_triple_barrier_label`
  (:829) and `_sigma_barriers` (:1399) -- supplied by the `_Trader` shim (:31-32).

### 2e. `quality_score` -- the fundamentals interaction is REAL and asymmetric

`quality_score` is built in `backend/backtest/historical_data.py:215-260` from
four QMJ dimensions and is assigned **only inside the `if fundamentals:` branch**
(the branch that also sets `features["sector"]` at :265). Fundamentals come from
`cache.cached_fundamentals(ticker, cutoff_date)` (`historical_data.py:44`).
With no fundamentals row, `quality_score` is never set, so `fv.get("quality_score")`
is None. Consequences differ per strategy:

- `_compute_quality_momentum_label:1200` -- `quality_score = fv.get("quality_score", 0) or 0`
  coerces **None -> 0.0**. Then `momentum_6m > 5 and quality_score > 0.3` can
  NEVER be true, while `momentum_6m < -5 and quality_score < 0.1` is true purely
  on momentum. So on a fundamentals-free span the label degenerates into
  "**-1 if 6m momentum < -5%, else 0; +1 unreachable**". That is a SECOND,
  independent defect stacked on the non-forward one.
- `_compute_factor_label:1313-1314` -- `if pb is None and pe is None: return None`.
  Both are fundamentals, so on a fundamentals-free span factor_model labels
  **nothing at all** (exactly what the 880-row None run demonstrates for a
  fixture missing those keys). Its `quality` leg separately falls back to a
  neutral 0.5 (:1337).

So fixing forward-lookingness alone does NOT make `quality_momentum` trainable
across the standard window if step 82.21's finding (no `historical_fundamentals`
rows before 2024-06-30) holds. Both must be addressed or the strategy stays
un-comparable. **I did not re-measure the 82.21 BQ row-date claim in this
session** (would need a BQ query); I am relying on it as stated by the caller and
flagging the dependency rather than asserting it.

### 2f. The tautology is MEASURABLE, not just inferred

Every input to both broken labels is itself a training feature. `_NUMERIC_FEATURES`
(backtest_engine.py:50-60) contains `momentum_6m` (:51) and `quality_score` (:57)
-- the complete input set of `_compute_quality_momentum_label` -- and
`momentum_1m`/`momentum_12m` (:51), `annualized_volatility` (:52),
`pe_ratio`/`pb_ratio` (:55), `quality_score`/`dividend_yield` (:57) -- the
complete input set of `_compute_factor_label`. Both labels are therefore
**deterministic functions of a subset of the columns handed to
`GradientBoostingClassifier.fit`** (`_train_model`, :881-898; X built at :768-770
from `_NUMERIC_FEATURES`). The claim in the step description is not rhetorical;
it is checkable by set containment.

### 2g. REMOVAL BLAST RADIUS (measured consumer-by-consumer)

| Consumer | file:line | What happens if a name leaves STRATEGY_REGISTRY | Severity |
|---|---|---|---|
| `AVAILABLE_STRATEGIES` literal | `backend/backtest/quant_optimizer.py:68` | Optimizer keeps PROPOSING the dead name; `backtest_engine.py:211` (`self.strategy = strategy if strategy in STRATEGY_REGISTRY else "triple_barrier"`) SILENTLY runs triple_barrier while `_log_experiment` writes `"strategy": "quality_momentum"` to the TSV. **A fabricated experiment record -- worse than the defect being fixed.** | **BLOCKER: must patch** |
| `IMPLEMENTED_STRATEGY_IDS` frozenset | `backend/meta_evolution/archetype_library.py:31-33`, archetypes at `:123` (quality_momentum) and `:171` (factor_model), validator at `:89-92` | INERT TODAY (hand-written literal). **After step 82.17 derives it from STRATEGY_REGISTRY, those two `is_implemented=True` archetypes raise at IMPORT time.** | **ORDERING HAZARD 82.16<->82.17** |
| rotation seed `qm_trend_tilt` | `backend/autoresearch/strategy_registry.py:104` (`"strategy": "quality_momentum"`) | `backend/autoresearch/strategy_backtest_adapter.py:105-107` raises `ValueError(f"unknown strategy {strategy!r}...")` (also `:198`) -> producer SKIPS the seed. Correct fail-loud behaviour, but the seed set silently drops 4 -> 3 | MEDIUM: update the seed |
| LLM optimizer skill | `backend/agents/skills/quant_strategy.md:53` | Skill text still describes `_compute_quality_momentum_label`; the LLM proposer will keep suggesting a removed strategy | MEDIUM: patch the doc |
| Optimizer history TSV | `backend/backtest/experiments/quant_results.tsv` | MEASURED `grep -o '"strategy": "[a-z_]*"' ... | sort | uniq -c` -> **503 triple_barrier, 7 factor_model, 4 quality_momentum, 2 mean_reversion, 1 meta_label, 1 blend**. So **11 historical optimizer rows carry the tautological strategies** -- the "numbers that have appeared in optimizer comparisons" are real and countable | LOW: do not delete history; disclose |
| Live incumbent | `backend/backtest/experiments/optimizer_best.json` -> `"strategy": "triple_barrier"` | **UNAFFECTED.** No live-money exposure from either broken strategy | NONE (good news, measured) |
| Frontend | `grep -rn "quality_momentum\|factor_model\|triple_barrier" frontend/src/` -> **0 hits** | No UI consumer | NONE |
| Handoff artifact | `handoff/allocator_output.json:38` (`"name": "factor_model"`) | Data artifact, not code | NONE |

Registry cardinality MEASURED:
`python -c "from backend.backtest.backtest_engine import STRATEGY_REGISTRY as R; print(len(R), len(set(R.values())))"`
-> **8 keys, 7 distinct methods**
(`factor_model, mean_reversion, meta_label, qarp, quality_momentum, reversion_sigma, stretch_regime, triple_barrier`).

### 2h. EXISTING TEST PRECEDENT -- a near-miss test already exists

`backend/tests/test_phase_82_2_candidate_strategies.py:304-328`
`test_candidate_label_depends_on_post_entry_prices` is **already the exact
mutation test 82.16 asks for** -- but parametrized over
`NEW = ["stretch_regime", "qarp", "reversion_sigma"]` (`:18`), i.e. **a
hand-written list**, which is precisely what criterion 2 forbids. 82.16's job is
to GENERALISE it to the registry, add the non-empty assertion, add the
negative-control stub, and fix-or-remove the two offenders. Reusable shapes:

- engine builder `_engine()` `:35-41` (extend with `tp_pct`/`sl_pct`)
- feature-provider shim `_Provider` `:26-28`, trader shim `_Trader` `:31-32`
- price patch `patch.object(BE.cache, "cached_prices", ...)` `:49`
- mutation closure `_run(mutate)` `:314-320` (collapse `df.iloc[1:]` to `0.5 x` entry)
- committed fixture `backend/tests/fixtures/phase_82_2_label_fixture.py` (10 tickers, `entry_dates()` `:74-78`)

Also relevant precedent: `test_market_stretch_is_backward_looking_only` `:283-299`
proves the OPPOSITE direction (a helper must NOT read past `entry_date`) by
capturing the `end` argument. Same monkeypatch seam.

**No other test in `backend/tests/` asserts forward-lookingness.** Measured:
`grep -rln "cached_prices\|_compute_.*_label" backend/tests/` -> 4 files
(`test_phase_75_mcp_truth.py`, `test_phase_82_2_candidate_strategies.py`,
`test_gate_correctness_69.py`, `fixtures/phase_82_2_label_fixture.py`); only the
82.2 file contains a mutation assertion. **Do not build a duplicate -- build the
generalisation.**

---

## 3. EXTERNAL RESEARCH

### 3a. Search queries run (three-variant discipline)

| # | Query | Variant |
|---|-------|---------|
| 1 | `Lopez de Prado triple barrier method labeling financial machine learning why labels must be forward looking` | YEAR-LESS canonical |
| 2 | `target leakage taxonomy "illegitimate features" leakage reproducibility crisis machine learning based science Kapoor Narayanan` | YEAR-LESS canonical |
| 3 | `permutation test shuffled labels verify target is non-trivial classifier baseline sanity check` | YEAR-LESS canonical |
| 4 | `deflated Sharpe ratio number of trials selection bias candidate strategies must be removed from selection pool Bailey Lopez de Prado` | YEAR-LESS canonical |
| 5 | `label leakage financial machine learning 2026 forward-looking target validation backtest` | CURRENT-YEAR (2026) |
| 6 | `2025 metamorphic testing machine learning pipeline detect target leakage mutate future data assert prediction changes` | LAST-2-YEAR (2025) |

### 3b. Read IN FULL via WebFetch / curl+pypdf (counts toward the gate) -- accessed 2026-08-05

| # | URL | Tier | Fetched how | What it establishes |
|---|-----|------|-------------|---------------------|
| 1 | https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | peer-reviewed (Bailey & Lopez de Prado 2014, *J. Portfolio Management*) | `curl -sL` + `pypdf` (22 pages, 47,819 chars extracted) | The selection-pool argument. *"Not controlling for the number of trials involved in a particular discovery leads to over-optimistic performance expectations."* *"a backtest where the researcher has not controlled for the extent of the search involved in his or her finding is worthless, regardless of how excellent the reported performance might be."* DSR's deflation term uses **`V[{SR_n}]` = "the variance across the trials' estimated SR" and `N` = "the number of independent trials"** -- i.e. the trial POOL is an input to the statistic, so a degenerate member corrupts it. |
| 2 | https://ar5iv.labs.arxiv.org/html/2207.07048 | peer-reviewed preprint (Kapoor & Narayanan; published *Patterns* 2023) | WebFetch (ar5iv HTML; cell.com returned **403**) | The naming question. Category **[L2] "Model uses features that are not legitimate"**: *"One instance when this can happen is if a feature is a proxy for the outcome variable."* Also *"leakage usually leads to inflated estimates of model performance."* Remedy = **model info sheets** requiring an explicit argument per feature. |
| 3 | https://ar5iv.labs.arxiv.org/html/2401.13796 | preprint (Apicella, Isgro, Prevete 2024) | WebFetch (ar5iv; `arxiv.org/html/...v1` returned 404) | The closest named match. **Direct target leakage** = *"target labels are incorporated within the input data chosen features"*, which *"leads the ML model to over-fit on the training data, as it essentially possesses access to the correct label during training."* **Indirect target leakage** = *"the correct labels are subtly integrated into the features but in an indirect way (e.g., proxy features)."* |
| 4 | https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/ | peer-reviewed (Kriegeskorte et al. 2009, *Nature Neuroscience*) -- CROSS-DOMAIN | WebFetch | The exact structural analogue, and the best available NAME: **circular analysis / "double dipping"** -- *"the use of the same data set for selection and selective analysis -- will give distorted descriptive statistics and invalid statistical inference whenever the results statistics are not inherently independent of the selection criteria under the null hypothesis."* Their Example 1 produced *"high decoding accuracies significantly above chance"* **in completely random data**. |
| 5 | https://scikit-learn.org/stable/auto_examples/model_selection/plot_permutation_tests_for_classification.html | official docs | WebFetch | Criterion-1's literature basis. `permutation_test_score` *"generates a null distribution by calculating the accuracy of the classifier on 1000 different permutations of the dataset, where features remain the same but labels undergo different random permutations"*; null = *"there is no dependency between the features and labels"*. Caveat quoted verbatim: *"this test has been shown to produce low p-values even if there is only weak structure in the data."* |
| 6 | https://scikit-learn.org/stable/common_pitfalls.html | official docs | WebFetch | The canonical leakage definition: *"Data leakage occurs when information that would not be available at prediction time is used when building the model."* NOTE for honesty: this page has **no** section on target leakage -- it is entirely about train/test separation. Recorded so the brief does not over-claim. |
| 7 | https://mlfinpy.readthedocs.io/en/latest/Labelling.html | official docs (AFML reference implementation) | WebFetch | The primary framing of what a label IS. Triple barrier: *"The upper barrier represents the threshold an observation's **return needs to reach**..."*, vertical barrier = *"the amount of time an observation has to reach its given return"*. Fixed-time horizon uses *"returns over a fixed horizon h"*. Trend scanning fits *"multiple regressions from time t to t + L (L is a maximum look-forward window)"*. **Every labelling method in the AFML family is defined over a forward window; none is defined at t alone.** |
| 8 | https://reasonabledeviations.com/notes/adv_fin_ml/ | authoritative blog (chapter-by-chapter AFML notes) | WebFetch | Ch.3 in the author's own terms: label from `r_t = p_{t+h}/p_t - 1`; triple barrier = *"a stop loss, a profit take, and an expiration"*. Ch.7: *"K-fold CV vastly over-inflates results because of the lookahead bias"*; purge principle *"if a test set label Y_j depends on information Phi_j, training set labels that depend on Phi_j should be removed"*. Selection hygiene: *"Marcos' Third Law: Every backtest must be reported with all trials involved in its production."* |
| 9 | https://arxiv.org/html/2605.23959v1 | preprint, 2026-05-12 (Zhang, Li, Peng, Chen) | WebFetch (arXiv native HTML) | The 2026 frontier + the closest methodological match to criterion 1. *"At each decision time t, the signal must be constructed using only information that would have been available before the decision is made."* *"Chronological order alone does not define what information a model was allowed to use."* Method = a **one-switch paired benchmark**: *"hold fixed the data, walk-forward split, model family, portfolio rule, forecast horizon, and transaction-cost convention, and change only one protocol element at a time"*, scoring a **"Leakage Gain"** delta. Measured LG-SR@5bps *"often exceeding 15-26 points"* for the worst switches. |

**External sources read in full: 9.** (Floor for `complex` = 5.)

### 3c. Identified but snippet-only (context; does NOT count toward the gate)

| URL | Tier | Why not read in full |
|---|---|---|
| https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9 | peer-reviewed | **HTTP 403** -- substituted the ar5iv preprint (#2) |
| https://pubmed.ncbi.nlm.nih.gov/37720327/ | index | abstract-only record of the same paper |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | peer-reviewed | SSRN landing page; read the author-hosted PDF instead (#1) |
| https://www.researchgate.net/publication/286121118_... | index | duplicate of #1 |
| https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf | peer-reviewed | Bailey et al. "Statistical Overfitting and Backtest Performance" -- same authors, redundant with #1 |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | community | tertiary summary of #1 |
| https://en.wikipedia.org/wiki/Purged_cross-validation | community | tertiary; purging is out of scope for 82.16 |
| https://arxiv.org/pdf/2311.04179 | preprint | "On Leakage in Machine Learning Pipelines" -- pipeline/preprocessing leakage, not target degeneracy |
| https://arxiv.org/pdf/2509.15971 | preprint (2025) | LeakageDetector 2.0 -- Jupyter static analysis; no target-triviality check |
| https://arxiv.org/pdf/2604.10965 | preprint (2026) | bioLeak (R) -- has "permutation tests, and target-leakage scans"; R-only, no reusable design |
| https://arxiv.org/pdf/2604.04199 | preprint (2026) | "Which Leakage Types Matter? ... 2,047 Benchmark Datasets" -- quantifies leakage prevalence, not label triviality |
| https://arxiv.org/pdf/2604.06899 | preprint (2026) | Data leakage in automotive perception -- different domain, no transfer |
| https://arxiv.org/pdf/2308.07832 | preprint | REFORMS reporting standards -- reporting, not testing |
| https://arxiv.org/html/2511.02108v1 | preprint (2025) | Metamorphic testing of LLMs for NLP -- confirms the metamorphic-relation vocabulary; not finance |
| https://arxiv.org/pdf/2604.21579 | preprint (2026) | Metamorphic testing for LLM memorization -- adjacent, not applicable |
| https://arxiv.org/pdf/2301.03318 | preprint | "The Dutch Draw" input-independent baseline -- a *baseline*, not a label check |
| https://arxiv.org/pdf/2303.04581 | preprint | Supervised learning in Chinese futures -- applies TB, adds nothing normative |
| https://arxiv.org/pdf/2507.07107 | preprint (2025) | ML multi-factor trading with bias correction -- applies DSR downstream |
| https://www.emergentmind.com/papers/2207.07048 | aggregator | summary of #2 |
| https://www.newsletter.quantreo.com/p/the-triple-barrier-labeling-of-marco | community | popular explainer of TB |
| https://williamsantos.me/posts/2022/triple-barrier-labelling-algorithm/ | community | implementation walkthrough |
| https://github.com/nkonts/barrier-method | community | code repo |
| https://github.com/eslazarev/purged-cross-validation | community | code repo (purging/DSR) |
| https://medium.com/@yairoz/the-triple-barrier-method-labeling-financial-time-series-for-ml-in-elixir-e539301b90d6 | community | Elixir port |
| https://www.mql5.com/en/articles/19253 | community | trend-scanning labelling |
| https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/ | practitioner | CV hygiene, out of scope |
| https://pub.towardsai.net/the-combinatorial-purged-cross-validation-method-363eb378a9c5 | community | CPCV explainer |
| https://medium.com/data-science-at-microsoft/how-to-leverage-permutation-tests-and-bootstrap-tests-for-baselining-your-machine-learning-models-f1010bf22e71 | practitioner | permutation-baseline how-to; superseded by #5 |
| http://mvpa.blogspot.com/2012/12/which-labels-to-permute.html | community | "which labels to permute" -- useful nuance, low tier |
| https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/ | community | leakage explainer |
| https://www.tradinginterview.com/courses/machine-learning/lessons/cross-validation-for-financial-data-purging-and-embargoing/ | community | course page |
| https://paperswithbacktest.com/course/deflated-sharpe-ratio | community | DSR course page |
| https://mlfinpy.readthedocs.io/en/latest/ (index) | official docs | navigation page for #7 |

**Total unique URLs collected: 42** (9 read in full + 33 snippet-only).

### 3d. Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

Searched explicitly for 2024/2025/2026 work (queries #5 and #6 above).
**Result: 3 new findings that COMPLEMENT, and 0 that supersede, the canonical
sources.**

1. **arXiv:2605.23959 (2026-05-12), "When Alpha Disappears: A One-Switch
   Benchmark for Decision-Time Leakage in Financial Backtests"** -- the strongest
   new result and the most directly transferable. It formalises exactly the
   testing method criterion 1 mandates: hold the whole pipeline fixed, flip ONE
   protocol element, and measure the delta ("Leakage Gain"). Our mutation test is
   the same construction with the switch applied to the *label* rather than the
   *feature*. It also supplies the crisp normative sentence for the contract:
   *"At each decision time t, the signal must be constructed using only
   information that would have been available before the decision is made."*
   **Note the asymmetry**: that paper polices the *upper* bound on information
   (no future in the FEATURES). 82.16 polices the *lower* bound (there must be
   future in the LABEL). Both are decision-time-semantics violations; the
   literature is heavily skewed to the first.
2. **arXiv:2401.13796 (2024), Apicella et al.** -- supplies the vocabulary
   "direct / indirect target leakage" (source #3). Useful but, again, the mirror
   image of our defect.
3. **arXiv:2604.10965 (2026) bioLeak; arXiv:2509.15971 (2025) LeakageDetector
   2.0; arXiv:2604.04199 (2026) leakage landscape** -- an active 2025-2026 tooling
   line around automated leakage scans, including *"permutation tests, and
   target-leakage scans"*. Snippet-only; none provides a check for a target with
   NO forward content, which reinforces finding K3 below.

Nothing in the window supersedes Lopez de Prado (2018) Ch.3 on what a label is,
Bailey & Lopez de Prado (2014) on trial-pool hygiene, or Kriegeskorte (2009) on
circular analysis.

### 3e. Key findings (cited per claim)

**K1 -- A label is BY DEFINITION a forward-window observable; a label at t is not
a label.** Every AFML-family method (fixed-horizon `r_t = p_{t+h}/p_t - 1`,
triple barrier, trend scanning over `[t, t+L]`, meta-labelling) is defined over a
window ending strictly after the event time
(https://mlfinpy.readthedocs.io/en/latest/Labelling.html ;
https://reasonabledeviations.com/notes/adv_fin_ml/ , accessed 2026-08-05). So
`_compute_quality_momentum_label` and `_compute_factor_label` are not
mis-specified labels -- they are **not labels at all**; they are screens.

**K2 -- The right NAME is "circular analysis / double dipping", NOT leakage.**
Kriegeskorte et al. 2009: *"the use of the same data set for selection and
selective analysis ... will give distorted descriptive statistics and invalid
statistical inference whenever the results statistics are not inherently
independent of the selection criteria under the null hypothesis"*
(https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/). Their random-data example
produced *"high decoding accuracies significantly above chance"* with no real
effect -- the precise analogue of a GradientBoosting model scoring near-perfectly
on a threshold rule over its own inputs. The leakage taxonomies get close but do
not name it: Kapoor & Narayanan's **[L2] "model uses features that are not
legitimate ... a feature is a proxy for the outcome variable"** and Apicella
et al.'s **direct target leakage** ("target labels are incorporated within the
input data chosen features") both describe FEATURE -> LABEL contamination. Our
defect is the SAME identity relation traversed the other way (LABEL := f(FEATURES)),
so those definitions apply verbatim in the limit -- an identity is symmetric.
**Recommended wording for the contract: "the label is a deterministic function of
the training features -- a tautological target (Kriegeskorte's circular analysis;
the degenerate limit of Kapoor & Narayanan's [L2] illegitimate-feature leakage)."**
Do NOT call it look-ahead bias; that is the opposite failure.

**K3 -- There is no off-the-shelf test for a target with NO forward content.**
The standard hygiene tool is the **permutation / shuffled-label test**
(https://scikit-learn.org/stable/auto_examples/model_selection/plot_permutation_tests_for_classification.html),
which asks "does the model beat chance on a SHUFFLED label?" -- it detects a
model that has learned nothing. It **cannot** detect our defect: a tautological
label yields a genuine, hugely significant feature-label dependency, so
`permutation_test_score` would return `p ~ 0` and PASS. Shuffling is therefore
the WRONG instrument here, and the contract should say so explicitly to stop a
reviewer from asking for it. The right instrument is the **mutation / metamorphic
relation on the FUTURE**, exactly as arXiv:2605.23959's one-switch design does
(https://arxiv.org/html/2605.23959v1) -- change only the post-decision data and
require the output to move. Criterion 1 is well-founded and is, as far as this
search reached, ahead of the published tooling.

**K4 -- A non-comparable candidate must be REMOVED from the selection pool, not
caveated.** Bailey & Lopez de Prado 2014: the deflation term is built from
*"the variance across the trials' estimated SR"* and *"N ... the number of
independent trials"*, so the trial POOL is a direct input to DSR
(https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf). A tautological
candidate contaminates the pool twice: it inflates `N`, and its in-sample Sharpe
is drawn from a different (degenerate) distribution, corrupting `V[{SR_n}]`. Worse,
it can WIN: *"a backtest where the researcher has not controlled for the extent of
the search involved in his or her finding is worthless, regardless of how
excellent the reported performance might be."* Leaving it registered "with a
caveat" is not a weaker version of the fix -- it keeps the corrupted input. This
is the literature basis for criterion 2's fix-OR-remove disjunction.

**K5 -- Prevention is a per-item ARGUMENT, not a global scan.** Kapoor &
Narayanan's remedy is model info sheets in which the researcher must *"argue why
each feature used in their model is legitimate"*. The registry-enumerating test is
the executable form of that: every strategy must, individually and automatically,
carry an argument that its target is legitimate.

### 3f. Consensus vs debate

**Consensus (unanimous across all 9 sources):** a supervised target must be
defined over information unavailable at decision time; contaminating the
feature-label relation with an identity destroys the validity of any performance
statistic derived from it; and the resulting inflation is not detectable from the
performance number alone.

**Debate / gap:** the literature is almost entirely about *too much* future in the
features. The mirror case -- *no* future in the label -- is unnamed and untooled.
Kriegeskorte's circular-analysis framing is the only source that covers both
directions natively, because it is stated about *dependence between the selection
criterion and the result statistic*, not about time. Minor tension: sklearn warns
permutation tests *"produce low p-values even if there is only weak structure"*,
which is a further reason not to lean on them here.

### 3g. Pitfalls from the literature, mapped to this step

- **Do not use a shuffled-label test as the guard** (K3) -- it passes on a
  tautological label.
- **Do not evaluate the guard on a sample that cannot exercise the method** --
  Kriegeskorte's remedy is *"an independent data set"*; the operational analogue
  here is that the fixture must actually reach the code path under test. See 2a:
  the current fixture cannot reach `_compute_factor_label` at all.
- **Do not leave a contaminated candidate in the pool with a footnote** (K4).
- **Prefer a one-switch paired design over a bespoke assertion** (arXiv:2605.23959):
  hold everything fixed, flip exactly the post-entry prices, diff the output.

---

## 4. RECOMMENDATION FOR THE CONTRACT

### 4a. The test (new file `backend/tests/test_phase_82_16_label_forward_information.py`)

Build the check as **one plain function + three consumers**, so the negative
control exercises the SAME code the real assertion uses (a negative control that
re-implements the check proves nothing --
`feedback_mutation_test_guards_and_fixtures`):

```
def label_response(eng, method_name, tickers, dates) -> tuple[int, int]:
    """Return (n_labelled, n_changed) under post-entry price mutation."""
```

1. `test_registry_enumeration_is_non_empty` -- NON-parametrized, asserts
   `len(STRATEGY_REGISTRY) > 0` **and** that the parametrize argvalues equal
   `set(STRATEGY_REGISTRY)`. **Criterion 3 exists because
   `@pytest.mark.parametrize` over an empty collection collects ZERO tests and
   pytest exits 0** -- an empty registry would otherwise turn the whole gate green.
2. `test_every_registered_strategy_label_is_forward_looking[name]` --
   `@pytest.mark.parametrize("name", sorted(STRATEGY_REGISTRY))`, resolved via
   `getattr(eng, STRATEGY_REGISTRY[name])` (NOT via `_compute_label`, whose
   `.get(..., "_compute_triple_barrier_label")` fallback at :1184 would mask a
   missing method). Assert **both**:
   - `n_labelled > 0` -- with its own message ("the fixture cannot exercise this
     method"), because `factor_model` currently returns None on all 880 rows and
     a bare `base != mutated` cannot tell that apart from "no forward info";
   - `n_changed >= 1` -- criterion 1's "at least one fixture row".
3. `test_guard_detects_a_non_forward_stub` (criterion 4) -- pass
   `{**STRATEGY_REGISTRY, "stub_non_forward": "_stub_non_forward_label"}` (a
   **copy**; do not mutate the module global, it is imported by
   `strategy_backtest_adapter.py:63`) plus
   `monkeypatch.setattr(BacktestEngine, "_stub_non_forward_label", <returns a
   constant from fv only>, raising=False)`, and assert `label_response` reports
   `n_labelled > 0, n_changed == 0`. **Add the mirror control too** -- a stub that
   DOES walk forward and is reported as changed -- otherwise the guard is only
   shown to fail, never to discriminate.

Engine builder: copy `_engine()` (82.2 `:35-41`) **and add `tp_pct` + `sl_pct`**
(MEASURED AttributeError, section 2c). Do not import the private `_engine` from
the 82.2 test module.

### 4b. Fix vs remove -- recommendation

- **`factor_model`: REMOVE.** (i) Making it forward-looking is not a fix but a
  replacement -- the composite at :1342-1351 is a cross-sectional *score*, and
  "forward return of a factor-ranked book" is a different strategy; (ii) it needs
  `pb_ratio`/`pe_ratio` (:1313-1314), i.e. fundamentals, which per step 82.21 do
  not exist before 2024-06-30, so even a fixed version is un-evaluable over ~81%
  of the standard window; (iii) the registry already carries a forward-looking
  fundamentals-gated cousin, `qarp` (:1489), which is the shape a fixed
  factor_model would converge to.
- **`quality_momentum`: fix OR remove -- present both, let PLAN choose; do not
  decide silently.** The cheap faithful fix is the house idiom: keep
  `momentum_6m`/`quality_score` as an entry GATE at `entry_date`, then label the
  forward move with `self._walk_barriers(ticker, entry_date, self.holding_days,
  up, down)` using `self._sigma_barriers(fv, self.holding_days)` (:1379-1455) --
  ~10 lines, reusing already-tested helpers, cost-adjusted and sigma-scaled like
  the 82.2 candidates. **But** it still reads `quality_score`, so the 82.21
  fundamentals gap applies; and the `or 0` coercion at :1200 must be replaced with
  an explicit `None -> return None` or the +1 branch stays unreachable on
  fundamentals-free spans (section 2e). If PLAN does not want to own that, remove
  it and let `qarp` carry the quality lens.
- **`meta_label`: LEAVE AS IS.** It maps to `_compute_triple_barrier_label` (:37),
  which is forward-looking; the meta stage at `_run_window:518-527` is a sizing
  model applied after labels exist. It passes the enumerating test on its own
  merits (MEASURED: 880 labelled, 375 changed). Do not add a special case.

### 4c. Mandatory companion edits if anything is REMOVED

1. **`backend/backtest/quant_optimizer.py:68` `AVAILABLE_STRATEGIES`** -- derive
   from `STRATEGY_REGISTRY` (plus `"blend"`), or at minimum delete the removed
   names. **Skipping this ships a worse bug than the one being fixed**: the
   optimizer proposes the dead name, `backtest_engine.py:211` silently substitutes
   `triple_barrier`, and the TSV records a strategy that never ran. It also drifts
   the other way today (missing `stretch_regime`/`qarp`/`reversion_sigma`).
2. **`backend/meta_evolution/archetype_library.py`** -- the archetypes at `:123`
   (`quality_momentum`) and `:171` (`factor_model`) declare `is_implemented=True`
   and are validated at `:89-92`. Inert today, but **step 82.17 will derive
   `IMPLEMENTED_STRATEGY_IDS` from `STRATEGY_REGISTRY`, at which point those two
   archetypes raise at import.** Whichever step lands second must flip them to
   `is_implemented=False`. State this dependency in the contract.
3. **`backend/autoresearch/strategy_registry.py:104`** -- the `qm_trend_tilt` seed
   would start raising in `strategy_backtest_adapter.py:105-107`. Correct
   fail-loud behaviour, but repoint the seed so the rotation set stays at 4.
4. **`backend/agents/skills/quant_strategy.md:53`** -- LLM-facing description of
   `_compute_quality_momentum_label`; stale text keeps the proposer suggesting it.
5. **`STRATEGY_REGISTRY`'s own comment block, backtest_engine.py:38-43 and
   :1353-1377** -- both currently *document the 82.16 defect in the present tense*
   ("`quality_momentum` and `factor_model` above do not"). After the fix those
   become the stale artifact. Update them in the same commit.
6. **Do NOT rewrite `quant_results.tsv`.** 11 historical rows carry the
   tautological strategies (7 factor_model, 4 quality_momentum -- MEASURED);
   deleting history is worse than disclosing it. `optimizer_best.json` is
   `triple_barrier`, so the incumbent and the live book are unaffected -- say so
   in `experiment_results.md`, it is the reassuring half of the finding.

### 4d. Traps (each one measured or read, not guessed)

1. **Empty-parametrize green.** `@pytest.mark.parametrize` over an empty registry
   collects 0 tests -> exit 0. Criterion 3's non-empty assertion must be a
   separate, non-parametrized test.
2. **`factor_model` returns None on all 880 fixture rows** -- `base == mutated`
   for the wrong reason. Assert `n_labelled > 0` per strategy, and extend the
   fixture with `momentum_12m` / `momentum_1m` / `pb_ratio` / `dividend_yield`
   before claiming a fixed factor_model is tested.
3. **`_engine()` lacks `tp_pct`/`sl_pct`** -> `AttributeError` on `triple_barrier`
   AND `meta_label` (2 of 8). Measured.
4. **The cache assertion masks it.** Un-patched, you get
   `AssertionError: Cache not initialized` from `backend/backtest/cache.py` first;
   the AttributeError only appears once `patch.object(BE.cache, "cached_prices", ...)`
   is active. Do not conclude "cache problem" and stub the wrong seam.
5. **`mean_reversion` moves on only 119/880 rows (13.5%)** -- because
   `if not is_oversold and not is_overbought: return 0` (:1242-1243) precedes the
   forward fetch. A threshold like "25% of rows must change" would FAIL a correct
   method. Keep "at least one row".
6. **`_compute_label`'s silent fallback (:1184) and the ctor clamp (:211)** both
   swallow unknown names -- never route the test through them.
7. **Removal without patching `AVAILABLE_STRATEGIES`** = fabricated TSV rows (4c.1).
8. **82.16 <-> 82.17 ordering hazard** on `archetype_library` (4c.2).
9. **`quality_score` None -> 0.0 coercion (:1200)** makes `+1` unreachable and
   `-1` momentum-only whenever fundamentals are absent -- a second, independent
   degeneracy that a forward-looking rewrite would silently inherit.
10. **Do not call this look-ahead bias / leakage in prose.** It is the mirror
    image (K2). Mislabelling it will send a future reader to purging/embargo code
    that is already correct (`_label_overlaps_test`, `_build_training_data:725-731`).
11. **A shuffled-label permutation test does not detect this** (K3) -- if a
    reviewer asks for one, the answer is "it would pass".
12. **SPY exemption in the mutation closure** (82.2 `:317`) is correct but for a
    non-obvious reason: `_market_stretch` (:1402-1432) requests
    `cached_prices("SPY", start, entry_date)` -- it ends AT entry_date, so it is
    unaffected either way. Keep the exemption for parity; do not "fix" it.

### 4e. Where the masterplan step text is WRONG or STALE (high-value)

| Claim in `.claude/masterplan.json` step 82.16 | Status | Correction (measured 2026-08-05) |
|---|---|---|
| "two of the **five** STRATEGY_REGISTRY strategies" | **STALE** | The registry has **8 keys / 7 distinct methods** since phase-82.2 added `stretch_regime`, `qarp`, `reversion_sigma`. A "5" framing would under-enumerate the test by 3. |
| `_compute_quality_momentum_label` at `:1124` | **STALE** | `:1190` |
| `_compute_factor_label` at `:1208` | **STALE** | `:1274` |
| `_compute_triple_barrier_label` at `:742` | **STALE** | `:808` |
| `_compute_mean_reversion_label` at `:1145` | **STALE** | `:1211` |
| "`_compute_mean_reversion_label` likewise has a forward validation stage" | **TRUE but incomplete** | Only 119/880 fixture rows are forward-sensitive; the no-signal path returns 0 before the fetch (:1242-1243). |
| "Audit `meta_label` too" | **RESOLVED, no action** | It aliases TB (:37); meta-labelling is a post-label sizing stage at `_run_window:518-527`. Forward-looking. Do not special-case it. |
| 82.2 artifact: "factor_model changed **0/880**" (`handoff/archive/phase-82.2/experiment_results.md:149`) | **VACUOUS** | factor_model returns None on all 880 rows of that fixture (missing `momentum_12m`). The verdict survives on source reading, but the NUMBER is not evidence. |
| Step 82.21's text: "both already shown by **82.16** to carry NO forward information" | **MIS-ATTRIBUTED** | 82.16 is still `pending`; it was **82.2**'s research gate + `experiment_results.md` that showed it. |
| Implied "this is a new test to write" | **PARTIALLY EXISTS** | `test_phase_82_2_candidate_strategies.py:304-328` already implements the mutation check for 3 hand-listed names. 82.16 = generalise, don't duplicate. |

---

## 5. RESEARCH GATE CHECKLIST

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (**9**: 3 peer-reviewed, 2 preprints, 3 official docs, 1 authoritative blog)
- [x] 10+ unique URLs total (**42**)
- [x] Recency scan (2024-2026) performed + reported (section 3d)
- [x] Full papers/pages read, not abstracts (DSR read via `pypdf`, 22pp/47,819 chars; leakage papers via ar5iv HTML; cell.com 403 disclosed and substituted)
- [x] file:line anchor for every internal claim, each with the command that produced it

Soft checks:
- [x] Internal exploration covered `backtest_engine.py`, `historical_data.py`, `quant_optimizer.py`, `archetype_library.py`, `autoresearch/strategy_registry.py`, `autoresearch/strategy_backtest_adapter.py`, the 82.2 test + fixture, `quant_results.tsv`, `optimizer_best.json`, `masterplan.json`, `agents/skills/quant_strategy.md`, `frontend/src/` (negative result)
- [x] Contradictions/consensus noted (3f) -- incl. the honest negative that `common_pitfalls.html` does NOT cover target leakage
- [x] Per-claim citation with URL + access date (2026-08-05)
- [ ] **NOT measured in this session:** step 82.21's BQ claim that `historical_fundamentals` has no rows before 2024-06-30. Relied on as stated; flagged as a dependency, not asserted.

---

## 6. JSON GATE ENVELOPE

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 33,
  "urls_collected": 42,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "CONFIRMED: quality_momentum (backtest_engine.py:1190) and factor_model (:1274) never read a post-entry price; the other five methods do. MEASURED on the committed 82.2 fixture (880 rows): quality_momentum 880 labelled / 0 changed under post-entry mutation; factor_model 0 labelled / 0 changed -- its 0/880 is VACUOUS (returns None everywhere; the fixture omits momentum_12m), so the new test must assert per-strategy non-None coverage. mean_reversion changes on only 119/880, so keep criterion 1's 'at least one row'. meta_label aliases triple_barrier and passes (375/880) -- no special case. The 82.2 test at :304-328 is already this test but over a hand-written 3-name list; 82.16 generalises it. Traps: empty-parametrize collects 0 tests (why criterion 3 exists); _engine() lacks tp_pct/sl_pct (AttributeError on 2 of 8); removal without patching quant_optimizer.py:68 AVAILABLE_STRATEGIES makes the optimizer log a strategy that never ran; 82.17 will make archetype_library raise at import. Literature: this is circular analysis (Kriegeskorte 2009), not look-ahead; a shuffled-label permutation test would PASS it; Bailey & LdP require removing a non-comparable candidate from the trial pool rather than caveating it.",
  "brief_path": "handoff/current/research_brief_82.16.md",
  "gate_passed": true
}
```
