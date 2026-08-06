# Research Brief -- masterplan step 82.46

**Topic:** Deliberate composition of the optimizer's categorical strategy trial pool
(`AVAILABLE_STRATEGIES`), the `blend` phantom strategy, and the DSR/PBO trial-count
consequences.

**Tier:** complex | **Audit-class:** false | **Started:** 2026-08-06
**Status:** IN PROGRESS (write-first; sections appended as sources are read)

---

## Q0. Structure of this brief

- Q1 -- CRUX: what quantity does this repo's DSR use as N (trials)?
- Q2 -- What Bailey & Lopez de Prado actually count as a "trial"
- Q3 -- Per-strategy evidence table (6 registry strategies + `blend`)
- Q4 -- `blend`: implement or remove?
- Q5 -- Cost of the re-run / cheapest honest experiment
- Q6 -- Guard-design traps for a "decide a set" step

---

## Read in full (7; >=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|-----|----------|------|-------------|----------------------|
| 1 | https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-08-06 | peer-reviewed (JPM 40(5):94, 2014) | WebFetch + `pdfplumber` full-text extract (45,402 chars) | **"It is critical to understand that the N used to compute E[max{SR}] corresponds to the number of independent trials. Suppose that we run M trials, where only N trials are independent, N<M. Clearly, using M instead of N will overstate E[max{SR}]."** (Appendix A.3) |
| 2 | https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf | 2026-08-06 | peer-reviewed (J. Computational Finance, 2016; SSRN 2326253) | WebFetch (extraction failed) -> `pdfplumber` full-text extract (63,988 chars) | **"the researcher ends up running a number N of alternative model configurations (or trials), out of which one is chosen according to some performance evaluation criterion"** + Algorithm 2.3: **"each column n = 1,...,N represents a vector of profits and losses ... associated with a particular model configuration tried by the researcher"** |
| 3 | https://arxiv.org/html/2507.07107 | 2026-08-06 | preprint (2025) | WebFetch (arXiv native HTML per the fetch chain) | **"With N≈50 effective configurations tried during development (architectures, losses, γ values, augmentation sizes, covariance estimators)"** -- counts configurations TRIED, not search-space size. Caveat quoted: **"DSR cannot correct for biases that affect every configuration uniformly"** |
| 4 | https://portfoliooptimizationbook.com/book/8.3-dangers-backtesting.html | 2026-08-06 | textbook (Palomar, *Portfolio Optimization*, CUP) | WebFetch | **"Keep track of the number of backtests conducted on a dataset so that the probability of backtest overfitting may be estimated and the Sharpe ratio may be properly deflated."** Explicitly: no guidance on optimal candidate-pool SIZE. |
| 5 | https://arxiv.org/html/2512.22476 | 2026-08-06 | preprint (2025, AutoQuant) | WebFetch | Counts evaluations, not space: **"Stage I runs Nopt∈{40,120} TPE trials ... Stage II then re-evaluates a fixed candidate pool across a small cost-scenario grid"**. Pool CURATION by ex-ante thresholds: **"absolute robustness thresholds are applied to remove underperforming configurations ... the top K candidates are retained"** |
| 6 | https://optuna.readthedocs.io/en/stable/faq.html | 2026-08-06 | official docs (cross-domain: ML HPO) | WebFetch | Dynamic search spaces are legal and sampler-dependent: **"it is possible to, in a single study, alter the range by sampling parameters from different search spaces in different trials. The behavior when altered is defined by each sampler individually."** No trial-count-vs-space-size rule. |
| 7 | https://www.risklab.ai/research/backtesting/backtesting_cross_validation | 2026-08-06 | academic lab (RiskLab AI / Seco) | WebFetch | CPCV path count `φ[N,k]`; **"The variance of this mean, σ²(μi), is much lower than the variance of a single backtest."** Notably does NOT mention PBO or DSR -- a gap worth knowing before citing it as a PBO source. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | SSRN landing (DSR) | Landing page; the same paper was read in full as #1 |
| https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4686376_code4361537.pdf | 2024 peer-reviewed (Arian/Norouzi/Seco, *Expert Systems with Applications*) | **Attempted, HTTP 403 Forbidden.** Recency finding taken from search snippet only -- flagged as such below. |
| https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 | 2024 peer-reviewed (same paper, journal version) | Paywalled abstract only |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | encyclopedia | **Attempted, HTTP 404** |
| https://www.pm-research.com/content/iijpormgmt/40/5/94 | journal landing | Paywalled |
| https://marti.ai/qfin/2018/05/30/deflated-sharpe-ratio.html | practitioner blog | Lower tier; primary already read |
| https://paperswithbacktest.com/course/deflated-sharpe-ratio | practitioner course | Lower tier |
| https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464 | community | Lowest tier |
| https://www.insightbig.com/post/traditional-backtesting-is-outdated-use-cpcv-instead | blog (CPCV) | Lower tier, no primary content |
| https://www.quantbeckman.com/p/with-code-combinatorial-purged-cross | blog + code | Lower tier |
| https://arxiv.org/pdf/2603.20319 | preprint (Implementation Risk in Portfolio Backtesting) | Adjacent topic, not trial-counting |
| https://arxiv.org/pdf/2407.17645 | preprint (Hopfield Networks for Asset Allocation) | Adjacent, uses PBO but not the question |
| https://www.turbinefi.com/blog/why-backtests-lie-prediction-market-overfitting-2026 | blog (2026) | Community tier |
| https://pdfs.semanticscholar.org/c215/d0a2064ce1a3565d276475abc84305418f0f.pdf | mirror of #1 | Duplicate |
| https://www.researchgate.net/publication/286121118_... | mirror of #1 | Duplicate, gated |

**URLs collected: 22 unique (7 read in full + 15 snippet-only).**

### Search-query composition (the mandatory three variants)

1. **Year-less canonical** -- `Deflated Sharpe Ratio Bailey Lopez de Prado number of trials definition what counts as a trial`; and `combinatorial purged cross-validation CPCV probability backtest overfitting ... out-of-sample testing methods synthetic`. Surfaced the two 2014/2016 primaries (#1, #2) and the Palomar textbook (#4).
2. **Current-year frontier (2026)** -- `arxiv 2026 deflated Sharpe ratio probability backtest overfitting strategy selection trial count`. **Result: no 2026 primary literature on trial-counting surfaced** (only 2014/2016 primaries plus one 2026 community blog).
3. **Last-2-year window (2024-2025)** -- `effective number of trials multiple testing backtest overfitting 2025 2026 deflated Sharpe`; `"deflated Sharpe" OR "probability of backtest overfitting" 2025 strategy pool selection number of trials hyperparameter search space`. Surfaced #3, #5 and the 2024 Arian et al. paper (403/paywalled).

---

## Q2 -- What Bailey & Lopez de Prado ACTUALLY count as a "trial" (verbatim, from the primaries)

**Both primaries count CONFIGURATIONS ACTUALLY EVALUATED, not the size of the menu.**

PBO paper (source #2), the definitional sentence:
> "the researcher ends up running a number N of alternative model configurations (or
> trials), out of which one is chosen according to some performance evaluation
> criterion, such as the Sharpe ratio."

and Algorithm 2.3, which is the operational definition:
> "we form a matrix M by collecting the performance series from the N trials. In
> particular, each column n = 1,...,N represents a vector of profits and losses over
> t = 1,...,T observations associated with a particular model configuration **tried by
> the researcher**. M is therefore a real-valued matrix of order (T x N)."

DSR paper (source #1), the five inputs:
> "DSR deflates SR by taking into consideration five additional variables: The
> non-Normality of the returns, the length of the returns series (T), the variance of
> the SRs **tested** (V[{SR}]), as well as the number of independent trials involved in
> the selection of the investment strategy (N)."

**ANSWER TO THE STEP'S PREMISE, in the paper's own terms:**

- **"strategy variants tried" vs "parameter configurations evaluated" -- the papers make
  NO distinction.** A trial is one evaluated configuration, whatever dimension varied.
  A strategy swap and a `max_depth` tweak each cost exactly one trial.
- **Does enlarging a categorical dimension increase N multiplicatively, additively, or
  not at all?** **Not at all, if the extra options are never selected.** The
  multiplicative story ("10 windows x 5 thresholds x 3 logics = 150") is only correct
  for an EXHAUSTIVE grid, where the space size and the evaluation count coincide.
  pyfinagent's optimizer is a fixed-budget random single-parameter perturbation loop
  (`_propose_random`, `quant_optimizer.py:423-458`), so `max_iterations` alone sets the
  trial count. **The premise "adding three strategies inflates the trial pool" is
  therefore wrong twice over: wrong about this code (Q1) AND wrong about the theory
  (here).** It would only be right for an exhaustive-grid searcher.
- **Where the pool DOES enter the theory: the effective-N correction, and it points the
  OTHER WAY.** Appendix A.3 (quoted above) says N should be the number of *independent*
  trials, N̂ ≤ M, estimated from the average pairwise correlation of the trials. The repo
  passes the raw M (`quant_optimizer.py:256`), which over-states E[max SR] and therefore
  **over-deflates -- the SAFE direction** (the repo already knows this:
  `backend/autoresearch/strategy_backtest_adapter.py:45`,
  `strategy_candidate_producer.py:40`). A MORE diverse pool makes the trials LESS
  correlated, pushing N̂ up toward M, so a diverse pool makes the repo's crude M a
  *better* approximation, not a worse one. **Under the actual theory, enlarging the pool
  with genuinely different strategies IMPROVES the honesty of this repo's DSR.**
- **PBO is the statistic that genuinely cares about pool composition.** Its N is the
  number of COLUMNS in M, and the paper is explicit that N must be large and granular:
  > "if the investor is sensitive to values of φ < 1/10, it is clear that the range of
  > values that the logits can adopt must be greater than 10, and so N >> 10 is required."

  `analytics.py:205` (`PBO_MIN_TRIALS_GATE_GRADE = 10`) is a faithful transcription of
  that sentence. So **the pool-composition decision is a PBO question, not a DSR
  question** -- which is exactly the re-framing the contract needs.

---

## Q3 -- Per-candidate evidence table (derived from artifacts + code, not prose)

Columns: **FWD** = carries forward information (82.16 criterion, `backtest_engine.py:55-67`);
**TRADES** = produces non-zero trades on a measured sample; **FUND** = fundamentals-dependent
(82.21 AST rule, `backend/backtest/fundamentals_coverage.py:240`); **PICKED** = ever selected
and scored by the optimizer (derived from `quant_results.tsv` `params_json.strategy`,
537 data rows).

| Name | In registry | FWD | TRADES (measured) | FUND | PICKED by optimizer | Verdict for the pool |
|---|---|---|---|---|---|---|
| `triple_barrier` | yes (`backtest_engine.py:70`) | yes | YES -- 982-1020 trades full sample; 80 short window | no | 503 rows (incumbent; 23 BASELINE, 2 keep, 185 discard, 291 crash) | **KEEP** |
| `mean_reversion` | yes (`:71`) | yes | **UNMEASURED on the 82.3 sample** -- absent from BOTH 82.3 passes. Optimizer history: 2 rows, Sharpe -6.1324 and -3.8598, both `discard` | no | 2 rows, both discard | **KEEP but flag**: it can train, it just performed badly. Different case from qarp. |
| `meta_label` | yes (`:72`) | yes (shares `_compute_triple_barrier_label`) | **UNMEASURED** -- absent from both 82.3 passes | no | 1 row, `crash` | **KEEP with a diversity caveat**: registry maps it to the SAME label fn as `triple_barrier` (`:72`), differing only in the meta-layer in `_run_window`. Its PBO column will be near-collinear with TB's -- material for the pool-level PBO in Q5. |
| `stretch_regime` | yes (`:79`) | yes (82.2) | YES -- 714-792 trades full sample; 80 short window | no | **NEVER** (0 rows; only became selectable at 82.16) | **ADD**: best measured full-sample evidence of the three -- Sharpe max 0.8246 vs TB 0.6127, **PBO 0.1960 vs TB 0.7486** on the identical 2018-2025 sample. |
| `reversion_sigma` | yes (`:81`) | yes (82.2) | YES -- 1030-1058 trades full sample; 80 short window | no | **NEVER** | **ADD with a caveat**: full-sample Sharpe max 0.6781, PBO 0.3968 (both beat TB). But the 82.3 artifact's own note: *"reversion_sigma is purged at holding_days*1.5=135d against a 15d label horizon (backtest_engine.py:665 is strategy-blind, queued as 82.19) -- a WIN is clean, a LOSS is CONFOUNDED"*. Its win is therefore clean. |
| `qarp` | yes (`:80`) | yes (82.2) | **NO on the standard window** -- the 2026-08-03 smoke over 2018-01-01..2025-12-31 records `sharpe 0.0, dsr 0.0, n_trades 0`, and qarp was consequently OMITTED from the full-sample pass entirely. Trades only on 2024-07..2025-12 (40-80 trades, 55 daily returns) | **YES** (82.21 measured; `historical_fundamentals` has no rows before 2024-06-30) | **NEVER** | **DO NOT ADD (or add only behind the 82.21 coverage guard)**: qarp CANNOT TRAIN on the default `optimizer_best.json` window (`start_date 2018-01-01`, `end_date 2025-12-31`). Offering it to the optimizer means offering a member that returns a 0-trade, Sharpe-0.0 run ~1/7 of the time `strategy` is proposed -- a guaranteed `discard` that still consumes a ~20-min backtest and still increments `num_trials`. |
| `quality_momentum` | **NO** -- demoted (`:56`) | **no** (measured 880 labelled / 0 changed) | n/a | yes | 4 rows (3 crash, 1 discard @ -0.5942) | **STAYS OUT** (82.16) |
| `factor_model` | **NO** -- demoted (`:62`) | **no** (measured 0 labelled / 0 changed) | n/a | yes | 7 rows (4 crash, 3 discard @ -1.06..-1.21) | **STAYS OUT** (82.16) |
| `blend` | **NO** -- never a key since `9fbd9cd6` | n/a (no implementation) | n/a | 1 row (`fb7d367b-exp58`, `crash`) | **REMOVE** -- see Q4 |

**The distinction the step asked for, stated explicitly:** `qarp` is a *cannot-train* case
(the data does not exist on the configured window). `mean_reversion` is a *can-train,
performs-badly* case (two real runs, real negative Sharpes). These warrant different
treatment: a cannot-train member should be gated on data coverage, not deleted; a
can-train/bad member is exactly what a search is FOR and should stay.

**Recommended pool (7 members -> 5, with one conditional):**
`{triple_barrier, mean_reversion, meta_label, stretch_regime, reversion_sigma}` +
`qarp` **only when** the configured window is covered by the 82.21 fundamentals
availability check; `blend` removed. Deriving this needs a rule, not a literal -- see Q6.

---

## Q5 -- Cost of "re-run the DSR/PBO numbers": SPLIT THE TWO HALVES, they cost 4 orders of magnitude apart

**Measured runtime anchors, from the artifacts (not from docs):**

| Pass | Sample | Strategies x K | Wall clock | Per-config |
|---|---|---|---|---|
| `20260804T025319Z_phase_82_3_full_sample_3strat.json` | 2018-01-01..2025-12-31 | 3 x 8 = 24 | **30830.3 s = 8.56 h** | ~1167-1436 s |
| `20260804T041628Z_phase_82_3_short_window_4strat.json` | 2024-07-01..2025-12-31 | 4 x 8 = 32 | **4987.7 s = 1.39 h** | ~153-159 s |
| `20260803T175308Z_phase_82_3_candidate_comparison.json` | full sample, K=1 smoke | 1 x 1 | 166.8 s | 166.8 s |

So the project's "~20 min per walk-forward run" figure is confirmed for the FULL sample
(~19.5 min mean) and is ~2.6 min on the short window. All of it is $0 (quant-only regime,
no LLM).

### Half A -- the DSR comparison costs ZERO backtest time and MUST NOT be run as a backtest

Per Q1, `num_trials` and `variance_of_srs` are both pool-invariant. The honest measurement
is a **deterministic function evaluation**, not a backtest: hold the observed
`(sharpe, T, skew, kurt, V)` fixed at the values in `optimizer_best.json`
(`sharpe = 1.1704633657934074`) and evaluate `compute_deflated_sharpe` under both pools.
The result is **exact bit-identity**, provable and reproducible in milliseconds, and
recording *"DSR before = DSR after = X, identical to the last bit, because the pool never
enters the statistic"* is a STRONGER artifact than "we asserted it was negligible" --
it satisfies criterion 2's "MEASURED ... rather than asserted to be negligible" by
producing two numbers that are measured and equal. **Budget: seconds.**

### Half B -- the PBO comparison is real, and it CANNOT be done from existing artifacts

`compute_pbo` needs the raw `(T, N)` daily-return matrix. **The 82.3 artifacts persist only
`pbo_matrix_shape` (e.g. `[1661, 8]`), the scalar PBO and the column-correlation summary --
the matrices themselves were NOT saved** (verified: the per-run dicts carry only
`sharpe/dsr/net_of_cost_return_pct/turnover_rate/total_commission/n_trades/max_drawdown/n_daily_returns/elapsed_s`).
So the existing artifacts answer *"PBO of each strategy's own K=8 config sweep"* but cannot
be recombined into a *pool-level* PBO.

Also note what the existing per-strategy PBOs are NOT: they are within-strategy config
sweeps (Bailey Algorithm 2.3 as the artifact's own note says). The pool-composition
question is a DIFFERENT matrix -- columns spanning the pool members.

**Cheapest honest design, in increasing cost:**

| Option | What it measures | Runs | Wall clock | Verdict |
|---|---|---|---|---|
| **B0** (recommended floor) | DSR identity + re-use the EXISTING per-strategy full-sample PBOs as the pool members' individual PBOs, and report the pool-level statistic that IS derivable: the set of PBOs a pool exposes (pre-change pool exposes only TB's 0.7486; post-change pool exposes {0.7486, 0.1960, 0.3968}) | **0 new** | **0** | Honest, artifact-backed, zero cost. Its limitation must be stated: it is a per-member comparison, not a joint CSCV. |
| **B1** | True pool-level CSCV on the SHORT window, columns = configs across all runnable pool members, matrices persisted | 6 strategies x K=8 = 48 @ ~156 s | **~2.1 h** | Viable in one sitting. Needs `mean_reversion` + `meta_label` added (never run in 82.3). |
| **B2** | Same on the FULL sample | 48 @ ~1250 s | **~16.7 h** | NOT viable; do not start. |

**Recommendation for Main: do B0 + Half A to satisfy the criteria, and queue B1 as its own
step** (or fold it into 82.26, which is already about raising K). Starting B2 blind is the
failure mode the step's own prompt warns against.

**Interaction with step 82.26 (queued, `pending`) -- asked explicitly:**
82.26's criteria include *"a test asserts the phase-82.3 artifacts are correctly marked as
N=8 and therefore not gate-grade"*. `analytics.py:205` sets
`PBO_MIN_TRIALS_GATE_GRADE = 10`, and every 82.3 PBO is N=8, so **every number in the Q3
table above is DIRECTIONAL, not gate-grade, by this repo's own constant** -- 82.46 must say
so in its artifact or it will be asserting gate-grade evidence that 82.26 is about to mark
non-gate-grade. Worse for the short window: `analytics.py:266-271` marks columns
non-diverse above 0.99 mean correlation, and the short-window columns sit at 0.9670-0.9789
-- close to, but under, the boundary; the artifact's own `trial_diversity_note` already
says to read them as weak evidence. **The 82.46 artifact must label its PBO numbers
`gate_grade: false` and cite 82.26.** If 82.46 raises the pool, it also raises the N
available for a future pool-level CSCV, which is a genuine (positive) interaction with
82.26 worth recording.

---

## Q6 -- Guard design for a "decide a set" step: the traps, and the one that WILL bite

### TRAP 0 (measured, will happen on the first commit): closing 82.46 turns a currently-GREEN 82.16 test RED

`backend/tests/test_phase_82_16_label_forward_information.py` currently passes
**34 passed in 7.99s** (run 2026-08-06). Two of those tests hard-code `blend`:

- `:226` -- `extra = set(AVAILABLE_STRATEGIES) - set(STRATEGY_REGISTRY) - {"blend"}`
  (a carve-out; survives removal as a vacuous subtraction, but then silently re-permits
  a future `blend`).
- `test_optimizer_trial_pool_composition_is_pinned` (`:376-395`) asserts
  `previously_offered - now == {"quality_momentum", "factor_model"}` where
  `previously_offered` literally contains `"blend"`. **Remove `blend` from
  `AVAILABLE_STRATEGIES` and this assertion becomes
  `{"quality_momentum", "factor_model", "blend"} != {"quality_momentum", "factor_model"}`
  -> FAIL.** This is the same shape as the 82.39 finding (closing a step turns a green
  guard red). It is a TEST, not immutable criteria, so it may be updated -- but it was
  authored precisely to make this change visible, so updating it IS the "recorded decision
  being updated" that criterion 4 demands. **Main must update it in the same commit and say
  so, not discover it in CI.**

### TRAP 1 -- the tautology. A test that restates the list is not a guard.

`assert set(AVAILABLE_STRATEGIES) == {"triple_barrier", ...}` is a mirror, not a check: it
fails on ANY change including a correct one, and it proves nothing about the decision. The
house pattern (82.21 `label_fundamentals_dependent_strategies`,
`fundamentals_coverage.py:240-300`) is the opposite: **derive the set by an executable RULE
from a source of truth, then assert properties of the derived set.** For 82.46 the rule
has three clauses, each independently checkable:
1. member is in `STRATEGY_REGISTRY` (kills `blend` structurally, forever);
2. member is not in `NON_COMPARABLE_STRATEGIES` (already implied by (1), but assert it so
   a future re-registration cannot sneak past);
3. member is not fundamentals-dependent on an uncovered window, i.e. `qarp` is admitted
   only when `label_fundamentals_dependent_strategies()` says it is safe for the
   configured dates -- **reusing 82.21's function rather than writing a second rule**.

### TRAP 2 -- the "decision record" must be MACHINE-READ, or criterion 4 is unsatisfiable

Criterion 4: *"a guard fails if a strategy is added to or removed from the pool without the
recorded decision being updated."* This only works if the recorded decision is a data
structure the test imports, not prose in a markdown file. Shape that works:

```python
POOL_DECISION: dict[str, str] = {"triple_barrier": "<reason>", ...}   # name -> recorded rationale
```
then the guard asserts (a) `set(derived_pool) == set(POOL_DECISION)`, (b) every value is
non-empty (82.16 already uses this idiom on `NON_COMPARABLE_STRATEGIES[name].strip()`,
test at `:202`), and (c) the derivation rule -- not the literal -- produces
`derived_pool`. Adding a strategy to `STRATEGY_REGISTRY` then changes `derived_pool` and
fails (a) until `POOL_DECISION` gains an entry WITH a reason. **That is a real guard: it
cannot be satisfied by editing the test to match, only by recording a decision.**

### TRAP 3 -- the source-scan trap that bit 82.43

82.43's lesson, recorded verbatim at `backtest_engine.py:139-150`: *"the only guard that
could reach it was a source scan for token names -- which a mutation run defeated twice:
replacing `int(X.shape[1])` with a literal survived (the scan checked the key, not the
value), and deleting the `macro_series_min` entry survived because that same string still
appeared in a logger call below. A guard that reads source text cannot see either."*
Applied here: `blend` appears in a COMMENT at `backtest_engine.py:102` and in the
`quant_optimizer.py:73-76` comment block. **Any guard of the form "assert 'blend' not in
the source of quant_optimizer.py" is defeated by the comments and must not be used.**
Assert on the imported VALUE (`"blend" not in AVAILABLE_STRATEGIES`) and on BEHAVIOUR
(`resolve_strategy(name)[0] == name for every name in the pool` -- criterion 3's literal
requirement, and an executable one).

### TRAP 4 -- assert non-empty, and mutate the guard

Per the house rule (82.16 criterion 3, `feedback_mutation_test_guards_and_fixtures`):
assert `len(derived_pool) > 0` so an empty registry cannot pass vacuously, and ship a
fixture that registers a deliberately-bad member in a TEMPORARY copy of the registry and
proves the guard detects it (82.16's `test_demoted_methods_are_kept...` and its stub
fixture are the in-repo template). Mutate the STUB too.

### TRAP 5 -- do not let the guard depend on a live BigQuery read

82.21's coverage function reaches for `historical_fundamentals`. If clause (3) is wired
naively the pool guard becomes network-dependent and flaky. Inject the coverage verdict
(the 82.21 function already accepts `registry` / `engine_path` / `historical_data_path`
overrides at `fundamentals_coverage.py:240-243` -- that injectability is the precedent to
copy).

---

## Q4 -- `blend`: it is a REVERT ORPHAN. Remove it. (Definitive; git-archaeology)

`blend` was NOT vestigial-by-neglect and NOT an unimplemented aspiration. **It was a
real, complete implementation that got deleted by a bulk revert which forgot the
other half of the wiring.**

- **Born** `1f270641` "Phase 1.3: Amihud liquidity filter + strategy blending"
  (2026-03-25). The commit body: *"New 'blend' strategy: weighted vote across TB, QM,
  MR, and FM labels ... Threshold: normalized weighted sum > 0.3 for BUY, < -0.3 for
  SELL. Source: Dietterich (2000) 'Ensemble Methods in Machine Learning'."* It added
  `"blend": "_compute_blend_label"` to `STRATEGY_REGISTRY`, the method
  `_compute_blend_label(self, ticker, entry_date) -> int | None`, the four weight
  params, and `"blend"` to `AVAILABLE_STRATEGIES`.
- **Killed** `9fbd9cd6` "validate: pre-Phase-1.2 code confirms Sharpe 1.0142"
  (2026-03-28). Commit body: *"This confirms the Phase 1.2-1.9 improvements broke the
  strategy. Pre-Phase-1.2 simple inverse-vol sizing is the validated baseline.
  Reverting to this as the production code. Phase 1 branch preserved at
  'phase1-experimental' for reference."*
- **THE ASYMMETRY IS THE BUG.** `git show 9fbd9cd6 --stat` changes exactly five files:
  `backend.log`, `backend/backtest/backtest_engine.py`,
  `backend/backtest/backtest_trader.py`,
  `backend/backtest/experiments/mda_cache.json`,
  `frontend/src/app/backtest/page.tsx`. **`quant_optimizer.py` is NOT in that list.**
  The revert deleted the implementation and the registry entry; the *offer* survived
  untouched. `blend` has been a name with no referent for 4+ months.

**Collateral revert-orphans in the SAME file (found while answering this; they are
part of the same decision and the step should sweep them):**

- `quant_optimizer.py:109-113` -- `tb_weight` / `qm_weight` / `mr_weight` / `fm_weight`
  in `_PARAM_BOUNDS`, with the comment *"Strategy blend weights (Dietterich 2000):
  active when strategy='blend'"*. **Nothing reads them.** `grep -rn
  "tb_weight\|qm_weight\|mr_weight\|fm_weight" --include="*.py" backend/` returns
  ZERO hits outside `quant_optimizer.py` itself. They are 4 of the 24 numeric params,
  so ~4/26 = **15% of every random proposal is spent tuning parameters no code
  reads** -- each such iteration still increments `num_trials` at `:256` and so still
  deflates the DSR. This is a REAL and MEASURABLE trial-pool cost, unlike the one the
  step's premise names.
- `quant_optimizer.py:588` -- comment "Blend weights (read from `_strategy_params` by
  `_compute_blend_label`)" inside `_compute_feature_cache_key`; the named method does
  not exist.
- (Prior art, independently: `handoff/archive/misc/research_brief_phase_48_3_rotation_runner.md:130`
  already recorded "**NO -- dead key** (`_compute_blend_label` does not exist)".
  It was found in phase-48.3 and never actioned.)

**The corruption the step warns about has ALREADY HAPPENED, on disk, measured:**

`backend/backtest/experiments/results/baseline_blend_20260326T085310.json` records
`strategy_params.strategy == "blend"`, `analytics.sharpe == 0.7386211176221463`,
`deflated_sharpe == 0.9990331026133727`, `num_trials == 1` -- and commit `1ee4e5fe`
is literally titled *"Blend strategy baseline: Sharpe 0.7386, DSR 0.9990, +43.3%
return"*. That run predates the revert so it was genuinely blend at the time; but
**any re-run of that artifact today produces triple_barrier's numbers under the label
`blend`**, which is exactly the attribution corruption 82.16 made loud at
`backtest_engine.py:100-112`. There is also one optimizer TSV row that selected it:
`fb7d367b-exp58`, `2026-04-06T18:45:16`, `strategy: triple_barrier -> blend`,
`status=crash`.

**RECOMMENDATION: REMOVE, do not implement.** Reasons, in order of weight:
1. Implementing it means resurrecting a weighted vote over TB + **QM + MR + FM** --
   and 82.16 demoted QM and FM as non-comparable (no forward information), so the
   original ensemble is now 2/4 built on labels this repo has ruled inadmissible. A
   faithful re-implementation would be *knowingly* re-admitting them through a side
   door; a re-specified 3-way TB+MR+? blend is a NEW strategy that belongs in its own
   step with its own evidence, not in a pool-composition step.
2. The revert commit is an explicit, operator-authored judgment that the Phase-1.2-1.9
   family (blend included) "broke the strategy". Removal HONORS an existing decision;
   implementing REVERSES one, and reversing it needs its own evidence.
3. The implementation is not lost -- branch `phase1-experimental` is named in the
   revert commit as the preservation point, so removal is reversible.

**What breaks if `blend` is removed -- checked, not assumed:**

| Referent | Effect of removal |
|---|---|
| `backend/tests/test_phase_82_16_label_forward_information.py:226` | `extra = set(AVAILABLE_STRATEGIES) - set(STRATEGY_REGISTRY) - {"blend"}` -- the `- {"blend"}` becomes a no-op subtraction of an absent element. Still passes (set difference is total), but the carve-out should be deleted with it or it silently re-permits a future `blend`. **This is the one test that must be touched.** |
| `backend/backtest/experiments/optimizer_best.json` | `"strategy": "triple_barrier"` -- no reference. Unaffected. |
| `quant_results.tsv` | 1 historical row with `blend` in `params_json` (`fb7d367b-exp58`, status=crash). Read-only history; nothing parses `params_json.strategy` against the current pool. |
| `backend/meta_evolution/archetype_library.py` | see internal inventory -- `IMPLEMENTED_STRATEGY_IDS` is the one place named by `resolve_strategy`'s docstring (`backtest_engine.py:96-97`); VERIFY whether `blend` is in it (it lists QM/FM). |
| Frontend | no `blend` literal in `frontend/src/` (grep). Unaffected. |
| `resolve_strategy` | the `blend` mention at `backtest_engine.py:102` is a COMMENT naming the motivating example; removing the offer does not make the warning dead -- it still guards any unregistered name. |

---

## Q1 -- CRUX: what quantity is N in this repo's Deflated Sharpe? (MEASURED, traced end-to-end)

**ANSWER: N is the cumulative count of optimizer ITERATIONS actually executed
(plus any warm-started prior). It is NOT the pool size, NOT the number of
distinct parameter configurations in the search space, and `AVAILABLE_STRATEGIES`
never touches it.**

Full trace, every hop re-derived on the live tree 2026-08-06:

| Hop | file:line | What happens |
|-----|-----------|--------------|
| 1 | `backend/backtest/quant_optimizer.py:151` | `self.num_trials = 0` in `__init__` |
| 2 | `backend/backtest/quant_optimizer.py:226` | `self.num_trials = 1` after the cold-start baseline |
| 2b | `backend/backtest/quant_optimizer.py:852` / `:875` | warm start: `self.num_trials = prior` (from `optimizer_best.json`) or `= self._UNKNOWN_PRIOR_FLOOR` (82.25) |
| 3 | `backend/backtest/quant_optimizer.py:256` | `self.num_trials += 1` -- **once per `while` iteration**, unconditionally, before the trial even runs |
| 4 | `backend/backtest/quant_optimizer.py:285` | `report = generate_report(result, num_trials=self.num_trials)` |
| 5 | `backend/backtest/analytics.py:766-777` | `compute_deflated_sharpe(observed_sr=..., num_trials=max(num_trials, 1), variance_of_srs=sr_variance, ...)` |
| 6 | `backend/backtest/analytics.py:429-432` | `e_max_sr = sqrt(var_srs) * [(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(N·e))]` with `N = max(num_trials, 2)` |

**Grep proof of the negative:** `grep -n "AVAILABLE_STRATEGIES" backend/` returns exactly
two live hits -- `quant_optimizer.py:86` (definition) and `:126`
(`_CATEGORICAL_PARAMS["strategy"]`). It reaches `_propose_random` and nothing else.
There is no path from `len(AVAILABLE_STRATEGIES)` to `compute_deflated_sharpe`.

**SECOND, LESS OBVIOUS HALF -- the V input is also pool-invariant.**
Bailey's deflation has TWO inputs: N and V (`variance_of_srs`), and V enters
`e_max_sr` multiplicatively as `sqrt(V)` (`analytics.py:429`). In the paper V is the
variance of the Sharpe ratios **across the N trials**. This repo computes it
differently -- `analytics.py:753-754`:

```python
window_sharpes = [w.sharpe_ratio for w in result.windows if w.sharpe_ratio != 0]
sr_variance = float(np.var(window_sharpes)) if len(window_sharpes) > 1 else 0.5
```

That is the variance of per-WINDOW Sharpes **inside a single backtest** -- a
within-run dispersion statistic, computed per trial, with no memory of the other
trials in the run. So enlarging the strategy pool cannot move V either, except
indirectly through whichever single config happens to be the reported best.

**CONSEQUENCE FOR THE STEP'S PREMISE (say it plainly, as instructed):**
the step's framing -- *"the trial pool is a DIRECT input to Deflated Sharpe"* --
is **FALSE for this implementation as written**. Adding qarp / reversion_sigma /
stretch_regime to `AVAILABLE_STRATEGIES` changes the DSR of a run by **exactly
zero** at fixed iteration count. It is not a small effect; it is an identically-zero
effect, and the 82.16 comment at `quant_optimizer.py:77-83` overstates the coupling.
(NOTE the same comment is right about the *symmetry* of the argument -- it is only
the mechanism it names that does not exist.)

**What the pool size ACTUALLY affects (this is the real content of the step):**

1. **Search-trajectory dilution, not deflation.** `_propose_random` picks one
   param uniformly from `list(_PARAM_BOUNDS) + list(_CATEGORICAL_PARAMS)`
   (`quant_optimizer.py:430`), then for a categorical draws uniformly from its
   choices (`:436-437`). With 24 numeric params + 2 categoricals, `strategy` is
   proposed on ~1/26 of iterations; conditional on that, each pool member is drawn
   with probability 1/|pool|. Going 6 -> 7 pool members drops the per-strategy draw
   rate from 1/6 to 1/7 of an already-1/26 event. **The pool size is a prior over
   the search, not a term in the statistic.**
2. **Attribution and the recorded provenance**, which is where the real damage is
   (see Q4 on `blend`).
3. **The honest multiple-testing story.** Bailey's N is the number of trials
   CONDUCTED, so a bigger menu that is never sampled costs nothing -- but a bigger
   menu that IS sampled and then discarded still counts, and the repo already counts
   it correctly because `:256` increments per iteration regardless of outcome.

**Therefore the contract should NOT be written as "re-run DSR to measure the pool
effect on DSR".** That experiment has a provable answer (zero) and spending
~20 min/run to observe it would be theatre. See Q5 for what to measure instead.

### Q1 addendum -- I MEASURED the invariance rather than asserting it (2026-08-06)

`inspect.signature(compute_deflated_sharpe)` ->
`['observed_sr', 'num_trials', 'variance_of_srs', 'skewness', 'kurtosis', 'T',
'periods_per_year']`. **There is no pool parameter.** Evaluating at
`observed_sr = 1.1704633657934074` (the live `optimizer_best.json` Sharpe), `V=0.5`,
`skew=0`, `kurt=3`, `T=1661`, `ppy=252`:

```
DSR(pool_before, len=6) = 0.5581552457854054
DSR(pool_after,  len=7) = 0.5581552457854054   identical: True
```

**But look at the N sensitivity on the SAME inputs -- this is the number that matters:**

| N | DSR |
|---|-----|
| 1 | 0.9802364550 |
| 2 | 0.9802364550 |
| 10 | 0.5581552458 |
| 26 | 0.2578993685 |
| 100 | 0.0562703344 |

(N=1 and N=2 coincide because of `max(num_trials, 2)` at `analytics.py:430-431` -- the
82.25 finding.) **DSR collapses from 0.98 to 0.26 between N=2 and N=26.** So the DSR is
violently sensitive to *how many iterations you burn* and completely insensitive to
*what is on the menu*. That reframes the whole step: the four dead `*_weight` params
found in Q4 waste ~15% of every proposal, and each wasted iteration is a real,
steeply-priced DSR cost. **Removing the dead params is a bigger DSR intervention than
any pool-membership decision could be.**

---

## Recency scan (2024-2026) -- MANDATORY SECTION

**Searched** (three variants; see the composition subsection above) for 2024-2026
literature on trial-counting for DSR/PBO, effective-N, and strategy-pool composition.

**Result: 2 new findings that COMPLEMENT (do not supersede) the 2014/2016 primaries,
plus 1 gap.**

1. **arXiv:2507.07107 (2025)** -- current practice still counts *evaluations*, not
   search-space size: *"With N≈50 effective configurations tried during development"*.
   It also adds a caveat the 2014 paper does not stress and which bears directly on
   82.16/82.21: *"DSR cannot correct for biases that affect every configuration
   uniformly (e.g. look-ahead in factor construction)."* **This is the literature
   endorsement for why 82.16 REMOVED non-forward strategies instead of deflating for
   them: a uniform label defect is invisible to DSR at any N.**
2. **arXiv:2512.22476 AutoQuant (2025)** -- pool CURATION by ex-ante thresholds is the
   contemporary pattern: *"absolute robustness thresholds are applied to remove
   underperforming configurations ... the top K candidates are retained"*, and it
   downgrades DSR/PBO to *"a robustness check"* rather than a primary result. This is
   direct support for 82.46 deciding the pool by a RULE with recorded thresholds rather
   than by a literal.
3. **GAP / honest negative:** the 2024 peer-reviewed comparison (Arian, Norouzi M.,
   Seco, *Expert Systems with Applications*, "Backtest Overfitting in the Machine
   Learning Era") is the most relevant recent work -- its headline is that CPCV beats
   walk-forward and K-fold on both PBO and DSR. **I could not read it in full: SSRN
   returned HTTP 403 and ScienceDirect is paywalled.** It is recorded snippet-only and
   is NOT counted toward the gate. Its relevance to 82.46 is indirect (this repo uses
   walk-forward, not CPCV) but it is a live argument for 82.26.
4. **No 2026 primary literature on trial-counting surfaced at all.** The 2014/2016
   definitions remain the operative ones; nothing supersedes them.

---

## Key findings (1-line each, cited)

1. **This repo's DSR N is the optimizer iteration count, not the pool size** --
   `quant_optimizer.py:256` -> `:285` -> `analytics.py:766-777` -> `:429-432`; measured
   invariance printed above.
2. **The theory agrees**: a trial is *"a particular model configuration tried by the
   researcher"* (PBO paper, Algorithm 2.3, https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
   Menu size is not trial count for a fixed-budget random search.
3. **The step's stated premise is therefore FALSE as written** and the contract must say
   so rather than quietly measure something else.
4. **Effective-N points the opposite way**: *"using M instead of N will overstate
   E[max{SR}]"* (DSR paper A.3) -- the repo's raw-M is conservative, and a MORE diverse
   pool makes it less conservative-by-error.
5. **PBO, not DSR, is the pool-sensitive statistic** -- *"N >> 10 is required"* (PBO
   paper); `analytics.py:205` already encodes it as `PBO_MIN_TRIALS_GATE_GRADE = 10`.
6. **`blend` is a revert orphan from `9fbd9cd6`** -- implementation deleted, offer left
   behind because that commit never touched `quant_optimizer.py`. Remove it.
7. **Four dead params (`tb/qm/mr/fm_weight`) burn ~15% of every proposal** and each
   burned iteration costs real DSR (0.98 -> 0.26 from N=2 to N=26).
8. **`qarp` cannot train on the configured window** (0 trades, Sharpe 0.0 on
   2018-2025); `mean_reversion` can train and performs badly. Different cases, different
   remedies.
9. **`stretch_regime` is the strongest add**: full-sample PBO 0.1960 vs incumbent
   triple_barrier's 0.7486 on the identical sample.
10. **Every 82.3 PBO is N=8 and therefore NOT gate-grade by this repo's own constant** --
    82.46 must label it and cite 82.26.
11. **Closing this step turns two currently-green 82.16 tests red** (34 passed today);
    `test_optimizer_trial_pool_composition_is_pinned` fails on `blend` removal.

---

## Internal code inventory

| File | Lines (re-derived 2026-08-06) | Role | Status |
|---|---|---|---|
| `backend/backtest/quant_optimizer.py` | 960 total; `:67-86` pool derivation, `:89-119` `_PARAM_BOUNDS`, `:109-113` DEAD blend weights, `:125-128` `_CATEGORICAL_PARAMS`, `:151/:226/:256/:285` num_trials, `:318` DSR gate, `:423-458` `_propose_random`, `:588` stale comment, `:631-651` `_log_experiment`, `:746-881` warm-start | Trial pool + DSR consumer | **Primary change site.** Contains 4 dead params + 1 dead pool member |
| `backend/backtest/backtest_engine.py` | 1841 total; `:55-67` `NON_COMPARABLE_STRATEGIES`, `:69-82` `STRATEGY_REGISTRY` (6 keys), `:84-121` `resolve_strategy`, `:1421` dispatch | Source of truth for the pool | Correct; `blend` mention at `:102` is a COMMENT (source-scan trap) |
| `backend/backtest/analytics.py` | 882 total; `:197-205` PBO constants, `:208-273` `compute_pbo_checked`, `:276-328` `compute_pbo`, `:384-447` `compute_deflated_sharpe`, `:741-796` `generate_report` (`:753-754` the non-Bailey V) | DSR/PBO implementation | **`variance_of_srs` is per-WINDOW dispersion, not the paper's across-trial variance -- an undocumented deviation, worth its own queued step** |
| `backend/backtest/fundamentals_coverage.py` | 326 total; `:240-300` `label_fundamentals_dependent_strategies` | 82.21 AST-derived subset | **The precedent to copy** (injectable params at `:240-243`) |
| `backend/tests/test_phase_82_16_label_forward_information.py` | `:205-212`, `:216-229`, `:376-395` | Pool guards | 34 passed today; **2 tests hard-code `blend`** |
| `backend/meta_evolution/archetype_library.py` | `:31-33` `IMPLEMENTED_STRATEGY_IDS`, `:42-45` docstring, `:89` validator | Second, DRIFTED copy of the pool | **`{triple_barrier, quality_momentum, mean_reversion, factor_model, meta_label, blend}` -- still lists BOTH 82.16-demoted names AND `blend`, and omits all three 82.2 candidates. `resolve_strategy`'s docstring (`backtest_engine.py:96-97`) names it as the live caller that can request a demoted name. This is a THIRD drifted list the step should sweep or queue.** |
| `backend/autoresearch/strategy_backtest_adapter.py` | `:132-268` matrix assembly + `compute_pbo_checked` | PBO producer | Uses `num_trials=K` (over-deflates, safe) |
| `backend/backtest/experiments/optimizer_best.json` | 26 lines | Live best | `strategy: triple_barrier`, sharpe 1.1704633657934074, dsr 0.9525811126193078, **no `num_trials` key** (82.25) |
| `backend/backtest/experiments/quant_results.tsv` | 539 lines / 537 data rows | Trial history | Per-strategy counts in Q3 |
| `scripts/harness/run_82_3_candidate_backtests.py` | -- | The 82.3 runner | **The script to extend for Q5 option B1; must be changed to PERSIST the matrices** |
| `scripts/harness/run_optimizer.py` | `:110-112` | Sets `optimizer.lock_strategy = True` | Only writer of `lock_strategy`; when set the categorical is excluded entirely (`quant_optimizer.py:431-432`) |

---

## Consensus vs debate (external)

**Consensus:** (a) a trial = one evaluated configuration; (b) the trial count must be
reported (Bailey 2014/2016, Palomar textbook, both 2025 preprints); (c) N must be large
for PBO (N >> 10).

**Debate / unsettled:** (a) how to estimate *effective* N -- Bailey A.3 offers the
average-correlation interpolation and then immediately concedes *"two problematic
aspects should be highlighted in connection with 'average correlation' formulat[ions]"*;
later Lopez de Prado work substitutes ONC clustering. This repo does neither and uses
raw M -- defensible, conservative, but should be labelled. (b) Whether DSR/PBO are
primary results or robustness checks -- AutoQuant (2025) demotes them to checks;
Bailey treats DSR as the decision statistic. (c) Whether walk-forward is adequate at all
vs CPCV (Arian et al. 2024) -- relevant to 82.26, out of scope here.

## Pitfalls (from the literature)

1. **A uniform bias is invisible to DSR at any N** (arXiv:2507.07107). Deflation is not a
   substitute for removing a broken label -- the 82.16 doctrine is literature-correct.
2. **Using M where N̂ is meant over-states E[max SR]** (DSR A.3) -- conservative, but
   don't call it exact.
3. **A small-N PBO is not merely weak, it is granularity-broken** (PBO paper: too few
   distinct ω values -> discontinuous logits -> estimation error). `compute_pbo`'s
   silent `0.0` at N<2 is the local instance, already guarded by `compute_pbo_checked`.
4. **Correlated columns make PBO uninformative however large N is** -- the repo's own
   `columns_diverse` check (`analytics.py:269-271`) and the 82.3 short-window
   correlations of 0.9670-0.9789.
5. **Reporting a pool decision without the thresholds is the same failure as reporting a
   backtest without the trial count** (AutoQuant's ex-ante threshold table is the
   counter-pattern to copy).

## Application to pyfinagent

1. **Re-frame criterion 2 honestly.** Report DSR-before == DSR-after as an exact,
   measured identity with the trace (Q1) explaining why, and put the real
   pool-sensitivity measurement on PBO. Do NOT run an 8.6-hour full-sample sweep to
   observe a zero (`quant_optimizer.py:285` + `analytics.py:766`).
2. **Derive the pool by a rule** in `quant_optimizer.py:86`, replacing
   `list(_STRATEGY_REGISTRY.keys()) + ["blend"]` with a function whose clauses are
   registry-membership, non-demotion, and (for `qarp`) the 82.21 coverage verdict --
   mirroring `fundamentals_coverage.py:240`.
3. **Delete `blend` and the four dead weight params**; update the stale comment at
   `quant_optimizer.py:588` and the 82.16 tests at `:226` and `:376-395` in the same
   commit, and say in the artifact that you did.
4. **Sweep or queue `archetype_library.py:31-33`** -- it is a third drifted copy and
   `resolve_strategy`'s own docstring names it as the reason the coercion path is live.
5. **Label every PBO number `gate_grade: false` (N=8 < 10)** and cite 82.26.
6. **Queue, do not do here:** (a) `analytics.py:753-754`'s non-Bailey `variance_of_srs`;
   (b) the pool-level CSCV (Q5 option B1, ~2.1 h).

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL -- **7** (2 peer-reviewed primaries
      via `pdfplumber` full-text, 2 preprints, 1 textbook, 1 official doc, 1 academic lab)
- [x] 10+ unique URLs total -- **22**
- [x] Recency scan (2024-2026) performed + reported -- dedicated section above, incl. one
      honest negative (403/paywall) and one "no 2026 primaries" result
- [x] Full papers/pages read, not abstracts -- the two primaries were extracted to
      45,402 and 63,988 characters respectively and quoted verbatim from body/appendix
- [x] file:line anchors for every internal claim -- all re-derived on the live tree
      2026-08-06, none cited from memory

Soft checks:
- [x] Internal exploration covered every module named in the prompt, plus two the prompt
      did not name (`archetype_library.py`, `scripts/harness/run_optimizer.py`)
- [x] Contradictions/consensus noted -- incl. the finding that the step's own premise and
      the 82.16 comment at `quant_optimizer.py:77-83` are wrong
- [x] Claims cited per-claim
- [ ] **Brief length: ~5,200 words vs the `complex` tier's <=1,500-word guide. OVER by
      ~3.5x, declared honestly rather than trimmed** -- the overrun is measured evidence
      (per-strategy artifact tables, verbatim primary-source quotes, the DSR sensitivity
      table). Trimming would have removed measurements Main needs to write the contract.
- [ ] Tool-call budget: ~44 vs the `complex` guide of <=30. Driven by the >=5 read-in-full
      floor plus two PDF-extraction fallbacks and two dead URLs (404/403).

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 15,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_82.46.md",
  "gate_passed": true
}
```

