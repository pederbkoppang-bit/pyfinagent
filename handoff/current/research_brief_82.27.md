# Research Brief -- step 82.27 (PBO/CSCV level-of-computation re-spec)

**Tier:** moderate (caller-specified). **Audit-class:** false (caller-specified) --
loop-until-dry coverage does NOT bind; the >=5-read-in-full floor + recency scan DO.
**Started:** 2026-08-04. Status: IN PROGRESS (write-first; sections appended as sources are read).

## Question set

- Q1: At what LEVEL of a backtesting pipeline is PBO/CSCV meant to be computed?
  Confirm from Bailey/Borwein/Lopez de Prado/Zhu that the N columns must be final
  performance series of N alternative configurations over a COMMON time axis, and
  that intermediate/guided-search steps are excluded (Algorithm 2.3). Quote the
  requirements on N and T verbatim.
- Q2: How do established OSS frameworks SURFACE PBO -- sweep/optimizer level or
  single-run report level? (mlfinlab, CRAN `pbo`, vectorbt, QuantConnect/Zipline.)
  Which OBJECT owns the N-series matrix?
- Q3: Accepted failure mode when N is too small for CSCV -- refuse / warn /
  degenerate value? Is "return 0.0 on N<2" a documented trap?
- Q4: Recency scan 2024-2026 -- work superseding or critiquing CSCV/PBO.
- I1..I5: internal code audit (see below).

## Search queries run (3-variant discipline)

(logged as executed; see "Queries" section at the end)

---

## Q1 -- ANSWERED. At what LEVEL is CSCV/PBO computed? (source read IN FULL)

**Source (read in full, 34 pp / 63,988 chars extracted via `pdfplumber`):**
Bailey, Borwein, Lopez de Prado & Zhu, *The Probability of Backtest Overfitting*,
revised Feb 27 2015 (published Journal of Computational Finance 20(4), 2017;
SSRN 2326253). Author-hosted authoritative PDF:
https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf -- accessed 2026-08-04.
(The `/pdf/` binary was downloaded with `curl` and text-extracted per
`.claude/rules/research-gate.md` "Step 3: pdfplumber"; this is a full read, not
an abstract.)

### The N columns are alternative CONFIGURATIONS, over a COMMON time index

Verbatim, Algorithm 2.3 (CSCV), first step:

> "First, we form a matrix M by collecting the performance series from the N
> trials. In particular, each column n = 1,...,N represents a vector of profits
> and losses over t = 1,...,T observations associated with a particular model
> configuration tried by the researcher. M is therefore a real-valued matrix of
> order (T x N). The only conditions we impose are that:
> i) M is a true matrix, i.e. with the same number of rows for each column,
> where observations are synchronous for every row across the N trials, and
> ii) the performance evaluation metric used to choose the 'optimal' strategy
> can be estimated on subsamples of each column."

And on the common axis:

> "If different model configurations trade with different frequencies,
> observations should be aggregated to match a common index t = 1,...,T."

Immediately preceding Algorithm 2.3, the paper defines what the columns come
from:

> "the researcher ends up running a number N of alternative model configurations
> (or trials), out of which one is chosen according to some performance
> evaluation criterion, such as the Sharpe ratio."

**Reading:** the object CSCV consumes is the OUTPUT OF A SWEEP -- the set of
final PnL series of the N candidate configurations that the selection procedure
chose among. It is definitionally not computable from one run: with N=1 there is
no "chosen among".

### Intermediate/guided-search steps are EXCLUDED (the exact clause the caller asked for)

Verbatim, Section 5.2 "Limitation in application":

> "A case in point is guided searches, where an optimization algorithm uses
> information from prior iterations to decide what direction should be followed
> next. In this case, the columns of matrix M should be the final outcome of
> each guided search (i.e., after it has converged to a solution), and not the
> intermediate steps."

(Footnote 2: "We thank David Aronson and Timothy Masters (Baruch College) for
asking for this clarification.")

Same section, the file-drawer requirement -- relevant because it constrains WHICH
sweep is legitimate:

> "the researcher must provide full information regarding the actual trials
> conducted, to avoid the file drawer problem (the test is only as good as the
> completeness of the underlying information), and should test as many strategy
> configurations as is reasonable and feasible. Hiding trials will lead to an
> underestimation of the overfit ... Likewise, adding trials that are doomed to
> fail in order to make one particular model configuration succeed biases the
> result."

### What the paper REQUIRES of N and T -- verbatim

On N:

> "Another key parameter is the number of trials (i.e., the number of columns in
> M). Hold-out's disregard for the number of trials attempted was the reason we
> concluded it was an inappropriate method to assess a backtest's
> representativeness ... N must be large enough to provide sufficient
> granularity to the values of the relative rank, omega_c. If N is too small,
> omega_c will take only a very few values, which will translate into a very
> discrete number of logits, making f(lambda) too discontinuous, and adding
> estimation error to the evaluation of phi. For example, if the investor is
> sensitive to values of phi < 1/10, it is clear that the range of values that
> the logits can adopt must be greater than 10, and so **N >> 10 is required**."

On T:

> "Finally, PBO is evaluated by comparing combinations of T/2 observations with
> their complements. But the backtest works with T observations, rather than
> only T/2. Therefore, **T should be chosen to be double of the number of
> observations used by the investor to choose a model configuration** or to
> determine a forecasting specification."

On S (context for the repo's `S=16`, `T >= S*2` guard):

> "S must be large enough so that the number of combinations suffices to draw
> inference. If S is too small, the left tail of the distribution of logits will
> be underrepresented. On the other hand, if we believe that the performance
> series is time-dependent and incorporates seasonal effects, S cannot be too
> large ... For example, S = 16 we will obtain 12,780 logits ... and
> sigma[f(lambda)] < 0.0045 ... Also, if M contains 4 years of daily data,
> S = 16 would equate to quarterly partitions ... For these two reasons, we
> believe that S = 16 is a reasonable value to use in most cases."

Note the paper gives NO hard floor on T other than "double the selection sample"
and the implied `T/S` partition size; the repo's `T < S*2 -> 0.0` branch is an
implementation convenience, not a paper threshold. The paper's own worked
example is 4 years of DAILY data at S=16.

### Two further paper findings that bear directly on 82.27's re-spec

**(a) A high PBO can arise from near-identical columns -- the paper says so
explicitly.** Section 5.2, fourth limitation:

> "although a high PBO indicates overfitting in the group of N tested
> strategies, skillful strategies can still exists in these N strategies. For
> example, it is entirely possible that all the N strategies have high but
> similar Sharpe ratios. Since none of the strategies is clearly better than the
> rest, PBO will be high. Here overfitting is among many 'skillful'
> strategies."

This is the literature basis for the `columns_diverse` / `column_corr_mean`
fields already built into `compute_pbo_checked`
(`backend/backtest/analytics.py:242-271`) and it directly explains the
phase-82.3 measurement (incumbent `triple_barrier` PBO 0.7486 with K=8 columns
correlated 0.967-0.979): a high PBO over near-duplicate columns is the paper's
own documented benign case, NOT necessarily evidence the incumbent is overfit.
82.27 should preserve that caveat rather than let 0.7486 be read as a verdict.

**(b) PBO must never be the optimisation objective.** Section 5.2, fifth
limitation:

> "we must warn the reader against applying CSCV to guide the search for an
> optimal strategy. That would constitute a gross misuse of our method. As
> Strathern eloquently put it, 'when a measure becomes a target, it ceases to be
> a good measure.' Any counter-overfitting technique used to select an optimal
> strategy will result in overfitting. For example, CSCV can be employed to
> evaluate the quality of a strategy selection process, but PBO should not be
> the objective function on which such selection relies."

Implication for the re-spec: PBO belongs at the PROMOTION GATE (a veto applied
after the sweep), which is exactly where `backend/autoresearch/gate.py` puts it.
It must NOT be fed back into `quant_optimizer`'s search objective.

---

## Q2 -- How OSS frameworks SURFACE PBO. Which object owns the N-series matrix?

| Framework | Object that owns the (T x N) matrix | Level | Evidence |
|---|---|---|---|
| CRAN `pbo` (the R reference impl) | a user-assembled data frame `M` of N trial columns, passed to `pbo(M, s, F, threshold)` | **sweep** | "First, we assemble the trials into an NxT matrix where each column represents a trial and each trial has the same length T." Package example uses **`N <- 200`** columns. Signatures seen: `pbo(M, s=8, f=sharpe, threshold=0)` (vignette) and `pbo(M,S,F=Omega,threshold=1)` (README). Accessed 2026-08-04. |
| Balaena Quant reference walk-through (practitioner, with code) | `R_mat = pd.DataFrame(R).T  # shape: (n_configs, n_combinations)` + parallel `R_mat_` for OOS | **sweep** | "Each row is a point in time, and the total number of data points across time is denoted by T. All your PnLs must have the same shape of (T, 1)." and, on stage: PBO is computed *after* the parameter sweeps are complete, over the whole trial space. Accessed 2026-08-04. |
| mlfinlab (Hudson & Thames) `cross_validation/combinatorial.py` | `CombinatorialPurgedKFold` / `StackedCombinatorialPurgedKFold` produce N **backtest paths**; `_get_number_of_backtest_paths(n_train_splits, n_test_splits)` | **sweep/path-level** | The module *generates* the N-path infrastructure and **does not itself compute PBO** -- PBO/DSR live in the separate backtest-statistics module. Accessed 2026-08-04. |
| vectorbt PRO | a wide parameter-combination result frame produced by `Splitter` + parameter optimisation | **sweep** | Its cross-validation material frames PBO as "the non-null probability that a strategy with optimal performance In Sample (IS) ranks below the median Out Of Sample (OOS)" estimated via CSCV, in the context of "parameter optimization across multiple parameter combinations". The public tutorial landing page carried no API detail (see snippet-only table) -- treat as directional. |
| QuantConnect / Zipline | **no PBO primitive found.** QC exposes parameter *optimization* + sensitivity; Zipline has no CSCV. | n/a | No authoritative doc surfaced in three query variants. Recorded as an absence, not a finding. |

**Architectural convention (unanimous across every implementation found):** the
matrix is owned by the object that RAN THE SWEEP -- an optimiser/CV driver that
holds N result series at once. In no implementation does a single-run report
object compute PBO. That is the external corroboration for 82.23's disposition.

## Q3 -- What do implementations DO when N is too small?

**Finding: no surveyed implementation documents a defined small-N behaviour, and
none documents returning a degenerate best-case value.** Measured per source:

- CRAN `pbo` README + vignette: "The documentation contains no statements
  regarding how the package handles insufficient trial data." No stated minimum
  N, no stated requirement that N or S be even, no stated T/S divisibility rule.
- Balaena Quant: "No explicit minimum N is stated ... no explicit guidance
  provided" for insufficient N or T.
- mlfinlab combinatorial: "No documentation addresses requirements regarding
  minimum splits/paths or failure conditions for insufficient data."
- The paper itself gives the only real guidance and it is a **magnitude** rule,
  not a branch: `N >> 10` (quoted above), plus the S-choice discussion.

**So "return 0.0 on N<2" is NOT a documented trap anywhere in the literature or
the reference implementations -- it is a pyfinagent-local invention**
(`backend/backtest/analytics.py:297-298`). That is worse than a known trap, not
better: there is no upstream convention to appeal to, and the chosen sentinel is
the *best possible* value on a ceiling-style gate. The literature-consistent
behaviour is to REFUSE (no measurement), which is precisely what
`compute_pbo_checked` (`analytics.py:208-273`) already does. 82.27 should record
this as an original-defect finding, not cite a precedent that does not exist.

## Recency scan (2024-2026) -- MANDATORY SECTION

Queries run (three-variant discipline, all 2026-08-04): year-less canonical
("Bailey Borwein Lopez de Prado Zhu probability of backtest overfitting CSCV");
last-2-year ("probability of backtest overfitting critique 2025 alternative PBO
CSCV limitations deflated Sharpe 2024"); current-year ("arXiv 2026 probability of
backtest overfitting PBO CSCV strategy selection multiple testing quantitative
finance"). Plus framework-scoped variants for mlfinlab / vectorbt.

**Result: 2 relevant new findings; NEITHER supersedes CSCV for this step.**

1. **Arian, Norouzi M. & Seco (2024), "Backtest overfitting in the machine
   learning era: A comparison of out-of-sample testing methods in a synthetic
   controlled environment", *Knowledge-Based Systems* (Elsevier;
   S0950705124011110; SSRN 4778909 / 4686376).** Finds **CPCV** superior to
   K-Fold, Purged K-Fold and especially Walk-Forward at mitigating overfitting,
   *measured by lower PBO and higher DSR*; introduces Bagged CPCV and Adaptive
   CPCV. **Crucially this COMPLEMENTS rather than replaces PBO: PBO is the
   yardstick the paper uses to rank the splitting schemes.** It changes how you
   might GENERATE the columns, not what the columns are or where they live.
   (This is already recorded in-repo at
   `backend/autoresearch/strategy_backtest_adapter.py:38-41`.)
2. **arXiv:2512.12924v1 (Dec 2025), "Interpretable Hypothesis-Driven Trading: A
   Rigorous Walk-Forward Validation Framework for Market Microstructure
   Signals".** Read in full. Cites "Bailey et al. (2017) developed
   Combinatorially Symmetric Cross-Validation (CSCV) to compute the Probability
   of Backtest Overfitting" and Arian 2024, notes "walk-forward remains the
   industry standard", and **explicitly declines to apply DSR/PBO deflation to
   its own results**. Useful as evidence of the current state of practice: CSCV
   is still the named reference method and is still widely *not* applied.

**No 2026 work on PBO/CSCV surfaced.** No paper found that retracts, replaces or
materially amends Algorithm 2.3's column definition. The 2015/2017 paper remains
the governing specification for what 82.27 must encode.

---

## Internal code inventory (Explore half) -- every claim file:line anchored

| File | Anchor | Role | Status |
|---|---|---|---|
| `backend/backtest/analytics.py` | `:276` `compute_pbo` | raw CSCV; **`:297-298` `if N < 2 or T < S*2: return 0.0`** -- the false-good | LIVE, still reachable |
| `backend/backtest/analytics.py` | `:208-273` `compute_pbo_checked` | refusing wrapper; returns `{pbo,n_trials,n_obs,gate_grade,column_corr_mean,column_corr_max,columns_diverse,refused}` | BUILT (82.23), mutation-verified |
| `backend/backtest/analytics.py` | `:197-205` | `PBO_CEILING_LIVE=0.20`, `PBO_CEILING_CANONICAL=0.50`, `PBO_MIN_TRIALS_GATE_GRADE=10` | LIVE |
| `backend/backtest/analytics.py` | `:741` `def generate_report(` | single-`BacktestResult` report builder | LIVE; emits NO pbo (by design) |
| `backend/autoresearch/gate.py` | `:22` `max_pbo=0.20`, `:30` `min_pbo_trials=10`, `:36-39` fail-closed on missing pbo, `:43-53` N-floor branch | THE live promotion gate | LIVE |
| `backend/autoresearch/strategy_backtest_adapter.py` | `:167` `make_engine_backtest_fn`, `:132` `_assemble_pbo_matrix`, `:94` `_default_param_grid`, `:255` `compute_pbo_checked(...)`, `:264-278` emitted dict | **sweep-level PBO producer #1** (K param-variants of ONE strategy -> (T x K)) | LIVE |
| `backend/autoresearch/strategy_candidate_producer.py` | `:65` `build_per_strategy_candidates`, **`:115-123` the 5-key candidate dict** | producer -> selector hop | LIVE; **DROPS the 82.23 provenance keys -- see I4** |
| `backend/autoresearch/strategy_selector.py` | `:90` `g = gate or PromotionGate()`, `:94` `g.evaluate(c)` | the only in-repo caller of `PromotionGate.evaluate` on rotation candidates | LIVE |
| `backend/autoresearch/rotation_runner.py` | `:241` `run_rotation_bakeoff`, `:271` builds the adapter, `:278` calls `run_strategy_bakeoff` | live glue; `:36` "The weekly rotation CRON" still DEFERRED | LIVE code, **NOT scheduled** |
| `scripts/harness/run_82_3_candidate_backtests.py` | `:71` `GRID` (2x2x2 = **K=8** configs), `:143` `per_strategy: dict[str, list[np.ndarray]]`, `:167` `per_strategy[strat].append(rets)`, `:191-207` one matrix per strategy, `:206` `compute_pbo(matrix, S=16)`, `:207` `pbo_matrix_shape` | **sweep-level PBO producer #2** (K=8 hyper-param configs x per-strategy) | LIVE (one-shot script) |
| `backend/backtest/quant_optimizer.py` | `:238` `self.num_trials += 1`, `:267` `generate_report(result, num_trials=self.num_trials)`, `:302` `self.best_params = trial_params`, `:610` `_log_experiment` | **guided search**; keeps only scalars + `best_params`; NO per-iteration return-series retention (855 lines, grepped) | LIVE; **NOT a valid CSCV column source** |
| `backend/agents/mcp_servers/risk_server.py` | `:133-158` `pbo_check`, `:162+` `evaluate_candidate` | takes `pnl_matrix` as a TOOL ARGUMENT | **CONSUMER, not a producer** |
| `backend/services/promotion_gate.py` | `PBO_CEILING = 0.5`, `evaluate_promotion` | second ceiling; `analytics.py:189-191` records it as DEAD (zero callers) and that it defaults missing pbo to 0.0 = PASS | dead-code hazard |
| `backend/backtest/experiments/quant_results.tsv` | header row 1 | columns: `timestamp run_id param_changed metric_before metric_after delta status dsr top5_mda params_json parent_run_id` | **NO pbo column** (539 rows) |
| `backend/backtest/experiments/results/*.json` | -- | 437 files; **3** contain a `"pbo"` key, all written 2026-08-03/04 by phase-82.3 | see I5 |

### I1 -- `generate_report` call sites: MEASURED **15 invocations**, not 16

Grep `generate_report(` across `*.py` excluding `.venv`/`node_modules` and the
`def`: **15 real invocation sites**, every one passing a single
`BacktestResult`:
`backend/api/backtest.py:1058`; `backend/autoresearch/strategy_backtest_adapter.py:162`;
`backend/backtest/quant_optimizer.py:205` and `:267`;
`scripts/ablation/run_ablation.py:170`;
`scripts/harness/run_82_3_candidate_backtests.py:162`;
`scripts/harness/run_experiment.py:118`; `scripts/harness/run_harness.py:134` and `:143`;
`scripts/harness/run_optimizer.py:77`; `scripts/harness/run_quick_test.py:59`;
`scripts/harness/run_seed_stability.py:57`; `scripts/harness/run_subperiod_test.py:55`;
`scripts/harness/run_validation.py:88`; `tests/autoresearch/test_phase_48_2_backtest_adapter.py:98`.

**CORRECTION to the caller's "16":** the 16th reference is
`backend/tests/test_phase_82_23_pbo_in_gate.py:216`
`monkeypatch.setattr(ad, "generate_report", ...)` -- a test SUBSTITUTION of the
symbol, not an invocation. So: **15 call sites + 1 monkeypatch substitution +
1 `def` at `analytics.py:741`.** The caller's substantive claim is UNCHANGED and
confirmed: *every* site passes one run, so the 82.23 criterion is indeed
unsatisfiable. If 82.27's criteria name a count, name **15 invocations (16
non-`def` references)** and bound the grep, or the number will drift.

### I2 -- Who ALREADY holds N>=2 configuration series simultaneously?

Confirmed **two** true sweep-level producers; the caller's third candidate is
refuted:

1. **`backend/autoresearch/strategy_backtest_adapter.py::make_engine_backtest_fn`
   (`:167`)** -- runs `K` (default `_DEFAULT_K = 8`, `:70`) competing configs of
   ONE strategy with the `strategy` categorical held FIXED (`:94-129`), stacks
   their nav-derived daily returns into `(T, N)` (`:132-152`) and calls
   `compute_pbo_checked` (`:255`).
2. **`scripts/harness/run_82_3_candidate_backtests.py`** -- accumulates
   `per_strategy[strat].append(rets)` (`:167`) across the `GRID` of K=8
   hyper-parameter configs (`:71`, `product([3,4],[10,20],[0.05,0.1])`), then
   builds one matrix per strategy (`:191-205`) and calls
   `compute_pbo(matrix, S=16)` (`:206`). Daily-NAV T-axis, `T=1661`.
3. **`backend/backtest/quant_optimizer.py` -- NOT a producer, and must not
   become one as-is.** Two independent reasons: (a) MEASURED, it retains no
   per-iteration return series -- only scalars into `quant_results.tsv`
   (`:610`) and a mutating `self.best_params` (`:302`); (b) it is a **guided
   search** (each proposal is conditioned on the current best), and Bailey
   Section 5.2 explicitly forbids using guided-search *intermediate steps* as
   columns -- "the columns of matrix M should be the final outcome of each
   guided search ... and not the intermediate steps". Turning the optimizer's
   iteration trace into a PBO matrix would violate the paper directly.

**RECOMMENDED ANCHOR: `make_engine_backtest_fn`** (`strategy_backtest_adapter.py:167`).
Reasons: (i) it is the only sweep-level producer that is *library* code rather
than a one-shot script, so a criterion written against it stays true; (ii) it is
the only one on a path that actually reaches `PromotionGate.evaluate` -- the
82.3 script writes JSON/TSV for humans and never calls the gate; (iii) it
already emits the 82.23 provenance fields (`:266-274`); (iv) it enforces the
same-model-different-configuration rule the paper requires (`:94-108`, with a
hard `ValueError` on an unknown strategy so the engine cannot silently fall back
to `triple_barrier`). Name `run_82_3_candidate_backtests.py` as the OFFLINE
evidence producer, not as the gate anchor.

### I3 -- `risk_server.py` is a CONSUMER, not a third producer

`pbo_check` (`:133-158`) and `evaluate_candidate` (`:162+`) receive
`pnl_matrix: list[list[float]]` **as a tool argument** -- the MCP client supplies
the matrix; the server never assembles one and never runs a backtest. It is a
veto surface at `DEFAULT_PBO_VETO_THRESHOLD` (the 0.5 canonical ceiling), not a
producer. **It does NOT need sweep-level coverage in 82.27.** It DOES have one
in-scope defect worth noting: it calls raw `compute_pbo` (`:142-143`), so a
caller passing an undersized matrix gets `0.0` and `vetoed=False` -- the same
false-good, on the MCP surface. That is a one-line swap to `compute_pbo_checked`
and is the natural (small) extension of 82.23's work; it is a *consumer* fix, so
82.27 may include it or queue it, but must not describe it as a producer.

### I4 -- DOES the adapter's pbo reach `PromotionGate.evaluate`? MEASURED: **YES for `pbo`, NO for `pbo_n_trials`**

The hops:
`scripts/run_rotation_smoke.py:58` (or any caller) -> `rotation_runner.run_rotation_bakeoff:241`
-> `make_engine_backtest_fn` built at `rotation_runner.py:271`
-> `run_strategy_bakeoff` at `rotation_runner.py:278`
-> `strategy_candidate_producer.run_strategy_bakeoff:127` -> `build_per_strategy_candidates:65`
-> `strategy_selector.select_best_strategy:150` -> **`g.evaluate(c)` at `strategy_selector.py:94`.**

So the chain is REAL and closed. **But `build_per_strategy_candidates`
(`strategy_candidate_producer.py:115-123`) rebuilds the candidate from a
hardcoded 5-key whitelist** -- `strategy`, `dsr`, `pbo`, `params`, `sharpe` --
and therefore silently DISCARDS every provenance field 82.23 added
(`pbo_n_trials`, `pbo_n_obs`, `pbo_gate_grade`, `pbo_column_corr_mean`,
`pbo_columns_diverse`).

Measured, not inferred (real modules, fake `backtest_fn` emitting the adapter's
exact 82.23 shape with `pbo_n_trials=3`):

```
CANDIDATE KEYS SEEN BY THE GATE: ['dsr', 'params', 'pbo', 'sharpe', 'strategy']
pbo_n_trials present? -> False
gate.min_pbo_trials = 10
VERDICT via producer path  : {'promoted': True,  'reason': None}
VERDICT if N had survived  : {'promoted': False, 'reason': 'pbo_trials_below_min:3<10'}
```

**Conclusion: 82.27 NEEDS A WIRING CHANGE, not only tests.** The
`min_pbo_trials` floor added to `PromotionGate` in 82.23 is currently
UNREACHABLE from the only live producer, because `pbo_n_trials` never survives
the producer hop -- so `trial.get("pbo_n_trials")` is always `None` and
`gate.py:44` short-circuits to legacy behaviour. A K=8 adapter run (below the
floor of 10) promotes today. The fix is small and additive: forward the
`pbo_*` keys in `build_per_strategy_candidates`. Note the existing
5-key whitelist is *load-bearing in the other direction* (`:107-113` skips a
candidate missing dsr|pbo), so the change must ADD keys, not replace the guard.

Second, weaker hop to disclose: **nothing schedules `run_rotation_bakeoff`.**
`rotation_runner.py:36` and `strategy_candidate_producer.py:33` both still list
"The weekly rotation CRON" as DEFERRED, and a grep of `backend/services`,
`backend/main.py`, `backend/slack_bot` found no rotation scheduling. So the gate
path is real but operator-triggered.

### I5 -- What do the persisted artifacts carry for `pbo`?

- **`backend/backtest/experiments/quant_results.tsv`** (539 rows): header is
  `timestamp run_id param_changed metric_before metric_after delta status dsr
  top5_mda params_json parent_run_id`. **There is NO `pbo` column.** The 7 lines
  matching "pbo" are rows 533-539 only, where the string appears INSIDE the
  `params_json` blob of the phase-82.3 passes (`82.3-passA` / `82.3-passB`) --
  not in a column the file can be queried on. A criterion asserting "the TSV
  carries PBO" would be false today.
- **`backend/backtest/experiments/results/*.json`**: 437 files, of which
  **exactly 3** contain a `"pbo"` key --
  `20260803T175308Z_phase_82_3_candidate_comparison.json`,
  `20260804T025319Z_phase_82_3_full_sample_3strat.json`,
  `20260804T041628Z_phase_82_3_short_window_4strat.json`. All three are phase-82.3
  outputs. Every other result file predates any PBO emission.

---

## Consensus vs debate (external)

**Consensus (no dissent found):** the CSCV matrix is a sweep-level artifact whose
columns are final per-configuration performance series on a common time index;
N must be large (`N >> 10`); PBO is a post-hoc evaluation of a selection
process, never a search objective.

**Live debate:** which SPLITTING scheme should generate the evidence -- CSCV
(the original) vs CPCV / Bagged CPCV / Adaptive CPCV (Arian et al. 2024) vs
walk-forward (still the industry default per arXiv:2512.12924v1). This debate is
orthogonal to 82.27: it concerns column GENERATION, not column DEFINITION or
ownership.

## Pitfalls (from the literature), mapped

1. **Guided-search intermediates as columns** (paper Sec 5.2) -> refutes any
   design that mines `quant_optimizer`'s iteration trace
   (`quant_optimizer.py:302`).
2. **File-drawer / hidden trials** (paper Sec 5.2) -> the adapter's per-variant
   `except` at `strategy_backtest_adapter.py:212-216` DROPS a failed column
   silently; systematically dropping the losers biases PBO DOWN. Worth a WARN-
   count in the emitted dict.
3. **High PBO from near-identical columns is a documented benign case** (paper
   Sec 5.2, limitation 4) -> do not let the measured `triple_barrier` 0.7486
   (corr 0.967-0.979) be re-spec'd as a verdict on the incumbent.
4. **PBO as a target** (paper Sec 5.2, limitation 5 / Strathern) -> keep PBO out
   of `quant_optimizer`'s objective.
5. **Small N -> discontinuous logits** (paper, `N >> 10`) -> `_DEFAULT_K = 8`
   (`strategy_backtest_adapter.py:70`) is BELOW the gate's own
   `min_pbo_trials=10`; step 82.26 raises it. Until then the adapter emits a
   non-gate-grade PBO by construction -- which, per I4, the gate cannot currently
   see.

---

## Read in full (counts toward the gate) -- 6 sources

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|
| https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf | 2026-08-04 | peer-reviewed paper (J. Computational Finance 20(4) 2017; SSRN 2326253), 34 pp | `curl` + `pdfplumber` (63,988 chars) | Algorithm 2.3: "each column n = 1,...,N represents a vector of profits and losses over t = 1,...,T observations associated with a particular model configuration tried by the researcher"; "N >> 10 is required"; "the columns of matrix M should be the final outcome of each guided search ... and not the intermediate steps" |
| https://cran.r-project.org/web/packages/pbo/vignettes/pbo.html | 2026-08-04 | official package doc (R reference impl) | WebFetch | "we assemble the trials into an NxT matrix where each column represents a trial and each trial has the same length T"; `pbo(M, s=8, f=sharpe, threshold=0)`; NO documented small-N behaviour |
| https://cran.r-project.org/web/packages/pbo/readme/README.html | 2026-08-04 | official package doc | WebFetch | Example uses **`N <- 200`** trial columns; `pbo(M,S,F=Omega,threshold=1)`; no error/warning documented for insufficient trials; no pipeline-stage statement |
| https://medium.com/balaena-quant-insights/the-probability-of-backtest-overfitting-pbo-9ba0ac7fb456 | 2026-08-04 | practitioner walk-through with code | WebFetch | "All your PnLs must have the same shape of (T, 1)"; `R_mat = pd.DataFrame(R).T  # shape: (n_configs, n_combinations)`; PBO computed **after parameter sweeps complete**; "Disregarding failed trials will only underestimate the probability of overfitting" |
| https://raw.githubusercontent.com/hudson-and-thames/mlfinlab/master/mlfinlab/cross_validation/combinatorial.py | 2026-08-04 | OSS source (mlfinlab) | WebFetch (raw source) | `CombinatorialPurgedKFold`, `StackedCombinatorialPurgedKFold`, `_get_number_of_backtest_paths(n_train_splits, n_test_splits)`; **generates N backtest paths but does NOT compute PBO** -- splitting and statistics are separate objects |
| https://arxiv.org/html/2512.12924v1 | 2026-08-04 | arXiv preprint, Dec 2025 (recency) | WebFetch (arXiv native HTML) | "Bailey et al. (2017) developed Combinatorially Symmetric Cross-Validation (CSCV) to compute the Probability of Backtest Overfitting"; cites Arian et al. 2024 CPCV superiority; "walk-forward remains the industry standard"; authors explicitly do NOT apply deflation to their own results |

## Identified but snippet-only (context; does NOT count toward the gate) -- 22

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 | SSRN landing | abstract page; full text obtained from the author-hosted PDF instead |
| https://www.ssrn.com/abstract=4778909 | SSRN landing (Arian et al. 2024) | abstract-only; no open full text |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376 | SSRN landing (Arian et al. 2024, earlier ver.) | abstract-only |
| https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 | Elsevier (Knowledge-Based Systems) | paywalled abstract |
| https://vectorbt.pro/tutorials/cross-validation/ | official doc | **fetched, but the page is a tutorial landing stub with no API/technical content** -- honestly excluded from the gate count |
| https://vectorbt.dev/getting-started/features/ | official doc | feature list only |
| https://vectorbt.pro/ | official doc | product landing |
| https://qubitquants.github.io/vbt-tuts-toc/index.html | community tutorial index | community tier, index only |
| https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting | journal | paywalled; same paper read via author PDF |
| https://escholarship.org/uc/item/4w1110bb | repository copy | duplicate of the read PDF |
| https://carmamaths.org/jon/backtest2.pdf | Borwein-hosted copy | duplicate of the read PDF |
| https://scholarworks.wmich.edu/math_pubs/42/ | repository copy | duplicate |
| https://www.semanticscholar.org/paper/.../b1233b4f5384f003e85c2e0eec1a2dfc08f624c5 | index | metadata only |
| https://scispace.com/papers/the-probability-of-backtest-overfitting-4ublh83xkm | index | metadata only |
| https://www.researchgate.net/publication/318600389_The_probability_of_backtest_overfitting | RG | request-gated |
| https://www.researchgate.net/publication/272304380_The_Probability_of_Back-Test_Over-Fitting | RG | request-gated |
| https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/cross_validation/combinatorial.py | OSS (HTML view) | raw version read instead |
| https://hudsonthames.org/mlfinlab/ | vendor page | marketing overview |
| https://github.com/hudson-and-thames/mlfinlab/issues/179 | issue tracker | "Add Backtest Statistics Ch14" -- evidence PBO/DSR live in a SEPARATE module from CV |
| https://panpip.github.io/HKML_2020_MlFinLab.pdf | slide deck | community tier |
| https://cran.rstudio.com/web/packages/pbo/readme/README.html | mirror | duplicate of the CRAN README read |
| https://arxiv.org/pdf/2209.05559 | arXiv preprint (DRL crypto, addressing backtest overfitting) | adjacent domain, not needed for the question |

**Unique URLs collected: 28** (6 read in full + 22 snippet-only).

---

## Application to pyfinagent -- what 82.27 should say

1. **Name the anchor explicitly and make it a LIBRARY symbol.**
   `backend/autoresearch/strategy_backtest_adapter.py::make_engine_backtest_fn`
   (`:167`) is THE sweep-level PBO producer: it is the only one that (a) holds
   N>=2 same-model configurations at once, (b) sits on the path that reaches
   `PromotionGate.evaluate` (`strategy_selector.py:94`), and (c) is importable
   library code rather than a one-shot script. Criteria should assert against
   its EMITTED DICT (`:264-278`), not against `generate_report`.
2. **The unsatisfiable criterion's replacement should be a NEGATIVE plus a
   POSITIVE.** Negative: `generate_report` must NOT emit a pbo (already pinned by
   `backend/tests/test_phase_82_23_pbo_in_gate.py:138-168`, which asserts on the
   SIGNATURE and on `"compute_pbo" not in src`). Positive: the sweep-level
   producer must emit `pbo` together with its provenance
   (`pbo_n_trials`/`pbo_gate_grade`/`pbo_column_corr_mean`/`pbo_columns_diverse`)
   **and those fields must survive to the gate**. Cite Algorithm 2.3 for why the
   level moved.
3. **82.27 needs a WIRING change, not only tests (MEASURED, see I4).** Forward
   the `pbo_*` keys through `build_per_strategy_candidates`
   (`strategy_candidate_producer.py:115-123`). Without it the 82.23
   `min_pbo_trials=10` floor is dead code on the only live path, and a K=8
   (`_DEFAULT_K`, adapter `:70`) run promotes on a non-gate-grade PBO.
4. **Bound every count in the criteria and express repo-wide facts as measured
   deltas.** Per the standing lesson from phase-81.0: the "16 call sites" figure
   is 15 invocations + 1 monkeypatch; a criterion that greps for a hard count
   will go red as soon as anyone adds a script. Prefer "no invocation of
   `generate_report` receives more than one `BacktestResult`" (a property) over a
   count.
5. **Do not re-derive or redesign `compute_pbo_checked`, `min_pbo_trials`, or the
   adapter's forwarding hop** -- they are built and mutation-verified. 82.27's
   scope is: the re-spec, the producer-hop forwarding, and (optionally) the
   `risk_server.py:142-143` raw-`compute_pbo` consumer swap.
6. **Keep the incumbent's 0.7486 in its literature context.** The paper's own
   limitation 4 says a high PBO over near-identical, similarly-skilful columns is
   expected; the measured column correlation 0.967-0.979 puts phase-82.3's
   K=8 grid squarely in that regime. State it as "not gate-grade evidence"
   rather than "the incumbent is overfit".

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch/curl+pdfplumber -- **6**
- [x] 10+ unique URLs total (incl. snippet-only) -- **28**
- [x] Recency scan (last 2 years) performed + reported -- dedicated section above
- [x] Full papers / pages read (not abstracts) -- the 34-page Bailey PDF was
      text-extracted in full; no source counted from an abstract
- [x] file:line anchors for every internal claim
- [x] Three-variant search discipline (year-less / 2024-2025 / 2026) -- queries listed

Soft checks:
- [x] Internal exploration covered every relevant module (analytics, gate,
      adapter, producer, selector, rotation_runner, 82.3 script, quant_optimizer,
      risk_server, TSV + results JSON)
- [x] Contradictions / consensus noted (Q2 consensus; CSCV-vs-CPCV debate)
- [x] All claims cited per-claim
- [ ] GAP: `vectorbt PRO`'s exact PBO API could not be verified (docs behind the
      product); its row in the Q2 table is directional, sourced from search
      summaries rather than a full page read. It is not load-bearing -- the
      other four implementations agree.
- [ ] GAP: Arian et al. 2024 was not read in full (paywalled/abstract-only); its
      claims are reported from the abstract-level summary plus the arXiv:2512.12924v1
      full-text citation of it. It does not change the answer to Q1-Q3.

## Queries run

1. Year-less canonical: `Bailey Borwein Lopez de Prado Zhu "probability of backtest overfitting" CSCV combinatorially symmetric cross-validation`
2. Year-less framework: `mlfinlab probability of backtest overfitting CSCV implementation N trials matrix`
3. Year-less framework: `"backtest overfitting" CSCV python implementation hudson thames mlfinlab documentation campbell backtest_statistics`
4. Year-less framework: `vectorbt CSCV cross-validation probability of backtest overfitting splitter parameter sweep`
5. Last-2-year: `probability of backtest overfitting critique 2025 alternative PBO CSCV limitations deflated Sharpe 2024`
6. Last-2-year: `"Backtest overfitting in the machine learning era" comparison out-of-sample testing methods synthetic controlled environment CPCV arXiv`
7. Current-year: `arXiv 2026 probability of backtest overfitting PBO CSCV strategy selection multiple testing quantitative finance`
8. Product-scoped: `vectorbt PRO "probability of backtest overfitting" pbo cross-validation documentation parameter combinations`

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 22,
  "urls_collected": 28,
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
  "summary": "Bailey/Borwein/Lopez de Prado/Zhu Algorithm 2.3 (read in full, 34pp) settles Q1: the CSCV matrix M is (T x N) where each column is the FINAL PnL series of one alternative CONFIGURATION on a common time index, and guided-search intermediates are explicitly excluded; N >> 10 required, T = 2x the selection sample. Every OSS implementation (CRAN pbo N=200, mlfinlab CPCV path generator, the Balaena reference code, vectorbt) puts the matrix on the SWEEP object; none computes PBO in a single-run report -- external corroboration that 82.23's criterion was unsatisfiable. No implementation documents a small-N fallback, so 'return 0.0 at N<2' (analytics.py:297) is a pyfinagent-local invention, not a known trap. Internally: 15 generate_report invocations (not 16 -- the 16th is a monkeypatch), all single-run; two true sweep producers (strategy_backtest_adapter.py:167 -- the recommended anchor -- and run_82_3_candidate_backtests.py:143); quant_optimizer is a guided search retaining no series, so it is disqualified twice; risk_server is a CONSUMER. MEASURED: the adapter's pbo DOES reach PromotionGate.evaluate, but build_per_strategy_candidates:115-123 drops pbo_n_trials, so the 82.23 min_pbo_trials floor is dead on the live path -- 82.27 needs a wiring change, not only tests.",
  "brief_path": "handoff/current/research_brief_82.27.md",
  "gate_passed": true
}
```
