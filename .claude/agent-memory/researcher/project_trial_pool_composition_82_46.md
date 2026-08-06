---
name: trial-pool-composition-82-46
description: 82.46 optimizer trial-pool -- the step's premise is FALSE (DSR N is the iteration count, not the pool size), blend is a revert orphan, 4 dead weight params burn 15% of proposals, and closing the step turns a green 82.16 test red
metadata:
  type: project
---

**The step's own premise is FALSE and the contract must say so.** 82.46 asserts "the
trial pool is a DIRECT input to Deflated Sharpe" (echoing the 82.16 comment at
`quant_optimizer.py:77-83`). Measured: `compute_deflated_sharpe`'s signature is
`(observed_sr, num_trials, variance_of_srs, skewness, kurtosis, T, periods_per_year)` --
**there is no pool parameter**. N is the cumulative optimizer ITERATION count
(`quant_optimizer.py:151` -> `:226`/warm-start `:852` -> `:256 +=1` -> `:285` ->
`analytics.py:766` -> `:429-432`). `AVAILABLE_STRATEGIES` has exactly two live hits
(`:86` def, `:126` categorical) and reaches only `_propose_random`. Adding three
strategies changes DSR by IDENTICALLY ZERO, not "a little".

**Second, less obvious half: V is pool-invariant too, and it is not Bailey's V.**
`analytics.py:753-754` computes `variance_of_srs` as `np.var(window_sharpes)` -- the
variance of per-WINDOW Sharpes inside ONE backtest. Bailey's V is the variance of the
Sharpes ACROSS TRIALS ("the variance of the SRs tested"). Undocumented deviation; worth
its own step.

**What IS steeply sensitive: N itself.** Same inputs (`sr=1.1704633657934074`, V=0.5,
T=1661, ppy=252): N=2 -> DSR 0.9802, N=10 -> 0.5582, N=26 -> 0.2579, N=100 -> 0.0563.
(N=1==N=2 via `max(num_trials,2)`.) So **wasted iterations are the real DSR cost.**

**`blend` is a REVERT ORPHAN, not an aspiration.** Born `1f270641` (2026-03-25) as a
real `_compute_blend_label` weighted vote (Dietterich 2000) WITH a registry entry.
Killed by `9fbd9cd6` (2026-03-28, "Phase 1.2-1.9 improvements broke the strategy").
**That revert touched exactly 5 files and `quant_optimizer.py` is NOT one of them** --
so the implementation died and the offer survived. Implementation preserved on branch
`phase1-experimental` (exists). Collateral orphans in the same file:
`tb_weight/qm_weight/mr_weight/fm_weight` (`:109-113`) are read by NOTHING -- **4 of 26
proposable params, ~15% of every random proposal, each burning a ~20-min backtest and a
DSR-costing increment.** Stale comment at `:588` names the deleted method.
`handoff/archive/misc/research_brief_phase_48_3_rotation_runner.md:130` already found
this in phase-48.3 and it was never actioned.

**Closing this step turns a currently-GREEN test RED (34 passed 2026-08-06).**
`test_phase_82_16_label_forward_information.py::test_optimizer_trial_pool_composition_is_pinned`
(`:376-395`) asserts `previously_offered - now == {"quality_momentum","factor_model"}`
where `previously_offered` literally contains `"blend"`. Also `:226` carries a
`- {"blend"}` carve-out. Same shape as 82.39. Update both in the SAME commit.

**Theory (read in full):** PBO paper Algorithm 2.3 -- "each column n represents ... a
particular model configuration **tried by the researcher**"; DSR A.3 -- "the N ...
corresponds to the number of INDEPENDENT trials ... using M instead of N will overstate
E[max{SR}]" (so raw-M over-deflates = SAFE, and a MORE diverse pool makes raw-M *more*
accurate). Menu size multiplies trials only for an EXHAUSTIVE grid; this optimizer is
fixed-budget random single-param perturbation, so |pool| changes only the search prior.
**PBO, not DSR, is the pool-sensitive statistic** ("N >> 10 is required" -> already
transcribed as `analytics.py:205 PBO_MIN_TRIALS_GATE_GRADE = 10`).

**Cost reality (Q5).** The 82.3 artifacts persist only `pbo_matrix_shape`, NOT the
matrices -- a pool-level CSCV cannot be recombined from them. Measured runtimes:
full-sample 3x8 = **30830 s (8.6 h)**; short-window 4x8 = **4988 s (1.4 h, ~156 s/config)**.
A 6-strategy pool-level short-window sweep is ~2.1 h; full sample ~16.7 h (don't).

**Per-candidate facts.** `qarp` CANNOT train on the configured 2018-2025 window
(smoke: 0 trades, Sharpe 0.0; omitted from the full-sample pass) -- fundamentals-dependent,
82.21. `mean_reversion` CAN train and performs badly (-6.13, -3.86, both discard) --
a DIFFERENT case. `meta_label` maps to the SAME label fn as triple_barrier
(`backtest_engine.py:72`) so its PBO column is near-collinear. `stretch_regime` is the
strongest add: full-sample PBO **0.1960 vs incumbent triple_barrier 0.7486**.
TSV history (537 rows): triple_barrier 503, factor_model 7, quality_momentum 4,
mean_reversion 2, meta_label 1, blend 1 (crash); the three 82.2 candidates: NEVER.

**A THIRD drifted list nobody has swept:** `backend/meta_evolution/archetype_library.py:31-33`
`IMPLEMENTED_STRATEGY_IDS` still holds both 82.16-demoted names AND `blend`, and omits
all three 82.2 candidates -- and `resolve_strategy`'s docstring
(`backtest_engine.py:96-97`) names it as the live caller that can request a demoted name.

See [[project_dsr_trial_count_reset_82_25]], [[project_pbo_level_and_dead_gate_82_27]],
[[project_non_forward_labels_82_16]], [[project_fundamentals_coverage_82_21]],
[[project_phantom_columns_82_39]].
