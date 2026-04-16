# Sprint Contract -- Cycle 1
Generated: 2026-04-16T18:14:36.068688+00:00

## Hypothesis
Continue parameter optimization with random perturbation

## Current Baseline
- Sharpe: 1.1705

## Success Criteria (from evaluator_criteria.md)
- Statistical Validity: DSR >= 0.95, Sharpe > 0
- Robustness: ALL sub-periods Sharpe > 0
- Reality Gap: 2x costs Sharpe > 0.5

## Planner Suggestions
- PLATEAU: Last 10 experiments all discarded. Consider strategy change.
- SATURATED: trailing_distance_pct has 19 consecutive discards. Excluding.
- SATURATED: rsi_weight has 21 consecutive discards. Excluding.
- SATURATED: n_estimators has 21 consecutive discards. Excluding.
- SATURATED: sl_pct has 15 consecutive discards. Excluding.
- SATURATED: volatility_weight has 16 consecutive discards. Excluding.
- SATURATED: qm_weight has 23 consecutive discards. Excluding.
- SATURATED: mr_holding_days has 12 consecutive discards. Excluding.
- SATURATED: frac_diff_d has 6 consecutive discards. Excluding.
- SATURATED: top_n_candidates has 14 consecutive discards. Excluding.
- SATURATED: vol_barrier_multiplier has 15 consecutive discards. Excluding.
- SATURATED: min_samples_leaf has 13 consecutive discards. Excluding.
- SATURATED: momentum_weight has 21 consecutive discards. Excluding.
- SATURATED: mr_weight has 8 consecutive discards. Excluding.
- SATURATED: target_vol has 22 consecutive discards. Excluding.
- SATURATED: learning_rate has 20 consecutive discards. Excluding.
- SATURATED: holding_days has 21 consecutive discards. Excluding.
- SATURATED: fm_weight has 18 consecutive discards. Excluding.
- SATURATED: max_positions has 14 consecutive discards. Excluding.
- SATURATED: trailing_stop_enabled has 19 consecutive discards. Excluding.
- SATURATED: tb_weight has 17 consecutive discards. Excluding.
- SATURATED: target_annual_vol has 16 consecutive discards. Excluding.
- SATURATED: trailing_trigger_pct has 12 consecutive discards. Excluding.
- SATURATED: tp_pct has 15 consecutive discards. Excluding.
- SATURATED: sma_weight has 14 consecutive discards. Excluding.
- SATURATED: strategy has 12 consecutive discards. Excluding.
- SATURATED: max_depth has 15 consecutive discards. Excluding.
- COORDINATED: barrier_shape group (tp_pct, sl_pct) has 1 kept / 31 discarded. Try moving params together.
- STRATEGY: Current=triple_barrier. Consider switching to mean_reversion if plateau continues.

## Excluded Parameters
- trailing_distance_pct
- rsi_weight
- n_estimators
- sl_pct
- volatility_weight
- qm_weight
- mr_holding_days
- frac_diff_d
- top_n_candidates
- vol_barrier_multiplier
- min_samples_leaf
- momentum_weight
- mr_weight
- target_vol
- learning_rate
- holding_days
- fm_weight
- max_positions
- trailing_stop_enabled
- tb_weight
- target_annual_vol
- trailing_trigger_pct
- tp_pct
- sma_weight
- strategy
- max_depth
