# Experiment Results -- Phase 4.4.1.3 Seed Stability

**Date:** 2026-04-16
**Cycle:** 20
**Duration:** ~103 min (5 seeds x ~20 min each)

## Results

| Seed | Sharpe | DSR  | Return % | MaxDD %  | Trades | Hit Rate |
|------|--------|------|----------|----------|--------|----------|
| 42   | 0.5867 | 1.00 | 46.28    | -12.40   | 680    | 55.63%   |
| 123  | 0.5756 | 1.00 | 45.65    | -12.40   | 680    | 55.56%   |
| 456  | 0.5861 | 1.00 | 46.27    | -12.40   | 680    | 55.26%   |
| 789  | 0.6044 | 1.00 | 47.51    | -12.40   | 680    | 55.41%   |
| 2026 | 0.5917 | 1.00 | 46.59    | -12.40   | 680    | 55.48%   |

## Aggregate Statistics

- **Mean Sharpe:** 0.5889
- **Std Sharpe:** 0.0094
- **Min Sharpe:** 0.5756 (seed 123)
- **Max Sharpe:** 0.6044 (seed 789)
- **Range:** 0.0288

## Criteria Assessment

| Criterion | Threshold | Actual | Result |
|-----------|-----------|--------|--------|
| Std < 0.1 | 0.1 | 0.0094 | PASS |
| All seeds > 0.9 | 0.9 | min=0.5756 | FAIL |
| Range < 0.3 (sanity) | 0.3 | 0.0288 | PASS |

## Drill Test: 11/14 PASS

Failed checks: S5 (mean Sharpe < 0.9), S7 (not all seeds > 0.9), S12 (verdict != PASS)

## Verdict: FAIL

**The strategy IS seed-stable** (std=0.0094, range=0.029, identical trade counts across all seeds), but all Sharpe values are well below the 0.9 floor. The absolute Sharpe has degraded from the optimizer's best (1.1705, recorded 2025-03-28) to ~0.59 across all seeds.

## Root Cause Analysis

The Sharpe degradation is NOT caused by seed sensitivity. All 5 seeds produce nearly identical results (680 trades each, MaxDD identical to 4 decimal places). The degradation is likely caused by:

1. **Data drift**: BigQuery price/fundamental data has been updated since the March 28 optimizer run, changing the feature landscape
2. **Market regime shift**: The walk-forward windows now extend into different market conditions
3. **The strategy needs re-optimization** with current data before the 0.9 floor can be met

The checklist item 4.4.1.3 cannot be flipped until the strategy is re-optimized to produce Sharpe > 0.9 on current data.
