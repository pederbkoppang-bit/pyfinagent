# Phase 3.2.1 PLAN Detail — Evaluator Spot Checks Implementation

**Phase:** Phase 3.2.1 PLAN
**Status:** In Progress (08:56 UTC)
**Duration Est:** 30 min
**Author:** Ford (autonomous execution)
**Next:** GENERATE phase (09:30 UTC)

---

## Overview

This document details the PLAN phase for Phase 3.2.1. The harness protocol requires:
1. **Identify test data sets** (baseline, 2× cost, regime-shifted)
2. **Design spot check invocation patterns** (how to inject checks into evaluator)
3. **Define expected outputs** (what success looks like)
4. **Prepare for GENERATE** (code structure ready)

---

## Best Parameters (Current)

From `backend/backtest/experiments/optimizer_best.json`:

```json
{
  "sharpe": 1.1705,
  "dsr": 0.9526,
  "sl_pct": 12.923,
  "tp_pct": 10.0,
  "holding_days": 90,
  "train_window_months": 12,
  "test_window_months": 3,
  "embargo_days": 5,
  "max_positions": 20,
  "max_depth": 4,
  "n_estimators": 200
}
```

**Baseline Sharpe:** 1.1705 → Success threshold: ≥ 1.0535 (90%)

---

## Spot Check #1: Cost Stress Test (2× Costs)

### Objective
Verify evaluator detects strategy failure under doubled transaction costs.

### Implementation
1. **Run baseline harness** with `transaction_cost=0.001` (default)
2. **Run cost-stress harness** with `transaction_cost=0.002` (doubled)
3. **Compare Sharpe:**
   - **SUCCESS:** Cost-stress Sharpe ≥ 1.0535 (≥90% baseline)
   - **CONDITIONAL:** Cost-stress Sharpe 0.95-1.05 (95-100% baseline, edge case)
   - **FAIL:** Cost-stress Sharpe < 0.95 (<95% baseline, overfitting to costs)

### Test Data Set
- **Date Range:** 2018-01-01 to 2025-12-31 (same as best params)
- **Window:** Train 12mo, test 3mo, embargo 5 days (from best params)
- **Cost Modification:** Double transaction cost in backtest engine
- **Expected Output:** Sharpe, drawdown, num trades, return (in `results/2x_cost_*.json`)

### Success Criteria (Research-Backed)
- **Sharpe ≥ 90% baseline** — Roncelli (2020): "robustness coefficient ≥ 0.9"
- **Drawdown < 2× baseline** — Cost stress shouldn't triple drawdowns
- **Num trades unchanged** — Cost doesn't change signal generation

### Evaluator Integration
When evaluator scores a proposal, it must:
1. Run baseline harness (existing)
2. Run 2× cost harness (new)
3. Check: `sharpe_2x_cost >= 0.9 * sharpe_baseline`
4. Output: `cost_stress_pass: true/false` (hard constraint)

---

## Spot Check #2: Regime Shift Detection

### Objective
Verify evaluator detects when strategy fails across market regime boundaries.

### Implementation
1. **Identify regime boundaries** using HMM or GMM on factor correlations
   - Two Sigma (2021) identifies 4 distinct regimes
   - We'll use 2-3 regime boundaries from 2018-2025 data
   - Example regimes: Bull (2017-2020), COVID crash (Mar 2020), Recovery (2021-2022), Rate hikes (2022-2025)

2. **Split backtest by regime:**
   - Regime 1: Train all, test Regime 1
   - Regime 2: Train Regime 1, test Regime 2
   - Regime 3: Train Regime 1-2, test Regime 3
   - **CRITICAL:** No look-ahead bias across regime boundaries

3. **Compare Sharpe across regimes:**
   - **SUCCESS:** Sharpe ≥ baseline in all regimes
   - **CONDITIONAL:** Sharpe ≥ 80% baseline in 2/3 regimes
   - **FAIL:** Sharpe < 80% baseline in any regime, or crashes

### Test Data Set
- **Regime Detection:** Fit HMM to factor correlations (VIX, VIX slope, DXY, 10Y yield, credit spreads)
- **Regimes:** Will be identified algorithmically from historical data
- **Requirement:** Strategy must not regress >20% in any regime

### Success Criteria (Research-Backed)
- **Sharpe ≥ baseline across regimes** — Two Sigma: strategies fail at regime boundaries
- **No regime-specific collapse** — Strategy must adapt, not break
- **Drawdown stable** — No unexpected recovery periods

### Evaluator Integration
When evaluator scores a proposal:
1. Run baseline harness (whole period)
2. Identify market regimes (HMM)
3. Run harness for each regime (walk-forward)
4. Check: `sharpe_regime_i >= 0.8 * sharpe_baseline` for all i
5. Output: `regime_shift_pass: true/false` (hard constraint)

---

## Spot Check #3: Parameter Sweep (Sensitivity Analysis)

### Objective
Verify evaluator detects parameter overfitting via sensitivity test.

### Implementation
1. **Define parameter ranges** around best params:
   - `sl_pct`: 12.923 ± 10% = [11.6, 14.2]
   - `tp_pct`: 10.0 ± 10% = [9.0, 11.0]
   - `holding_days`: 90 ± 20% = [72, 108]
   - `max_depth`: 4 ± 1 = [3, 5]
   - `learning_rate`: 0.1 ± 20% = [0.08, 0.12]

2. **Generate 10 random parameter combos** from these ranges:
   - 5 combos with random parameters
   - 5 combos with best params + small variations
   - Example combo 1: `sl=11.8, tp=9.5, holding=80, depth=3, lr=0.09`
   - Example combo 2: `sl=13.5, tp=10.8, holding=95, depth=4, lr=0.11`

3. **Run harness for each combo**, collect Sharpe

4. **Measure variance:**
   - σ (Sharpe across 10 combos) = `std([sharpe_1, sharpe_2, ..., sharpe_10])`
   - **SUCCESS:** σ ≤ 5% (params are stable)
   - **CONDITIONAL:** σ ≤ 10% (slightly unstable)
   - **FAIL:** σ > 10% (severe overfitting, params are brittle)

### Test Data Set
- **Date Range:** Same as baseline (2018-2025)
- **Window:** Train 12mo, test 3mo (same)
- **10 Parameter Combos:** Generated from ranges above
- **Output:** Sharpe, σ, top 3 combos, bottom 3 combos (in `results/param_sweep_*.json`)

### Success Criteria (Research-Backed)
- **σ(Sharpe) ≤ 5%** — BuildAlpha: threshold for robust params
- **Top params close to baseline** — Best combo should be ≤ 5% better than baseline
- **No large outliers** — No combo with Sharpe > 1.5× baseline (sign of overfitting)

### Evaluator Integration
When evaluator scores a proposal:
1. Run baseline harness
2. Generate 10 nearby parameter combos
3. Run harness for each (parallel if possible)
4. Compute σ(Sharpe)
5. Check: `sigma_sharpe <= 0.05 * mean_sharpe`
6. Output: `param_sweep_pass: true/false, sigma_pct: X%` (hard constraint)

---

## Harness Invocation Pattern

### Current Harness Structure (from `run_harness.py`)

```python
def run_harness(proposal_params, cycles=1, dry_run=False):
    """Run walk-forward backtest with given parameters."""
    # Returns: {sharpe, dsr, return, maxdd, ...}
```

### New Evaluator Flow (Phase 3.2.1)

```python
def evaluate_proposal(proposal_params):
    """
    Multi-check evaluator for proposal validation.
    
    Returns: {
        baseline_sharpe: float,
        cost_stress_pass: bool,
        regime_shift_pass: bool,
        param_sweep_pass: bool,
        overall_pass: bool,  # AND of all 3 checks
        reasoning: str
    }
    """
    # Step 1: Baseline harness
    baseline = run_harness(proposal_params, cycles=1)
    baseline_sharpe = baseline['sharpe']
    
    # Step 2: Cost stress test
    cost_params = proposal_params.copy()
    cost_params['transaction_cost'] *= 2.0
    cost_result = run_harness(cost_params, cycles=1)
    cost_pass = cost_result['sharpe'] >= 0.9 * baseline_sharpe
    
    # Step 3: Regime shift test
    regime_results = evaluate_regimes(proposal_params)  # Run walk-forward per regime
    regime_pass = all(r['sharpe'] >= 0.8 * baseline_sharpe for r in regime_results)
    
    # Step 4: Parameter sweep
    sweep_results = sweep_params(proposal_params, n=10)  # 10 random nearby params
    sigma_sharpe = np.std([r['sharpe'] for r in sweep_results])
    mean_sharpe = np.mean([r['sharpe'] for r in sweep_results])
    param_pass = sigma_sharpe <= 0.05 * mean_sharpe
    
    # Step 5: Aggregate
    overall_pass = cost_pass and regime_pass and param_pass
    reasoning = f"Cost:{cost_pass}, Regime:{regime_pass}, ParamStab:{param_pass}"
    
    return {
        baseline_sharpe: baseline_sharpe,
        cost_stress_pass: cost_pass,
        regime_shift_pass: regime_pass,
        param_sweep_pass: param_pass,
        overall_pass: overall_pass,
        reasoning: reasoning
    }
```

### Parallel Execution (Time Optimization)
- **Baseline:** 30 min (baseline harness is slow; walk-forward with many windows)
- **Cost Stress:** 30 min (same structure, just 2× cost)
- **Regime Shift:** 45 min (3 regimes × 30 min each, but can run in parallel → ~30 min total)
- **Param Sweep:** 90 min (10 combos × 30 min each, can parallelize → ~30 min with 10 workers)

**Total Sequential:** 30 + 30 + 45 + 90 = 195 min (~3.2 hours)
**Total Parallel:** max(30, 30, 30, 30) = ~1.5-2 hours (if we have 10 workers)

For now, we'll run sequentially (simplicity) to ensure data consistency. Can optimize later.

---

## Expected Outputs (GENERATE Phase)

After running GENERATE, we'll have:

1. **`results/2026-04-06_cost_stress_baseline.json`**
   ```json
   {
     "test_type": "baseline",
     "sharpe": 1.1705,
     "dsr": 0.9526,
     "return": 0.802,
     "maxdd": -0.120,
     "num_trades": 632
   }
   ```

2. **`results/2026-04-06_cost_stress_2x.json`**
   ```json
   {
     "test_type": "2x_cost",
     "sharpe": 1.0535,
     "dsr": 0.9100,
     "return": 0.710,
     "maxdd": -0.125,
     "num_trades": 632
   }
   ```

3. **`results/2026-04-06_regime_shift.json`**
   ```json
   {
     "test_type": "regime_shift",
     "regimes": [
       {"name": "Bull", "sharpe": 1.18, "dsr": 0.96},
       {"name": "COVID", "sharpe": 0.92, "dsr": 0.85},
       {"name": "Recovery", "sharpe": 1.22, "dsr": 0.98}
     ],
     "all_pass": true
   }
   ```

4. **`results/2026-04-06_param_sweep.json`**
   ```json
   {
     "test_type": "param_sweep",
     "n_combos": 10,
     "sharpes": [1.10, 1.12, 1.15, 1.14, 1.13, 1.12, 1.11, 1.13, 1.12, 1.14],
     "mean": 1.1260,
     "sigma": 0.0143,
     "sigma_pct": 1.27,
     "pass": true
   }
   ```

5. **`results/2026-04-06_spot_checks_summary.json`**
   ```json
   {
     "proposal_id": "best_params_52eb3ffe",
     "baseline_sharpe": 1.1705,
     "cost_stress_pass": true,
     "cost_stress_sharpe": 1.0535,
     "regime_shift_pass": true,
     "param_sweep_pass": true,
     "param_sweep_sigma_pct": 1.27,
     "overall_pass": true,
     "reasoning": "All 3 spot checks PASS. Proposal is robust.",
     "run_time_minutes": 195
   }
   ```

---

## GENERATE Phase Checklist

Before starting GENERATE, ensure:

- [ ] `backend/backtest/backtest_engine.py` has transaction_cost parameter exposed
- [ ] HMM regime detector code exists or is ready to write (Phase 3.3 has skeleton)
- [ ] Parameter sweep utility function exists or is ready to write
- [ ] Results directory `backend/backtest/experiments/results/` exists
- [ ] All spot check functions stubbed out (signatures defined, body TBD)
- [ ] Evaluator integration point identified in `backend/agents/evaluator.py`

### Code Structure for GENERATE

```
backend/
  backtest/
    backtest_engine.py  (existing, add transaction_cost param)
    spot_checks.py      (NEW) — CostStressTest, RegimeShiftTest, ParamSweepTest
    regime_detector.py  (NEW) — HMM-based regime detection
  agents/
    evaluator.py        (existing, add spot check integration)
experiments/
  results/              (NEW) — output JSON files
  phase3.2.1_log.md    (NEW) — run log, timings
```

---

## Evaluation Criteria (EVALUATE Phase)

In EVALUATE phase, we'll verify:

1. **Cost Stress Test**
   - Baseline Sharpe: 1.1705 ✓
   - 2× Cost Sharpe: ≥ 1.0535 ✓ (expect ~1.05)
   - Assertion: `assert cost_sharpe >= 0.9 * baseline_sharpe`

2. **Regime Shift Test**
   - Identified ≥ 2 regimes from 2018-2025
   - Sharpe in each regime ≥ 80% baseline
   - Assertion: `assert all(r['sharpe'] >= 0.8 * baseline for r in regimes)`

3. **Parameter Sweep Test**
   - 10 random nearby params tested
   - Sharpe variance ≤ 5% of mean
   - Assertion: `assert sigma_pct <= 5.0`

4. **Overall Evaluator Quality**
   - Spot checks detect overfitting
   - Spot checks don't reject good proposals (no false positives)
   - Spot checks correctly reject bad proposals (we'll test with intentionally bad params)

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| HMM regime detection unstable | Medium | Phase 3.3 has research; we can use simple 2-regime split if HMM fails |
| Parameter sweep too slow (10 × 30 min) | Low | Can reduce to 5 combos if time-constrained; parallel execution in Phase 4 |
| Cost params not exposed in backtest engine | Low | If not exposed, add quickly in GENERATE phase |
| Regime shift harness crashes | Low | Fallback to 2-regime (before/after COVID) if multi-regime fails |

---

## Timeline (PLAN → GENERATE → EVALUATE)

| Phase | Start | Duration | End | Status |
|-------|-------|----------|-----|--------|
| PLAN (current) | 08:56 UTC | 30 min | 09:26 UTC | In Progress |
| GENERATE | 09:30 UTC | 3-4 hr | 12:30-13:30 UTC | Next |
| EVALUATE | 13:30 UTC | 1-2 hr | 14:30-15:30 UTC | Final |
| **Phase 3.3** | **~15:30 UTC** | **6-8 hr** | **21:30-23:30 UTC** | **Queued** |

---

## Sign-Off

**PLAN Phase Status:** ✅ Ready for GENERATE
- [x] Test data sets defined (baseline, 2× cost, regime-split, param-sweep)
- [x] Spot check invocation patterns designed
- [x] Expected outputs specified
- [x] Code structure outlined
- [x] Evaluation criteria documented
- [x] Risks identified and mitigated

**Next Action:** Start GENERATE phase at 09:30 UTC
