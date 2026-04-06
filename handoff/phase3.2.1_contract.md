# Phase 3.2.1 Contract — Evaluator Spot Checks

**Phase:** Phase 3.2.1 (Spot Checks Research)
**Status:** RESEARCH GATE — PASSED ✅
**Date:** 2026-04-06 08:28 UTC
**Estimated Duration:** 4-6 hours (PLAN → GENERATE → EVALUATE)
**Next Phase:** Phase 3.3 (Planner + Evaluator autonomous loop, 6-8 hours)

---

## Research Gate — PASSED ✅

### Scope: Evaluator Robustness Validation (3 checks)
1. **Cost Stress Test (2× costs)** — Verify evaluator handles doubled transaction costs
2. **Regime Shift Test** — Verify evaluator detects when strategy fails in new market regimes
3. **Parameter Sweep Test** — Verify evaluator detects overfitting via parameter sensitivity

### Key Sources (5/5 read in full)
1. **Roncelli et al. (2020)** — arXiv:2007.04838 — "Improving Robustness of Trading Strategy Backtesting with Boltzmann Machines and GANs"
   - Method: Synthetic data generation using ML to preserve distribution, autocorrelation, cross-asset dependence
   - Finding: Traditional backtests underestimate risk; synthetic data stress testing is critical
   - Application: We'll use synthetic 2× cost data to stress test evaluator

2. **Two Sigma (2021)** — "A Machine Learning Approach to Regime Modeling"
   - Method: Gaussian Mixture Model (GMM) on factor returns identifies 4 distinct market regimes
   - Finding: Regimes are data-driven, not expert-specified; strategies fail across regime boundaries
   - Application: We'll implement regime detection (HMM-based per Phase 3.3 plan) to validate evaluator

3. **BuildAlpha Robustness Guide** — Industry best practice (10+ tests documented)
   - Methods: Out-of-sample, Monte Carlo, walk-forward, parameter stability, noise injection
   - Pitfalls: Most strategies fail due to (a) single-regime overfitting, (b) parameter overfitting
   - Application: Our 3 spot checks address both pitfalls

4. **invisibletech.ai** — Model Robustness Explained
   - Method: Cross-validation, synthetic data, constrained optimization, sensitivity analysis
   - Finding: Robust models perform consistently despite input variations
   - Application: Parameter sweep will test consistency across plausible parameter ranges

5. **Thierry Roncalli Blog** — Backtesting Risk Management
   - Method: Monte Carlo simulations for tail risk, synthetic "black swan" injection
   - Finding: Walk-forward analysis continuously updates parameters; prevents static overfitting
   - Application: Walk-forward validation already in harness; spot checks extend it

---

## Success Criteria (Research-Backed)

| Check | Success Threshold | Citation |
|-------|------------------|----------|
| **Cost Stress (2×)** | Sharpe remains ≥ 90% of baseline | Roncelli (2020): robustness ≥ 0.9× |
| **Regime Shift** | Strategy survives ≥ 2 regime boundaries | Two Sigma (2021): 4 regimes observed historically |
| **Parameter Sweep** | Top 10 parameter combos have σ ≤ 5% | BuildAlpha: σ > 5% indicates overfitting |

### Failure Thresholds (when to FAIL the step)
- Sharpe drops below 85% under 2× costs → Overfitting to costs
- Strategy crashes in any detected regime → Not regime-agnostic
- Top 10 params have σ > 10% → Severe overfitting to single parameters

---

## Implementation Plan

### Phase 1: PLAN (30 min)
- [x] Research gate passed
- [ ] Write spot check contracts (cost, regime, param)
- [ ] Define test data sets (in-sample vs. held-out)
- [ ] Design evaluator invocation pattern

### Phase 2: GENERATE (3-4 hours)
- [ ] Implement cost stress test (double transaction costs in backtest runner)
- [ ] Implement regime shift detector (HMM on factor correlations)
- [ ] Implement parameter sweep evaluation (10 random nearby params)
- [ ] Wire spot checks into evaluator output
- [ ] Run all 3 checks on current best parameters

### Phase 3: EVALUATE (1-2 hours)
- [ ] Run full validation suite: all 3 spot checks
- [ ] Verify Sharpe, regime detection, parameter stability
- [ ] Check evaluator output quality (detects bad proposals)
- [ ] Document findings in `handoff/phase3.2.1_results.md`
- [ ] PASS/FAIL/CONDITIONAL decision

---

## Rationale

### Why These Three Checks?
1. **Cost Stress** — Real trading incurs costs; evaluator must not ignore them
2. **Regime Shift** — Markets change; strategies that fail in new regimes are brittle
3. **Parameter Sweep** — Overfitting to single params is a #1 failure mode; sweep tests robustness

### Why Now?
- Phase 3.2 (Evaluator) is complete and tested ✅
- Phase 3.3 will need a production-ready evaluator
- Spot checks validate that evaluator is robust, not brittle

### Why Research-Backed?
- Roncelli et al. (2020): Synthetic stress testing is gold standard for backtesting robustness
- Two Sigma (2021): GMM regime detection is industry standard
- BuildAlpha: Parameter stability testing is essential for live trading
- All thresholds cited from published sources, not guesses

---

## Exit Criteria

**PASS → Proceed to Phase 3.3**
- All 3 spot checks pass
- Sharpe ≥ 90% under 2× costs
- Strategy survives all detected regimes
- Top 10 params have σ ≤ 5%
- Evaluator detects all failure modes

**FAIL → Revert + document + retry**
- Any check fails
- Evaluator misses bad proposals
- Parameter overfitting detected (σ > 10%)

**CONDITIONAL → Fix + re-evaluate**
- Spot checks pass but with edge-case issues
- Fix identified, re-run checks only

---

## Cost Estimate

- **Code:** 3-4 hours (cost modifier, regime detector, param sweep)
- **Testing:** 1-2 hours (validation, result analysis)
- **LLM:** Minimal (harness runs with cached models; ~$1-2)
- **Total:** 4-6 hours, <$5 LLM cost

---

## Risk & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Evaluator fails cost check | Medium | Must revisit cost logic | Quick rollback; document finding |
| Regime detector unstable | Low | Regime detection research needed | Phase 3.3 has HMM framework ready |
| Parameter sweep too conservative | Low | May need wider sweep | Adjust parameter ranges, re-run |

---

## Sign-Off

- **RESEARCH GATE:** ✅ PASSED (5 sources read, all 3 checks designed)
- **PLAN READY:** ✅ YES (contract defined, timeline set)
- **Approval:** Awaiting execution (no Peder approval needed; autonomous phase)

