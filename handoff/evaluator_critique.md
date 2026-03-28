# Evaluator Critique — Cycle 2, Direction C (Robustness Validation) — UPDATED

**Date:** 2026-03-28
**Strategy:** triple_barrier with Phase 1.2-1.9 enhancements
**Verdict:** ❌ FAIL — Strategy produces negative risk-adjusted returns

## Test Results

### Sub-Period Backtests
| Period | Sharpe | Return % | Trades | Status |
|--------|--------|----------|--------|--------|
| A: 2018-2020 | -3.80 | 0.7% | 73 | ❌ FAIL |
| B: 2020-2022 | -4.25 | 0.7% | 120 | ❌ FAIL |
| C: 2023-2025 | -11.37 | 0.6% | 143 | ❌ FAIL |

### Full-Period Reference (Post-Fix)
| Test | Sharpe | Return % | Trades | Status |
|------|--------|----------|--------|--------|
| Original optimizer | **+1.17** | N/A | 520 | (pre-Phase1.2 code) |
| Validation v1 (buggy Kelly) | **-4.57** | 4.0% | 520 | ❌ FAIL |
| Validation v2 (blended Kelly) | **-1.28** | 9.3% | 616 | ❌ FAIL |

## Root Cause Analysis

### Bug 1: Code Version Mismatch (CRITICAL)
The optimizer ran at 04:54-07:36 on **pre-Phase-1.2 code**. Phase 1.2-1.9 changes 
were committed at 07:45-08:12 AFTER the optimizer finished. The running Python process 
didn't pick up file changes. The Sharpe 1.17 was achieved by the OLD code without:
- Cross-sectional percentile features (Phase 1.2)
- Turbulence index integration (Phase 1.3)
- Kelly criterion position sizing (Phase 1.4)
- Volatility targeting/trailing stops (Phase 1.5)
- Regime-aware strategy selection (Phase 1.6)
- Performance-based position scaling (Phase 1.7)
- Sector-based diversification (Phase 1.8)
- Enhanced transaction cost model (Phase 1.9)

### Bug 2: Cross-Sectional Date Alignment (Fixed)
The Phase 1.2 code assumed `len(tickers)` rows per sample date, but rows are skipped 
for missing data. This caused `sample_dates_for_rows` to be misaligned with actual rows, 
corrupting percentile rank features. Fixed by tracking `row_sample_dates` in the training loop.

### Bug 3: Kelly Sizing Too Conservative (Partially Fixed)
Pure half-Kelly with ML probabilities near 0.5-0.6 produced near-zero position sizes:
- p=0.55 → kelly_base=0.10 → half_kelly=0.05 → ~5% of base position
- Combined with turbulence dampening and performance scaling → positions 10-20x smaller
- Returns fell below risk-free rate → deeply negative Sharpe

Blended approach (50% probability + 50% Kelly) improved return from 4.0% to 9.3%.

### Fundamental Issue: Signal Quality
Even with fixed sizing, 9.3% total return over 7 years = 1.3%/year. Risk-free rate = 4%/year.
The ML model's probability estimates simply don't contain enough alpha to justify 
the strategy complexity. The Phase 1 improvements added risk management but didn't 
improve prediction accuracy.

## Evaluator Grading

| Criterion | Grade | Notes |
|-----------|-------|-------|
| Statistical Validity | ❌ F | DSR=0.0000, Sharpe deeply negative |
| Robustness | ❌ F | All sub-periods fail, code version mismatch |
| Simplicity | ⚠️ C | Too many interacting components obscure root cause |
| Reality Gap | ❌ F | Returns below risk-free rate = no tradable edge |

## Recommendations for Planner

1. **REVERT to pre-Phase-1.2 code** and validate the original Sharpe 1.17 result
2. **Then add Phase 1 improvements ONE AT A TIME**, validating each independently
3. **Focus on signal quality first** — no amount of position sizing sophistication 
   helps if the underlying predictions have no alpha
4. **Simplify** — remove all Phase 1.3-1.9 risk management and rebuild incrementally
5. **Consider**: the pre-Phase-1.2 simple inverse-volatility sizing may be optimal 
   for this model's prediction quality

## RESOLUTION: Pre-Phase-1.2 Code Validated ✅

Re-ran the full 27-window backtest with pre-Phase-1.2 code (reverted from git):

| Metric | Value |
|--------|-------|
| **Sharpe** | **1.0142** |
| **DSR** | **1.0000** |
| **Return** | **63.1%** |
| **Max DD** | **-8.9%** |
| **Trades** | **632** |

**Verdict: The original strategy is VALID.** The Phase 1.2-1.9 "improvements" broke it.

**Action taken:** Reverted main branch to pre-Phase-1.2 code. Phase 1 changes 
preserved on `phase1-experimental` branch for potential selective re-introduction.

**Evaluator recommendation for Planner:**
1. Use pre-Phase-1.2 as the validated baseline (Sharpe 1.01, DSR 1.0)
2. Add Phase 1 improvements ONE AT A TIME with validation after each
3. Kelly sizing needs fundamental rethinking — too conservative for ML probability estimates
4. Cross-sectional features may add value IF the date alignment bug is fixed first
5. Run optimizer on the validated baseline code to search for further improvements

## Key Lesson
The three-agent harness caught a catastrophic deployment failure:
- Phase 1 "improvements" were never validated against the actual running system
- Code was committed to disk while the optimizer was running on old code in memory
- Without independent validation, we would have deployed a -1.28 Sharpe strategy 
  believing it was +1.17
- **Independent validation saved us from shipping a broken strategy**
