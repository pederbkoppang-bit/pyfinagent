# Quant Evaluator — Grading Criteria

> Modeled on Anthropic's harness design: concrete, gradable terms that turn
> subjective judgments ("is this strategy good?") into specific, testable checks.
> A score below 6 on ANY criterion is a FAIL.

---

## 1. Statistical Validity (weight: 40%)

**What we're checking:** Is the improvement real, or noise/overfitting?

| Check | Pass | Fail |
|-------|------|------|
| Deflated Sharpe Ratio | DSR ≥ 0.95 | DSR < 0.95 |
| Seed stability | Sharpe std < 0.10 across 5 seeds | std ≥ 0.10 |
| Window concentration | No single window > 30% of total return | Any window > 30% |
| Autocorrelation | Ljung-Box p > 0.05 on daily returns | p ≤ 0.05 (serial correlation) |
| Lo (2002) correction | Adjusted Sharpe within 15% of raw | Adjusted drops > 15% |

**Scoring:**
- 9-10: All checks pass, DSR > 0.99, rock solid
- 7-8: All checks pass, minor concerns
- 5-6: One check borderline, needs investigation
- 1-4: One or more checks fail

---

## 2. Robustness (weight: 30%)

**What we're checking:** Does it work across market regimes, or just one period?

| Check | Pass | Fail |
|-------|------|------|
| Sub-period: 2018-2020 | Sharpe > 0.3 | Sharpe ≤ 0.0 |
| Sub-period: 2020-2022 | Sharpe > 0.3 | Sharpe ≤ 0.0 |
| Sub-period: 2022-2025 | Sharpe > 0.3 | Sharpe ≤ 0.0 |
| Feature stability | Top-5 MDA overlap ≥ 3/5 across sub-periods | Overlap < 2/5 |
| 2× transaction costs | Sharpe still > 0.7 | Sharpe drops below 0.7 |

**Scoring:**
- 9-10: Positive Sharpe in all sub-periods, features stable, survives 2× costs
- 7-8: Positive in all sub-periods, minor feature drift
- 5-6: One sub-period weak but positive, features somewhat unstable
- 1-4: Negative in any sub-period, or collapses under 2× costs

---

## 3. Simplicity (weight: 15%)

**What we're checking:** Is the complexity justified, or are we curve-fitting?

| Check | Pass | Fail |
|-------|------|------|
| Parameter count | ≤ 15 active params | > 20 active params |
| Marginal contribution | Each param contributes ≥ +0.05 Sharpe | Any param < +0.02 |
| t-statistic | New factors have t ≥ 3.0 (Harvey et al. 2016) | t < 2.0 |
| Ablation | Removing any single improvement drops Sharpe < 5% | Any single removal > 10% drop |

**Scoring:**
- 9-10: Minimal parameters, each clearly justified, clean ablation
- 7-8: Reasonable parameter count, most justified
- 5-6: Some questionable parameters, ablation shows concentration risk
- 1-4: Over-parameterized, improvements driven by 1-2 fragile additions

---

## 4. Reality Gap (weight: 15%)

**What we're checking:** Will this work with real money, not just in backtests?

| Check | Pass | Fail |
|-------|------|------|
| Transaction costs | ≥ 10 bps round-trip modeled | < 5 bps or zero |
| Execution timing | Not assuming exact close price | Assuming perfect execution |
| Position limits | Max position < 10% of portfolio | Any position > 15% |
| Universe diversity | Includes mid-cap, not just mega-cap | Only top-20 S&P |
| Survivorship bias | Addressed or documented | Ignored |
| Turnover | Annual turnover < 500% | Extreme turnover (> 1000%) |

**Scoring:**
- 9-10: Realistic costs, conservative execution, diverse universe, bias addressed
- 7-8: Mostly realistic, minor gaps documented
- 5-6: Some assumptions too optimistic, needs fixes before live
- 1-4: Backtest is fantasy, would lose money in reality

---

## Verdict Rules

- **PASS**: All criteria ≥ 7, no criterion below 6
- **CONDITIONAL**: One criterion at 5-6, others ≥ 7 (fix required before next cycle)
- **FAIL**: Any criterion below 5, or two criteria below 6

## Anti-Leniency Protocol

From Anthropic's research: "Claude is a poor QA agent out of the box. It identifies legitimate issues, then talks itself into deciding they aren't a big deal."

To counter this:
1. Grade each criterion BEFORE writing the overall verdict
2. Never upgrade a score after seeing the overall picture
3. If uncertain between two scores, pick the lower one
4. The cost of approving a bad strategy is losing real money
5. False negatives (rejecting a good strategy) cost time; false positives (approving a bad one) cost money
