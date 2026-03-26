# PyFinAgent — Optimization Plan (Ford's Master Plan)

> **Goal**: Make money. Ship a validated, evidence-based trading signal system by May 2026.
> **Budget**: TIGHT — currently negative cash flow (-$10K, $200/month costs). Every dollar must have ROI. BigQuery (low cost OK), GitHub Models (free via Copilot Pro), Claude Max (via OpenClaw/Ford), Vertex AI (avoid unless proven high-ROI). **⚠️ LLM API costs require Peder's explicit approval.**
> **Timeline**: March-April = validate + fix + research. May = go live with real money (Slack signals → manual trading).

---

## Phase 0: Audit & Validate (Week 1 — NOW)
*Fix what's broken before optimizing. No point tuning a leaky engine.*

### 0.1 Formula Validation (academic cross-check)
- [ ] **Sharpe Ratio** — validate against Sharpe (1994) and Lo (2002) √T annualization. Current impl looks correct (daily returns, √252). Verify risk-free rate handling (currently 4% annualized, divided by periods_per_year).
- [ ] **Deflated Sharpe Ratio** — validate against Bailey & López de Prado (2014). Current E[max(SR)] uses Euler-Mascheroni approximation. Cross-check with original paper's exact formula.
- [ ] **Sample Weights (Average Uniqueness)** — validate O(n²) overlap computation against AFML Ch. 4. Correct intent but verify edge cases (non-overlapping labels should get weight=1).
- [ ] **Fractional Differentiation** — validate against AFML Ch. 5. Current fixed-width window implementation. Verify d=0.4 achieves stationarity while preserving memory (ADF test).
- [ ] **Triple Barrier Labels** — validate against AFML Ch. 3. Check that holding_days counts trading days vs calendar days (current: iterates price rows, which are trading days ✓).
- [ ] **Monte Carlo VaR** — validate GBM assumption. GBM assumes log-normal returns; reality has fat tails. Document limitation. Consider adding Student-t or historical simulation as alternative.
- [ ] **Position Sizing (Inverse Vol)** — validate against AQR literature. Current: `probability × (target_vol / stock_vol) × nav / max_positions`. Verify this matches Kelly-inspired fractional sizing.
- [ ] **Scalar Metric** — validate the tx cost penalty. `risk_adjusted_return × (1 - min(0.3, turnover × tx_cost))`. This is a custom extension — document its theoretical justification.

### 0.2 Bug Fixes (known issues from code review)
- [ ] **Quality Score incomplete** — add `fcf_yield`, `dividend_yield`, `revenue_growth_yoy` to quality composite per Asness et al. (2019) QMJ definition
- [ ] **Factor Model hardcoded normalization** — replace with cross-sectional percentile ranking within the universe at each training date
- [ ] **Mean Reversion label doesn't use mr_holding_days** — wire `mr_holding_days` into `_compute_mean_reversion_label()` as forward-looking barrier
- [ ] **Meta-Label is a stub** — implement proper 2-stage model per AFML Ch. 3.6 (primary model → meta-label → bet sizing)
- [ ] **No transaction cost in TB labels** — labels should account for spread/slippage (at minimum, shift TP down and SL up by estimated spread)
- [ ] **Sample weight O(n²)** — optimize using interval tree or sorted-endpoint sweep (O(n log n))

### 0.3 Overfitting Diagnostics
- [ ] **Walk-forward leakage audit** — verify no future data leaks into training features. Check `build_feature_vector` uses only data available as-of `cutoff_date`. Verify embargo gap is working.
- [ ] **Label leakage check** — Triple Barrier looks ahead by `holding_days`. Verify training labels don't peek into test windows.
- [ ] **Feature importance stability** — run baseline backtest, check if MDA top-10 features are stable across windows. High variance = regime-dependent = overfitting risk.
- [ ] **DSR validation** — run 5+ baseline backtests with different random seeds. If Sharpe varies wildly, the result is not robust regardless of DSR.
- [ ] **Backtest vs reality gap analysis** — document all assumptions that differ from live trading (no slippage model, no partial fills, close-price execution, no market impact).

---

## Phase 1: Quant Engine Optimization (Weeks 2-3)
*Zero LLM cost. Pure math. This is where we find real alpha.*

### 1.1 Data Quality & Depth
- [ ] Verify BQ historical data completeness — check for gaps in prices, fundamentals, macro
- [ ] Extend historical data to 2018 or earlier if needed (more walk-forward windows = more reliable DSR)
- [ ] Add survivorship bias check — are we only testing current S&P 500 members? Need historical constituents
- [ ] Add corporate actions adjustment — stock splits, dividends (yfinance adjusted close should handle this, verify)

### 1.2 Feature Engineering (research-backed additions)
- [ ] **Volatility-adjusted TB barriers** — replace fixed tp_pct/sl_pct with `daily_vol × multiplier` per AFML recommendation
- [ ] **12-1 Momentum** — compute `momentum_12m - momentum_1m` to avoid short-term reversal per Jegadeesh & Titman (1993)
- [ ] **Bollinger Band features** — add `bb_upper_distance`, `bb_lower_distance` for mean reversion
- [ ] **Amihud filter in labels** — penalize illiquid stocks in position sizing
- [ ] **Turbulence Index integration** — already coded but not used in backtest. High turbulence → reduce position sizes
- [ ] **Cross-sectional features** — rank features within universe (percentile momentum, percentile value) for regime independence

### 1.3 Strategy Improvements
- [ ] **Implement proper meta-labeling** (AFML Ch. 3.6) — two-stage model, secondary predicts primary correctness
- [ ] **Strategy blending** — weighted vote across strategies (new optimizer params: tb_weight, qm_weight, etc.)
- [ ] **Regime detection** — use turbulence index or HMM to switch strategies based on market regime
- [ ] **Portfolio optimization** — replace equal-weight with mean-variance or risk parity

### 1.4 Autoresearch Loop Improvements
- [ ] Align `trading_agent.md` and `quant_strategy.md` with actual Karpathy autoresearch principles:
  - **One file to modify** → we modify strategy params (not a file, but same concept ✓)
  - **Fixed time budget** → each backtest takes same data range ✓
  - **Single metric** → Sharpe ratio ✓ (DSR is the gate, not the metric)
  - **Simplicity criterion** → need to add: reject complexity that doesn't justify delta
  - **NEVER STOP** → optimizer runs until stopped ✓
- [ ] Add **experiment analysis** capability — after N experiments, analyze patterns in kept vs discarded
- [ ] Implement **warm cache** improvements — preload BQ data once, reuse across all optimizer iterations

---

## Phase 2: Academic Research Deep Dive (Ongoing, parallel with Phase 1)
*Every decision must be evidence-based. We are not guessing.*

### 2.1 Required Reading & Implementation
- [ ] **López de Prado, "Advances in Financial Machine Learning" (2018)** — we implement Ch. 3-8. Need to verify Ch. 9 (bet sizing), Ch. 10 (cross-validation in finance), Ch. 12 (backtesting through cross-validation)
- [ ] **Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014)** — verify our DSR implementation matches the paper exactly
- [ ] **Lo, "The Statistics of Sharpe Ratios" (2002)** — verify non-IID return adjustments
- [ ] **Asness, Frazzini & Pedersen, "Quality Minus Junk" (2019)** — fix quality score to match full QMJ definition
- [ ] **Fama & French, "A Five-Factor Model" (2015)** — verify factor construction
- [ ] **Jegadeesh & Titman, "Returns to Buying Winners and Selling Losers" (1993)** — momentum implementation
- [ ] **Harvey, Liu & Zhu, "...and the Cross-Section of Expected Returns" (2016)** — 316 factors tested, most are false. Use their t-stat threshold (3.0+) for new factors.

### 2.2 New Research to Evaluate
- [ ] **Machine Learning for Asset Managers (López de Prado, 2020)** — newer book, more ML techniques
- [ ] **Deep Learning for Finance (recent papers)** — evaluate if GradientBoosting is still SOTA for tabular financial data (likely yes — see Grinsztajn et al. 2022 "Why do tree-based models still outperform deep learning on tabular data?")
- [ ] **Transaction cost modeling** — Almgren & Chriss (2000) for market impact, Kissell (2013) for realistic cost modeling
- [ ] **Portfolio construction** — Black-Litterman model for combining quantitative signals with views
- [ ] **Alternative data** — evaluate satellite data, credit card data, web scraping for alpha (budget-constrained)

---

## Phase 3: LLM-Guided Research & Pipeline Optimization (Weeks 4-5)
### ⚠️ GATE: REQUIRES PEDER'S EXPLICIT APPROVAL BEFORE STARTING
*Negative cash flow — every LLM call must justify its cost. Ford does NOT proceed without sign-off.*

### 3.0 LLM-as-Researcher (Karpathy-inspired autoresearch for trading)
*Inspired by Karpathy's autoresearch + Kou et al. (2024) "Automate Strategy Finding with LLM in Quant Investment" (53% return on SSE50).*

**The idea:** Instead of a dumb parameter sweep, use an LLM to reason about experiment results and hypothesize what to try next — like a quant researcher, not a grid search.

**Our approach (budget-conscious):**
- [ ] **Batch-then-reason:** Run 10-20 optimizer experiments (free CPU), then feed the log to an LLM once for analysis. Not a tight loop.
- [ ] **Max 5-10 LLM reasoning calls per day** — analyze patterns in kept vs discarded experiments, suggest structural changes
- [ ] **LLM can propose code changes** to strategy (new features, different blending weights, regime-aware params) — but Ford reviews before running
- [ ] **Experiment log as context:** maintain structured TSV of all experiments so the LLM can spot patterns humans miss
- [ ] **Overfitting guard:** DSR validation on every proposed change. If DSR drops, discard regardless of Sharpe improvement.

**What this is NOT:**
- ❌ Not a tight autoresearch loop (too expensive, ~$50+/day in API calls)
- ❌ Not unreviewed code changes (overfitting factory)
- ❌ Not a replacement for the current optimizer (complements it)

**Cost estimate:** ~$2-5/day if throttled properly. Peder approves budget before starting.

**Key references:**
- Kou et al. (2024) — "Automate Strategy Finding with LLM in Quant Investment" [arXiv:2409.06289]
- FactorEngine (March 2026) — LLM-driven alpha factor mining
- AlphaForgeBench (Feb 2026) — benchmark for LLM trading strategy design

### 3.1 Regime Detection (zero LLM cost, high ROI)
*Classical quant — no API costs. This is the single highest-ROI improvement for market adaptation.*
- [ ] **HMM-based regime detector** — identify 2-3 market regimes (bull, bear, high-vol) from returns + volatility
- [ ] **Turbulence index switching** — already coded, wire into param selection: low turbulence → aggressive params, high → defensive
- [ ] **Per-regime parameter sets** — optimizer finds best params for each regime independently
- [ ] **Rolling re-optimization** — weekly/monthly re-run optimizer on latest N months (cron job, zero LLM cost)

### 3.2 Skill Optimization (SkillOpt loop)
- [ ] Run SkillOpt on each of the 29 agent skills, starting with highest-impact agents (Synthesis, Moderator, Risk Judge)
- [ ] Use outcome_tracker feedback to identify which agents' signals correlate most with actual returns
- [ ] Wire MetaCoordinator: MDA importance → Agent targeting (if MDA says `nlp_sentiment_score` matters, optimize NLP Sentiment Agent skill)

### 3.3 Signal Quality Validation
- [ ] Run analysis on 20+ tickers, compare recommendations vs actual price movement over 30/60/90 days
- [ ] Measure signal accuracy per enrichment tool — which tools actually add alpha?
- [ ] Consider dropping tools that don't contribute (reduce cost, reduce noise)

### 3.4 Model Selection
- [ ] Benchmark gpt-4.1 (current, via GitHub Models) vs alternatives on signal quality
- [ ] Document cost per analysis for each model option
- [ ] Evaluate if DEEP_THINK_MODEL justifies its cost for Critic/Synthesis

---

## Phase 4: Production Readiness (Week 6 — Late April)
*Get ready for real money in May.*

### 4.1 Slack Integration
- [ ] Configure Slack bot for daily signal delivery
- [ ] Design alert format: ticker, signal, confidence, key reasons, risk level, position size suggestion
- [ ] Set up morning digest (top opportunities + portfolio rebalance suggestions)

### 4.2 Paper Trading Validation
- [ ] Run paper trading for 2+ weeks with the optimized system
- [ ] Track every signal vs actual outcome
- [ ] Validate that live signals match backtest expectations (backtest-reality gap)

### 4.3 Risk Management
- [ ] Define max portfolio size, max single position, max daily loss
- [ ] Implement stop-loss monitoring
- [ ] Define when to override signals (earnings week? FOMC? major events?)

### 4.4 Go-Live Checklist
- [ ] All formulas validated ✓
- [ ] DSR ≥ 0.95 on out-of-sample backtest ✓
- [ ] Paper trading matches backtest within tolerance ✓
- [ ] Slack signals working reliably ✓
- [ ] Risk limits defined and tested ✓
- [ ] Peder's manual review process defined ✓

---

## File Restructuring Plan

### trading_agent.md — Rewrite
Current file mixes architecture description, implementation status, research notes, and API docs. Rewrite to follow Karpathy's program.md pattern:
1. **Mission** (what we optimize)
2. **What you CAN modify** (strategy params, skill prompts, cache TTLs)
3. **What you CANNOT modify** (pipeline structure, schemas, evaluation harness)
4. **The metric** (Sharpe ratio, gated by DSR ≥ 0.95)
5. **The loop** (baseline → modify → measure → keep/discard → log)
6. **Research context** (condensed from current §2 and §6)
7. **Experiment log** (reference to TSV files)

### Agent Skills — Audit
- Verify all 29 skills follow SKILL_TEMPLATE.md structure
- Ensure every skill has correct Research Foundations section
- Verify all {{variable}} placeholders match prompts.py injection
- Check that Anti-Patterns reflect actual observed failures (not just theoretical)

### AGENTS.md — Update
- Add Phase 1-3 changes from trading_agent.md
- Update directory structure if files moved
- Ensure Quick Start works on macOS (not just Windows)

---

## Budget Tracking

| Item | Monthly Cost (est.) | Notes |
|------|-------------------|-------|
| BigQuery storage | ~$5 | Historical data tables, analysis results |
| BigQuery queries | ~$5-20 | Depends on backtest frequency |
| GitHub Models (Copilot Pro) | $0 (included) | gpt-4.1 via GitHub token |
| Claude Max | Already paid | Used via OpenClaw (Ford), not API |
| Alpha Vantage | Free tier | 5 req/min, sufficient |
| FRED | Free | No cost |
| Vertex AI | $0 (avoid) | Only if proven high-ROI |
| **Total** | **~$10-25/month** | |

---

## Success Criteria

1. **Backtest Sharpe > 1.0** with DSR ≥ 0.95 on out-of-sample data
2. **Beat SPY** over the backtest period (after transaction costs)
3. **Paper trading matches backtest** within 20% tolerance over 2+ weeks
4. **All formulas verified** against original academic papers
5. **No known overfitting** — stable feature importance, robust to random seed variation

---

*This plan is a living document. Ford updates it as work progresses.*
*Last updated: 2026-03-25 by Ford*
