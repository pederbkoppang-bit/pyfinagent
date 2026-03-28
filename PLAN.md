# PyFinAgent — Master Plan v2

> **Goal**: Make money. Ship a validated, evidence-based trading signal system by May 2026.
> **Budget**: TIGHT — negative cash flow (-$10K, $200/month costs). Every dollar must have ROI. **⚠️ LLM API costs require Peder's explicit approval.**
> **Timeline**: March-April = validate + optimize + research. May = go live (Slack signals → manual trading).
> **Architecture**: Three-agent harness (Planner → Generator → Evaluator) inspired by [Anthropic's harness design for long-running apps](https://www.anthropic.com/engineering/harness-design-long-running-apps).

---

## Architectural Philosophy

### Why a multi-agent harness, not a monolith

Our previous approach was a single optimizer loop: propose param → run backtest → keep/discard. This is the "solo agent" equivalent. It works but hits two ceilings described by Anthropic's research:

1. **Self-evaluation failure.** The optimizer can't judge whether its own improvements are real alpha or overfitting. It "praises its own work" — a Sharpe improvement passes DSR and gets kept, even when a human quant would flag it as fragile.

2. **Context degradation.** Long optimization runs lose coherence. The optimizer doesn't remember why earlier experiments failed, can't spot patterns across 40+ experiments, and can't propose structural changes (new features, strategy rewrites) — only parameter perturbations.

### The Anthropic harness pattern applied to quant research

From the article's core insight: **separate the agent doing the work from the agent judging it.** Tuning a standalone evaluator to be skeptical is far more tractable than making a generator critical of its own work.

Applied to our domain:

| Anthropic Pattern | PyFinAgent Equivalent |
|---|---|
| **Planner** — expands 1-sentence prompt into full spec | **Research Planner** — reads experiment log + academic literature, proposes next research direction |
| **Generator** — implements features sprint by sprint | **Strategy Generator** — implements code changes, runs backtests, produces experiment results |
| **Evaluator** — tests the running app via Playwright, grades against criteria | **Quant Evaluator** — runs independent validation: overfitting tests, robustness checks, out-of-sample verification, academic cross-checks |
| **Sprint contracts** — generator and evaluator negotiate "done" criteria before work | **Experiment contracts** — evaluator defines what constitutes a valid improvement BEFORE the generator runs it |
| **Context resets** — fresh agent with structured handoff artifact | **Run boundaries** — each optimization cycle starts fresh with a handoff file containing: best params, experiment history, evaluator critique, next research directions |
| **Grading criteria** — concrete, gradable terms for subjective quality | **Validation criteria** — concrete, gradable terms for strategy quality (see below) |

### Grading criteria for strategy quality

Inspired by the article's four design criteria, adapted for quantitative finance:

1. **Statistical validity** (most important): Is the improvement statistically significant? DSR ≥ 0.95. Sharpe stable across 5+ random seeds. No look-ahead bias.

2. **Robustness**: Does performance hold across regimes? Test on 2008 crisis, 2020 COVID crash, 2022 bear market separately. Feature importance stable across windows.

3. **Simplicity**: Is the complexity justified? Every new parameter must improve Sharpe by at least +0.05 to be kept. Reject complexity that doesn't justify its delta. (Harvey, Liu & Zhu 2016: t-stat ≥ 3.0 for new factors.)

4. **Reality gap**: How close is backtest to real trading? Transaction costs modeled realistically. No partial fills assumed. Survivorship bias addressed. Slippage estimated.

---

## Phase 0: Audit & Validate ✅ COMPLETE
*Completed March 25-26. Foundation is solid.*

- [x] Walk-forward leakage audit — no future data leaks found
- [x] Formula validation — Sharpe, DSR, sample weights verified against papers
- [x] Quality Score — full Asness (2019) QMJ implementation (4 dimensions)
- [x] Mean reversion — mr_holding_days correctly wired
- [x] Cross-sectional percentile ranking — replaces hardcoded normalization

---

## Phase 1: Quant Engine Optimization ✅ COMPLETE
*Completed March 27-28. Sharpe 0.9848 → 1.1705 (+19%).*

### Implemented improvements (all zero LLM cost):
- [x] **1.2** Cross-sectional percentile features (13 new regime-independent features)
- [x] **1.3** Turbulence index integration (market stress → position sizing)
- [x] **1.4** Fractional Kelly Criterion (half-Kelly, Thorp recommendation)
- [x] **1.5** Volatility targeting + trailing stops + dynamic risk-free rate (FRED)
- [x] **1.6** Regime-aware strategy selection (normal/stressed/crisis)
- [x] **1.7** Performance-based position scaling (20-day adaptive risk)
- [x] **1.8** Sector-based diversification (correlation penalties)
- [x] **1.9** Market microstructure transaction costs (Almgren-Chriss model)

### Current best: Sharpe 1.1705 | DSR 0.9984 | Return 80.2% | MaxDD -12.0%

---

## Phase 2: Three-Agent Harness (Weeks 3-4) — NOW
*This is where we apply the Anthropic harness pattern. The solo optimizer found local optima. The harness finds structural improvements.*

### 2.0 Harness Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH PLANNER                              │
│  Reads: experiment_log.tsv, evaluator_critique.md, papers/      │
│  Writes: research_plan.md (next research direction + hypothesis) │
│  Runs: Once per cycle (every 15-20 experiments)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ research_plan.md
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STRATEGY GENERATOR                             │
│  Reads: research_plan.md, current codebase, best params         │
│  Does: Implements code changes, runs optimizer (15 iterations)   │
│  Writes: experiment_results.md, code diffs, new params           │
│  Runs: Continuous (hours of CPU time per cycle)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ experiment_results.md
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUANT EVALUATOR                                │
│  Reads: experiment_results.md, code diffs, backtest data         │
│  Does: Independent validation, overfitting checks, critique      │
│  Writes: evaluator_critique.md (pass/fail + detailed feedback)   │
│  Grades: statistical_validity, robustness, simplicity, reality   │
│  Runs: Once after each generator cycle                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ evaluator_critique.md
                           ▼
                    (feeds back to Planner)
```

### 2.1 Handoff Artifacts (structured context between agents)

Each agent communicates via files, not conversation. This is the key unlock from the article — files survive context resets and carry precise state.

**`handoff/research_plan.md`** — Planner output:
```markdown
## Cycle N Research Direction
### Hypothesis: [what we think will improve Sharpe]
### Evidence: [patterns from experiment log that support this]
### Proposed changes: [specific code modifications or new params]
### Success criteria: [what the evaluator should check]
### Risk: [what could go wrong / overfitting concerns]
```

**`handoff/experiment_results.md`** — Generator output:
```markdown
## Cycle N Results
### Changes made: [code diffs, new params]
### Experiments run: [N iterations, best/worst Sharpe, kept/discarded]
### Best result: [params, Sharpe, DSR, return, drawdown]
### MDA feature importance: [top 10 features and stability]
### Observations: [anything unexpected]
```

**`handoff/evaluator_critique.md`** — Evaluator output:
```markdown
## Cycle N Evaluation
### Statistical Validity: [score 1-10, detailed assessment]
### Robustness: [score 1-10, regime-specific tests]
### Simplicity: [score 1-10, complexity vs improvement tradeoff]
### Reality Gap: [score 1-10, backtest-to-live concerns]
### Verdict: PASS / FAIL / CONDITIONAL
### Required fixes: [if FAIL, what must change]
### Suggestions for next cycle: [what the planner should consider]
```

### 2.2 Research Planner Agent

**Role:** The "quant researcher" who reads all experiment data and decides what to investigate next. This replaces random parameter perturbation with directed research.

**Implementation:** Ford (me) acts as planner between optimization runs — reading experiment logs, analyzing patterns, proposing structural changes. For budget-approved LLM-guided research, this becomes a Claude agent with experiment context.

**Planner reads:**
- Full experiment TSV (40+ experiments with params, Sharpe, DSR, kept/discarded)
- Evaluator's previous critique
- Academic papers relevant to current research direction
- MDA feature importance trends across experiments

**Planner decides:**
- What type of change to try next (param tuning vs feature engineering vs strategy change)
- Which parameters to explore (based on sensitivity analysis from experiment log)
- Whether to continue current direction or pivot (analogous to the article's "refine or pivot" decision)
- Whether we've hit diminishing returns on current approach

### 2.3 Strategy Generator Agent

**Role:** The "quant developer" who implements changes and runs experiments. This is our current optimizer, enhanced with the ability to make structural changes (not just param sweeps).

**Implementation:** The existing `QuantStrategyOptimizer` for parameter exploration. For structural changes (new features, strategy logic), Ford implements code changes then runs the optimizer.

**Generator capabilities (expanding):**
- [x] Single-parameter perturbation (current)
- [ ] Multi-parameter coordinated proposals (param groups that interact)
- [ ] Feature addition/removal experiments (add a feature, measure impact)
- [ ] Strategy switching experiments (try blend vs triple_barrier vs quality_momentum)
- [ ] Ablation studies (remove one improvement at a time, measure drop)

### 2.4 Quant Evaluator Agent

**Role:** The "skeptical reviewer" who independently validates results. This is the key missing piece from our current system. The article's core insight: self-evaluation is unreliable; external evaluation drives real quality.

**Implementation:** A separate validation pipeline that runs AFTER the optimizer, applying independent tests the generator didn't run.

**Evaluator checks (grading criteria):**

#### Statistical Validity (weight: 40%)
- [ ] DSR ≥ 0.95 on the kept result
- [ ] Sharpe stable across 5 random seeds (std < 0.1)
- [ ] No single window drives >30% of total return (concentration check)
- [ ] Ljung-Box test on returns (reject if significant autocorrelation)
- [ ] Compare raw vs Lo (2002) adjusted Sharpe

#### Robustness (weight: 30%)
- [ ] Run backtest on 3 sub-periods: 2018-2020, 2020-2022, 2022-2025
- [ ] Sharpe positive in all 3 sub-periods (not just aggregate)
- [ ] Feature importance top-5 stable across sub-periods
- [ ] Performance under 2x transaction costs (reality buffer)

#### Simplicity (weight: 15%)
- [ ] Count active parameters vs baseline (penalize complexity)
- [ ] Each new parameter must contribute ≥ +0.05 Sharpe (ablation test)
- [ ] Reject if improvement < t-stat 3.0 (Harvey, Liu & Zhu threshold)

#### Reality Gap (weight: 15%)
- [ ] Transaction costs ≥ 10 bps round-trip
- [ ] No execution at exact close price (add 5 bps slippage)
- [ ] Max position < 10% of portfolio
- [ ] Universe includes at least some mid-cap stocks (not just mega-cap)

### 2.5 Sprint Contracts (Experiment Contracts)

Before each optimization cycle, the planner and evaluator negotiate what "done" looks like. This prevents the generator from optimizing the wrong thing.

**Example contract:**
```markdown
## Cycle 5 Contract
Hypothesis: Adding correlation-based position sizing will improve risk-adjusted returns.
Generator will: Implement pairwise correlation computation, modify size_position(), run 15 iterations.
Evaluator will verify:
  1. Sharpe improves by ≥ +0.05 vs baseline
  2. Max drawdown improves (less negative)
  3. No single sector > 40% of portfolio at any point
  4. Feature importance remains stable
  5. Performance holds under 2x transaction costs
Fail conditions: DSR < 0.95, Sharpe improvement < +0.03, drawdown worsens.
```

### 2.6 Context Management Strategy

From the article: "Context resets — clearing the context window entirely and starting a fresh agent, combined with a structured handoff — addresses both [coherence loss and context anxiety]."

**Our approach:**
- Each optimization cycle is a **clean context** (optimizer restarts from handoff file)
- The handoff file contains: best params (JSON), experiment log (TSV), evaluator critique (markdown)
- No reliance on conversation history — everything is in files
- The `optimizer_best.json` already implements this for params; extend to include evaluator state

**Warm-start protocol:**
```
1. Read handoff/evaluator_critique.md → understand what failed last cycle
2. Read handoff/research_plan.md → understand what to try next
3. Read optimizer_best.json → current best params
4. Read quant_results.tsv → full experiment history
5. Execute research plan → run optimizer with new direction
6. Write handoff/experiment_results.md → results for evaluator
```

---

## Phase 3: LLM-Guided Research (Weeks 4-5)
### ⚠️ GATE: REQUIRES PEDER'S EXPLICIT APPROVAL BEFORE STARTING

*With the harness in place, we can now add LLM reasoning to the Planner and Evaluator agents.*

### 3.0 LLM-as-Planner (batch reasoning, not tight loop)

**The article's lesson applied:** "Taking inspiration from GANs, I designed a multi-agent structure with a generator and evaluator agent."

Our adaptation for budget constraints:
- Run 15-20 experiments (free CPU) → feed experiment log to LLM once
- LLM acts as Research Planner: analyzes patterns, proposes next research direction
- Max 3-5 LLM reasoning calls per optimization cycle (~$2-5/cycle)
- LLM can propose code changes but Ford reviews before running

**Planner prompt structure:**
```
You are a quantitative researcher reviewing experiment results.

## Current best: Sharpe {X}, DSR {Y}
## Last {N} experiments: [TSV data]
## Evaluator's last critique: [markdown]

Your job:
1. Identify patterns in kept vs discarded experiments
2. Hypothesize what structural change would improve Sharpe
3. Propose specific, testable changes (params, features, or strategy logic)
4. Estimate expected impact and risk of overfitting
5. Define success criteria for the evaluator

Do NOT propose changes that:
- Add complexity without justification (t-stat < 3.0)
- Ignore the evaluator's previous critique
- Violate the budget constraint (no external API calls)
```

### 3.1 LLM-as-Evaluator (skeptical, calibrated)

**The article's lesson:** "Out of the box, Claude is a poor QA agent. I watched it identify legitimate issues, then talk itself into deciding they weren't a big deal."

Our calibration approach:
- Explicitly prompt for skepticism: "Your job is to find problems, not praise."
- Provide few-shot examples of overfitting patterns (high Sharpe but fragile)
- Grade against concrete criteria (not "is this good?" but "does this pass each specific test?")
- Hard thresholds that can't be argued away

**Evaluator prompt structure:**
```
You are a skeptical quant reviewer. Your reputation depends on catching overfitting.

## Experiment results: [data]
## Changes made: [code diffs]
## Previous best: Sharpe {X}
## New result: Sharpe {Y}

Grade each criterion 1-10 with specific evidence:
1. Statistical Validity: [DSR, seed stability, window concentration]
2. Robustness: [sub-period performance, regime sensitivity]
3. Simplicity: [parameter count, marginal contribution per param]
4. Reality Gap: [transaction costs, execution assumptions]

A score below 6 on ANY criterion is a FAIL.
If the improvement is real, say so. If it smells like overfitting, say that.
Do not be generous. The cost of approving a bad strategy is losing real money.
```

### 3.2 Regime Detection (zero LLM cost)
- [ ] HMM-based regime detector (2-3 states from returns + volatility)
- [ ] Per-regime parameter optimization
- [ ] Rolling re-optimization via cron

### 3.3 Agent Skill Optimization
- [ ] SkillOpt on highest-impact agents (Synthesis, Moderator, Risk Judge)
- [ ] MDA → Agent bridge (feature importance drives agent targeting)

---

## Phase 4: Production Readiness (Week 6 — Late April)
*Get ready for real money. The evaluator agent becomes the live QA system.*

### 4.1 Slack Signal Delivery
- [ ] Daily morning digest: top opportunities + rebalance suggestions
- [ ] Alert format: ticker, signal, confidence, reasons, risk level, position size
- [ ] The evaluator validates signals before they're sent (catch bad recommendations)

### 4.2 Paper Trading (evaluator as live QA)
- [ ] Run paper trading 2+ weeks
- [ ] Evaluator compares paper results vs backtest expectations daily
- [ ] Track signal accuracy per enrichment tool → drop tools that don't add alpha
- [ ] If paper Sharpe < 0.7 × backtest Sharpe → STOP and investigate

### 4.3 Risk Management
- [ ] Max portfolio size, max single position, max daily loss — all defined
- [ ] Stop-loss monitoring with automatic position reduction
- [ ] Event calendar integration (earnings, FOMC → reduce exposure)
- [ ] Kill switch: if drawdown > 15%, system goes to cash automatically

### 4.4 Go-Live Checklist
- [ ] All evaluator criteria passing (statistical validity, robustness, simplicity, reality gap)
- [ ] DSR ≥ 0.95 on out-of-sample data
- [ ] Paper trading matches backtest within 20% tolerance
- [ ] Slack signals tested and reliable
- [ ] Peder's manual review process defined and working
- [ ] Risk limits hardcoded (not configurable without code change)

---

## Harness Simplification Principles

From the article: "Every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing."

### What to strip as models improve:
- If Opus 4.6 can maintain coherence for 2+ hours → drop sprint decomposition ✅ (already simplified)
- If evaluator catches <5% of issues → reduce evaluation frequency
- If planner's suggestions match what optimizer finds naturally → simplify planner
- **Always test:** remove one component, measure impact, keep only what's load-bearing

### What remains load-bearing regardless of model quality:
- **Separation of generation and evaluation** — self-evaluation stays unreliable
- **Structured handoff files** — context resets need precise state
- **Grading criteria** — concrete terms beat vague quality judgments
- **DSR gate** — statistical validation can't be model-intuited away

---

## Budget Tracking

| Item | Monthly Cost | Notes |
|------|-------------|-------|
| BigQuery | ~$10-25 | Storage + queries |
| GitHub Models (Copilot Pro) | $0 | gpt-4.1 included |
| Claude Max (OpenClaw/Ford) | Already paid | Planner + Evaluator via Ford |
| FRED / Alpha Vantage | $0 | Free tiers |
| LLM-guided research (Phase 3) | $2-5/cycle | ⚠️ Needs approval |
| **Total** | **~$10-30/month** | |

---

## Success Criteria

1. **Backtest Sharpe > 1.0** with DSR ≥ 0.95 ✅ ACHIEVED (1.1705)
2. **Evaluator passes** all four criteria (statistical validity, robustness, simplicity, reality gap)
3. **Paper trading** matches backtest within 20% tolerance over 2+ weeks
4. **Beat SPY** over backtest period after realistic transaction costs
5. **No known overfitting** — stable features, robust to seeds, holds across regimes

---

## Progress Log

| Date | Milestone | Sharpe | Notes |
|------|-----------|--------|-------|
| 2026-03-25 | Phase 0 complete | 0.905 | Audit, validation, bug fixes |
| 2026-03-26 | First optimizer run | 0.9848 | 12 experiments, baseline established |
| 2026-03-27 | Target achieved | 1.0391 | min_samples_leaf 20→18 |
| 2026-03-28 | Phase 1 complete | 1.1705 | 9 improvements, +19% over baseline |
| 2026-03-28 | v2 Plan (harness) | — | Three-agent architecture adopted |

---

*This plan follows the Anthropic harness design pattern: Planner → Generator → Evaluator.*
*"The space of interesting harness combinations doesn't shrink as models improve. Instead, it moves."*
*Last updated: 2026-03-28 by Ford*
