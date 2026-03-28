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

## Phase 2: Three-Agent Harness (Weeks 3-4) — ✅ CORE IMPLEMENTED
*Automated harness loop operational. `run_harness.py` runs autonomous Planner → Generator → Evaluator cycles.*

### 2.0 Harness Architecture ✅ OPERATIONAL

**Implementation:** `run_harness.py` (718 lines) — autonomous cycle orchestrator.

```
Usage: python run_harness.py [--cycles N] [--iterations-per-cycle N] [--dry-run]
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH PLANNER ✅                            │
│  Reads: experiment_log.tsv, evaluator_critique.md               │
│  Writes: research_plan.md, contract.md (sprint contract)        │
│  Heuristic: plateau detection, param saturation, weak periods   │
│  Runs: Once per cycle (automatic)                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ research_plan.md + contract.md
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STRATEGY GENERATOR ✅                           │
│  Wraps: QuantStrategyOptimizer.run_loop() (Karpathy autoresearch)│
│  Reads: optimizer_best.json, research_plan.md                    │
│  Writes: experiment_results.md, updated optimizer_best.json      │
│  Runs: N iterations per cycle (configurable, default 10)         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ experiment_results.md
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUANT EVALUATOR ✅                             │
│  Runs: 5 INDEPENDENT backtests (separate engine instances)       │
│  Tests: 3 sub-periods + 2× costs + full-period DSR              │
│  Grades: 4 criteria independently (anti-leniency protocol)       │
│  Writes: evaluator_critique.md (PASS/FAIL/CONDITIONAL + scores)  │
│  Decision: PASS → save | FAIL → auto-revert | CONDITIONAL → warn │
└──────────────────────────┬──────────────────────────────────────┘
                           │ evaluator_critique.md
                           ▼
                    (feeds back to Planner — next cycle)
```

**Cycle time:** ~25-40 minutes. Three cycles = ~2 hours autonomous work.

### 2.1 Handoff Artifacts ✅ IMPLEMENTED

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

### 2.2 Research Planner Agent ✅ IMPLEMENTED (Heuristic)

**Role:** The "quant researcher" who reads all experiment data and decides what to investigate next. This replaces random parameter perturbation with directed research.

**Implementation:** Heuristic rule-based planner in `run_harness.py`. Reads experiment TSV, detects plateaus (10+ discards), saturated params (5+ consecutive discards on same param), diminishing returns (<0.02 delta), and evaluator-flagged weak sub-periods. Writes sprint contracts. LLM planner upgrade deferred to Phase 3 (requires budget approval).

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

### 2.3 Strategy Generator Agent ✅ INTEGRATED

**Role:** The "quant developer" who implements changes and runs experiments. This is our current optimizer, enhanced with the ability to make structural changes (not just param sweeps).

**Implementation:** `run_generator()` in `run_harness.py` wraps `QuantStrategyOptimizer.run_loop()`. Creates fresh engine each cycle. Exports results to `handoff/experiment_results.md`. The optimizer itself is Karpathy autoresearch-style (zero LLM cost, param perturbation + walk-forward backtest).

**Generator capabilities (expanding):**
- [x] Single-parameter perturbation (current, Karpathy autoresearch-style)
- [x] Warm-start from previous best (skip baseline if optimizer_best.json exists)
- [x] Feature caching (reuse features when only ML hyperparams change)
- [x] Early stopping (abort experiments below 85% of best Sharpe)
- [ ] Multi-parameter coordinated proposals (param groups that interact) — Phase 2.7
- [ ] Feature addition/removal experiments (add a feature, measure impact) — Phase 2.7
- [ ] Strategy switching experiments (try blend vs triple_barrier vs quality_momentum) — Phase 2.7
- [ ] Ablation studies (remove one improvement at a time, measure drop) — Phase 2.7

### 2.4 Quant Evaluator Agent ✅ IMPLEMENTED

**Role:** The "skeptical reviewer" who independently validates results. This is the key missing piece from our current system. The article's core insight: self-evaluation is unreliable; external evaluation drives real quality.

**Implementation:** `run_evaluator()` in `run_harness.py`. Runs 5 independent backtests with SEPARATE engine instances. Grades 4 criteria independently with anti-leniency protocol (score before verdict, never upgrade after). Writes `handoff/evaluator_critique.md` with pass/fail verdict and detailed scoring.

**Evaluator checks (grading criteria):**

#### Statistical Validity (weight: 40%)
- [x] DSR ≥ 0.95 on the kept result — **automated in evaluator**
- [x] Full-period Sharpe check — **automated in evaluator**
- [ ] Sharpe stable across 5 random seeds (std < 0.1) — Phase 2.7
- [ ] No single window drives >30% of total return (concentration check) — Phase 2.7
- [ ] Ljung-Box test on returns — Phase 2.7
- [ ] Compare raw vs Lo (2002) adjusted Sharpe — Phase 2.7

#### Robustness (weight: 30%)
- [x] Run backtest on 3 sub-periods: 2018-2020, 2020-2022, 2023-2025 — **automated in evaluator**
- [x] Sharpe positive in all 3 sub-periods (not just aggregate) — **automated in evaluator**
- [ ] Feature importance top-5 stable across sub-periods — Phase 2.7
- [x] Performance under 2x transaction costs (reality buffer) — **automated in evaluator**

#### Simplicity (weight: 15%)
- [x] Count active parameters vs baseline (penalize complexity) — **automated in evaluator**
- [ ] Each new parameter must contribute ≥ +0.05 Sharpe (ablation test) — Phase 2.7
- [ ] Reject if improvement < t-stat 3.0 (Harvey, Liu & Zhu threshold) — Phase 2.7

#### Reality Gap (weight: 15%)
- [x] Transaction costs ≥ 10 bps round-trip — **automated in evaluator (tests at 20 bps)**
- [ ] No execution at exact close price (add 5 bps slippage) — Phase 2.7
- [ ] Max position < 10% of portfolio — Phase 2.7
- [ ] Universe includes at least some mid-cap stocks (not just mega-cap) — Phase 2.7

### 2.5 Sprint Contracts (Experiment Contracts) ✅ IMPLEMENTED

Before each optimization cycle, the planner writes `handoff/contract.md` with hypothesis, current baseline, success criteria, and excluded params. This prevents the generator from optimizing the wrong thing.

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

### 2.6 Context Management Strategy ✅ IMPLEMENTED

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

### 2.7 Harness Hardening & Advanced Evaluator (Next)
*Enhance the automated harness with deeper statistical tests and generator capabilities.*

- [ ] **Seed stability test:** Run best params with 5 random seeds, check Sharpe std < 0.1
- [ ] **Concentration check:** No single window drives >30% of total return
- [ ] **Ljung-Box autocorrelation test** on returns
- [ ] **Lo (2002) adjusted Sharpe** comparison
- [ ] **Feature importance stability** across sub-periods
- [ ] **Ablation studies:** Remove one Phase 1 improvement at a time, measure drop
- [ ] **Multi-param proposals:** Coordinate param groups (e.g., tp_pct + sl_pct together)
- [ ] **Strategy switching:** Generator can propose trying different strategies
- [ ] **Slippage modeling:** 5 bps execution slippage on top of transaction costs
- [ ] **Position concentration limits:** Max position < 10% in evaluator checks

### 2.8 Karpathy Autoresearch Integration
*The Generator (QuantStrategyOptimizer) already follows Karpathy's autoresearch pattern for zero-cost param optimization. The harness adds the missing evaluation and planning layers. If the harness proves beneficial on pyfinAgent, apply the same three-agent pattern to the upstream [autoresearch](https://github.com/karpathy/autoresearch) optimizer — wrapping its research loop with independent evaluation and heuristic planning.*

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
| 2026-03-28 | Sub-period validation | 1.01 | ALL pass: A=0.89, B=0.92, C=1.88, 2×costs=0.91 |
| 2026-03-28 | Hang bug fixed | — | Macro preload + BQ timeouts, 5× speedup |
| 2026-03-28 | **Phase 2 core** | 1.1705 | `run_harness.py` operational (Planner→Generator→Evaluator) |

---

*This plan follows the Anthropic harness design pattern: Planner → Generator → Evaluator.*
*"The space of interesting harness combinations doesn't shrink as models improve. Instead, it moves."*
*Last updated: 2026-03-28 17:15 by Ford — Phase 2 core implemented*
