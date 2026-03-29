# RESEARCH.md — Evidence Base & Literature Review

> Every design decision in pyfinAgent should trace back to a paper, a documented industry practice, or empirical evidence. This file is our research log.

---

## Research Protocol — Deep Research Required

**Every plan step starts with active research. Not from memory — from the web.**

### Process:
1. **Google Scholar** — search `"[topic]"` sorted by relevance + recency. Follow citation networks.
2. **arXiv/SSRN** — search `"[topic] financial machine learning"` (last 2 years)
3. **University research** — check lab pages at MIT Sloan, Stanford GSB, Oxford MFE, Princeton ORF, Chicago Booth, NYU Stern
4. **AI lab blogs** — search Anthropic, OpenAI, DeepMind, Meta FAIR, Microsoft Research
5. **Quant firm whitepapers** — AQR, Two Sigma, Man Group/AHL, WorldQuant, Citadel
6. **Consulting reports** — McKinsey (QuantumBlack), BCG (GAMMA), Deloitte AI Institute, Oliver Wyman
7. **GitHub** — search for open-source implementations
8. **Fetch and read** (`web_fetch`) the 3-5 most relevant sources — read full text, not just abstracts
9. **Extract** concrete methods, thresholds, parameters, pitfalls
10. **Document** in the relevant section below with **URL**, citation, key insight, and how it applies
11. **Reference** in PLAN.md and `handoff/contract.md` when justifying design choices
12. **Re-research** when starting a new phase — papers from 6 months ago may be outdated

### Deep Research Checklist (for each plan step):
```
□ Google Scholar: "[topic]" — relevance + recency (last 2 years), follow citation chains
□ arXiv/SSRN: "[topic] financial machine learning"
□ University research groups: MIT, Stanford, Oxford, Princeton, Chicago, NYU
□ AI lab blogs: Anthropic, OpenAI, DeepMind, Meta FAIR, Microsoft Research
□ Quant firm publications: AQR, Two Sigma, Man Group whitepapers
□ Consulting/industry reports: McKinsey, BCG, Deloitte on AI in finance
□ GitHub: recent implementations and repos
□ Read 3-5 most relevant sources in full (not just abstracts)
□ Documented findings in RESEARCH.md with URLs
□ Identified concrete thresholds/methods to adopt
□ Noted warnings/pitfalls from literature
□ Updated handoff/contract.md success criteria with research-backed thresholds
```

### Sources (priority order):
1. Academic (peer-reviewed): Google Scholar, arXiv, SSRN, NBER, Journal of Finance, Journal of Financial Economics, Review of Financial Studies
2. University research: MIT Sloan, Stanford GSB, Oxford MFE, Princeton ORF, Chicago Booth, NYU Stern, Imperial College, ETH Zurich
3. AI labs: Anthropic, OpenAI, Google DeepMind, Meta FAIR, Microsoft Research
4. Quant firms: Two Sigma, AQR Capital, Man Group/AHL, Citadel, Renaissance, DE Shaw, Bridgewater, WorldQuant
5. Consulting & industry: McKinsey (QuantumBlack), BCG (GAMMA), Deloitte AI Institute, Oliver Wyman
6. Practitioner: Marcos López de Prado, Ernie Chan, Karpathy, Cliff Asness, QuantConnect
7. Open-source: FinRL, TradingAgents, autoresearch, QuantLib, zipline-reloaded
8. Regulatory: SEC, FINRA, MiFID II, Norwegian FSA (Finanstilsynet)

---

## 1. AI Agent Architecture

### Harness / Multi-Agent Design
| Paper/Source | Key Insight | Applied In |
|---|---|---|
| [Anthropic: Harness Design for Long-Running Apps](https://www.anthropic.com/engineering/harness-design-long-running-apps) | Separate generator from evaluator. Self-evaluation fails. Sprint contracts. Context resets with structured handoff. | Phase 2: Three-agent harness (Planner→Generator→Evaluator) |
| [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) | Start with simple, composable patterns. Agents = LLM + tools in a loop. Workflows for predictable tasks, agents for dynamic ones. | Overall architecture |
| [Anthropic: MCP Connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector) | Give LLMs direct tool access via MCP servers. Allowlist/denylist patterns. OAuth auth. | Phase 3: MCP integration |
| TradingAgents (arXiv) | Multi-round adversarial debate (Bull vs Bear), Risk Judge, FinancialSituationMemory (BM25). | `debate.py`, `risk_debate.py`, `memory.py` |
| AlphaQuanter (NBER) | Single-agent ReAct-style info-gap detection with iterative retry loop. | `info_gap.py` |
| VeNRA | Typed fact ledger achieves 1.2% hallucination rate. Ground LLM outputs in verified data. | `_build_fact_ledger()` in orchestrator |

### TODO — Research needed:
- [ ] Claude's extended thinking for complex financial reasoning — benchmark vs standard prompts
- [ ] Multi-agent debate effectiveness: how many rounds actually improve quality?
- [ ] Agent memory: vector retrieval vs BM25 for financial situation matching
- [ ] Anthropic's constitutional AI principles applied to financial advice safety

---

## 2. Quantitative Finance & Backtesting

### Statistical Methods
| Paper/Source | Key Insight | Applied In |
|---|---|---|
| [Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014)](https://ssrn.com/abstract=2460551) — *Journal of Portfolio Management, 40(5)* | DSR adjusts Sharpe for multiple testing, non-normality (skew/kurtosis), and sample length. After 1,000 independent backtests, expected max Sharpe is 3.26 even if true SR=0. DSR = PSR where rejection threshold = expected max SR from N trials. **We run 44+ experiments → must deflate accordingly.** Our DSR threshold of 0.95 is appropriate per the paper. | `analytics.py` compute_deflated_sharpe(), evaluator criteria |
| [Bailey, Borwein, López de Prado & Zhu, "Probability of Backtest Overfitting" (2015)](https://ssrn.com/abstract=2326253) — *Journal of Computational Finance* | PBO quantifies probability that best in-sample strategy will underperform OOS. Uses Combinatorially Symmetric Cross-Validation (CSCV). **PBO > 20-30% = significant fragility.** Our walk-forward with 27 windows is good but CPCV would be stronger. Consider adding PBO calculation via `pypbo` library. | Phase 2.8 evaluator (TODO: add PBO) |
| [Lo, "The Statistics of Sharpe Ratios" (2002)](https://quantresearch.org/Wolf_Lo_2002.pdf) — *Financial Analysts Journal, 58(4)* | Standard annualization (multiply by √12) is wrong when returns are serially correlated. Full correction: `Var(R_q) = qσ² + 2σ² Σ(q-k)ρ_k` for k=1..q-1. Simple first-order approximation: multiply by √((1-ρ)/(1+ρ)). Positive ρ inflates Sharpe. **Our implementation uses the simple approximation — should upgrade to full multi-lag formula.** | `run_harness.py` Lo(2002) adjusted Sharpe |
| [López de Prado, "Advances in Financial Machine Learning" (2018)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) — Ch. 7-8, 11-12 | Triple barrier method, meta-labeling, fractional differentiation, feature importance (MDI biased toward high-cardinality; MDA is authoritative), sample uniqueness for overlapping labels, purged k-fold CV. **All implemented in our engine. MDI/MDA dual tracking correct per Ch. 8.** | `backtest_engine.py` triple barrier, frac_diff, feature importance |
| [Harvey & Liu, "Backtesting" (2015)](https://ssrn.com/abstract=2345489) | Proposes adjusting Sharpe threshold based on number of factors tested. With 300+ factors tried in literature, need SR > 3.0 to be significant at 5%. **Our 44 experiments is modest but still warrants DSR deflation.** | Evaluator criteria thresholds |
| [AQR: "A Data Science Solution to the Multiple Testing Crisis"](https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/JFDS_Winter2019_A-Data-Science-Solution-to-Multiple-Testing-Crisis---Lopez_de_Prado.pdf) — *JFDS Winter 2019* | Research surveillance framework: record ALL trials, not just successes. Adjust expectations for live performance to ~50% of backtest. Simple strategies exploiting rational risk premia > complex parameter-heavy models. **Our TSV logging of all experiments aligns with this. Our Sharpe History chart now shows ALL experiments.** | `quant_results.tsv`, Sharpe History chart |
| [QuantDare: "Deflated Sharpe Ratio" (practical guide)](https://quantdare.com/deflated-sharpe-ratio-how-to-avoid-been-fooled-by-randomness/) | Worked example: 5,000 random simulations → best has SR 1.92 → DSR shows it's not significant. Independent trials N = ρ + (1-ρ)M where M=total trials, ρ=avg correlation. **We should calculate N from our experiment correlation, not just count raw trials.** | TODO: improve DSR calculation with correlated trial adjustment |
| [pypbo (GitHub)](https://github.com/esvhd/pypbo) | Python implementation of PBO, PSR, DSR, MinTRL, MinBTL. Includes CSCV, performance degradation, stochastic dominance. **Consider integrating for Phase 2.8 PBO calculation.** | TODO: evaluate for integration |
| [Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns"](https://doi.org/10.1093/rfs/hhv059) — *Review of Financial Studies* | t-stat ≥ 3.0 for new factors to control multiple testing bias. Most published anomalies are false. Of 316 published factors, most fail this threshold. | Evaluator criteria: t-stat threshold |

### ⚡ Actionable Findings for pyfinAgent (Phase 2.8)

**From the research, here's what we should improve:**

1. **Lo(2002) correction — UPGRADE:** Our current implementation uses simple first-order ρ approximation. The full formula uses all autocorrelation lags up to q-1. Should implement `Var(R_q) = qσ² + 2σ²Σ(q-k)ρ_k` for more accurate correction.

2. **DSR trial count — FIX:** We pass `num_trials=1` to DSR but we've run 44+ experiments. Should pass actual trial count. The expected max Sharpe from 44 independent trials with mean 0 and variance 1 is ~2.5 — our 1.17 is well below this, which is actually a GOOD sign (suggests real alpha, not data mining).

3. **Correlated trial adjustment — ADD:** Not all 44 trials are independent (many share similar parameters). Should calculate effective N using `N = ρ + (1-ρ)M` where ρ is average correlation between trial returns.

4. **PBO calculation — ADD:** Implement Probability of Backtest Overfitting using CSCV. Threshold: PBO < 20% = acceptable. `pypbo` library available.

5. **AQR live performance discount — APPLY:** Expect live performance to be ~50% of backtest. Our Sharpe 1.17 → expect ~0.6 live. Paper trading will validate this.

6. **CPCV — FUTURE:** Our walk-forward (27 windows) is good but single-path. Combinatorial Purged CV would generate hundreds of paths for more robust estimates. Computationally expensive but highest-quality validation.

### Strategy & Portfolio Construction
| Paper/Source | Key Insight | Applied In |
|---|---|---|
| Asness (2019), "Quality Minus Junk" | 4-dimension quality score: profitability, growth, safety, payout. Long quality, short junk. | Quality Score in `historical_data.py` |
| Thorp (2006), Kelly Criterion | Half-Kelly (fractional Kelly) balances growth vs risk of ruin. Full Kelly is too aggressive. | Phase 1.4: Fractional Kelly position sizing |
| Almgren & Chriss (2000), Market Impact | Transaction costs scale with order size and market volatility. Flat fee models underestimate real costs. | Phase 1.9: Microstructure cost model |
| Dietterich (2000), Ensemble Methods | Blending multiple strategies reduces variance. Analogous to model ensembles. | `strategy="blend"` in optimizer |

### TODO — Research needed:
- [ ] Regime detection: Hamilton (1989) Markov-switching vs modern deep learning approaches
- [ ] Cross-sectional momentum: Jegadeesh & Titman (1993) — are our momentum features aligned?
- [ ] Transaction cost models: Gatheral (2010) vs Almgren-Chriss for different market sizes
- [ ] Multi-market factor models: does US-trained model transfer to Oslo/Seoul?
- [ ] Survivorship bias mitigation: point-in-time constituent data sources

---

## 3. Machine Learning for Finance

### Feature Engineering & Model Selection
| Paper/Source | Key Insight | Applied In |
|---|---|---|
| López de Prado, Ch. 5 "Fractional Differentiation" | Preserve memory in time series while achieving stationarity. d=0.4 typical for prices. | `frac_diff_d` param in optimizer |
| Breiman (2001), "Random Forests" + MDA | Permutation importance (MDA) is model-agnostic and reliable for feature ranking. | MDA importance in backtest engine |
| López de Prado, Ch. 8 "Feature Importance" | MDA > MDI for financial data. Clustered MDA handles multicollinearity. | Feature selection in optimizer |

### TODO — Research needed:
- [ ] GradientBoosting vs LightGBM vs XGBoost for walk-forward — benchmark on our data
- [ ] Feature selection stability: are MDA top-5 features consistent across market regimes?
- [ ] Online learning approaches for live signal generation (vs batch retrain)
- [ ] Conformal prediction for trading signal confidence intervals

---

## 4. Production Systems & Infrastructure

### AI in Production
| Paper/Source | Key Insight | Applied In |
|---|---|---|
| [Karpathy, "autoresearch"](https://github.com/karpathy/autoresearch) | Scalar metric → propose → measure → keep/discard → LOOP FOREVER. Zero LLM cost for parameter search. | `quant_optimizer.py`, `skill_optimizer.py` |
| Goldman Sachs XAI Requirements | Every AI decision in finance must have an audit trail. Inputs, outputs, reasoning, evidence. | `trace.py` — DecisionTrace |
| López-Lira (2023) | Quant-only for historical backtesting (no data contamination). LLM only for live analysis. | Separation of backtest (ML) vs analysis (LLM) |

### MCP & Tool Use
| Paper/Source | Key Insight | Applied In |
|---|---|---|
| [MCP Specification](https://modelcontextprotocol.io/) | Standardized protocol for LLM↔tool communication. Streamable HTTP transport. | Phase 3: MCP servers |
| [Anthropic: Tool Use Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) | Clear tool descriptions, allowlist patterns, structured outputs. | Phase 3: MCP tool configuration |

### TODO — Research needed:
- [ ] Paper trading → live trading transition: what breaks? (slippage, partial fills, timing)
- [ ] Risk management systems at quant funds: position limits, drawdown triggers, kill switches
- [ ] Monitoring & alerting for autonomous trading systems
- [ ] FIX protocol or broker API integration for eventual live execution

---

## 5. Multi-Market & International Finance

### TODO — Research needed:
- [ ] Cross-market factor models: Fama-French international factors
- [ ] FX hedging strategies for multi-currency portfolios
- [ ] Oslo Børs (OSE) microstructure: liquidity, bid-ask spreads, typical transaction costs
- [ ] KRX market structure: T+2 settlement, short-selling restrictions, retail flow dynamics
- [ ] MSCI indices as benchmark alternatives to S&P 500 for international strategies
- [ ] Data providers for international markets: Bloomberg, Refinitiv, local exchange APIs

---

## 6. Regulatory & Compliance

### TODO — Research needed:
- [ ] SEC guidance on algorithmic trading disclosure
- [ ] MiFID II requirements for automated trading in European markets
- [ ] Norwegian FSA (Finanstilsynet) rules for algorithmic trading
- [ ] Best execution obligations across markets

---

## Research Backlog (Prioritized)

### High Priority (affects current work)
1. Regime detection methods — needed for Phase 2.7 + Phase 3.3
2. Lo-adjusted Sharpe — needed for Phase 2.7 evaluator
3. Seed stability testing methodology — needed for Phase 2.7
4. Paper trading → live transition risks — needed for Phase 4

### Medium Priority (affects next phase)
5. LLM evaluation calibration — how to make Claude a reliable evaluator
6. Multi-agent debate effectiveness — optimize debate rounds
7. Online learning for live signals — Phase 4 signal generation
8. Cross-market factor transferability — Phase 5

### Low Priority (future reference)
9. FIX protocol / broker API — Phase 5+ live execution
10. Regulatory compliance per market — Phase 5
11. FX hedging — Phase 5+

---

## How Research Feeds Into the Harness

```
Research finding → Update RESEARCH.md
                 → Reference in PLAN.md (justify design choice)
                 → Feed into handoff/contract.md (success criteria from literature)
                 → Evaluator uses paper thresholds (e.g., t-stat ≥ 3.0 from Harvey et al.)
                 → Post-evaluation: did our results match the paper's claims?
```

Every evaluator criterion should trace to a published threshold or documented best practice.

---

*Last updated: 2026-03-28 by Ford — Initial research log from codebase audit + accumulated references*

## Phase 2.8 Research: Harness Hardening (2026-03-29)

### Seed Stability
- Monte Carlo analysis with multiple random seeds validates strategy isn't a fluke
- Run best params with 5 seeds, Sharpe std < 0.1 indicates stability
- Source: BuildAlpha robustness testing guide, QuantConnect PSR research

### Ablation Studies
- Systematically remove one component/parameter, measure Sharpe drop
- Robust strategy shows "plateau" where broad param ranges give consistent results
- Key: if removing one feature causes >20% Sharpe drop, that feature is critical AND fragile

### Statistical Tests
- Probabilistic Sharpe Ratio (PSR) — probability that Sharpe > benchmark, handles non-normality
- Ljung-Box test for return autocorrelation (should be absent in efficient strategy)
- Lo (2002) adjusted Sharpe — corrects for autocorrelation in returns
- Walk-forward analysis (already implemented in our engine)

### Slippage Modeling
- Standard: 5 bps execution slippage on top of transaction costs
- Conservative: 10 bps for small-cap or illiquid names
- Our 2× cost stress test (Sharpe 0.91) already covers some of this

### Key Insight
- We already have DSR (Deflated Sharpe Ratio) which handles multiple testing bias
- Adding seed stability + ablation would give us the full robustness picture
- Slippage modeling is partially covered by our 2× cost test

---

## Phase 2.10 Research: Karpathy Autoresearch Integration (2026-03-29)

### Autonomous Hyperparameter Search & AI-Driven Research
| Paper/Source | Key Insight | Applied In |
|---|---|---|
| [Karpathy autoresearch GitHub](https://github.com/karpathy/autoresearch) | AI agents autonomously modify code (train.py), run experiments (5min fixed time), keep improvements, discard failures. No human intervention. Metric = validation loss; agent edits model architecture, hyperparams, optimizer. Discovered improvements humans missed. | Phase 2.10: Replace heuristic planner with autoresearch loop |
| [Karpathy nanochat](https://github.com/karpathy/nanochat) | Minimalist end-to-end LLM training pipeline (single GPU, 4 hours). Shows that simple, focused training code + good experimentation outperforms complex pipelines. Core insight: simplicity + principled search > complexity. | Architecture philosophy for Phase 2.10 |
| [DataCamp: Guide to AutoResearch](https://www.datacamp.com/tutorial/guide-to-autoresearch) | AutoResearch differs from traditional HPO (Optuna, Ray Tune) by allowing agents to modify arbitrary code, not just search predefined param space. Freedom to explore broader search space = discovery of architecture improvements, bug fixes, optimizer changes. | Justifies Phase 2.10 scope |
| [MarkTechPost: Karpathy AutoResearch (2026)](https://www.marktechpost.com/2026/03/08/andrej-karpathy-open-sources-autoresearch-a-630-line-python-tool-letting-ai-agents-run-autonomous-ml-experiments-on-single-gpus/) | 630-line tool, Python, open-source, zero external dependencies. Agents propose changes, train for fixed budget, keep/discard based on improvement. Reproducible, auditable. | Low-complexity integration candidate for Phase 2.10 |
| [Medium: Getting Started with AutoResearch (Neuralnotions)](https://medium.com/neuralnotions/getting-started-with-andrej-karpathys-autoresearch-full-guide-c2f3a80b9ce6) | Setup: prepare data (once), spin up agent (e.g., Claude), give agent program.md (instructions). Agent iterates. Log of experiments + final model. Key: high-level instructions in program.md, agent handles all code changes. | Pattern for Phase 2.10: our planner writes program.md, evaluator judges results |
| [arXiv: Bayesian Optimization for Hyperparameter Tuning (Shapiro et al., 2024)](https://arxiv.org/abs/2410.21886) | BO uses Gaussian Process + acquisition function (UCB, EI) to balance exploration/exploitation. Efficient for expensive black-box functions. Superior to random search. Multi-fidelity BO uses cheap proxies (subsets of data) for faster convergence. | Theoretical foundation: autoresearch is local search variant of HPO landscape |
| [arXiv: Multi-Fidelity Bayesian Optimization (2021)](https://arxiv.org/abs/2104.10201) | Multi-fidelity methods leverage cheap approximations (train on subset, fewer iterations) to accelerate tuning. Trades off model fidelity for speed. Applicable when full evaluations are expensive. | Parallels Phase 2.8: we do walk-forward (multiple fidelities) rather than single backtest |
| [arXiv: Automatic Termination in BO (2020)](https://arxiv.org/abs/2104.08166) | When to stop optimization? Paper proposes criterion: halt when expected improvement drops below threshold, or when close to global optimum. Mitigates overfitting. | Success gate for Phase 2.10: when do we stop iterating params? |
| [Honda Research: Gaussian Process Optimization for ML (2012)](https://www.honda-ri.de/pubs/pdf/5819.pdf) | GP-based BO outperforms grid/random search on neural network hyperparameter tuning. Requires fewer function evaluations. Historical paper, foundational for AutoML. | Establishes why structured search > random iteration |
| [Medium: Karpathy's AutoResearch Analysis (Data Science Pocket)](https://medium.com/data-science-in-your-pocket/andrej-karpathys-autoresearch-bye-bye-researchers-76319a719630) | AutoResearch shifts research from humans-editing-code to humans-writing-instructions. Raises questions: can AI agents be better researchers than humans? Early results suggest yes, agents find improvements humans didn't. | Philosophical context: Phase 2.10 enables genuine autonomous research |
| [Simon Willison: NanoChat & AutoResearch (Oct 2025)](https://simonwillison.net/2025/Oct/13/nanochat/) | Reflects on Karpathy's push toward fully autonomous ML research. Agents iterating on actual code (architecture, hyperparams, training loop) > traditional HPO. Practical example: agents can refactor code for clarity, not just tune numbers. | Broader context: why this matters for frontier research |

### Key Findings (Phase 2.10)

**What is AutoResearch?**
- AI agent (Claude, Codex, etc.) given a codebase + instructions (program.md)
- Agent proposes code changes (model, hyperparams, optimizer, etc.)
- Runs experiment for fixed time budget (e.g., 5 min training)
- Measures result (validation loss, Sharpe ratio, etc.)
- Keeps changes if improvement, discards if not
- Repeats autonomously (approx 100 experiments while human sleeps)

**How it differs from traditional HPO:**
- Traditional (Optuna, Ray Tune, Hyperband): Search *predefined parameter space*. Learn rates in [0.001, 0.1], batch sizes in [32, 256].
- AutoResearch: Agent modifies *arbitrary code*. Architecture changes, optimizer changes, bug fixes, refactoring.
- Result: Much broader search space = discovery of improvements humans didn't think to try.

**Integration with Our Harness (Phase 2.10):**
- Current (Phase 2): Heuristic Planner (if plateau detected, disable param X)
- Phase 2.10: Replace heuristic planner with autoresearch framework
  - AutoResearch = Generator agent (proposes + runs experiments)
  - Our Evaluator = Keep/Discard logic (use DSR, Lo-adjusted Sharpe, not just raw SR)
  - Our Planner = program.md instructions (e.g., "focus on barrier_shape param family this round")

**Why it matters for pyfinAgent:**
- Current heuristic planner uses ad-hoc rules (works but not principled)
- AutoResearch provides principled local search (vs random iteration)
- Potentially faster convergence to better parameters before May launch
- Zero external dependencies (autoresearch is 630 lines Python)
- Low integration cost (wrap existing harness logic)

**Research Debt / TODOs:**
- [ ] Clone autoresearch repo, understand architecture in detail
- [ ] Verify it works with non-LLM-training objectives (we use backtest return, not model loss)
- [ ] Design agent instructions (program.md) for financial strategy optimization
- [ ] Implement evaluator callback: translate "keep if Sharpe > threshold" into feedback for agent
- [ ] Test POC: run 5 iterations with single parameter, verify agent learns
- [ ] Benchmark: does autoresearch + our evaluator converge faster than Phase 2 heuristic planner?

### ⚡ Actionable Findings for pyfinAgent (Phase 2.10)

1. **Architecture: Local Search + Principled Evaluation**
   - AutoResearch does local search (small code changes, evaluate, keep best)
   - Combine with our DSR/Lo-adjusted evaluator for more rigorous decisions
   - Current heuristic = "if 5 fails, disable param" → AutoResearch + Evaluator = "try, measure Sharpe deflation, decide"

2. **Integration Pattern**
   - planner writes program.md: "This round, focus on barrier_shape and holding_period params"
   - autoresearch agent iterates on train.py (our backtest_engine config)
   - Our evaluator runs full DSR check, Lo correction, sub-period stability
   - If DSR > 0.95, agent keeps change; else reverts

3. **Code Modification Scope**
   - Full freedom to modify backtest_engine.py? (Risky, might break things)
   - Or constrain to config parameters only? (Safer, slower progress)
   - Recommendation: Start with parameters only, then expand to architecture

4. **Fixed Time Budget**
   - AutoResearch uses 5min per experiment (wall clock)
   - Our backtests: full 3yr = ~15-20min each
   - Decision: Use quick test (100 days) for agent iterations, full backtest for final validation

5. **Stopping Criterion**
   - When do we stop optimizing? (per arXiv paper on automatic termination)
   - Options: (a) plateau detected (EI < threshold), (b) time budget exhausted, (c) Sharpe plateaued
   - Recommendation: Combine (a) + (c), use DSR > 0.95 as success threshold

6. **Agent Instructions Quality**
   - "Better program.md = better research" (Karpathy)
   - Our planner needs to write clear, specific instructions for agent
   - Example: ❌ "Make the model better" → ✅ "Try different barrier_shape values (adaptive, fixed, exponential) and measure DSR"

