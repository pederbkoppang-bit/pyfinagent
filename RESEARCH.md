# RESEARCH.md — Evidence Base & Literature Review

> Every design decision in pyfinAgent should trace back to a paper, a documented industry practice, or empirical evidence. This file is our research log.

---

## Research Protocol

Before implementing any major feature or making architectural decisions:
1. **Search** for relevant academic papers, industry documentation, and peer implementations
2. **Document** findings in the relevant section below with citation + key insight
3. **Reference** the research in PLAN.md when justifying a design choice
4. **Update** this file when new evidence emerges that changes our approach

**Sources (priority order):**
1. Academic papers (arXiv, SSRN, NBER, Journal of Finance, Journal of Financial Economics)
2. Industry documentation (Anthropic, OpenAI, Google DeepMind, Two Sigma, AQR, Man Group)
3. Practitioner blogs (Marcos López de Prado, Ernie Chan, Karpathy, AQR Cliff Asness)
4. Open-source implementations (FinRL, TradingAgents, autoresearch)
5. Regulatory guidance (SEC, FINRA, MiFID II)

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
| López de Prado, "Advances in Financial Machine Learning" (2018) | Deflated Sharpe Ratio, walk-forward validation, sample weights, embargo periods, fractional differentiation | `analytics.py`, `walk_forward.py`, `historical_data.py` |
| Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns" | t-stat ≥ 3.0 for new factors to control multiple testing bias. Most published anomalies are false. | Evaluator criteria: t-stat threshold |
| Lo (2002), "The Statistics of Sharpe Ratios" | Sharpe ratio is autocorrelation-sensitive. Adjusted Sharpe accounts for serial correlation in returns. | Phase 2.7: Lo-adjusted Sharpe comparison |
| Bailey & López de Prado (2014), "The Deflated Sharpe Ratio" | DSR adjusts for multiple testing, non-normal returns, and short track records. | `compute_deflated_sharpe()` in analytics.py |

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
