# Phase 3.0 Research: MCP Server Architecture for LLM Integration

**Status:** RESEARCH GATE CHECKPOINT (10+ sources collected, 3 read in full)

**Date:** 2026-03-29 09:45 UTC

**Goal:** Design three MCP servers (data, backtest, signals) to enable Claude to autonomously propose features, optimize parameters, and generate trading signals.

---

## Deep Research Findings

### ✅ Source 1: MCP Specification 2025-11-25 (Official)
- **URL:** https://modelcontextprotocol.io/specification/2025-11-25
- **Citation:** Anthropic (2025). "Model Context Protocol Specification." modelcontextprotocol.io
- **Key concepts:**
  - MCP uses JSON-RPC 2.0 for stateful communication between Host (Claude) → Client → Server
  - Three core server primitives: **Tools** (callable functions), **Resources** (read-only data), **Prompts** (templated workflows)
  - Security first: user consent required for all data access and tool execution
  - Servers provide "context" (data) and "capabilities" (tools) to LLMs
- **Application to pyfinAgent:**
  - `pyfinagent-data` server: Resources for prices, fundamentals, macro, universe (read-only to Claude)
  - `pyfinagent-backtest` server: Tools for running backtests, getting experiments, ablation tests
  - `pyfinagent-signals` server: Tools to generate signals, validate, publish, risk-check
  - All three servers expose their capabilities to Claude via MCP

### ✅ Source 2: FastMCP Python Framework (Practical Implementation)
- **URL:** https://gofastmcp.com/tutorials/create-mcp-server
- **Citation:** FastMCP Documentation (2025). "How to Create an MCP Server in Python."
- **Key learnings:**
  - FastMCP abstracts all JSON-RPC boilerplate (session management, message routing)
  - Decorators make it simple: `@mcp.tool` for callables, `@mcp.resource` for data
  - Resource templates: `@mcp.resource("greetings://{name}")` for dynamic data generation
  - Type hints auto-generate JSON schemas for LLM understanding
  - Transports: stdio (local), SSE (HTTP), or custom
- **Application to pyfinAgent:**
  ```python
  from fastmcp import FastMCP
  
  mcp_data = FastMCP(name="pyfinagent-data")
  
  @mcp_data.resource("prices://{ticker}")
  def get_prices(ticker: str) -> dict:
      """Get OHLCV data for ticker"""
      return fetch_from_bq(ticker)
  
  @mcp_data.resource("features://{ticker}")
  def get_features(ticker: str) -> dict:
      """Get computed features (momentum, value, etc.)"""
      return compute_features(ticker)
  ```

### ✅ Source 3: LLM Agents in Quantitative Trading (Academic + Industry)
- **URL:** IEEE, arXiv (2024-2025), Anthropic blog
- **Citation:** Multiple: AlphaAgent paper, Chain-of-Alpha, Anthropic "Claude for Financial Services"
- **Key patterns:**
  - **Multi-agent architecture:** Generator LLM proposes alphas, Evaluator LLM critiques (Anthropic harness pattern!)
  - **Claude for Finance:** Native integration with S&P Global, FactSet, Bloomberg for real data
  - **Alpha decay prevention:** Constraints enforcing originality, economic logic, simplicity
  - **Dynamic rebalancing:** Portfolio optimization responds to sentiment + price signals in real-time
  - **MCP for finance:** LLMs use tools to query data, run backtests, and retrieve experiment history
- **Application to pyfinAgent:**
  - **Planner LLM:** Uses `pyfinagent-data` MCP to analyze feature correlations, suggest new features
  - **Generator LLM:** Uses `pyfinagent-backtest` MCP to test features in backtests
  - **Evaluator LLM:** Uses `pyfinagent-backtest` MCP to independently validate, spot overfitting
  - **Signal Generator:** Uses `pyfinagent-signals` MCP to propose trades, validate against constraints
  - All LLMs coordinate via MCP, reducing hallucinations (grounded in actual data + backtest results)

---

## Candidate Sources (Collected, Not Yet Deeply Read)

1. **Anthropic harness design pattern** — https://www.anthropic.com/engineering/harness-design-long-running-apps (already used in Phase 2)
2. **MCP GitHub examples** — https://github.com/modelcontextprotocol/ (official implementations)
3. **Chain-of-Alpha paper** — LLM iterative alpha generation with feedback loops
4. **AlphaAgent paper** — Sustainable quant alpha via LLM constraints (originality, simplicity)
5. **Anthropic Claude for Financial Services** — https://www.anthropic.com/news/claude-for-financial-services
6. **Multi-agent LLM trading frameworks** — TradingAgents, FinRL ecosystem
7. **Black-Litterman portfolio optimization with sentiment** — LLM-derived sentiment for portfolio weighting
8. **Efficient frontier discovery via LLM** — Can LLMs solve combinatorial portfolio optimization?
9. **Regime detection (HMM + ML)** — Separate from LLM (Phase 3.3)
10. **OpenClaw MCP integration** — Already partially integrated into OpenClaw, needs documentation

---

## Architecture Design (From Research)

### Three MCP Servers (Phase 3.0 Implementation)

#### **1. pyfinagent-data (FastMCP server)**
**Purpose:** Read-only data access for Claude (prices, fundamentals, macro, universe, features)

**Resources:**
```
prices://[TICKER]              → OHLCV data (open, high, low, close, volume, dates)
fundamentals://[TICKER]        → P/E, P/B, ROE, debt, earnings, growth rates
macro://[SERIES]               → VIX, yield curve, GDP, inflation (by date)
universe://[MARKET]            → List of tickers in each market (US, NO, CA, EU, KR)
features://[TICKER]            → Computed features (momentum, value, sentiment)
experiments://list             → All historical backtest results
best_params://current          → Current best parameters from optimizer
```

**Tools:**
- None (read-only). All operations are resources.

#### **2. pyfinagent-backtest (FastMCP server)**
**Purpose:** Callable backtest operations for Claude (propose features, run tests, validate)

**Tools:**
```
run_backtest(params: dict, date_range: str) → BacktestResult
  "Run full walk-forward backtest with given parameters"
  Returns: {sharpe, return_pct, max_dd, trades, by_period}

run_single_feature_test(feature_code: str) → FeatureResult
  "Test one new feature on holdout period (2-week validation)"
  Returns: {sharpe, correlation_with_existing, acceptance_verdict}

run_ablation_study(feature_to_remove: str) → AblationResult
  "What happens if we remove this feature? Impact on Sharpe?"
  Returns: {sharpe_delta, pvalue, affected_periods}

get_feature_importance() → dict
  "MDI + MDA feature importance across all 27 windows"
  Returns: {feature_name: importance_score}
```

**Resources:**
```
quant_results://all              → TSV of all 70+ experiments run so far
experiments://recent             → Last 10 backtest results (summary)
best_experiment://metadata       → Sharpe history, DSR, deflated Sharpe
```

#### **3. pyfinagent-signals (FastMCP server)**
**Purpose:** Signal generation, validation, and publishing (for autonomous trading)

**Tools:**
```
generate_signal(ticker: str, date: str) → Signal
  "Generate buy/sell signal for ticker on given date"
  Returns: {signal: BUY|SELL|HOLD, confidence: 0-1, factors: [factor1, factor2, ...]}

validate_signal(signal: Signal) → ValidationResult
  "Check signal against constraints: market hours, liquidity, exposure limits"
  Returns: {valid: bool, violations: [list], adjusted_signal: Signal}

publish_signal(signal: ValidatedSignal) → PublishResult
  "Post signal to Slack + portfolio (paper trading or live)"
  Returns: {published: bool, channel: slack_id, timestamp}

risk_check(portfolio: Portfolio, proposed_trade: Trade) → RiskResult
  "Can we add this position? Check exposure, correlation, margin"
  Returns: {allowed: bool, current_exposure, max_exposure, conflict_factors}
```

**Resources:**
```
portfolio://current              → Current holdings (tickers, shares, PnL)
constraints://risk               → Risk limits (max exposure, max drawdown, Sharpe floor)
signals://history                → All generated signals (this month/week)
```

---

## Security Considerations (MCP 2025 Spec)

1. **User Consent:** Claude must ask for approval before:
   - Accessing sensitive data (prices, portfolio, features)
   - Running backtests (compute-heavy, ~5-10 min each)
   - Publishing signals (affects real money)

2. **Data Privacy:** Don't expose:
   - Full portfolio holdings (summary only)
   - Proprietary feature definitions (results only)
   - Historical trade logs with PnL (summary statistics only)

3. **Tool Safety:** Backtest and signal tools must:
   - Have hard limits (timeout after 15 min)
   - Validate inputs (ticker exists, date in range, params reasonable)
   - Log all calls (audit trail for Peder's review)

4. **LLM Sampling Controls:** When Claude samples (re-runs a backtest with different params), Peder must see:
   - What parameters were changed
   - Why Claude thinks it will improve
   - Expected vs actual results

---

## Implementation Roadmap (Phase 3.0 → 3.4)

### Phase 3.0: MCP Server Architecture (THIS RESEARCH)
- Research ✅ Complete (10+ sources, 3 read in full)
- Contract: Define success criteria for 3 MCP servers
- Implementation: Build pyfinagent-data, pyfinagent-backtest, pyfinagent-signals
- Validation: Test all server tools/resources with mock Claude client
- Estimated effort: 20-30 hours

### Phase 3.1: LLM-as-Planner
- Claude with MCP tools analyzes experiment log + feature correlations
- Proposes next research direction (new feature class, parameter combo, regime detection)
- Autonomously runs handoff/contract.md generation
- Estimated effort: 15-20 hours

### Phase 3.2: LLM-as-Evaluator
- Claude independently validates generator's work via MCP backtest tools
- Spot-checks for overfitting, sub-period crashes, feature leakage
- Writes handoff/evaluator_critique.md
- Estimated effort: 15-20 hours

### Phase 3.3: Regime Detection (HMM)
- Separate from LLM (zero cost)
- Detect market regimes: bull, consolidation, bear, high-vol
- Adjusts feature weights per regime
- Estimated effort: 10-15 hours

### Phase 3.4: Agent Skill Optimization
- Learning-to-optimize: which agent types, prompts, constraints yield best alpha?
- Run experiments: Planner A vs B, Evaluator strict vs lenient, different feature generators
- Auto-improve agent instructions
- Estimated effort: 20-30 hours

### Total Phase 3: ~80-115 hours (2-3 weeks)

---

## Success Criteria (Research-Backed)

Phase 3.0 PASS if:
1. ✅ All 3 MCP servers implemented with FastMCP
2. ✅ All tools/resources work with mock LLM client (Claude SDK)
3. ✅ No crashes on backtest/signal operations
4. ✅ Latency < 5s for data resources, < 30s for backtest tools
5. ✅ Security audit: user consent flow working, no private data leaks
6. ✅ Integration test: Claude (via MCP) can query data + run backtest + interpret results

---

## Budget Impact

- **Phase 3.0 implementation:** $0 (FastMCP + pyfinAgent is CPU-only)
- **Phase 3.1 + 3.2 (LLM Planner + Evaluator):** ~$2-5 per research cycle (Claude API calls)
- **Phase 3.3 (Regime detection):** $0 (HMM is CPU-only)
- **Phase 3.4 (Agent optimization):** ~$5-10 per experiment (iterative Claude calls)
- **Total Phase 3 cost:** $20-50/month (awaiting Peder approval)

---

## References & Citations

1. MCP Specification 2025-11-25 — https://modelcontextprotocol.io/specification/2025-11-25
2. FastMCP Documentation — https://gofastmcp.com/tutorials/create-mcp-server
3. Anthropic Claude for Financial Services — https://www.anthropic.com/news/claude-for-financial-services
4. Chain-of-Alpha — LLM iterative alpha generation (arxiv)
5. AlphaAgent — Sustainable quant alpha via LLM (medium article)
6. Multi-agent LLM trading frameworks — TradingAgents, academic papers (2024-2025)
7. Anthropic Harness Design Pattern — https://www.anthropic.com/engineering/harness-design-long-running-apps

---

## Next Step: Phase 3.0 Contract

Upon approval, will write `handoff/phase30_contract.md` with:
- Hypothesis: MCP servers enable Claude to autonomously improve strategy
- Success criteria: all 3 servers working, integration test PASS
- Fail conditions: crashes, latency > 30s, security leaks
- Timeline: 20-30 hours over 2-3 work days
- Deliverables: 3 server implementations + integration tests + security audit
