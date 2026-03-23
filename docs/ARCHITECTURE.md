# PyFinAgent — Architecture Reference

> **For a quick overview**, see [AGENTS.md](../AGENTS.md). This is the full architecture reference.
> For frontend UX conventions, see [UX-AGENTS.md](../UX-AGENTS.md).
> For autonomous trading optimization, see [trading_agent.md](../trading_agent.md).
> For version history, see [CHANGELOG.md](../CHANGELOG.md).

This document provides the complete architecture, agent registry, API reference, data schemas, and conventions for the PyFinAgent project.

---

## Project Overview

*   **Project Name**: PyFinAgent
*   **Purpose**: An agentic AI financial analyst that performs deep-dive company analysis by orchestrating multiple specialized agents — including a multi-agent debate framework, Monte Carlo stress testing, anomaly detection, and LLM bias mitigation — to produce a comprehensive, auditable investment report.
*   **Core Function**: The system coordinates 20+ expert AI agents through a structured pipeline: data collection → parallel enrichment → adversarial debate (Bull vs Bear) → synthesis → critic validation → bias audit. This design is informed by research from Goldman Sachs, BlackRock, Morgan Stanley, Harvard Business School, Stanford University, and the University of Chicago Booth School of Business.
*   **Design Philosophy**: "Glass Box" — every agent's inputs, reasoning, and outputs are visible to the user. No black-box decisions.

---

## Architecture

PyFinAgent uses a **Next.js 15 + FastAPI** architecture:

| Layer | Technology | Port |
|-------|-----------|------|
| **Frontend** | Next.js 15 / React 19 / TypeScript 5.6 / Tailwind CSS / Geist Font / Phosphor Icons / Motion / Recharts | 3001 |
| **Backend** | FastAPI 0.115+ / Python 3.14 / Vertex AI (Gemini 2.0 Flash) | 8000 |
| **Storage** | Google BigQuery (reports), Google Cloud Storage (10-K/10-Q filings) | — |
| **AI Engine** | Google Vertex AI with Gemini model, RAG via Vertex AI Search datastore | — |
| **Embeddings** | Vertex AI text-embedding-005 (for NLP sentiment) | — |

### Directory Structure

```
pyfinagent/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py      # Main 15-step analysis pipeline
│   │   ├── debate.py            # Multi-round debate framework (Bull/Bear/Devil's Advocate/Moderator)
│   │   ├── risk_debate.py       # Risk Assessment Team (Aggressive/Conservative/Neutral + Risk Judge)
│   │   ├── info_gap.py          # AlphaQuanter-style info-gap detection + retry loop
│   │   ├── memory.py            # BM25-based FinancialSituationMemory (learns from outcomes)
│   │   ├── trace.py             # Decision trace logger (XAI)
│   │   ├── bias_detector.py     # LLM bias detection (tech bias, confirmation bias)
│   │   ├── conflict_detector.py # Knowledge conflict detection (parametric vs real-time)
│   │   ├── cost_tracker.py      # Per-agent token/cost tracking (dual LLM support)
│   │   ├── schemas.py           # Pydantic output schemas for Gemini structured output (Phase 3)
│   │   ├── skill_optimizer.py   # Autonomous skill optimization loop (autoresearch pattern)
│   │   ├── meta_coordinator.py  # Cross-loop sequencing for 3 optimization loops (MDA→Agent bridge)
│   │   └── skills/              # Agent skills.md files (29 agents) + experiments/ + quant_strategy.md optimizer skill
│   ├── config/
│   │   ├── settings.py          # Pydantic settings (env vars)
│   │   └── prompts.py           # Skill-loaded prompt wrappers (loads from skills/*.md)
│   ├── api/
│   │   ├── analysis.py          # POST /api/analysis/, GET /api/analysis/{id}
│   │   ├── auth.py              # HKDF + JWE token decrypt, email whitelist
│   │   ├── reports.py           # Reports CRUD + performance stats
│   │   ├── charts.py            # Price chart + financials endpoints
│   │   ├── signals.py           # Enrichment signals endpoints (12 routes)
│   │   ├── portfolio.py         # Portfolio tracking CRUD
│   │   ├── paper_trading.py     # Autonomous paper trading endpoints (8 routes)
│   │   ├── settings_api.py      # Model configuration + available models
│   │   ├── skills.py            # Skills optimization API endpoints
│   │   ├── backtest.py          # Walk-forward backtest + quant optimizer endpoints (11 routes); mutual exclusion (HTTP 409) prevents concurrent runs; engine_source tracks who started the engine; state dicts carry error+traceback for UI
│   │   └── performance_api.py   # Performance monitoring + cache stats + TTL optimizer (8 routes)
│   ├── backtest/
│   │   ├── analytics.py          # Sharpe, DSR, baselines (SPY/equal-weight/momentum with real Sharpe), reporting, compute_round_trips (FIFO BUY→SELL matching), compute_trade_statistics (23-field: profit_factor/win_rate/expectancy/SQN/streaks/cost metrics)
│   │   ├── backtest_engine.py    # Walk-forward ML backtest orchestrator — 8-step pipeline (preloading→screening→building_features→training→computing_mda→predicting→trading→finalizing), _current_window_id tracking fixes 0/N counter, emits finalizing step before cache.clear_cache(); skip_cache_clear param allows optimizer to keep warm cache across iterations; SPY preloaded alongside universe for baseline benchmarks; mr_holding_days param for mean reversion strategy; all_trades extraction (capped at 500) into BacktestResult; commission_model/commission_per_share forwarded to trader
│   │   ├── backtest_trader.py    # In-memory portfolio simulator (inverse-vol sizing); per-trade commission tracking (flat_pct + per_share models); Trade dataclass includes commission field
│   │   ├── cache.py              # BQ bulk-preload cache (2 queries for entire backtest) + hit/miss stats
│   │   ├── candidate_selector.py # S&P 500 screening at historical dates
│   │   ├── data_ingestion.py     # Bulk ingest prices/fundamentals/macro to BQ
│   │   ├── historical_data.py    # ~49-feature vector builder (point-in-time)
│   │   ├── quant_optimizer.py    # Autoresearch-style strategy optimization loop; 17 tunable params (including mr_holding_days); per-run UUID tagging, step-level progress (establishing_baseline/running_experiment/evaluated), skip_cache_clear for warm cache, explicit bq_cache.clear_cache() at end; LLM proposals load quant_strategy.md skill for research-backed guidance
│   │   ├── walk_forward.py       # Expanding-window walk-forward scheduler
│   │   └── experiments/          # quant_results.tsv experiment logs
│   ├── db/
│   │   ├── bigquery_client.py    # BigQuery report persistence (68-column ML schema) + paper trading CRUD
│   │   └── __init__.py
│   ├── services/
│   │   ├── outcome_tracker.py   # Evaluates past recs vs actual returns (feedback loop)
│   │   ├── paper_trader.py      # Virtual trade execution engine (buy/sell/MTM/snapshot)
│   │   ├── portfolio_manager.py # Trade decision logic (sell-first-then-buy, Risk Judge sizing)
│   │   ├── autonomous_loop.py  # Daily autonomous trading cycle (screen→analyze→trade→learn)
│   │   ├── api_cache.py         # Thread-safe TTL response cache (in-memory, lazy eviction)
│   │   ├── perf_tracker.py      # Per-endpoint latency recording (p50/p95/p99)
│   │   ├── perf_optimizer.py    # Autoresearch-style API cache TTL optimizer
│   │   └── perf_metrics.py      # Canonical P&L + scalar metric (PerformanceSkill)
│   ├── slack_bot/
│   │   ├── app.py               # Slack bot entry point (Socket Mode)
│   │   ├── commands.py          # /analyze, /portfolio, /report slash commands
│   │   ├── scheduler.py         # Morning digest cron job + proactive alerts
│   │   ├── formatters.py        # Block Kit message builders
│   │   └── Dockerfile           # Standalone container for slack bot
│   ├── tools/
│   │   ├── alphavantage.py      # Alpha Vantage news + competitor discovery
│   │   ├── yfinance_tool.py     # Comprehensive financials + price history
│   │   ├── sec_insider.py       # SEC EDGAR Form 4 insider trades
│   │   ├── options_flow.py      # Options chain P/C ratio + unusual volume
│   │   ├── social_sentiment.py  # Social/news sentiment velocity
│   │   ├── patent_tracker.py    # USPTO PatentsView innovation velocity
│   │   ├── earnings_tone.py     # Earnings transcript fetcher
│   │   ├── fred_data.py         # FRED 7-series macro indicators
│   │   ├── alt_data.py          # Google Trends momentum
│   │   ├── sector_analysis.py   # Sector ETF rotation + peer comparison
│   │   ├── nlp_sentiment.py     # Transformer NLP via Vertex AI embeddings
│   │   ├── anomaly_detector.py  # Multi-dimensional anomaly detection (Z-score)
│   │   ├── monte_carlo.py       # Monte Carlo VaR simulation engine
│   │   ├── slack.py             # Slack webhook notifications
│   │   ├── screener.py          # S&P 500 quant screener (momentum, RSI, composite score)
│   │   └── quant_model.py       # MDA-weighted quant model signal (12th enrichment tool)
│   └── main.py                  # FastAPI app + router registration; setup_logging() wraps stderr in UTF-8 TextIOWrapper
├── frontend/
│   ├── prisma/
│   │   └── schema.prisma        # SQLite auth DB (User, Account, Session, Authenticator)
│   ├── src/app/
│   │   ├── page.tsx             # Home overview (portfolio snapshot, recent reports, quick actions)
│   │   ├── login/page.tsx       # Login page (Google SSO + Passkey)
│   │   ├── api/auth/[...nextauth]/route.ts  # NextAuth catch-all handler
│   │   ├── analyze/page.tsx     # Deep analysis (15-step pipeline + 6-tab report)
│   │   ├── signals/page.tsx     # Signals exploration
│   │   ├── reports/page.tsx     # Reports (History + Compare tabs)
│   │   ├── performance/page.tsx # Performance stats
│   │   ├── portfolio/page.tsx   # Portfolio tracking + P&L
│   │   ├── paper-trading/page.tsx # Autonomous paper trading dashboard
│   │   ├── backtest/page.tsx    # Walk-forward backtest dashboard
│   │   └── settings/page.tsx    # Configuration
│   ├── src/components/
│   │   ├── AuthProvider.tsx     # SessionProvider wrapper (15-min refetch)
│   │   ├── DebateView.tsx       # Bull vs Bear debate visualization
│   │   ├── RiskDashboard.tsx    # Monte Carlo fan chart + VaR + anomalies
│   │   ├── SentimentDetail.tsx  # NLP sentiment deep-dive
│   │   ├── BiasReport.tsx       # LLM bias flags + knowledge conflicts
│   │   ├── SignalCards.tsx       # 12-signal enrichment grid + consensus bar
│   │   ├── SectorDashboard.tsx  # Sector rotation + relative strength
│   │   ├── MacroDashboard.tsx   # FRED indicator grid + warnings
│   │   ├── StockChart.tsx       # Candlestick + SMA/RSI chart
│   │   ├── EvaluationTable.tsx  # 5-pillar scoring matrix
│   │   ├── AnalysisProgress.tsx # 15-step progress tracker
│   │   ├── CostDashboard.tsx    # LLM cost/token analytics dashboard
│   │   ├── Skeleton.tsx         # Reusable loading skeleton components (pulse, card, grid, page)
│   │   └── Sidebar.tsx          # Navigation sidebar + user auth UI
│   ├── src/lib/
│   │   ├── auth.config.ts       # Edge-compatible auth config (providers, callbacks)
│   │   ├── auth.ts              # Full auth config (PrismaAdapter + WebAuthn)
│   │   ├── prisma.ts            # Singleton Prisma client
│   │   ├── types.ts             # Full TypeScript type definitions
│   │   └── api.ts               # API client (Bearer token + 401 handling)
│   └── src/middleware.ts        # Route protection (redirects unauthenticated → /login)
├── docker-compose.yml           # 3 services: backend, frontend, slack-bot
├── migrate_bq_schema.py         # Idempotent BQ schema migration (adds ML columns)
├── migrate_agent_memories.py    # Idempotent BQ migration (creates agent_memories table)
├── migrate_paper_trading.py     # Idempotent BQ migration (4 paper trading tables)
├── migrate_backtest_data.py     # Idempotent BQ migration (3 historical data tables)
├── quant-agent/                 # GCP Cloud Function
├── ingestion_agent/             # GCP Cloud Function
├── earnings-ingestion-agent/    # GCP Cloud Function
├── trading_agent.md             # Living memory/instruction file for autonomous trading optimization
└── AGENTS.md                    # This file
```

---

## Analysis Pipeline (15 Steps)

The orchestrator (`backend/agents/orchestrator.py`) executes a 15-step pipeline with **4 design pattern enhancements**: (1) **Reflection Loop** — Synthesis↔Critic iterative refinement (max 2 iterations), (2) **Session Memory** — AnalysisContext accumulates key findings across steps, (3) **Quality Gates** — data quality threshold skips debate/risk when insufficient, (4) **Sector Routing** — sector-aware tool skipping saves compute. Steps 6 and 7 run in parallel using `asyncio.gather`. Step 6b implements AlphaQuanter-style ReAct info-gap detection with retry. Step 6 now uses sector routing to skip irrelevant tools. The debate framework (Step 8) runs multi-round adversarial Bull↔Bear debate with Devil's Advocate stress-testing (TradingAgents pattern), but is conditionally skipped by the data quality gate. Step 11+12 implements an Evaluator-Optimizer reflection loop where Critic feedback triggers Synthesis revision.

| Step | Name | Agent/Tool | Description |
|------|------|-----------|-------------|
| 0 | Market Intel | `alphavantage.py` + `yfinance_tool.py` | Fetch Alpha Vantage news (50 articles) + yfinance fundamentals in parallel |
| 1 | Ingestion | `ingestion-agent` (Cloud Function) | Check for new SEC filings, ingest into BigQuery |
| 2 | Quant Data | `quant-agent` (Cloud Function) | Fetch SEC EDGAR financials, merge with yfinance data |
| 3 | Document Analysis | `RAG Agent` (Vertex AI Search) | Analyze 10-K/10-Q for moat, governance, risks with citations |
| 4 | Sentiment Analysis | `Market Agent` (LLM) | Detect sentiment velocity, price-sentiment divergence, catalyst keywords |
| 5 | Competitor Analysis | `Competitor Agent` (LLM) | Identify rivals from news co-occurrence, assess competitive positioning |
| 6 | **Data Enrichment** | 12 tools in parallel | Insider trades, options flow, social sentiment, patents, earnings transcripts, FRED macro, Google Trends, sector data, NLP sentiment, anomaly scan, Monte Carlo simulation, quant model signal. **Sector routing** skips irrelevant tools (e.g., patents for Financial Services) |
| 6b | **Info-Gap Detection** | `InfoGapDetector` (ReAct) | AlphaQuanter-style scan: assess sources for SUFFICIENT/PARTIAL/MISSING/SKIPPED status, retry critical failures (max 2 attempts), compute data quality score. SKIPPED tools excluded from quality score |
| 7 | **Enrichment Analysis** | 12 LLM agents | Each tool's data analyzed by a specialized LLM agent |
| 8 | **Agent Debate** | `Bull` + `Bear` + `Devil's Advocate` + `Moderator` | Multi-round iterative debate: N rounds of Bull↔Bear rebuttal (each sees opponent's prior argument), Devil's Advocate stress-test (hidden risks, groupthink detection), Moderator consensus with full debate history. **Quality gate**: skipped when data quality < threshold |
| 9 | Enhanced Macro | `Enhanced Macro Agent` (LLM + FRED) | Fed Funds, CPI, GDP, unemployment, yield curve, consumer sentiment analysis |
| 10 | Deep Dive | `Deep Dive Agent` (LLM + RAG) | Identify 3 contradictions between sources, probe with targeted 10-K questions |
| 11 | Synthesis | `Synthesis Agent` (LLM) | Combine all agent outputs + debate consensus + session memory into structured JSON report. **Reflection loop**: if Critic returns REVISE, re-runs with feedback (max 2 iterations) |
| 12 | Critic | `Critic Agent` (LLM) | Returns structured verdict {PASS/REVISE} with typed issues. Validates for hallucinations, logic errors, factual consistency against quant data |
| 12b | Bias Audit | `Bias Detector` + `Conflict Detector` | Check for tech-sector bias, confirmation bias, anchoring; flag knowledge conflicts |
| 12c | **Risk Assessment** | `Aggressive` + `Conservative` + `Neutral` + `Risk Judge` | TradingAgents-style round-robin risk debate: each analyst argues position sizing, Risk Judge delivers final verdict with risk limits |

### Pipeline Data Flow

```
User Input (ticker)
    │
    ├─ Step 0: Alpha Vantage + yfinance ──────────────────────────┐
    ├─ Step 1: Ingestion Agent (Cloud Function)                   │
    ├─ Step 2: Quant Agent (Cloud Function + yfinance merge)      │
    │                                                              │
    ├─ Step 3: RAG Agent (Vertex AI Search on 10-K/10-Q)         │
    ├─ Step 4: Market Agent (sentiment divergence)                │
    ├─ Step 5: Competitor Agent (rival analysis)                  │
    │                                                              │
    ├─ Step 6: Parallel Data Enrichment ──────────────────────────┤
    │   ├── sec_insider.py (SEC EDGAR Form 4)                     │
    │   ├── options_flow.py (yfinance options chain)              │
    │   ├── social_sentiment.py (Alpha Vantage social)            │
    │   ├── patent_tracker.py (USPTO PatentsView)                 │
    │   ├── earnings_tone.py (API Ninjas transcripts)             │
    │   ├── fred_data.py (FRED 7-series)                          │
    │   ├── alt_data.py (Google Trends via pytrends)              │
    │   ├── sector_analysis.py (11 SPDR ETFs)                     │
    │   ├── nlp_sentiment.py (Vertex AI embeddings)               │
    │   ├── anomaly_detector.py (Z-score detection)               │
    │   ├── monte_carlo.py (1,000 GBM simulations)               │
    │   └── quant_model.py (MDA-weighted factor score)           │
    │                                                              │
    ├─ Step 6b: Info-Gap Detection (AlphaQuanter ReAct) ──────────┤
    │   ├── Scan 12 sources → SUFFICIENT/PARTIAL/MISSING          │
    │   ├── Retry critical failures (max 2 attempts)              │
    │   └── Compute data quality score                            │
    │                                                              │
    ├─ Step 7: 12 LLM Enrichment Agents (analyze each dataset)   │
    │                                                              │
    ├─ Step 8: Agent Debate Framework ────────────────────────────┤
    │   ├── Round 1..N: Bull↔Bear iterative rebuttal              │
    │   │   (each sees opponent's prior argument)                 │
    │   ├── Devil's Advocate: hidden risks, groupthink detection  │
    │   └── Moderator: consensus with full debate history         │
    │                                                              │
    ├─ Step 9: Enhanced Macro Agent (AV + FRED combined)          │
    ├─ Step 10: Deep Dive Agent (contradiction probing via RAG)   │
    │                                                              │
    ├─ Step 11: Synthesis Agent ──────────────────────────────────┤
    │   (all agent outputs + debate consensus → structured JSON)  │
    │                                                              │
    ├─ Step 12: Critic Agent (hallucination/logic check)          │
    ├─ Step 12b: Bias Audit (tech bias, confirmation bias, etc.)  │
    └─ Step 12c: Risk Assessment Team ────────────────────────────┤
        ├── Aggressive Analyst → position sizing argument         │
        ├── Conservative Analyst → risk-focused argument          │
        ├── Neutral Analyst → balanced synthesis                  │
        └── Risk Judge → final verdict + risk limits              │
                                                                   │
    Final Output: Validated JSON report with:                      │
    ├── scoring_matrix (5 pillars, weighted)                      │
    ├── enrichment_signals (12 sources)                           │
    ├── debate_result (bull/bear/DA/consensus)                    │
    ├── risk_assessment (judge verdict + risk limits)             │
    ├── info_gap_report (data quality + gap status)               │
    ├── decision_traces (full XAI audit trail)                    │
    ├── anomalies (statistical outliers)                          │
    ├── risk_scenarios (Monte Carlo VaR)                          │
    └── bias_flags (detected LLM biases)                          │
```

---

## System Roles — Complete Agent Registry

### Backend Agents (GCP Cloud Functions)

| Agent | Location | Function |
|-------|----------|----------|
| `quant-agent` | `quant-agent/main.py` | Fetches quantitative data from SEC EDGAR filings and market data from yfinance |
| `ingestion-agent` | `ingestion_agent/main.py` | Loads fetched data into Google BigQuery for structured storage |
| `risk-management-agent` | `pyfinagent-app/risk-management-agent/main.py` | Guardrail: validates data and enforces analysis rules |
| `earnings-ingestion-agent` | `earnings-ingestion-agent/main.py` | Ingests and processes earnings call transcripts |

### In-App LLM Agents (Vertex AI Gemini)

All agents use `gemini-2.0-flash` via Vertex AI. Prompts are defined in `backend/config/prompts.py`.

#### Foundation Agents (Steps 3–5)

| Agent | Prompt Function | Input | Output | Signal Type |
|-------|----------------|-------|--------|-------------|
| **RAG Agent** | `get_rag_prompt()` | Ticker → Vertex AI Search on 10-K/10-Q datastore | Economic moat, governance assessment, risk factors with `[Source \| YYYY-MM-DD]` citations | N/A (text) |
| **Market Agent** | `get_market_prompt()` | Ticker + Alpha Vantage 50 articles | Sentiment velocity, price-sentiment divergence detection, institutional catalyst keywords | Divergence Warning / No Divergence |
| **Competitor Agent** | `get_competitor_prompt()` | Ticker + AV-derived rival tickers | Rival confirmation (true competitor vs partner), competitive positioning (winning/losing context) | N/A (text) |

#### Enrichment Agents (Step 7)

| Agent | Prompt Function | Input Data Source | Analysis Performed | Signal Values |
|-------|----------------|-------------------|-------------------|---------------|
| **Insider Activity Agent** | `get_insider_prompt()` | `sec_insider.py` → SEC EDGAR Form 4 | Cluster buy detection (3+ execs in 30d), buy/sell ratio, timing vs events, historical pattern comparison | STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH |
| **Options Flow Agent** | `get_options_prompt()` | `options_flow.py` → yfinance options chain | Put/call ratio (< 0.7 bullish, > 1.0 bearish), unusual volume (> 3x OI), skew analysis, institutional block trades | STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH |
| **Social Sentiment Agent** | `get_social_sentiment_prompt()` | `social_sentiment.py` → Alpha Vantage social API | Sentiment velocity (improving/deteriorating), source divergence (mainstream vs social vs financial press), contrarian risk | Score: -1.0 to +1.0 |
| **Patent/Innovation Agent** | `get_patent_prompt()` | `patent_tracker.py` → USPTO PatentsView API | YoY filing velocity (≥ 20% = INNOVATION_BREAKOUT), technology domain classification, moat assessment (defensive vs offensive), commercialization timeline | INNOVATION_BREAKOUT / ACCELERATING / STABLE / DECLINING |
| **Earnings Tone Agent** | `get_earnings_tone_prompt()` | `earnings_tone.py` → API Ninjas transcripts | Management confidence (1-10 scale), forward guidance extraction, red flag detection (evasive answers, topic changes), key themes | CONFIDENT / CAUTIOUS / EVASIVE |
| **Alt Data Agent** | `get_alt_data_prompt()` | `alt_data.py` → Google Trends via pytrends | Search interest momentum (recent 4w vs weeks 4-12), lead/lag relationship to revenue, related query analysis | RISING_STRONG / RISING / STABLE / DECLINING |
| **Sector Analysis Agent** | `get_sector_analysis_prompt()` | `sector_analysis.py` → yfinance + 11 SPDR ETFs | Sector rotation cycle positioning, relative strength (stock vs sector ETF vs SPY across 1M/3M/6M/1Y), peer valuation comparison | DOUBLE_TAILWIND / SECTOR_TAILWIND / STOCK_OUTPERFORMING / NEUTRAL / LAGGING |
| **NLP Sentiment Agent** | `get_nlp_sentiment_prompt()` | `nlp_sentiment.py` → Vertex AI text-embedding-005 | Contextual sentiment scoring (transformer embeddings, not keyword-based), semantic similarity to financial sentiment corpus, source reliability weighting | Score: -1.0 to +1.0 + confidence |
| **Anomaly Detection Agent** | `get_anomaly_detection_prompt()` | `anomaly_detector.py` → multi-dimensional Z-score | Interpret statistical anomalies (> 2σ deviation), classify as opportunity vs risk, prioritize by severity and actionability | ANOMALY_OPPORTUNITY / ANOMALY_RISK / NORMAL |
| **Scenario Analysis Agent** | `get_scenario_analysis_prompt()` | `monte_carlo.py` → 1,000 GBM simulations | Interpret VaR (95%/99%), expected shortfall, probability distributions, recommend position sizing based on risk tolerance | Risk profile classification |
| **Quant Model Agent** | `get_quant_model_prompt()` | `quant_model.py` → MDA-weighted factor scoring | Interpret backtest-derived factor weights, flag momentum-quality alignment/divergence, assess MDA source freshness, extreme value detection | STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH |

#### Debate Agents (Step 8)

| Agent | Prompt Function | Role | Input | Output |
|-------|----------------|------|-------|--------|
| **Bull Agent** | `get_bull_agent_prompt()` | Synthesize all bullish signals into the strongest investment case | All enrichment agent outputs where signal is bullish + opponent's prior argument (rounds 2+) | Structured bull thesis with evidence citations, confidence score, key catalysts |
| **Bear Agent** | `get_bear_agent_prompt()` | Synthesize all bearish signals into the strongest bear case | All enrichment agent outputs where signal is bearish + opponent's prior argument (rounds 2+) | Structured bear thesis with risk evidence, confidence score, key threats |
| **Devil's Advocate** | `get_devils_advocate_prompt()` | Stress-test both sides for hidden risks and groupthink | Bull case + Bear case + all enrichment signals | Challenges list, hidden risks, bull/bear weaknesses, groupthink flag, confidence adjustment |
| **Moderator Agent** | `get_moderator_prompt()` | Resolve contradictions, assign consensus, identify unresolved uncertainties | Bull case + Bear case + Devil's Advocate + full debate history | Final consensus (STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL), contradiction map, confidence-weighted score, dissent registry |

#### Risk Assessment Agents (Step 12c)

| Agent | Prompt Function | Role | Input | Output |
|-------|----------------|------|-------|--------|
| **Aggressive Analyst** | `get_aggressive_analyst_prompt()` | Argue for maximum position sizing based on upside potential | Synthesis result + enrichment signals | Position sizing argument with confidence, max position % |
| **Conservative Analyst** | `get_conservative_analyst_prompt()` | Argue for minimum position sizing based on downside risk | Synthesis result + enrichment signals | Risk-focused argument with confidence, max position % |
| **Neutral Analyst** | `get_neutral_analyst_prompt()` | Balanced synthesis of aggressive and conservative views | Synthesis + aggressive + conservative arguments | Balanced position with confidence, max position % |
| **Risk Judge** | `get_risk_judge_prompt()` | Final verdict on position sizing and risk limits | All three analyst arguments | Decision, risk-adjusted confidence, recommended position %, risk level, risk limits |

#### Validation Agents (Steps 10–12b)

| Agent | Prompt Function | Purpose |
|-------|----------------|---------|
| **Deep Dive Agent** | `get_deep_dive_prompt()` | Identifies 3 contradictions between quant/RAG/market/competitor data, generates targeted 10-K questions, probes via RAG model |
| **Synthesis Agent** | `get_synthesis_prompt()` | Combines ALL agent outputs + debate consensus into structured JSON: 5-pillar scoring matrix, investment thesis, key risks, recommendation |
| **Critic Agent** | `get_critic_prompt()` | Reviews draft JSON for hallucinations (numbers contradicting quant data), logic errors (Strong Buy with low score), JSON validity. Also checks for tech-sector bias, confirmation bias, anchoring bias, and source diversity |
| **Bias Detector** | `bias_detector.py` | Compares LLM recommendation against statistical base rates; flags systematic tech/large-cap favoritism, confirmation bias (ignoring contradictory signals), recency bias |
| **Conflict Detector** | `conflict_detector.py` | Compares Gemini's parametric knowledge (what it "knows") against real-time yfinance/SEC data; flags discrepancies as knowledge conflicts with severity ratings |

### Additional Agents

| Agent | Prompt Function | Description |
|-------|----------------|-------------|
| **Sector Catalyst Agent** | `get_sector_catalyst_prompt()` | Monitors "Innovation Velocity" (patent growth ≥ 20%) cross-referenced with R&D labor momentum to identify structural breakouts |
| **Supply Chain Agent** | `get_supply_chain_prompt()` | Analyzes co-occurrence data to determine if gains are sector-wide tailwind or company-specific market share capture |
| **Enhanced Macro Agent** | `get_enhanced_macro_prompt()` | Combines Alpha Vantage macro + FRED 7-series indicators for comprehensive monetary/fiscal policy impact analysis |

---

## Data Tools — Complete Registry

All tools are in `backend/tools/`. Each returns a consistent structure: `{ ticker, signal, summary, ... }`.

| Tool | File | Data Source | Key Metrics | Async | Rate Limits |
|------|------|-----------|-------------|-------|-------------|
| **Alpha Vantage Intel** | `alphavantage.py` | AV News API (50 articles) | Sentiment scores, derived competitor tickers via co-occurrence | Yes | 5 req/min |
| **yfinance Fundamentals** | `yfinance_tool.py` | Yahoo Finance | P/E, PEG, D/E, margins, ROE, ownership, 9 valuation metrics, OHLCV history | No | None |
| **SEC Insider Trades** | `sec_insider.py` | SEC EDGAR Form 4 | Buy/sell counts, buy/sell ratio, cluster flag (3+ execs in 30d), trade details | Yes | SEC User-Agent required |
| **Options Flow** | `options_flow.py` | yfinance options chain | P/C ratios, unusual volume (vol > 3x OI), skew, 2 nearest expirations | No | None |
| **Social Sentiment** | `social_sentiment.py` | AV Social Sentiment API | Avg sentiment, velocity, source breakdown (mainstream/social/financial) | Yes | 5 req/min |
| **Patent Tracker** | `patent_tracker.py` | USPTO PatentsView API | YoY filing velocity %, citations/patent, recent filings, assignee match | Yes | None |
| **Earnings Tone** | `earnings_tone.py` | API Ninjas | Latest transcript (8000-char excerpt), quarter, year | Yes | API key required |
| **FRED Macro** | `fred_data.py` | Federal Reserve FRED | 7 series (12M each): Fed Funds, CPI, unemployment, GDP, 10Y-2Y spread, consumer sentiment, 10Y Treasury | Yes | API key required |
| **Google Trends** | `alt_data.py` | pytrends (Google Trends) | Interest index, 12M momentum, recent 4w vs weeks 4-12 ratio | No | Rate limited, US-only |
| **Sector Analysis** | `sector_analysis.py` | yfinance + 11 SPDR ETFs | Stock/sector/SPY returns (4 periods), sector rotation chart, 5-peer comparison | No | None |
| **NLP Sentiment** | `nlp_sentiment.py` | Vertex AI text-embedding-005 + AV articles | Contextual embedding scores, semantic similarity to financial corpus, weighted source scoring | Yes | Vertex AI quota |
| **Anomaly Detector** | `anomaly_detector.py` | Multi-dimensional (yfinance + all enrichment) | Z-score per metric, IQR outlier detection, > 2σ deviation flags | No | None |
| **Monte Carlo VaR** | `monte_carlo.py` | yfinance daily returns | 1,000 GBM simulations, VaR (95%/99%), expected shortfall, P(±20%) over 3M/6M/1Y | No | None |
| **Slack Notifications** | `slack.py` | Slack Webhook | Formatted message blocks with color coding | No | None |
| **S&P 500 Screener** | `screener.py` | yfinance + Wikipedia | Momentum (1M/3M/6M), RSI_14, volatility, SMA distance, composite alpha score | No | None |
| **Quant Model Signal** | `quant_model.py` | MDA cache + yfinance live features | MDA-weighted factor score, top contributing factors, signal classification | No | None |

---

## Scoring System

### 5-Pillar Weighted Matrix

| Pillar | Key | Weight | What It Measures |
|--------|-----|--------|-----------------|
| Corporate Quality | `pillar_1_corporate` | 35% | Business model strength, moat durability, management quality, financial health |
| Industry Position | `pillar_2_industry` | 20% | Competitive landscape, market share trajectory, sector tailwinds/headwinds |
| Valuation | `pillar_3_valuation` | 20% | P/E, PEG, FCF yield, price-to-book relative to growth, margin of safety |
| Sentiment | `pillar_4_sentiment` | 15% | News/social sentiment, insider activity, options flow, institutional signals |
| Governance | `pillar_5_governance` | 10% | Executive compensation alignment, board independence, shareholder structure |

**Final Score** = Σ (pillar_score × weight) → 0.0 to 10.0

### Enrichment Signal Format

All enrichment tools return:
```json
{
  "ticker": "AAPL",
  "signal": "BULLISH",
  "summary": "Cluster buy detected: 4 executives purchased shares within 15-day window...",
  "data": { /* tool-specific raw data */ }
}
```

### Debate Result Format

```json
{
  "bull_case": { "thesis": "...", "confidence": 0.78, "key_catalysts": [...], "evidence": [...] },
  "bear_case": { "thesis": "...", "confidence": 0.45, "key_threats": [...], "evidence": [...] },
  "consensus": "BUY",
  "consensus_confidence": 0.72,
  "contradictions": [ { "topic": "...", "bull_view": "...", "bear_view": "...", "resolution": "..." } ],
  "dissent_registry": [ { "agent": "Options Flow", "position": "BEARISH", "reason": "..." } ],
  "debate_rounds": [ { "round": 1, "bull_argument": "...", "bear_argument": "..." }, { "round": 2, "bull_argument": "...", "bear_argument": "..." } ],
  "total_rounds": 2,
  "devils_advocate": { "challenges": [...], "hidden_risks": [...], "groupthink_flag": false, "confidence_adjustment": -0.05, "summary": "..." }
}
```

### Decision Trace Format (XAI)

Every agent call produces a trace:
```json
{
  "agent_name": "Insider Activity Agent",
  "timestamp": "2026-03-08T14:32:01Z",
  "input_data_hash": "a1b2c3d4",
  "output_signal": "STRONG_BULLISH",
  "confidence": 0.85,
  "evidence_citations": ["4 execs bought $12M in 15 days", "Buy/sell ratio: 8.5"],
  "reasoning_steps": ["Identified cluster buy pattern", "Cross-referenced with earnings date", "Historical: rare for this company"]
}
```

---

## Research Foundations

This system's design is informed by leading research on AI in financial trading:

| Research Source | Key Finding | Our Implementation |
|----------------|-------------|-------------------|
| **Harvard Business School** (NBER, ref 10) | Neural networks predict 71% of active fund trades; true alpha resides in the non-routine 29% | Our agents focus on non-routine analysis: contradictions, anomalies, cross-signal debate |
| **Stanford University** (ref 11) | Transformer embeddings achieve 0.07-0.13% price prediction error vs keyword sentiment | `nlp_sentiment.py` uses Vertex AI text-embedding-005 for contextual sentiment, not black-box keyword scoring |
| **Chicago Booth / Fama-Miller** (ref 12-13) | BERT models extract tradable signals from 8,000+ shareholder letters over 65 years | RAG Agent analyzes 10-K/10-Q filings; Earnings Tone Agent analyzes management language patterns |
| **Goldman Sachs** (ref 16) | 127-dimensional anomaly detection predicted Thai baht crisis 48h early; 5,000 scenario simulations every 5 min | `anomaly_detector.py` (multi-dimensional Z-score); `monte_carlo.py` (1,000 GBM simulations per analysis) |
| **BlackRock** (ref 4, 18) | Domain-specific LLMs on 400K earnings transcripts outperform general GPT; geospatial knowledge graph maps 1.8M locations to 8,750 equities | Prompt-engineered agents with curated financial analysis instructions; sector analysis maps 11 SPDR ETFs as proxy |
| **Morgan Stanley** (ref 21-22) | GPT-4 assistant synthesizes 100K internal reports; GenAI could affect 44% of occupations ($4.1T labor costs) | Our Synthesis Agent combines 20+ agent outputs; Glass Box dashboard democratizes institutional-grade analysis |
| **TradingAgents** (arXiv, ref 32) | Multi-agent debate frameworks yield more robust trading decisions than individual agents | `debate.py` implements 4-round Bull/Bear/Moderator adversarial debate before synthesis |
| **Wharton School** (NBER, ref 25-26) | RL agents autonomously learn collusive pricing without explicit communication | Our Bias Detector checks for algorithmic bias patterns; Critic Agent validates recommendation against base rates |
| **arXiv LLM Bias Study** (ref 33) | Financial LLMs exhibit tech-stock bias, large-cap favoritism, and confirmation bias | `bias_detector.py` checks for tech bias, confirmation bias, recency bias; `conflict_detector.py` flags knowledge conflicts |

---

## Frontend — Pages & Components

### Pages (10 routes)

| Route | File | Description |
|-------|------|-------------|
| `/login` | `login/page.tsx` | **Login**: Google SSO + Passkey authentication, PyFinAgent branding, error handling |
| `/` | `page.tsx` | **Home**: Portfolio snapshot hero (4 metric cards), recent reports table, quick actions (Run Analysis, View Signals, Run Backtest) |
| `/analyze` | `analyze/page.tsx` | **Deep Analysis**: Ticker input → real-time 15-step analysis → Alpha score card, investment thesis, 5-pillar evaluation, stock chart, enrichment signals, debate view, risk dashboard, bias report |
| `/signals` | `signals/page.tsx` | **Signals Explorer**: Enter ticker → view all 12 enrichment signals, consensus bar, sector dashboard, macro dashboard |
| `/reports` | `reports/page.tsx` | **Reports**: Tabbed — History (searchable list with filter chips) + Compare (select 2+ reports → side-by-side price overlay, radar chart, pillar score table, qualitative comparison) |
| `/performance` | `performance/page.tsx` | **Performance**: Historical accuracy metrics |
| `/portfolio` | `portfolio/page.tsx` | **Portfolio**: Position tracking, P&L, allocation pie chart, drawdown, recommendation accuracy scorecard |
| `/paper-trading` | `paper-trading/page.tsx` | **Paper Trading**: Autonomous fund dashboard, NAV chart, positions table, trades history, start/stop/run controls |
| `/backtest` | `backtest/page.tsx` | **Backtest**: Walk-forward ML backtest dashboard with 4 tabs (Results/Equity Curve/Features/Optimizer), data ingestion controls, analytics summary, strategy vs baselines; vertical Jira-style 8-step workflow timeline with window rail, client-side elapsed timer, sample sub-progress, cache hit rate footer; Optimizer tab has Karpathy progress chart (`OptimizerProgressChart`), live experiment polling, best strategy card |
| `/settings` | `settings/page.tsx` | **Settings**: 3-tab sub-navigation (Models & Analysis \| Cost & Weights \| Performance). Performance tab exposes API cache stats, TTL optimizer controls, and per-endpoint latency percentiles |

### Key Components

| Component | Purpose |
|-----------|---------|
| `DebateView.tsx` | Bull vs Bear argument cards side-by-side, contradiction highlights (red), confidence-weighted consensus bar, individual agent dissent badges |
| `RiskDashboard.tsx` | Monte Carlo fan chart (percentile bands over 1Y), VaR gauge (95%/99%), anomaly alert cards (red for risk, green for opportunity) |
| `SentimentDetail.tsx` | Contextual keyword cloud (embedding-weighted), sentiment time-series (30d), source breakdown chart |
| `BiasReport.tsx` | Bias flag cards with severity indicators, knowledge conflict table (LLM belief vs actual data), raw vs bias-adjusted score |
| `AuthProvider.tsx` | SessionProvider wrapper with 15-minute refetch interval |
| `SignalCards.tsx` | 12-card grid with color-coded badges (green=bullish, red=bearish, amber=neutral, gray=error). SignalSummaryBar shows consensus distribution |
| `SectorDashboard.tsx` | Sector rotation bar chart (11 SPDR ETFs by 3M return), relative performance table (stock/sector/SPY × 4 periods), peer comparison table |
| `MacroDashboard.tsx` | 7-indicator grid (current value + change), macro warnings section |
| `StockChart.tsx` | Candlestick + volume with toggleable SMA50/SMA200/RSI, 5 period options (1M–2Y) |
| `EvaluationTable.tsx` | 5-pillar horizontal bars with weights, individual pillar progress indicators |
| `AnalysisProgress.tsx` | 15-step real-time tracker with % bar, current-step spinner, Phosphor icon status indicators |
| `CostDashboard.tsx` | LLM cost analytics: 4 summary cards (total cost, tokens, calls, deep think), token distribution bar, cost by model, per-agent breakdown table |
| `Skeleton.tsx` | Reusable loading skeleton components: `SkeletonPulse` (atomic), `SkeletonCard` (card-sized), `SkeletonGrid` (N-card grid), `PageSkeleton` (full page: metric grid + content area) |
| `PerfProgressChart.tsx` | Autoresearch-style optimization progress chart (Recharts ComposedChart): kept/discarded scatter dots, running-best step line, text annotations on kept dots (karpathy/autoresearch style), hover tooltip with experiment details, click-to-expand changelog panel |
| `OptimizerProgressChart.tsx` | Karpathy-style Sharpe ratio progress chart for backtest optimizer: green running-best step line, kept/discarded/DSR-rejected dots, smart Y-axis scaling, staggered labels, clamped outlier triangles |
| `Sidebar.tsx` | Navigation sidebar with Phosphor icons + user auth UI (avatar, email, passkey registration, logout) |

---

## Environment & Setup

### Prerequisites

*   Node.js 18+
*   Python 3.12+ (workspace is standardized on `.venv312`)
*   GCP project with Vertex AI and BigQuery enabled
*   Application Default Credentials configured (`gcloud auth application-default login`)
*   Google OAuth Client ID (for SSO authentication)
*   Slack App with Socket Mode enabled (optional, for Slack bot)

### Quick Start

```bash
# Backend
./.venv312/Scripts/python.exe -m pip install -r backend/requirements.txt
./.venv312/Scripts/python.exe -m uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npx prisma migrate dev   # Initialize auth SQLite DB
npm run dev  # port 3000

# Docker (all services)
docker compose up --build
```

**Workspace note**: `.venv312` is the canonical backend environment for this repo and should be used for Pylance, debugging, and FastAPI startup.

**VS Code launch configs**: `.vscode/launch.json` includes `Backend: FastAPI (.venv312)`, `Frontend: Next.js`, and a compound `Full Stack: Backend + Frontend` launch so local development can be started from a consistent workspace configuration.

### Required Environment Variables

Backend uses `.env` file in `backend/` directory:

```env
# GCP
GCP_PROJECT_ID=<your-gcp-project>
GCP_LOCATION=us-central1
GEMINI_MODEL=gemini-2.0-flash
RAG_DATA_STORE_ID=<vertex-ai-search-datastore-id>

# Cloud Function URLs
INGESTION_AGENT_URL=<cloud-function-url>
QUANT_AGENT_URL=<cloud-function-url>

# API Keys
ALPHAVANTAGE_API_KEY=<key>
FRED_API_KEY=<key>
API_NINJAS_KEY=<key>

# Authentication (must match frontend/.env.local)
AUTH_SECRET=<base64-secret>  # openssl rand -base64 32
ALLOWED_EMAILS=user@example.com  # Comma-separated email whitelist

# Slack Bot (Socket Mode)
SLACK_BOT_TOKEN=xoxb-...     # Bot User OAuth Token
SLACK_APP_TOKEN=xapp-...     # App-Level Token (Socket Mode)
SLACK_CHANNEL_ID=C0123...    # Channel for alerts/digests
MORNING_DIGEST_HOUR=8        # Hour (0-23) for daily portfolio digest

# Optional
SLACK_WEBHOOK_URL=<optional>
USE_CELERY=false
MAX_DEBATE_ROUNDS=2          # Bull↔Bear rounds (1-5, default 2)
MAX_RISK_DEBATE_ROUNDS=1     # Risk analyst rounds (1-3, default 1)
DEEP_THINK_MODEL=            # Optional deep-think model (e.g., gemini-2.5-pro)
LITE_MODE=false              # Skip deep dive, devil's advocate, risk assessment; 1 debate round
MAX_ANALYSIS_COST_USD=0.50   # Per-analysis budget cap (warning logged when exceeded)
MAX_SYNTHESIS_ITERATIONS=2   # Reflection loop iterations (1-3, default 2)

# Paper Trading (Autonomous Fund)
PAPER_TRADING_ENABLED=false  # Enable autonomous paper trading loop
PAPER_STARTING_CAPITAL=10000.0
PAPER_MAX_POSITIONS=10
PAPER_MIN_CASH_RESERVE_PCT=5.0
PAPER_SCREEN_TOP_N=10        # Top candidates from quant screener
PAPER_ANALYZE_TOP_N=5        # Candidates to run full analysis on
PAPER_TRADING_HOUR=10        # Hour (0-23) for daily trading cycle
PAPER_REEVAL_FREQUENCY_DAYS=3  # Days between holding re-evaluation
PAPER_TRANSACTION_COST_PCT=0.1 # Simulated transaction cost (%)
PAPER_MAX_DAILY_COST_USD=2.0   # Max LLM spend per daily cycle
```

Frontend uses `.env.local` file in `frontend/` directory:

```env
AUTH_SECRET=<same-as-backend> # Must match backend AUTH_SECRET
AUTH_GOOGLE_ID=<google-oauth-client-id>
AUTH_GOOGLE_SECRET=<google-oauth-client-secret>
ALLOWED_EMAILS=user@example.com
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

---

## API Endpoints — Complete Reference

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analysis/` | Start analysis for a ticker. Returns `{ task_id }` |
| `GET` | `/api/analysis/{id}` | Poll analysis status. Returns `{ status, progress, result }` |

### Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/reports/` | List all past reports |
| `GET` | `/api/reports/{ticker}` | Get latest report for ticker |
| `GET` | `/api/reports/performance` | Performance statistics |
| `GET` | `/api/reports/cost-history` | LLM cost/token history per analysis |
| `GET` | `/api/reports/latest-cost-summary` | Per-agent cost breakdown from most recent analysis (for live cost estimator) |

### Charts

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/charts/{ticker}` | Price chart data (OHLCV) |
| `GET` | `/api/charts/{ticker}/financials` | Financial fundamentals |

### Signals (12 routes)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/signals/{ticker}` | All 12 enrichment signals in parallel |
| `GET` | `/api/signals/{ticker}/insider` | SEC Form 4 insider trading data |
| `GET` | `/api/signals/{ticker}/options` | Options flow analysis |
| `GET` | `/api/signals/{ticker}/sentiment` | Social/news sentiment |
| `GET` | `/api/signals/{ticker}/patents` | USPTO patent data |
| `GET` | `/api/signals/{ticker}/earnings-tone` | Earnings call transcript tone |
| `GET` | `/api/signals/{ticker}/alt-data` | Google Trends alternative data |
| `GET` | `/api/signals/{ticker}/sector` | Sector rotation + relative strength |
| `GET` | `/api/signals/{ticker}/nlp-sentiment` | Transformer NLP sentiment |
| `GET` | `/api/signals/{ticker}/anomalies` | Multi-dimensional anomaly scan |
| `GET` | `/api/signals/{ticker}/quant-model` | MDA-weighted quant model signal |
| `GET` | `/api/signals/macro/indicators` | FRED macro indicators |

### Portfolio

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/portfolio/` | List all portfolio positions |
| `POST` | `/api/portfolio/` | Add a new position |
| `DELETE` | `/api/portfolio/{id}` | Remove a position |
| `GET` | `/api/portfolio/performance` | P&L and recommendation accuracy |

### Settings

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/settings/` | All configurable settings (models, debate, weights, cost controls) |
| `PUT` | `/api/settings/` | Update any combination of settings (partial update, validates pillar weight sum) |
| `GET` | `/api/settings/models` | Current model configuration (legacy, backward-compatible) |
| `GET` | `/api/settings/models/available` | List of available Gemini models with pricing |

### Investigation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/investigate` | Run RAG investigation query against 10-K documents |

### Paper Trading

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/paper-trading/start` | Initialize fund + start daily scheduler |
| `POST` | `/api/paper-trading/stop` | Pause the daily scheduler |
| `GET` | `/api/paper-trading/status` | NAV, P&L, scheduler state, loop status |
| `GET` | `/api/paper-trading/portfolio` | All open positions with unrealized P&L |
| `GET` | `/api/paper-trading/trades` | Trade history (buys/sells with reasons) |
| `GET` | `/api/paper-trading/snapshots` | Daily NAV history for charting |
| `GET` | `/api/paper-trading/performance` | Sharpe ratio, win rate, alpha, cumulative costs |
| `POST` | `/api/paper-trading/run-now` | Trigger manual daily cycle (async) |

### Skills Optimization

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/skills/optimize` | Start autonomous skill optimization loop (background) |
| `POST` | `/api/skills/stop` | Stop the optimization loop gracefully |
| `GET` | `/api/skills/status` | Current loop status (running, experiments count, keep rate) |
| `GET` | `/api/skills/experiments` | Full experiment history from skill_results.tsv |
| `GET` | `/api/skills/analysis` | Summary stats, keep rates, delta chain, top hits |
| `GET` | `/api/skills/agents` | List all optimizable agents with skill file status |
| `GET` | `/api/skills/{agent_name}` | View an agent's current skills.md content |

### Backtest

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/backtest/run` | Start walk-forward backtest (async background task) |
| `GET` | `/api/backtest/status` | Poll backtest progress (status, run_id, progress) |
| `GET` | `/api/backtest/results` | Full backtest results with per-window analytics |
| `GET` | `/api/backtest/results/{window_id}` | Per-window detail (trades, predictions, feature importance) |
| `GET` | `/api/backtest/runs` | List all persisted backtest runs (newest first) |
| `GET` | `/api/backtest/runs/{run_id}` | Load a specific historical backtest result |
| `DELETE` | `/api/backtest/runs/{run_id}` | Delete a specific backtest result from disk |
| `POST` | `/api/backtest/ingest` | Ingest historical price, fundamental, and macro data into BigQuery |
| `GET` | `/api/backtest/ingest/status` | Row counts for historical data tables |
| `POST` | `/api/backtest/optimize` | Start quant strategy optimization loop (background) |
| `POST` | `/api/backtest/optimize/stop` | Stop the optimization loop gracefully |
| `GET` | `/api/backtest/optimize/status` | Current optimizer state (iterations, best Sharpe, kept/discarded) |
| `GET` | `/api/backtest/optimize/experiments` | Full experiment history from quant_results.tsv |
| `GET` | `/api/backtest/optimize/best` | Best strategy params + feature importance |
| `GET` | `/api/backtest/optimize/insights` | Optimizer insights (param bounds, experiments with full params, data scope) |

### Performance

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/perf/summary` | Per-endpoint latency percentiles (p50/p95/p99) |
| `GET` | `/api/perf/slow` | Slowest endpoints above configurable threshold |
| `GET` | `/api/perf/cache` | Cache hit/miss stats, entry count, hit rate |
| `POST` | `/api/perf/cache/clear` | Flush all cached responses |
| `POST` | `/api/perf/optimize` | Start autoresearch TTL optimizer loop (background) |
| `POST` | `/api/perf/optimize/stop` | Stop the optimizer loop gracefully |
| `GET` | `/api/perf/optimize/status` | Current optimizer state (iterations, kept/discarded) |
| `GET` | `/api/perf/optimize/experiments` | Full experiment history from perf_results.tsv |

---

## Key Conventions

### Agent Communication
All communication and data handoffs between agents **MUST** be in structured JSON format. This ensures predictable and parsable inputs/outputs. Refer to agent-specific prompt functions in `backend/config/prompts.py` for exact schemas.

### Glass Box Philosophy
The frontend is intentionally designed as a "Glass Box." When modifying the app, prioritize transparency:
*   Every agent's input and output must be visible to the user
*   The debate view shows bull/bear arguments and contradictions
*   Decision traces provide full audit trail for every recommendation
*   Bias flags are surfaced prominently, not hidden

### Signal Format
All enrichment tools return a consistent structure:
```json
{ "ticker": "AAPL", "signal": "BULLISH", "summary": "...", "data": { ... } }
```
Where `signal` is one of the tool-specific values (e.g., BULLISH, BEARISH, NEUTRAL, RISING, DECLINING, INNOVATION_BREAKOUT, ANOMALY_RISK, etc.).

### Decision Trace Convention
Every LLM agent call must produce a `DecisionTrace` (defined in `backend/agents/trace.py`) capturing:
*   Agent name and timestamp
*   Input data hash (for reproducibility)
*   Output signal and confidence
*   Evidence citations (specific data points)
*   Reasoning steps (chain of thought)

### Debate Protocol
The debate framework implements a multi-round adversarial structure (TradingAgents pattern):
1. **Round 1..N (Bull↔Bear Iterative Rebuttal)**: Bull and Bear agents alternate arguments. From round 2 onward, each agent receives the opponent's prior argument and must directly rebut it while strengthening their own case. Default: 2 rounds (configurable via `MAX_DEBATE_ROUNDS`, range 1-5).
2. **Devil's Advocate Round**: After debate rounds complete, a Devil's Advocate agent stress-tests both sides — identifying hidden risks, groupthink patterns, weaknesses in bull/bear arguments, and proposing a confidence adjustment.
3. **Moderation Round**: Moderator Agent receives the full debate history (all rounds), Devil's Advocate challenges, and resolves contradictions using evidence weight. Assigns final consensus with confidence score and registers any unresolved dissents.

Bull, Bear, and Moderator agents all receive **past_memory** from `FinancialSituationMemory` (BM25-retrieved lessons from prior analyses with similar market situations).

### Risk Assessment Protocol (Step 12c)
The risk assessment follows a TradingAgents-style **multi-round** round-robin debate (configurable via `MAX_RISK_DEBATE_ROUNDS`, default 1, range 1-3):
1. **Round 1..N (Aggressive → Conservative → Neutral)**: Each analyst argues their position. From round 2 onward, each analyst sees the other two's prior arguments and must directly rebut them.
2. **Risk Judge**: Reviews all analyst arguments (full debate history when multi-round) and delivers final verdict with risk-adjusted confidence, recommended position size (%), risk level, and specific risk limits (stop loss, max drawdown)

All risk analysts and the Risk Judge receive **past_memory** from `FinancialSituationMemory` (BM25-retrieved lessons from prior analyses with similar market situations).

### Skills System (Autoresearch Pattern)
Agent prompts are defined in skills.md files under `backend/agents/skills/`. Each file follows a standard template (SKILL_TEMPLATE.md) with sections: Goal, Identity, CAN/CANNOT boundaries, Data Inputs, Skills & Techniques, Anti-Patterns, Research Foundations, Evaluation Criteria, Output Format, Prompt Template, and Experiment Log.

*   `prompts.py` loads the `## Prompt Template` section from each skills.md via `load_skill()` and injects runtime variables via `format_skill()` using `{{variable}}` placeholders
*   Skill cache is keyed by file modification time — editing a skills.md file automatically picks up changes on next prompt call
*   `SkillOptimizer` (backend/agents/skill_optimizer.py) implements an autonomous experiment loop: establish baseline → propose LLM-generated modification to Prompt Template → apply → measure metric → keep/discard/crash → repeat
*   The single optimization metric is `scalar = risk_adjusted_return × (1 − min(0.3, turnover_ratio × tx_cost_pct))` where `risk_adjusted_return = avg(return_pct) × beat_benchmark_rate`, computed by `perf_metrics.get_scalar_metric()`. Transaction cost penalty prevents churn alpha.
*   Experiment results are logged to `backend/agents/skills/experiments/skill_results.tsv`
*   **Fixed harness** (UNTOUCHABLE by optimizer): data tools, orchestrator pipeline, output JSON schemas, BQ schema, evaluation formula, function signatures
*   **Modifiable** (the "train.py" equivalent): `## Prompt Template`, `## Skills & Techniques`, `## Anti-Patterns` sections of each skills.md

### Cost Management
The system implements multi-layered LLM cost controls:

**Output Token Limits** (per agent type):
| Agent Type | Max Output Tokens | Rationale |
|-----------|-------------------|----------|
| Enrichment agents (11) | 1,024 | Structured signal output, no prose needed |
| Debate agents (Bull/Bear/DA) | 1,536 | Argument + evidence citations |
| Moderator | 2,048 | Full consensus with contradiction map |
| Synthesis | 4,096 | Complete structured JSON report |
| Deep think (Critic) | 2,048 | Structured verdict + issues |
| Risk analysts (3) | 1,024 | Position sizing argument |
| Risk Judge | 1,536 | Final verdict + risk limits |

**Lite Mode** (`LITE_MODE=true`):
*   Skips Step 10 (Deep Dive Agent)
*   Skips Devil's Advocate in debate
*   Skips Step 12c (Risk Assessment Team)
*   Forces 1 debate round regardless of `MAX_DEBATE_ROUNDS`
*   Reduces LLM calls from ~39 → ~20 per analysis

**Budget Warning**: After analysis completes, `CostTracker.check_budget(max_analysis_cost_usd)` logs a warning if the run exceeded the configured budget. The `budget_warning` field is included in the final report JSON.

**Prompt Truncation**: In `run_synthesis_pipeline()`, each enrichment section is capped at 1,500 chars and the total market context at 12,000 chars to prevent unbounded input token costs.

**Configurable Synthesis Iterations**: `MAX_SYNTHESIS_ITERATIONS` (1-3, default 2) controls how many Synthesis↔Critic reflection loops are allowed, trading quality for cost.

### Info-Gap Detection Protocol (Step 6b)
AlphaQuanter-style ReAct loop for data quality assurance:
1. **Scan**: Assess all 12 enrichment sources against sector-specific criticality thresholds
2. **Classify**: Each source rated as SUFFICIENT (data present + meaningful), PARTIAL (data present but incomplete), or MISSING (no data or error)
3. **Retry**: Critical gaps (high-criticality sources with MISSING status) are retried up to 2 times
4. **Score**: Compute overall data quality score (0-100) and flag if recommendation is at risk

---

## Data Persistence & ML Training

### BigQuery Schema — `analysis_results` Table (88 columns)

Every analysis run persists an **88-column row** to BigQuery (`financial_reports.analysis_results`). The schema is designed for ML model training — all quantitative features that drive the recommendation are stored as first-class columns (not buried in JSON), enabling direct `SELECT`-based training queries and BQ ML `CREATE MODEL` syntax.

The schema is managed by `migrate_bq_schema.py` (idempotent — safe to re-run).

| Category | Columns | Description |
|----------|---------|-------------|
| **Identity** | `ticker`, `company_name`, `analysis_date` | Primary key fields |
| **Output** | `final_score`, `recommendation`, `summary`, `recommendation_justification` | LLM recommendation outputs |
| **Pillar Scores** | `pillar_1_corporate`, `pillar_2_industry`, `pillar_3_valuation`, `pillar_4_sentiment`, `pillar_5_governance` | 5-pillar weighted scoring matrix (0-10 each) |
| **Financial Fundamentals** | `price_at_analysis`, `market_cap`, `pe_ratio`, `peg_ratio`, `debt_equity`, `sector`, `industry` | Point-in-time market data from yfinance |
| **Risk Metrics** | `annualized_volatility`, `var_95_6m`, `var_99_6m`, `expected_shortfall_6m`, `prob_positive_6m`, `anomaly_count` | Monte Carlo VaR + anomaly detection outputs |
| **Debate Dynamics** | `debate_consensus`, `debate_confidence`, `bull_confidence`, `bear_confidence`, `bull_thesis`, `bear_thesis`, `contradiction_count`, `dissent_count`, `debate_rounds_count`, `devils_advocate_challenges` | Multi-agent debate meta-features including multi-round + DA metrics |
| **Enrichment Signals** | `insider_signal`, `options_signal`, `social_sentiment_score`, `nlp_sentiment_score`, `patent_signal`, `earnings_confidence`, `sector_signal`, `enrichment_signals_summary` | Individual signal values + compact JSON summary |
| **Validation** | `recommendation_confidence`, `critic_review`, `bias_flags`, `bias_count`, `bias_adjusted_score`, `conflict_count`, `overall_reliability`, `decision_trace_count`, `key_risks` | Bias audit, conflict detection, XAI completeness |
| **Info-Gap Quality** | `info_gap_count`, `info_gap_resolved_count`, `data_quality_score` | AlphaQuanter-style data completeness tracking |
| **Risk Assessment** | `risk_judge_decision`, `risk_adjusted_confidence`, `aggressive_analyst_confidence`, `conservative_analyst_confidence` | TradingAgents-style risk debate verdicts |
| **Macro Context** | `fed_funds_rate`, `cpi_yoy`, `unemployment_rate`, `yield_curve_spread` | Point-in-time FRED macro indicators for regime-aware ML |
| **Cost Metrics** | `total_tokens`, `total_cost_usd`, `deep_think_calls` | Per-analysis LLM cost tracking for dual-model architecture |
| **Reflection Loop** | `synthesis_iterations` | Tracks Evaluator-Optimizer reflection loop count per analysis |
| **Model Tracking** | `standard_model`, `deep_think_model` | Which LLM models were used for this analysis run |
| **Autoresearch Bridge** | `consumer_sentiment`, `revenue_growth_yoy`, `quality_score`, `momentum_6m`, `rsi_14` | FEATURE_TO_AGENT bridge features — enables MDA→Agent targeting in SkillOpt |
| **Enrichment Signal Parity** | `alt_data_signal`, `alt_data_momentum_pct`, `anomaly_signal`, `monte_carlo_signal`, `quant_model_signal`, `quant_model_score`, `social_sentiment_velocity`, `nlp_sentiment_confidence` | Complete enrichment tool signal values for ML training |
| **Risk Assessment Parity** | `risk_level`, `recommended_position_pct`, `neutral_analyst_confidence`, `risk_debate_rounds_count` | Full TradingAgents risk debate outputs for position sizing ML |
| **Debate Parity** | `groupthink_flag`, `da_confidence_adjustment` | Devil's Advocate stress-test results |
| **Cost Parity** | `grounded_calls` | Google Search Grounding call count per analysis |
| **Full Report** | `full_report_json` | Complete report dict (JSON) — fallback for feature engineering |

### Research Rationale for Schema Design

| Research Source | Schema Design Impact |
|----------------|---------------------|
| **Goldman Sachs** (127-dim anomaly detection) | `var_95_6m`, `var_99_6m`, `annualized_volatility`, `anomaly_count` as first-class columns |
| **Stanford** (transformer embeddings) | `nlp_sentiment_score` (numeric 0-1), not just categorical labels |
| **TradingAgents** (multi-agent debate) | `bull_confidence`, `bear_confidence`, `contradiction_count`, `dissent_count` as meta-features |
| **TradingAgents** (risk assessment team) | `risk_judge_decision`, `risk_adjusted_confidence`, `aggressive_analyst_confidence`, `conservative_analyst_confidence` for position sizing ML |
| **AlphaQuanter** (ReAct info-gap) | `info_gap_count`, `info_gap_resolved_count`, `data_quality_score` for data quality-aware training |
| **arXiv LLM Bias Study** | `sector`, `market_cap`, `bias_count`, `bias_adjusted_score` for bias correction training |
| **BlackRock** (regime-aware models) | `fed_funds_rate`, `cpi_yoy`, `unemployment_rate`, `yield_curve_spread` for macro context |

### Outcome Tracking Table — `outcome_tracking`

A separate table tracks actual price performance after each recommendation, enabling supervised learning:

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | STRING | Stock symbol |
| `analysis_date` | STRING | Links back to `analysis_results` |
| `recommendation` | STRING | Original recommendation |
| `price_at_recommendation` | FLOAT | Price when rec was made |
| `current_price` | FLOAT | Price at evaluation time |
| `return_pct` | FLOAT | Actual return (%) |
| `holding_days` | INT | Days since recommendation |
| `beat_benchmark` | BOOL | Did it beat SPY? |
| `evaluated_at` | STRING | Evaluation timestamp |

### Agent Memories Table — `agent_memories`

Stores learned lessons from past outcomes, used by `FinancialSituationMemory` (BM25 retrieval). Managed by `migrate_agent_memories.py` (idempotent).

| Column | Type | Description |
|--------|------|-------------|
| `agent_type` | STRING | Agent that generated the lesson (bull, bear, moderator, risk_judge) |
| `ticker` | STRING | Stock symbol the lesson relates to |
| `situation` | STRING | Textual description of the market situation |
| `lesson` | STRING | LLM-generated lesson from outcome evaluation |
| `created_at` | TIMESTAMP | When the memory was created |

### Paper Trading Tables (4 tables)

Managed by `migrate_paper_trading.py` (idempotent). Stores virtual fund state, positions, trades, and daily snapshots.

#### `paper_portfolio` — Fund State

| Column | Type | Description |
|--------|------|-------------|
| `portfolio_id` | STRING | Always "default" (single fund) |
| `starting_capital` | FLOAT64 | Initial capital (default $10,000) |
| `current_cash` | FLOAT64 | Available cash |
| `total_nav` | FLOAT64 | Cash + sum(position market values) |
| `total_pnl_pct` | FLOAT64 | ((NAV - starting_capital) / starting_capital) * 100 |
| `benchmark_return_pct` | FLOAT64 | SPY return over same period |
| `created_at` | TIMESTAMP | Fund inception date |
| `updated_at` | TIMESTAMP | Last update |

#### `paper_positions` — Open Positions

| Column | Type | Description |
|--------|------|-------------|
| `position_id` | STRING | UUID |
| `ticker` | STRING | Stock symbol |
| `quantity` | FLOAT64 | Shares held (fractional OK) |
| `avg_entry_price` | FLOAT64 | Weighted average cost |
| `cost_basis` | FLOAT64 | quantity × avg_entry_price |
| `current_price` | FLOAT64 | Last known price |
| `market_value` | FLOAT64 | quantity × current_price |
| `unrealized_pnl` | FLOAT64 | market_value - cost_basis |
| `unrealized_pnl_pct` | FLOAT64 | % gain/loss |
| `entry_date` | STRING | When first bought |
| `last_analysis_date` | STRING | Last analysis that confirmed hold |
| `recommendation` | STRING | Latest recommendation |
| `risk_judge_position_pct` | FLOAT64 | Risk Judge's recommended allocation % |
| `stop_loss_price` | FLOAT64 | Risk Judge's stop loss |

#### `paper_trades` — Trade History

| Column | Type | Description |
|--------|------|-------------|
| `trade_id` | STRING | UUID |
| `ticker` | STRING | Stock symbol |
| `action` | STRING | BUY / SELL |
| `quantity` | FLOAT64 | Shares traded |
| `price` | FLOAT64 | Execution price (market close) |
| `total_value` | FLOAT64 | quantity × price |
| `reason` | STRING | "new_buy_signal", "stop_loss", "signal_flip", "rebalance" |
| `analysis_id` | STRING | Links to analysis_results.analysis_date |
| `risk_judge_decision` | STRING | Risk Judge verdict at time of trade |
| `pnl_pct` | FLOAT64 | Realized P&L % (sells only) |
| `created_at` | TIMESTAMP | Trade timestamp |

#### `paper_portfolio_snapshots` — Daily NAV History

| Column | Type | Description |
|--------|------|-------------|
| `snapshot_date` | STRING | YYYY-MM-DD |
| `total_nav` | FLOAT64 | Portfolio NAV |
| `cash` | FLOAT64 | Available cash |
| `positions_value` | FLOAT64 | Sum of all position market values |
| `daily_pnl_pct` | FLOAT64 | Day-over-day change |
| `cumulative_pnl_pct` | FLOAT64 | Since inception |
| `benchmark_pnl_pct` | FLOAT64 | SPY cumulative return |
| `alpha_pct` | FLOAT64 | cumulative_pnl - benchmark |
| `position_count` | INT64 | Number of open positions |
| `trades_today` | INT64 | Trades executed today |
| `analysis_cost_today` | FLOAT64 | LLM cost for today's analyses |

### Backtest Data Tables (3 tables)

Managed by `migrate_backtest_data.py` (idempotent). Stores historical price, fundamental, and macro data downloaded once from yfinance and FRED for walk-forward backtesting.

#### `historical_prices` — OHLCV Price History

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | STRING | Stock symbol |
| `date` | STRING | Trading date (YYYY-MM-DD) |
| `open` | FLOAT64 | Open price |
| `high` | FLOAT64 | High price |
| `low` | FLOAT64 | Low price |
| `close` | FLOAT64 | Close price (adjusted) |
| `volume` | INT64 | Trading volume |
| `ingested_at` | TIMESTAMP | When the row was ingested |

#### `historical_fundamentals` — Quarterly Financial Statements

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | STRING | Stock symbol |
| `report_date` | STRING | Fiscal quarter end date |
| `filing_date` | STRING | SEC filing date (point-in-time) |
| `total_revenue` | FLOAT64 | Quarterly revenue |
| `net_income` | FLOAT64 | Net income |
| `total_debt` | FLOAT64 | Total debt |
| `total_equity` | FLOAT64 | Shareholders' equity |
| `total_assets` | FLOAT64 | Total assets |
| `operating_cash_flow` | FLOAT64 | Cash from operations |
| `shares_outstanding` | FLOAT64 | Diluted shares outstanding |
| `dividends_per_share` | FLOAT64 | Quarterly dividends per share (from cash flow / shares) |
| `sector` | STRING | GICS sector |
| `industry` | STRING | GICS industry |
| `ingested_at` | TIMESTAMP | When the row was ingested |

#### `historical_macro` — FRED Macro Indicators

| Column | Type | Description |
|--------|------|-------------|
| `series_id` | STRING | FRED series ID (e.g., FEDFUNDS, CPIAUCSL) |
| `date` | STRING | Observation date (YYYY-MM-DD) |
| `value` | FLOAT64 | Indicator value |
| `ingested_at` | TIMESTAMP | When the row was ingested |

### ML Training Query Pattern

```sql
-- Join analysis features with outcome labels for supervised learning
SELECT
  a.ticker, a.sector, a.pe_ratio, a.peg_ratio, a.debt_equity,
  a.var_95_6m, a.annualized_volatility, a.prob_positive_6m,
  a.bull_confidence, a.bear_confidence, a.contradiction_count,
  a.nlp_sentiment_score, a.social_sentiment_score,
  a.fed_funds_rate, a.yield_curve_spread,
  a.final_score, a.recommendation,
  o.return_pct, o.beat_benchmark  -- labels
FROM `financial_reports.analysis_results` a
JOIN `financial_reports.outcome_tracking` o
  ON a.ticker = o.ticker AND a.analysis_date = o.analysis_date
WHERE a.price_at_analysis IS NOT NULL
```

---

## Deployment & Testing

### Local Development

```bash
# Backend (with auto-reload)
./.venv312/Scripts/python.exe -m uvicorn backend.main:app --reload --port 8000

# Frontend (with hot-reload)
cd frontend && npm run dev

# Build verification
cd frontend && npm run build  # must produce 0 TypeScript errors
```

### Cloud Function Deployment

```bash
./deploy_agents.sh  # Deploys all GCP Cloud Functions
```

### Frontend Deployment

```bash
./pyfinagent-app/deploy.sh  # Deploy to Cloud Run
```

---

## Security

*   **Authentication**: NextAuth.js v5 with dual authentication (Google SSO + Passkey/WebAuthn). Frontend middleware redirects unauthenticated users to `/login`. Backend validates JWE tokens via HKDF key derivation + AES-256-CBC decryption. Email whitelist enforced on both frontend (NextAuth callback) and backend (`ALLOWED_EMAILS`).
    *   **Frontend**: `auth.config.ts` (Edge-compatible, no Prisma) for middleware + `auth.ts` (PrismaAdapter + WebAuthn) for route handler. JWT strategy with 8h maxAge.
    *   **Backend**: `auth.py` decrypts NextAuth JWE using `HKDF(AUTH_SECRET, info=b"Auth.js Generated Encryption Key", salt=b"", length=64)` → A256CBC-HS512/dir. Auth middleware skips `/api/health`, `/api/auth`, `/docs`, `/openapi.json`, `/redoc`.
*   **OWASP Security Headers**: All responses include `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`, `Referrer-Policy: strict-origin-when-cross-origin`, `Cache-Control: no-store`, `Permissions-Policy` (restricted).
*   **CORS**: Allows `localhost:*` and Tailscale IPs (`100.x.y.z:*`) via regex pattern.
*   **SEC EDGAR User-Agent**: When making requests to the SEC EDGAR database, you **MUST** declare a custom User-Agent string in the format `FirstName LastName email@domain.com`. Failure to do so will result in the IP address being blocked by the SEC. This is managed via secrets.
*   **Secret Management**:
    *   **Local**: Use the `backend/.env` file and `frontend/.env.local` for local development.
    *   **Production**: All production secrets (API keys, service account keys) are managed using **Google Cloud Secret Manager**. Do not hardcode secrets in the source code.
*   **Input Validation**: All API endpoints validate input parameters. Ticker symbols are sanitized. No raw user input is passed directly to LLM prompts without sanitization.
*   **Rate Limiting**: External API calls (Alpha Vantage, FRED, SEC EDGAR, USPTO) respect rate limits with automatic retry and exponential backoff.

---

