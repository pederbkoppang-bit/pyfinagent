# AGENTS.md for PyFinAgent

This document provides essential context and instructions for AI coding agents to effectively understand, operate, and contribute to the PyFinAgent project.

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
│   │   └── skills/              # Agent skills.md files (28 agents) + experiments/
│   ├── config/
│   │   ├── settings.py          # Pydantic settings (env vars)
│   │   └── prompts.py           # Skill-loaded prompt wrappers (loads from skills/*.md)
│   ├── api/
│   │   ├── analysis.py          # POST /api/analysis/, GET /api/analysis/{id}
│   │   ├── auth.py              # HKDF + JWE token decrypt, email whitelist
│   │   ├── reports.py           # Reports CRUD + performance stats
│   │   ├── charts.py            # Price chart + financials endpoints
│   │   ├── signals.py           # Enrichment signals endpoints (11 routes)
│   │   ├── portfolio.py         # Portfolio tracking CRUD
│   │   ├── paper_trading.py     # Autonomous paper trading endpoints (8 routes)
│   │   ├── settings_api.py      # Model configuration + available models
│   │   └── skills.py            # Skills optimization API endpoints
│   ├── db/
│   │   ├── bigquery_client.py    # BigQuery report persistence (68-column ML schema) + paper trading CRUD
│   │   └── __init__.py
│   ├── services/
│   │   ├── outcome_tracker.py   # Evaluates past recs vs actual returns (feedback loop)
│   │   ├── paper_trader.py      # Virtual trade execution engine (buy/sell/MTM/snapshot)
│   │   ├── portfolio_manager.py # Trade decision logic (sell-first-then-buy, Risk Judge sizing)
│   │   └── autonomous_loop.py  # Daily autonomous trading cycle (screen→analyze→trade→learn)
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
│   │   └── screener.py          # S&P 500 quant screener (momentum, RSI, composite score)
│   └── main.py                  # FastAPI app + router registration
├── frontend/
│   ├── prisma/
│   │   └── schema.prisma        # SQLite auth DB (User, Account, Session, Authenticator)
│   ├── src/app/
│   │   ├── page.tsx             # Dashboard (main analysis)
│   │   ├── login/page.tsx       # Login page (Google SSO + Passkey)
│   │   ├── api/auth/[...nextauth]/route.ts  # NextAuth catch-all handler
│   │   ├── signals/page.tsx     # Signals exploration
│   │   ├── compare/page.tsx     # Multi-stock comparison
│   │   ├── reports/page.tsx     # Past reports
│   │   ├── performance/page.tsx # Performance stats
│   │   ├── portfolio/page.tsx   # Portfolio tracking + P&L
│   │   ├── paper-trading/page.tsx # Autonomous paper trading dashboard
│   │   └── settings/page.tsx    # Configuration
│   ├── src/components/
│   │   ├── AuthProvider.tsx     # SessionProvider wrapper (15-min refetch)
│   │   ├── DebateView.tsx       # Bull vs Bear debate visualization
│   │   ├── RiskDashboard.tsx    # Monte Carlo fan chart + VaR + anomalies
│   │   ├── SentimentDetail.tsx  # NLP sentiment deep-dive
│   │   ├── BiasReport.tsx       # LLM bias flags + knowledge conflicts
│   │   ├── SignalCards.tsx       # 11-signal enrichment grid + consensus bar
│   │   ├── SectorDashboard.tsx  # Sector rotation + relative strength
│   │   ├── MacroDashboard.tsx   # FRED indicator grid + warnings
│   │   ├── StockChart.tsx       # Candlestick + SMA/RSI chart
│   │   ├── EvaluationTable.tsx  # 5-pillar scoring matrix
│   │   ├── AnalysisProgress.tsx # 15-step progress tracker
│   │   ├── CostDashboard.tsx    # LLM cost/token analytics dashboard
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
├── quant-agent/                 # GCP Cloud Function
├── ingestion_agent/             # GCP Cloud Function
├── earnings-ingestion-agent/    # GCP Cloud Function
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
| 6 | **Data Enrichment** | 11 tools in parallel | Insider trades, options flow, social sentiment, patents, earnings transcripts, FRED macro, Google Trends, sector data, NLP sentiment, anomaly scan, Monte Carlo simulation. **Sector routing** skips irrelevant tools (e.g., patents for Financial Services) |
| 6b | **Info-Gap Detection** | `InfoGapDetector` (ReAct) | AlphaQuanter-style scan: assess sources for SUFFICIENT/PARTIAL/MISSING/SKIPPED status, retry critical failures (max 2 attempts), compute data quality score. SKIPPED tools excluded from quality score |
| 7 | **Enrichment Analysis** | 11 LLM agents | Each tool's data analyzed by a specialized LLM agent |
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
    │   └── monte_carlo.py (1,000 GBM simulations)               │
    │                                                              │
    ├─ Step 6b: Info-Gap Detection (AlphaQuanter ReAct) ──────────┤
    │   ├── Scan 11 sources → SUFFICIENT/PARTIAL/MISSING          │
    │   ├── Retry critical failures (max 2 attempts)              │
    │   └── Compute data quality score                            │
    │                                                              │
    ├─ Step 7: 11 LLM Enrichment Agents (analyze each dataset)   │
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
    ├── enrichment_signals (11 sources)                           │
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

### Pages (8 routes)

| Route | File | Description |
|-------|------|-------------|
| `/login` | `login/page.tsx` | **Login**: Google SSO + Passkey authentication, PyFinAgent branding, error handling |
| `/` | `page.tsx` | **Dashboard**: Ticker input → real-time 15-step analysis → Alpha score card, investment thesis, 5-pillar evaluation, stock chart, enrichment signals, debate view, risk dashboard, bias report |
| `/signals` | `signals/page.tsx` | **Signals Explorer**: Enter ticker → view all 11 enrichment signals, consensus bar, sector dashboard, macro dashboard |
| `/compare` | `compare/page.tsx` | **Compare**: Select 2-5 past reports → side-by-side price overlay, radar chart, pillar score bars |
| `/reports` | `reports/page.tsx` | **Past Reports**: Searchable list of all completed analyses with quick-chip filters |
| `/performance` | `performance/page.tsx` | **Performance**: Historical accuracy metrics |
| `/portfolio` | `portfolio/page.tsx` | **Portfolio**: Position tracking, P&L, allocation pie chart, drawdown, recommendation accuracy scorecard |
| `/paper-trading` | `paper-trading/page.tsx` | **Paper Trading**: Autonomous fund dashboard, NAV chart, positions table, trades history, start/stop/run controls |
| `/settings` | `settings/page.tsx` | **Settings**: Model configuration, pillar weight configuration, debate depth |

### Key Components

| Component | Purpose |
|-----------|---------|
| `DebateView.tsx` | Bull vs Bear argument cards side-by-side, contradiction highlights (red), confidence-weighted consensus bar, individual agent dissent badges |
| `RiskDashboard.tsx` | Monte Carlo fan chart (percentile bands over 1Y), VaR gauge (95%/99%), anomaly alert cards (red for risk, green for opportunity) |
| `SentimentDetail.tsx` | Contextual keyword cloud (embedding-weighted), sentiment time-series (30d), source breakdown chart |
| `BiasReport.tsx` | Bias flag cards with severity indicators, knowledge conflict table (LLM belief vs actual data), raw vs bias-adjusted score |
| `AuthProvider.tsx` | SessionProvider wrapper with 15-minute refetch interval |
| `SignalCards.tsx` | 11-card grid with color-coded badges (green=bullish, red=bearish, amber=neutral, gray=error). SignalSummaryBar shows consensus distribution |
| `SectorDashboard.tsx` | Sector rotation bar chart (11 SPDR ETFs by 3M return), relative performance table (stock/sector/SPY × 4 periods), peer comparison table |
| `MacroDashboard.tsx` | 7-indicator grid (current value + change), macro warnings section |
| `StockChart.tsx` | Candlestick + volume with toggleable SMA50/SMA200/RSI, 5 period options (1M–2Y) |
| `EvaluationTable.tsx` | 5-pillar horizontal bars with weights, individual pillar progress indicators |
| `AnalysisProgress.tsx` | 15-step real-time tracker with % bar, current-step spinner, Phosphor icon status indicators |
| `CostDashboard.tsx` | LLM cost analytics: 4 summary cards (total cost, tokens, calls, deep think), token distribution bar, cost by model, per-agent breakdown table |
| `Sidebar.tsx` | Navigation sidebar with Phosphor icons + user auth UI (avatar, email, passkey registration, logout) |

---

## Environment & Setup

### Prerequisites

*   Node.js 18+
*   Python 3.11+
*   GCP project with Vertex AI and BigQuery enabled
*   Application Default Credentials configured (`gcloud auth application-default login`)
*   Google OAuth Client ID (for SSO authentication)
*   Slack App with Socket Mode enabled (optional, for Slack bot)

### Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npx prisma migrate dev   # Initialize auth SQLite DB
npm run dev  # port 3001

# Docker (all services)
docker compose up --build
```

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

### Signals (11 routes)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/signals/{ticker}` | All 11 enrichment signals in parallel |
| `GET` | `/api/signals/{ticker}/insider` | SEC Form 4 insider trading data |
| `GET` | `/api/signals/{ticker}/options` | Options flow analysis |
| `GET` | `/api/signals/{ticker}/sentiment` | Social/news sentiment |
| `GET` | `/api/signals/{ticker}/patents` | USPTO patent data |
| `GET` | `/api/signals/{ticker}/earnings-tone` | Earnings call transcript tone |
| `GET` | `/api/signals/{ticker}/alt-data` | Google Trends alternative data |
| `GET` | `/api/signals/{ticker}/sector` | Sector rotation + relative strength |
| `GET` | `/api/signals/{ticker}/nlp-sentiment` | Transformer NLP sentiment |
| `GET` | `/api/signals/{ticker}/anomalies` | Multi-dimensional anomaly scan |
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
*   The single optimization metric is `risk_adjusted_return = avg(return_pct) * beat_benchmark_rate` from the outcome_tracking table
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
1. **Scan**: Assess all 11 enrichment sources against sector-specific criticality thresholds
2. **Classify**: Each source rated as SUFFICIENT (data present + meaningful), PARTIAL (data present but incomplete), or MISSING (no data or error)
3. **Retry**: Critical gaps (high-criticality sources with MISSING status) are retried up to 2 times
4. **Score**: Compute overall data quality score (0-100) and flag if recommendation is at risk

---

## Data Persistence & ML Training

### BigQuery Schema — `analysis_results` Table (68 columns)

Every analysis run persists a **68-column row** to BigQuery (`financial_reports.analysis_results`). The schema is designed for ML model training — all quantitative features that drive the recommendation are stored as first-class columns (not buried in JSON), enabling direct `SELECT`-based training queries and BQ ML `CREATE MODEL` syntax.

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
cd backend && uvicorn backend.main:app --reload --port 8000

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

## Upgrade History

### v3.3 — Gemini 2.5 Flash + Extended Thinking: Phase 5 (March 2026)

Upgraded the deep-think model to `gemini-2.5-flash` and introduced opt-in extended thinking (chain-of-thought) for the four judge agents — Critic, Synthesis, Moderator, and Risk Judge. Tiered token budgets are set per agent. The feature is safe-defaulted to `false` so existing deployments on Gemini 2.0 are unaffected.

**New Settings (`backend/config/settings.py`)**:
*   `deep_think_model` default changed from `""` → `"gemini-2.5-flash"`
*   `enable_thinking: bool = False` — opt-in flag; must be `true` to activate thinking mode
*   `thinking_budget_critic: int = 8192` — token budget for Critic Agent reasoning
*   `thinking_budget_moderator: int = 8192` — token budget for Moderator Agent reasoning
*   `thinking_budget_risk_judge: int = 4096` — token budget for Risk Judge reasoning
*   `thinking_budget_synthesis: int = 4096` — token budget for Synthesis Agent reasoning

**New Environment Variables**:
```env
ENABLE_THINKING=false           # Set to true when using gemini-2.5-flash or later
THINKING_BUDGET_CRITIC=8192
THINKING_BUDGET_MODERATOR=8192
THINKING_BUDGET_RISK_JUDGE=4096
THINKING_BUDGET_SYNTHESIS=4096
```

**`backend/agents/orchestrator.py` Changes**:
*   Import: added `GenerationConfig` to vertexai imports
*   4 module-level thinking config dicts: `_THINKING_CRITIC_CONFIG`, `_THINKING_MODERATOR_CONFIG`, `_THINKING_RISK_JUDGE_CONFIG`, `_THINKING_SYNTHESIS_CONFIG` — each merges the existing structured output config with `{"thinking": {"type": "enabled", "budget_tokens": N}, "include_thoughts": True}`
*   `__init__`: stores `self.enable_thinking` and `self.thinking_budgets = {"Critic": N, "Moderator": N, "Risk Judge": N, "Synthesis": N}`
*   `_generate_with_retry()`: new thinking injection block — when `enable_thinking=True` and `is_deep_think=True` and `agent_name in self.thinking_budgets`, merges thinking config into `generation_config` before the API call
*   New `_extract_thoughts()` helper: safely reads `part.thinking` from Vertex AI response candidates; capped at 2000 chars for storage efficiency
*   `run_debate()` call site: added `enable_thinking=self.enable_thinking, thinking_budgets=self.thinking_budgets`
*   `run_risk_debate()` call site: added `enable_thinking=self.enable_thinking, thinking_budgets=self.thinking_budgets`

**`backend/agents/debate.py` Changes**:
*   `_generate_with_retry()`: added `thinking_budget: int = 0` parameter — when > 0, merges `{"thinking": {"type": "enabled", "budget_tokens": thinking_budget}, "include_thoughts": True}` into the config via dict spread
*   `run_debate()` signature: added `enable_thinking: bool = False, thinking_budgets: dict | None = None`
*   Moderator call: computes `_moderator_thinking_budget = (thinking_budgets or {}).get("Moderator", 0) if enable_thinking else 0` and passes it to `_generate_with_retry`

**`backend/agents/risk_debate.py` Changes**:
*   Same pattern as `debate.py`: `thinking_budget` in `_generate_with_retry`, new params on `run_risk_debate()`
*   Risk Judge call: computes `_judge_thinking_budget = (thinking_budgets or {}).get("Risk Judge", 0) if enable_thinking else 0`

**`backend/agents/trace.py` Change**:
*   `DecisionTrace` dataclass: added `thoughts: str = ""` field — stores extended thinking output for Glass Box audit trail rendering

**Tiered Thinking Budgets**:

| Agent | Budget Tokens | Rationale |
|-------|---------------|-----------|
| Critic | 8,192 | Deepest reasoning — Chain-of-Verification requires meticulous claim checking |
| Moderator | 8,192 | Must resolve Bull/Bear contradictions with full debate history |
| Risk Judge | 4,096 | Risk sizing verdict; moderate depth needed |
| Synthesis | 4,096 | Structured JSON output is constrained; budget limits runaway thinking |
| All others | 0 (disabled) | Enrichment agents use tight token limits; thinking cost outweighs benefit |

**Activation Requirement**: `ENABLE_THINKING=true` **requires** `DEEP_THINK_MODEL=gemini-2.5-flash` (or a later Gemini 2.5+ model). Do not enable on `gemini-2.0-flash` — the thinking config will be silently ignored or cause errors.

### v3.2 — Google Search Grounding: Phase 4 (March 2026)

Selective Google Search Grounding on 4 agents for live web fact-checking with source citations. Constraint: Schema + Grounding cannot combine on Gemini 2.0, so grounded agents use separate model instances and produce text responses (not structured output).

**New Model Instance**:
*   `orchestrator.py` — New `self.grounded_model = GenerativeModel(model, tools=[search_tool], generation_config=_gen_config)` using `Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())`

**Grounded Agents (4)**:
*   **Market Agent** (Step 4) — Breaking news supplement via Google Search
*   **Competitor Agent** (Step 5) — Real-time M&A, partnerships, competitive dynamics
*   **Enhanced Macro Agent** (Step 9) — Latest Fed speeches, policy context on top of FRED data
*   **Deep Dive Agent** (Step 10) — Verify claims against public record

**Grounding Metadata Extraction**:
*   New `_extract_grounding_metadata()` helper in `orchestrator.py` — Extracts `groundingChunks` (source URLs + titles) and `groundingSupports` (text→source mapping) from Vertex AI responses
*   All 4 grounded agent methods return `grounding_sources` in their output dict
*   Final report includes `grounding_sources` dict with per-agent metadata for Glass Box rendering

**DecisionTrace Extension (1 file)**:
*   `trace.py` — New `grounding_sources: list[dict]` field on `DecisionTrace` dataclass

**Cost Tracking Extension (1 file)**:
*   `cost_tracker.py` — New `is_grounded: bool` field on `AgentCostEntry`, tracked through `record()` and exposed in `summarize()` output

**Deep Dive Agent Return Type Change**:
*   `run_deep_dive_agent()` now returns `dict` (was `str`) with `text` and `grounding_sources` keys
*   Synthesis pipeline updated to extract `.text` from deep_dive dict

### v3.1 — Anti-Hallucination Stack: Phases 1-3 (March 2026)

Three-phase anti-hallucination system targeting temperature determinism, fact anchoring, and structured output enforcement. Research basis: VeNRA (Typed Fact Ledger), Chain-of-Verification (CoVe), OpenAI Structured Outputs, Google Gemini structured output.

**Phase 1: Temperature Determinism**:
*   All 8 generation configs set to `temperature=0.0, top_k=1` across `orchestrator.py`, `debate.py`, `risk_debate.py`
*   Preserved creative temperatures for `memory.py` (0.3) and `skill_optimizer.py` (0.7, 0.9)

**Phase 2: Fact Ledger + Prompt Hardening (11 files, 28 skills)**:
*   `orchestrator.py` — New `_build_fact_ledger()` builds a 26-field typed fact dict from `quant_data["yf_data"]` after Step 2, stored as `self._fact_ledger_json`
*   `prompts.py` — New `_build_fact_ledger_section()` helper; all 29 prompt functions accept `fact_ledger: str` and inject `{{fact_ledger_section}}` via `format_skill()`
*   28 `backend/agents/skills/*.md` — All updated with `{{fact_ledger_section}}` placeholder + 5 anti-hallucination rules (cite FACT_LEDGER, flag discrepancies, never fabricate, etc.)
*   `SKILL_TEMPLATE.md` — Updated with fact ledger section and anti-hallucination rules
*   All 31 call sites wired: orchestrator agent methods (via `self._fact_ledger_json`), `run_debate()`, `run_risk_debate()`, synthesis pipeline
*   `critic_agent.md` — Enhanced with Chain-of-Verification (CoVe) 3-step loop: extract claims → verify each against FACT_LEDGER → flag mismatches
*   `synthesis_agent.md` — Pillar anchoring: each pillar lists specific FACT_LEDGER fields it must cite; scoring guardrails (score >7 requires PEG <1.5 or P/E <25)

**Phase 3: Gemini Structured Output Enforcement (4 files)**:
*   NEW: `backend/agents/schemas.py` — 10 Pydantic models: `SynthesisReport` (+ `ScoringMatrix`, `Recommendation`), `CriticVerdict` (+ `CriticIssue`), `DevilsAdvocateResult`, `ModeratorConsensus` (+ `Contradiction`, `Dissent`), `RiskAnalystArgument`, `RiskJudgeVerdict` (+ `RiskLimits`)
*   Uses `Literal` type constraints on critical enum fields: `consensus` (5 values), `verdict` (PASS/REVISE), `decision` (4 values), `risk_level` (4 values)
*   `orchestrator.py` — `_generate_with_retry()` extended with `generation_config` parameter; 3 call sites (Synthesis draft, Synthesis revision, Critic) pass `_SYNTHESIS_STRUCTURED_CONFIG` / `_CRITIC_STRUCTURED_CONFIG` with `response_mime_type="application/json"` + `response_schema`
*   `debate.py` — New `_DA_STRUCTURED_CONFIG` and `_MODERATOR_STRUCTURED_CONFIG`; DA and Moderator calls use structured output schemas
*   `risk_debate.py` — New `_RISK_STRUCTURED_CONFIG` and `_JUDGE_STRUCTURED_CONFIG`; all 3 analyst calls + Judge call use structured output schemas
*   Existing `_clean_json_output` / `_parse_json_with_fallback` / `_clean_json` / `_parse_json` retained as safety fallbacks
*   SDK pattern: `{"response_mime_type": "application/json", "response_schema": PydanticModel}` (requires `google-cloud-aiplatform>=1.60.0`)
*   Constraint: Schema + Grounding cannot combine on Gemini 2.0 — structured output agents and grounded agents (Phase 4) use separate model instances

### v3.0 — Design System Overhaul: Geist + Phosphor Icons (March 2026)

Complete frontend design system migration: Geist self-hosted font replacing Google Fonts Inter, Phosphor Icons replacing all emoji icons across 20+ components and 9 pages, Motion animation library installed, and TradingView Lightweight Charts installed for future StockChart rewrite.

**Design System Foundation (4 config files)**:
*   `next.config.js` — Added `experimental.optimizePackageImports: ["@phosphor-icons/react"]` for tree-shaking
*   `layout.tsx` — Geist font loading via `geist/font/sans` and `geist/font/mono`, applied as CSS variable classes on `<html>`
*   `tailwind.config.js` — CSS variable font families (`var(--font-geist-sans/mono)`), navy color palette (`navy-500: "#243352"`, `navy-600: "#1a2744"`), shadow tokens (`card`, `card-hover`), border radius tokens (`card: 12px`, `button: 8px`, `badge: 6px`)
*   `globals.css` — `tabular-nums`, antialiasing, `skeleton` shimmer keyframe in `@layer base`

**Centralized Icon System (2 new files)**:
*   `frontend/src/lib/icons.ts` — ~110 aliased Phosphor re-exports organized by domain: Navigation (8), Pipeline Steps (16), Signals (11), Debate (7), Risk Team (4), Bias/Audit (12), Evaluation Pillars (5), Macro Indicators (8), GlassBox (4), Tabs (6), Utility (20+). All icons use `Icon` type from `@phosphor-icons/react` for TypeScript safety
*   `frontend/src/lib/motion.ts` — Shared motion variants (`fadeIn`, `slideUp`, `staggerContainer`, `staggerItem`) and spring presets (`springSnappy`, `springGentle`, `hoverTap`) for future animation integration

**Emoji → Phosphor Icon Conversion (15 components, 5 pages)**:
*   All emoji characters removed from entire frontend codebase (verified via grep: 0 matches)
*   Icon data types changed from `icon: string` to `icon: Icon` in all component interfaces (`TabDef`, `SIGNAL_META`, `PillarConfig`, `ProbabilityCard`, etc.)
*   Icon rendering changed from `{meta.icon}` string interpolation to `<meta.icon size={20} weight="duotone" />` JSX component rendering
*   Components converted: SignalCards, SignalDashboard, MacroDashboard, BiasReport, DecisionTraceView, DebateView, RiskDashboard, GlassBoxCards, EvaluationTable, CostDashboard, ValuationRange, PdfDownload, StockChart, ReportTabs, ResearchInvestigator
*   Pages converted: page.tsx, signals/page.tsx, compare/page.tsx, performance/page.tsx, reports/page.tsx
*   Previously converted (prior session): Sidebar.tsx, AnalysisProgress.tsx

**Package Changes**:
*   Added: `geist@1.7.0`, `@phosphor-icons/react@2.1.10`, `motion@12.38.0`, `lightweight-charts`
*   Removed: `lucide-react` (fully replaced by Phosphor Icons)

**Icon Naming Convention**: All icons are aliased with domain prefixes for discoverability:
*   Navigation: `Nav*` (NavDashboard, NavSignals, NavReports, …)
*   Pipeline: `Step*` (StepMarketIntel, StepIngestion, StepQuant, …)
*   Signals: `Signal*` (SignalInsider, SignalOptions, SignalSocial, …)
*   Debate: `Debate*` (DebateBull, DebateBear, DebateConsensus, …)
*   Risk: `Risk*` (RiskAggressive, RiskConservative, RiskNeutral, RiskJudge)
*   Tabs: `Tab*` (TabOverview, TabSignals, TabDebate, TabRisk, TabAudit, TabCost)
*   Utility: `Icon*` (IconWarning, IconSearch, IconDownload, IconChart, …)

### v2.9 — Autonomous Paper Trading System (March 2026)

Fully autonomous AI trading agent managing a virtual $10,000 portfolio. Daily cycle: quant screen S&P 500 → deep-analyze top candidates (lite mode) → execute virtual trades → track P&L vs SPY → learn from outcomes. Configurable via 11 new settings.

**New Backend Modules (3)**:
*   `backend/tools/screener.py` — S&P 500 quant screener: batch yfinance download, momentum/RSI/volatility/SMA filters, composite alpha score ranking. Zero LLM cost. Fallback 50-ticker list if Wikipedia scrape fails.
*   `backend/services/paper_trader.py` — `PaperTrader` class: virtual trade execution engine backed by BigQuery. `execute_buy()` (position averaging, cash check, max positions), `execute_sell()` (full/partial exit), `mark_to_market()` (live yfinance prices + SPY benchmark), `check_stop_losses()`, `save_daily_snapshot()`.
*   `backend/services/portfolio_manager.py` — `decide_trades()`: sell-first-then-buy logic. Sells on SELL/STRONG_SELL signal, signal downgrade (BUY→HOLD), or stop-loss hit. Buys sized by `min(risk_judge_pct * NAV, available_cash)`, sorted by final_score, respects max_positions and min_cash_reserve.
*   `backend/services/autonomous_loop.py` — `run_daily_cycle()`: 9-step async orchestrator (screen → filter → analyze candidates → re-evaluate holdings → MTM → decide trades → execute → snapshot → learn from closed trades). Module-level state tracking.

**New API Route (1)**:
*   `backend/api/paper_trading.py` — 8 endpoints: `POST /start` (init fund + scheduler), `POST /stop` (pause scheduler), `GET /status` (NAV + scheduler state), `GET /portfolio` (positions), `GET /trades` (history), `GET /snapshots` (daily NAV), `GET /performance` (Sharpe, win rate, alpha), `POST /run-now` (manual trigger). APScheduler cron job wired into FastAPI lifespan.

**BQ Schema (4 new tables)**:
*   `paper_portfolio` (8 cols) — Fund state (cash, NAV, benchmark)
*   `paper_positions` (14 cols) — Open positions with unrealized P&L
*   `paper_trades` (11 cols) — Trade history with reasons and realized P&L
*   `paper_portfolio_snapshots` (11 cols) — Daily NAV for charting and Sharpe calculation
*   Managed by `migrate_paper_trading.py` (idempotent)

**Frontend Dashboard (1 page)**:
*   `frontend/src/app/paper-trading/page.tsx` — Full dashboard: 6 summary cards (NAV, Cash, P&L, vs SPY, Sharpe, Positions), status banner, 3-tab view (Positions table, Trades table, NAV Chart with Recharts LineChart showing Portfolio/SPY/Alpha lines), Initialize Fund / Start / Pause / Run Now action buttons.
*   `frontend/src/lib/types.ts` — 6 new interfaces: `PaperPortfolio`, `PaperPosition`, `PaperTrade`, `PaperSnapshot`, `PaperTradingStatus`, `PaperPerformance`
*   `frontend/src/lib/api.ts` — 8 new functions for paper trading endpoints
*   `frontend/src/components/Sidebar.tsx` — "Paper Trading" (🤖) nav entry added

**Modified Backend Files**:
*   `backend/config/settings.py` — 11 new paper trading settings (capital, max positions, cash reserve, screen/analyze top N, trading hour, re-eval frequency, transaction cost, daily cost cap)
*   `backend/db/bigquery_client.py` — ~150 lines of paper trading CRUD methods (get/upsert portfolio, positions, trades, snapshots)
*   `backend/main.py` — Paper trading router registered, APScheduler init in lifespan startup

**Daily Cycle Cost Estimate**:
*   Screen: $0 (yfinance batch download)
*   Analyze 5 new candidates (lite mode): ~$0.50
*   Re-evaluate ~5 holdings (lite mode): ~$0.50
*   Daily total: ~$1.00/day, configurable via `PAPER_MAX_DAILY_COST_USD`

**New Environment Variables (11)**: `PAPER_TRADING_ENABLED`, `PAPER_STARTING_CAPITAL`, `PAPER_MAX_POSITIONS`, `PAPER_MIN_CASH_RESERVE_PCT`, `PAPER_SCREEN_TOP_N`, `PAPER_ANALYZE_TOP_N`, `PAPER_TRADING_HOUR`, `PAPER_REEVAL_FREQUENCY_DAYS`, `PAPER_TRANSACTION_COST_PCT`, `PAPER_MAX_DAILY_COST_USD`

**New Dependencies**: `APScheduler>=3.10.0` (already present from v2.8 Slack bot)

### v2.8 — Auth + Slack Bot + Deployment (March 2026)

End-to-end authentication (Google SSO + Passkey), Slack bot with slash commands and morning digest, Docker Compose overhaul, and OWASP security hardening.

**Frontend Authentication (12 files)**:
*   `prisma/schema.prisma` — SQLite database with 5 models: User, Account, Session, VerificationToken, Authenticator (WebAuthn)
*   `src/lib/auth.config.ts` — Edge-compatible auth config: Google + Passkey providers, JWT strategy (8h maxAge), email whitelist callback, `authorized` callback for middleware
*   `src/lib/auth.ts` — Full auth config extending auth.config with PrismaAdapter + `experimental: { enableWebAuthn: true }`
*   `src/lib/prisma.ts` — Singleton Prisma client with global hot-reload safety
*   `src/app/api/auth/[...nextauth]/route.ts` — NextAuth v5 catch-all route handler
*   `src/components/AuthProvider.tsx` — SessionProvider wrapper with 15-minute refetch interval
*   `src/app/login/page.tsx` — Login page: Google SSO button (SVG icon) + Passkey button (🔑), PyFinAgent branding, generic error messages, dark theme
*   `src/middleware.ts` — Route protection: imports from `auth.config` (Edge-safe), redirects unauthenticated → `/login`, skips `/api/auth`, `/_next`, `/favicon`
*   `src/lib/api.ts` — Added `getAuthToken()` (reads session cookie), Bearer token injection, 401 → redirect to `/login`, `Cache-Control: no-store`
*   `src/lib/types.ts` — Added `AuthUser` interface
*   `src/components/Sidebar.tsx` — Added user avatar/email display, passkey registration button, logout button
*   `.env.local` — Added `AUTH_SECRET`, `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`, `ALLOWED_EMAILS`

**Backend Authentication (3 files)**:
*   `api/auth.py` — HKDF key derivation + JWE A256CBC-HS512/dir decryption using `cryptography` library. `get_current_user()` validates Bearer token, checks email whitelist + token expiry
*   `main.py` — Auth middleware (skips public paths), OWASP security headers (6 headers), CORS updated for Tailscale IPs (`100.x.y.z`)
*   `config/settings.py` — Added `auth_secret`, `allowed_emails`, `slack_bot_token`, `slack_app_token`, `slack_channel_id`, `morning_digest_hour`

**Slack Bot Module (6 files in `backend/slack_bot/`)**:
*   `app.py` — Entry point: AsyncApp + AsyncSocketModeHandler, registers slash commands, starts scheduler
*   `commands.py` — 3 slash commands: `/analyze <TICKER>` (starts analysis, polls 5s intervals), `/portfolio` (P&L summary), `/report <TICKER>` (report card)
*   `scheduler.py` — APScheduler AsyncIOScheduler: morning digest cron job (configurable hour), `send_analysis_alert()` for proactive alerts after analysis completes
*   `formatters.py` — 4 Block Kit message builders: analysis result, portfolio summary, report card, morning digest
*   `Dockerfile` — python:3.11-slim, non-root appuser, Socket Mode (no inbound ports)

**Docker Compose Overhaul**:
*   3 services: `backend`, `frontend`, `slack-bot` (removed redis + celery-worker)
*   `slack-bot` service: depends on backend, Socket Mode (no ports exposed), `restart: unless-stopped`
*   `auth-db` named volume for SQLite persistence across container restarts
*   `restart: unless-stopped` on all services
*   Frontend gets `env_file: ./frontend/.env.local` and volume mount for Prisma

**New Dependencies**: `cryptography>=42.0.0`, `slack-bolt[async]>=1.18.0`, `slack-sdk>=3.27.0`, `APScheduler>=3.10.0`, `next-auth@5.0.0-beta.30`, `@prisma/client@6`, `@auth/prisma-adapter`, `@simplewebauthn/browser@9.0.1`, `@simplewebauthn/server@9.0.3`

**New Environment Variables**: `AUTH_SECRET`, `ALLOWED_EMAILS`, `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`, `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_CHANNEL_ID`, `MORNING_DIGEST_HOUR`

### v2.7 — Cost Management + Settings UI Overhaul (March 2026)

Multi-layered LLM cost controls, per-agent output token limits, lite mode, budget warnings, prompt truncation, and a complete Settings page overhaul with live cost estimation using real token data from the most recent analysis.

**Cost Controls (Backend)**:
*   `orchestrator.py` — 4 GenerationConfig tiers: `_enrichment_config` (1024 tokens), `_deep_think_config` (2048), `_synthesis_config` (4096), `_gen_config` (base, no limit)
*   `orchestrator.py` — Lite mode: skips deep dive, devil's advocate, risk assessment; forces 1 debate round. Controlled by `LITE_MODE` env var
*   `orchestrator.py` — Prompt truncation: `_MAX_SECTION=1500` per enrichment output, `_MAX_CONTEXT=12000` total market context in synthesis
*   `orchestrator.py` — Configurable synthesis iterations via `MAX_SYNTHESIS_ITERATIONS` (1-3). New `self.synthesis_model` for synthesis calls (uses `_synthesis_config`)
*   `orchestrator.py` — Budget warning: `CostTracker.check_budget()` after analysis completes, adds `budget_warning` to final report
*   `cost_tracker.py` — New `total_cost` property (thread-safe sum) + `check_budget(max_cost_usd)` method
*   `debate.py` — `_DEBATE_GEN_CONFIG` (1536 tokens), `_MODERATOR_GEN_CONFIG` (2048 tokens), `gen_config` parameter on `_generate_with_retry()`, `skip_devils_advocate` parameter
*   `risk_debate.py` — `_RISK_GEN_CONFIG` (1024 tokens), `_JUDGE_GEN_CONFIG` (1536 tokens), `gen_config` parameter on `_generate_with_retry()`
*   `settings.py` — 3 new fields: `lite_mode` (bool), `max_analysis_cost_usd` (float, default 0.50), `max_synthesis_iterations` (int, 1-3, default 2)

**Settings API Overhaul**:
*   `settings_api.py` — Complete rewrite: `FullSettings` model (13 readable fields), `SettingsUpdate` model (all optional, validated)
*   `GET /api/settings/` returns all settings, `PUT /api/settings/` accepts partial updates with validation (model whitelist, pillar weight sum = 1.0)
*   `_FIELD_TO_ENV` mapping persists changes to `.env` file. Legacy `/models` endpoints preserved
*   `reports.py` — New `GET /api/reports/latest-cost-summary` returns per-agent cost breakdown from most recent analysis

**Frontend Settings Overhaul**:
*   `settings/page.tsx` — Complete rewrite with 6 BentoCards:
    1. **Analysis Mode**: Full vs Lite toggle with descriptions
    2. **Live Cost Estimator**: Uses real per-agent token counts from last analysis, scales by model pricing, debate rounds, synthesis iterations, lite mode. Shows $/analysis, total tokens, LLM calls
    3. **Model Configuration**: Standard + Deep Think model dropdowns
    4. **Cost Controls**: Budget slider ($0.05-$5.00), synthesis iterations (1-3), min data quality (0-100%)
    5. **Debate Depth**: Bull↔Bear rounds (1-5), risk rounds (1-3) with lite mode override warning
    6. **Pillar Weights**: 5 weight sliders (0-50%) with live total validation (must = 100%)
*   Single "Save All Settings" button sends only changed fields as diff
*   `types.ts` — New `FullSettings` + `LatestCostSummary` interfaces
*   `api.ts` — New `getFullSettings()`, `updateSettings()`, `getLatestCostSummary()` functions

**New Environment Variables**: `LITE_MODE`, `MAX_ANALYSIS_COST_USD`, `MAX_SYNTHESIS_ITERATIONS` (all optional, with defaults)

### v2.6 — Agent Design Pattern Optimization (March 2026)

Implements 4 design pattern enhancements derived from research analysis of Google Cloud, Andrew Ng, and Anthropic's AI agent design patterns: Reflection Loop (Evaluator-Optimizer), Session Memory, Quality Gates, and Sector Routing. BQ schema expanded from 67 → 68 columns.

**Reflection Loop (Evaluator-Optimizer Pattern)**:
*   `critic_agent.md` restructured: now returns structured `{verdict: "PASS"|"REVISE", issues: [...], corrected_report: {...}}` instead of just corrected JSON
*   `orchestrator.py` `run_synthesis_pipeline()` rewritten: Synthesis → Critic → if REVISE and iteration < 2, re-run Synthesis with Critic's issues → Critic again. Max 2 iterations
*   `prompts.py` — new `get_synthesis_revision_prompt()` function that prepends revision instruction block with Critic issues to the base synthesis template
*   `get_critic_prompt()` now accepts optional `critic_feedback` parameter for iteration context

**Session Memory (AnalysisContext)**:
*   `trace.py` — new `AnalysisContext` dataclass: accumulates `key_findings`, `contradictions`, `signal_consensus` during a run (capped at 20 findings, each ≤100 chars)
*   Populated after Steps 2 (quant), 4 (market), 6 (enrichment signals), 6b (info-gap), 8 (debate)
*   Injected into Synthesis prompt via `format_for_prompt()` → prepended to market_context

**Quality Gates (Conditional Pipeline)**:
*   Data quality gate after Step 6b: if `data_quality_score < data_quality_min` (default 0.5), Steps 8 (debate) and 12c (risk assessment) are skipped with placeholder results
*   Skipped debate returns `consensus: "HOLD"` with low confidence; skipped risk returns `decision: "REJECT"` with 0% position
*   `settings.py` — 3 new configurable thresholds: `data_quality_min` (0.5), `conflict_escalation_threshold` (5), `critic_major_issues_threshold` (3)

**Sector Routing (Tool Skipping)**:
*   `orchestrator.py` — new `SECTOR_SKIP_MAP` dict: Financial Services skips patents, Utilities/Real Estate skip patents + alt_data
*   Step 6 enrichment now conditionally runs `_skip_placeholder()` for sector-irrelevant tools
*   `info_gap.py` — new `SKIPPED` status: skipped tools excluded from data quality denominator

**BQ Schema Expansion (1 new column)**:
*   `("synthesis_iterations", "INT64")` — tracks reflection loop count per analysis

**New Environment Variables**: `DATA_QUALITY_MIN`, `CONFLICT_ESCALATION_THRESHOLD`, `CRITIC_MAJOR_ISSUES_THRESHOLD` (all optional, with defaults)

### v2.5 — Skills System + Autonomous Optimization Loop (March 2026)

Autoresearch-inspired skills architecture: each agent's prompt is defined in a skills.md file, loaded dynamically by a skill loader with mtime caching. An autonomous optimization loop (SkillOptimizer) proposes, tests, and keeps/discards prompt modifications using a single metric. Feedback loop wired: outcome evaluation now generates LLM reflections and persists them to BQ agent_memories table.

**New Backend Modules (2)**:
*   `backend/agents/skill_optimizer.py` — `SkillOptimizer` class: `establish_baseline()`, `analyze_agent_performance()`, `propose_skill_modification()`, `apply_modification()`, `revert_modification()`, `handle_crash()`, `think_harder()`, `run_loop()`. Uses git for experiment tracking. Simplicity criterion: ≥0.5% delta per 10 lines added.
*   `backend/agents/skills/experiments/analyze_experiments.py` — Experiment analysis: keep rate (overall + per-agent), delta chain, running best chart data, top hits table, summary stats. CLI + importable.

**New API Route (1)**:
*   `backend/api/skills.py` — 7 endpoints: POST optimize/stop, GET status/experiments/analysis/agents/{agent_name}

**Skills Architecture (28 files)**:
*   `backend/agents/skills/*.md` — 28 agent skills files following SKILL_TEMPLATE.md format
*   Each contains: Goal, Identity, CAN/CANNOT, Prompt Template (with `{{variable}}` placeholders), Experiment Log
*   `backend/config/prompts.py` refactored: 950 lines of inline prompts → ~380 lines of thin wrappers using `load_skill()` + `format_skill()`

**Feedback Loop Wired**:
*   `outcome_tracker.py` — After evaluating outcomes, generates LLM reflections via `generate_reflection()` for 4 agent types (bull, bear, moderator, risk_judge) and persists to `agent_memories` table via `save_agent_memory()`
*   Closes the learning loop: outcomes → reflections → BM25 memory → future agent prompts

### v2.4 — Dual LLM Strategy + Cost Analytics (March 2026)

Dual-model architecture with "deep think" model for judge agents (Moderator, Risk Judge, Synthesis, Critic), per-agent token/cost tracking across all LLM calls, Cost tab in dashboard, model configuration UI, and cost history on Performance page. BQ schema expanded from 64 → 67 columns.

**New Backend Module (1)**:
*   `backend/agents/cost_tracker.py` — `CostTracker` dataclass: thread-safe per-agent token recording from `response.usage_metadata`, `MODEL_PRICING` dict (Flash/2.5-Flash/2.5-Pro), `summarize()` produces JSON-serializable breakdown by model and agent

**New API Route (1)**:
*   `backend/api/settings_api.py` — `GET /api/settings/models` (current config), `GET /api/settings/models/available` (model list with pricing)

**Dual LLM Architecture**:
*   `settings.py` — New `deep_think_model` field (defaults to empty → falls back to `gemini_model`)
*   `orchestrator.py` — Creates `self.deep_think_model` GenerativeModel, passes to debate/risk_debate; Synthesis + Critic use deep_think_model with `is_deep_think=True`
*   `debate.py` — Moderator uses `deep_think_model or model`; Bull/Bear/DA use standard model
*   `risk_debate.py` — Risk Judge uses `deep_think_model or model`; Aggressive/Conservative/Neutral use standard model
*   All `_generate_with_retry` calls auto-record to `CostTracker` via `getattr(self, "_cost_tracker", None)`

**BQ Schema Expansion (3 new columns)**:
*   **Cost Metrics (+3)**: `total_tokens`, `total_cost_usd`, `deep_think_calls`

**Frontend Enhancements**:
*   `CostDashboard.tsx` — New component: 4 summary cards, token distribution bar, cost by model breakdown, per-agent table sorted by cost
*   `page.tsx` — 6th tab "Cost" (💰) with badge showing total cost
*   `settings/page.tsx` — "Model Configuration" BentoCard with Standard/Deep Think model dropdowns + pricing display
*   `performance/page.tsx` — Cost history section: total spend, avg cost/analysis, per-analysis table from BQ
*   `types.ts` — New interfaces: `AgentCostEntry`, `ModelBreakdown`, `CostSummary`, `CostHistoryEntry`, `ModelPricing`, `ModelConfig`
*   `api.ts` — New functions: `getModelConfig()`, `getAvailableModels()`, `getCostHistory()`

**New Environment Variable**: `DEEP_THINK_MODEL` (optional, e.g., `gemini-2.5-pro`)

### v2.3 — FinancialSituationMemory + Prompt Hardening (March 2026)

BM25-based agent memory that learns from past outcomes, anti-HOLD bias prompts, multi-round risk debate with cross-visibility, and configurable debate depth. Inspired by TradingAgents `FinancialSituationMemory` and prompt-hardening research.

**New Backend Module (1)**:
*   `backend/agents/memory.py` — `FinancialSituationMemory` class: BM25Okapi lexical similarity retrieval, 5 cold-start seed archetypes, `generate_reflection()` for LLM-based lesson extraction from outcome evaluations

**Prompt Hardening (7 prompts updated)**:
*   Anti-HOLD bias in Moderator: "Choose HOLD only if strongly justified by specific arguments — not as a fallback"
*   Anti-HOLD bias in Risk Judge: "Choose REJECT only if strongly justified by specific downside evidence"
*   Anti-hallucination guards in Aggressive/Conservative/Neutral: "do not hallucinate their arguments"
*   `past_memory` parameter added to Bull, Bear, Moderator, all 3 Risk Analysts, and Risk Judge prompts

**Risk Debate Rewrite**:
*   `risk_debate.py` rewritten: single-pass → multi-round with cross-visibility (each analyst sees the other two's prior arguments)
*   Risk Judge now receives full debate history when `max_risk_rounds > 1`
*   Output includes `risk_debate_rounds` and `total_risk_rounds` fields

**Configurable Debate Depth**:
*   `max_debate_rounds` (default 2, range 1-5) — Bull↔Bear iterative rebuttal exchanges
*   `max_risk_debate_rounds` (default 1, range 1-3) — Aggressive/Conservative/Neutral exchanges
*   Controlled via `MAX_DEBATE_ROUNDS` and `MAX_RISK_DEBATE_ROUNDS` env vars

**BQ Persistence**:
*   New `agent_memories` table: `agent_type`, `ticker`, `situation`, `lesson`, `created_at`
*   `migrate_agent_memories.py` — Idempotent table creation script
*   `bigquery_client.py` — Added `get_agent_memories()` and `save_agent_memory()` methods

**Frontend Enhancements**:
*   `settings/page.tsx` — New "Debate Depth" BentoCard with Bull↔Bear and Risk Debate round sliders
*   `types.ts` — Added `RiskDebateRound` interface + `risk_debate_rounds`/`total_risk_rounds` to `RiskAssessment`

**New Dependency**: `rank-bm25>=0.2.2`

### v2.2 — TradingAgents + AlphaQuanter Enhancement (March 2026)

Implemented multi-round adversarial debate, Devil's Advocate stress-testing, round-robin Risk Assessment Team, and AlphaQuanter-style info-gap detection with retry. BQ schema expanded from 55 → 64 columns. Pipeline expanded from 13 → 15 steps.

**New Backend Modules (2)**:
*   `backend/agents/risk_debate.py` — TradingAgents-style round-robin: Aggressive → Conservative → Neutral → Risk Judge
*   `backend/agents/info_gap.py` — AlphaQuanter ReAct loop: scan 11 sources, classify SUFFICIENT/PARTIAL/MISSING, retry critical gaps

**Enhanced Debate Framework**:
*   `debate.py` rewritten: single-round → multi-round iterative Bull↔Bear (each sees opponent's prior argument)
*   New Devil's Advocate agent: stress-tests both sides for hidden risks + groupthink detection
*   Moderator now receives full debate history + DA challenges
*   7 new prompts in `prompts.py`: Devil's Advocate, 4 Risk Analysts, Info-Gap

**New Pipeline Steps (2)**:
*   Step 6b: Info-Gap Detection (between enrichment + enrichment analysis)
*   Step 12c: Risk Assessment Team (after bias audit)

**Frontend Enhancements (4 components)**:
*   `DebateView.tsx` — DA section (violet theme), multi-round timeline
*   `RiskDashboard.tsx` — New `RiskAssessmentPanel` component (judge verdict + analyst cards + risk limits)
*   `AnalysisProgress.tsx` — 2 new steps (info_gap + risk_assessment)
*   `types.ts` — 7 new interfaces (DebateRound, DevilsAdvocate, RiskAssessment, InfoGap, etc.)

**BQ Schema Expansion (9 new columns across 3 categories)**:
*   **Debate Dynamics (+2)**: `debate_rounds_count`, `devils_advocate_challenges`
*   **Info-Gap Quality (+3)**: `info_gap_count`, `info_gap_resolved_count`, `data_quality_score`
*   **Risk Assessment (+4)**: `risk_judge_decision`, `risk_adjusted_confidence`, `aggressive_analyst_confidence`, `conservative_analyst_confidence`

### v2.1 — ML-Training-Ready BigQuery Schema (March 2026)

Expanded BigQuery `analysis_results` table from 18 → 55 columns for ML model training. Every quantitative feature that drives the recommendation is now stored as a first-class column.

**Schema Expansion (37 new columns across 6 categories)**:
*   **Financial Fundamentals (7)**: `price_at_analysis`, `market_cap`, `pe_ratio`, `peg_ratio`, `debt_equity`, `sector`, `industry`
*   **Risk Metrics (6)**: `annualized_volatility`, `var_95_6m`, `var_99_6m`, `expected_shortfall_6m`, `prob_positive_6m`, `anomaly_count`
*   **Debate Dynamics (8)**: `bull_confidence`, `bear_confidence`, `bull_thesis`, `bear_thesis`, `contradiction_count`, `dissent_count`, `recommendation_confidence`, `key_risks`
*   **Enrichment Signals (7)**: `insider_signal`, `options_signal`, `social_sentiment_score`, `nlp_sentiment_score`, `patent_signal`, `earnings_confidence`, `sector_signal`
*   **Bias & Conflict Audit (5)**: `bias_count`, `bias_adjusted_score`, `conflict_count`, `overall_reliability`, `decision_trace_count`
*   **Macro Context (4)**: `fed_funds_rate`, `cpi_yoy`, `unemployment_rate`, `yield_curve_spread`

**New Files**:
*   `migrate_bq_schema.py` — Idempotent schema migration script
*   `backend/db/bigquery_client.py` — Expanded `save_report()` (55-field row insert)
*   `backend/services/outcome_tracker.py` — Outcome evaluation with benchmark comparison

**LLM Temperature Fix**: Set `temperature=0.2` on all Gemini model calls (orchestrator + debate) to reduce score variance across runs.

### v2.0 — AI Research-Driven Upgrade (March 2026)

Based on comprehensive research analysis from Goldman Sachs, BlackRock, Morgan Stanley, Harvard, Stanford, Chicago Booth, and Wharton:

**New Backend Tools (3)**:
*   `nlp_sentiment.py` — Transformer-based sentiment via Vertex AI embeddings (Stanford ref 11)
*   `anomaly_detector.py` — Multi-dimensional anomaly detection using Z-score/IQR (Goldman ref 16)
*   `monte_carlo.py` — Monte Carlo VaR simulation engine with 1,000 GBM paths (Goldman ref 16)

**New Agent Framework (5 agents)**:
*   `debate.py` — Bull/Bear/Moderator 4-round adversarial debate (TradingAgents ref 32)
*   `trace.py` — Decision trace logger for Explainable AI (Goldman XAI ref 16)
*   `bias_detector.py` — LLM bias detection: tech favoritism, confirmation bias, recency bias (arXiv ref 33)
*   `conflict_detector.py` — Knowledge conflict detection: parametric vs real-time data (arXiv ref 33)
*   Enhanced Critic Agent — Now checks for bias patterns in addition to hallucination/logic

**New Frontend Components (4)**:
*   `DebateView.tsx` — Bull vs Bear adversarial debate visualization
*   `RiskDashboard.tsx` — Monte Carlo fan chart + VaR gauge + anomaly alerts
*   `SentimentDetail.tsx` — NLP sentiment deep-dive with keyword cloud
*   `BiasReport.tsx` — LLM bias flags + knowledge conflict table

**New Pages (1)**:
*   `/portfolio` — Position tracking, P&L, allocation, recommendation accuracy

**Pipeline Expansion**: 11 steps → 13 steps (added Agent Debate + Bias Audit)

### v1.0 — Initial Migration (February 2026)

Migrated from Streamlit to Next.js 15 + FastAPI architecture. Implemented 8 enrichment data tools, 8 enrichment LLM agents, signals API, Glass Box dashboard.