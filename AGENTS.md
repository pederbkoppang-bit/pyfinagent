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
| **Frontend** | Next.js 15 / React 19 / TypeScript 5.6 / Tailwind CSS / Recharts | 3001 |
| **Backend** | FastAPI 0.115+ / Python 3.14 / Vertex AI (Gemini 2.0 Flash) | 8000 |
| **Storage** | Google BigQuery (reports), Google Cloud Storage (10-K/10-Q filings) | — |
| **AI Engine** | Google Vertex AI with Gemini model, RAG via Vertex AI Search datastore | — |
| **Embeddings** | Vertex AI text-embedding-005 (for NLP sentiment) | — |

### Directory Structure

```
pyfinagent/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py      # Main 13-step analysis pipeline
│   │   ├── debate.py            # Multi-agent debate framework (Bull/Bear/Moderator)
│   │   ├── trace.py             # Decision trace logger (XAI)
│   │   ├── bias_detector.py     # LLM bias detection (tech bias, confirmation bias)
│   │   └── conflict_detector.py # Knowledge conflict detection (parametric vs real-time)
│   ├── config/
│   │   ├── settings.py          # Pydantic settings (env vars)
│   │   ├── prompts.py           # All LLM prompt templates (25+ prompts)
│   │   └── synthesis_prompt.txt # Synthesis JSON schema template
│   ├── api/
│   │   ├── analysis.py          # POST /api/analysis/, GET /api/analysis/{id}
│   │   ├── reports.py           # Reports CRUD + performance stats
│   │   ├── charts.py            # Price chart + financials endpoints
│   │   ├── signals.py           # Enrichment signals endpoints (11 routes)
│   │   └── portfolio.py         # Portfolio tracking CRUD
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
│   │   └── slack.py             # Slack webhook notifications
│   └── main.py                  # FastAPI app + router registration
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx             # Dashboard (main analysis)
│   │   ├── signals/page.tsx     # Signals exploration
│   │   ├── compare/page.tsx     # Multi-stock comparison
│   │   ├── reports/page.tsx     # Past reports
│   │   ├── performance/page.tsx # Performance stats
│   │   ├── portfolio/page.tsx   # Portfolio tracking + P&L
│   │   └── settings/page.tsx    # Configuration
│   ├── src/components/
│   │   ├── DebateView.tsx       # Bull vs Bear debate visualization
│   │   ├── RiskDashboard.tsx    # Monte Carlo fan chart + VaR + anomalies
│   │   ├── SentimentDetail.tsx  # NLP sentiment deep-dive
│   │   ├── BiasReport.tsx       # LLM bias flags + knowledge conflicts
│   │   ├── SignalCards.tsx       # 8-signal enrichment grid + consensus bar
│   │   ├── SectorDashboard.tsx  # Sector rotation + relative strength
│   │   ├── MacroDashboard.tsx   # FRED indicator grid + warnings
│   │   ├── StockChart.tsx       # Candlestick + SMA/RSI chart
│   │   ├── EvaluationTable.tsx  # 5-pillar scoring matrix
│   │   ├── AnalysisProgress.tsx # 13-step progress tracker
│   │   └── Sidebar.tsx          # Navigation sidebar
│   └── src/lib/
│       ├── types.ts             # Full TypeScript type definitions
│       └── api.ts               # API client (all endpoint calls)
├── quant-agent/                 # GCP Cloud Function
├── ingestion_agent/             # GCP Cloud Function
├── earnings-ingestion-agent/    # GCP Cloud Function
└── AGENTS.md                    # This file
```

---

## Analysis Pipeline (13 Steps)

The orchestrator (`backend/agents/orchestrator.py`) executes a 13-step pipeline. Steps 6 and 7 run in parallel using `asyncio.gather`. The debate framework (Steps 8-8b) implements adversarial validation before synthesis.

| Step | Name | Agent/Tool | Description |
|------|------|-----------|-------------|
| 0 | Market Intel | `alphavantage.py` + `yfinance_tool.py` | Fetch Alpha Vantage news (50 articles) + yfinance fundamentals in parallel |
| 1 | Ingestion | `ingestion-agent` (Cloud Function) | Check for new SEC filings, ingest into BigQuery |
| 2 | Quant Data | `quant-agent` (Cloud Function) | Fetch SEC EDGAR financials, merge with yfinance data |
| 3 | Document Analysis | `RAG Agent` (Vertex AI Search) | Analyze 10-K/10-Q for moat, governance, risks with citations |
| 4 | Sentiment Analysis | `Market Agent` (LLM) | Detect sentiment velocity, price-sentiment divergence, catalyst keywords |
| 5 | Competitor Analysis | `Competitor Agent` (LLM) | Identify rivals from news co-occurrence, assess competitive positioning |
| 6 | **Data Enrichment** | 11 tools in parallel | Insider trades, options flow, social sentiment, patents, earnings transcripts, FRED macro, Google Trends, sector data, NLP sentiment, anomaly scan, Monte Carlo simulation |
| 7 | **Enrichment Analysis** | 11 LLM agents | Each tool's data analyzed by a specialized LLM agent |
| 8 | **Agent Debate** | `Bull Agent` + `Bear Agent` + `Moderator Agent` | 4-round adversarial debate: positions → bull case → bear case → moderator consensus with contradiction resolution |
| 9 | Enhanced Macro | `Enhanced Macro Agent` (LLM + FRED) | Fed Funds, CPI, GDP, unemployment, yield curve, consumer sentiment analysis |
| 10 | Deep Dive | `Deep Dive Agent` (LLM + RAG) | Identify 3 contradictions between sources, probe with targeted 10-K questions |
| 11 | Synthesis | `Synthesis Agent` (LLM) | Combine all agent outputs + debate consensus into structured JSON report |
| 12 | Critic | `Critic Agent` (LLM) | Validate for hallucinations, logic errors, factual consistency against quant data |
| 12b | Bias Audit | `Bias Detector` + `Conflict Detector` | Check for tech-sector bias, confirmation bias, anchoring; flag knowledge conflicts |

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
    ├─ Step 7: 11 LLM Enrichment Agents (analyze each dataset)   │
    │                                                              │
    ├─ Step 8: Agent Debate Framework ────────────────────────────┤
    │   ├── Round 1: Each agent submits position + confidence     │
    │   ├── Round 2: Bull Agent argues strongest bullish case     │
    │   ├── Round 3: Bear Agent argues strongest bearish case     │
    │   └── Round 4: Moderator resolves contradictions            │
    │                                                              │
    ├─ Step 9: Enhanced Macro Agent (AV + FRED combined)          │
    ├─ Step 10: Deep Dive Agent (contradiction probing via RAG)   │
    │                                                              │
    ├─ Step 11: Synthesis Agent ──────────────────────────────────┤
    │   (all agent outputs + debate consensus → structured JSON)  │
    │                                                              │
    ├─ Step 12: Critic Agent (hallucination/logic check)          │
    └─ Step 12b: Bias Audit (tech bias, confirmation bias, etc.)  │
                                                                   │
    Final Output: Validated JSON report with:                      │
    ├── scoring_matrix (5 pillars, weighted)                      │
    ├── enrichment_signals (11 sources)                           │
    ├── debate_result (bull/bear/consensus)                       │
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
| **Bull Agent** | `get_bull_agent_prompt()` | Synthesize all bullish signals into the strongest investment case | All enrichment agent outputs where signal is bullish | Structured bull thesis with evidence citations, confidence score, key catalysts |
| **Bear Agent** | `get_bear_agent_prompt()` | Synthesize all bearish signals into the strongest bear case | All enrichment agent outputs where signal is bearish | Structured bear thesis with risk evidence, confidence score, key threats |
| **Moderator Agent** | `get_moderator_prompt()` | Resolve contradictions, assign consensus, identify unresolved uncertainties | Bull case + Bear case + all raw signals | Final consensus (STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL), contradiction map, confidence-weighted score, dissent registry |

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
  "dissent_registry": [ { "agent": "Options Flow", "position": "BEARISH", "reason": "..." } ]
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

### Pages (7 routes)

| Route | File | Description |
|-------|------|-------------|
| `/` | `page.tsx` | **Dashboard**: Ticker input → real-time 13-step analysis → Alpha score card, investment thesis, 5-pillar evaluation, stock chart, enrichment signals, debate view, risk dashboard, bias report |
| `/signals` | `signals/page.tsx` | **Signals Explorer**: Enter ticker → view all 11 enrichment signals, consensus bar, sector dashboard, macro dashboard |
| `/compare` | `compare/page.tsx` | **Compare**: Select 2-5 past reports → side-by-side price overlay, radar chart, pillar score bars |
| `/reports` | `reports/page.tsx` | **Past Reports**: Searchable list of all completed analyses with quick-chip filters |
| `/performance` | `performance/page.tsx` | **Performance**: Historical accuracy metrics |
| `/portfolio` | `portfolio/page.tsx` | **Portfolio**: Position tracking, P&L, allocation pie chart, drawdown, recommendation accuracy scorecard |
| `/settings` | `settings/page.tsx` | **Settings**: Pillar weight configuration |

### Key Components

| Component | Purpose |
|-----------|---------|
| `DebateView.tsx` | Bull vs Bear argument cards side-by-side, contradiction highlights (red), confidence-weighted consensus bar, individual agent dissent badges |
| `RiskDashboard.tsx` | Monte Carlo fan chart (percentile bands over 1Y), VaR gauge (95%/99%), anomaly alert cards (red for risk, green for opportunity) |
| `SentimentDetail.tsx` | Contextual keyword cloud (embedding-weighted), sentiment time-series (30d), source breakdown chart |
| `BiasReport.tsx` | Bias flag cards with severity indicators, knowledge conflict table (LLM belief vs actual data), raw vs bias-adjusted score |
| `SignalCards.tsx` | 8-card grid with color-coded badges (green=bullish, red=bearish, amber=neutral, gray=error). SignalSummaryBar shows consensus distribution |
| `SectorDashboard.tsx` | Sector rotation bar chart (11 SPDR ETFs by 3M return), relative performance table (stock/sector/SPY × 4 periods), peer comparison table |
| `MacroDashboard.tsx` | 7-indicator grid (current value + change), macro warnings section |
| `StockChart.tsx` | Candlestick + volume with toggleable SMA50/SMA200/RSI, 5 period options (1M–2Y) |
| `EvaluationTable.tsx` | 5-pillar horizontal bars with weights, individual pillar progress indicators |
| `AnalysisProgress.tsx` | 13-step real-time tracker with % bar, current-step spinner, emoji status icons |

---

## Environment & Setup

### Prerequisites

*   Node.js 18+
*   Python 3.11+
*   GCP project with Vertex AI and BigQuery enabled
*   Application Default Credentials configured (`gcloud auth application-default login`)

### Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev  # port 3001
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

# Optional
SLACK_WEBHOOK_URL=<optional>
USE_CELERY=false
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

### Investigation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/investigate` | Run RAG investigation query against 10-K documents |

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
The debate framework follows a strict 4-round structure:
1. **Position Round**: Each enrichment agent submits its signal, confidence (0-1), and top 3 evidence points
2. **Bull Round**: Bull Agent synthesizes all bullish signals into the strongest possible investment case
3. **Bear Round**: Bear Agent synthesizes all bearish signals into the strongest possible risk case
4. **Moderation Round**: Moderator Agent identifies contradictions between bull/bear, resolves them using evidence weight, assigns final consensus with confidence score, and registers any unresolved dissents

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

*   **SEC EDGAR User-Agent**: When making requests to the SEC EDGAR database, you **MUST** declare a custom User-Agent string in the format `FirstName LastName email@domain.com`. Failure to do so will result in the IP address being blocked by the SEC. This is managed via secrets.
*   **Secret Management**:
    *   **Local**: Use the `backend/.env` file for local development.
    *   **Production**: All production secrets (API keys, service account keys) are managed using **Google Cloud Secret Manager**. Do not hardcode secrets in the source code.
*   **Input Validation**: All API endpoints validate input parameters. Ticker symbols are sanitized. No raw user input is passed directly to LLM prompts without sanitization.
*   **Rate Limiting**: External API calls (Alpha Vantage, FRED, SEC EDGAR, USPTO) respect rate limits with automatic retry and exponential backoff.

---

## Upgrade History

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