# AGENTS.md for PyFinAgent

> An agentic AI financial analyst orchestrating 20+ specialized agents through a 15-step pipeline:
> data collection -> parallel enrichment -> adversarial debate -> synthesis -> critic validation -> bias audit.

**Design Philosophy**: "Glass Box" -- every agent's inputs, reasoning, and outputs are visible. No black-box decisions.

---

## Tech Stack

| Layer | Technology | Port |
|-------|-----------|------|
| **Frontend** | Next.js 15 / React 19 / TypeScript 5.6 / Tailwind CSS / Geist / Phosphor Icons / Recharts | 3000 |
| **Backend** | FastAPI 0.115+ / Python 3.14 / Vertex AI (Gemini 2.0 Flash) | 8000 |
| **Storage** | Google BigQuery (`financial_reports` dataset), Google Cloud Storage (10-K/10-Q filings) |
| **AI** | Vertex AI (Gemini multi-provider via `llm_client.py`), RAG via Vertex AI Search, text-embedding-005 |

## Quick Start

```bash
# Backend (macOS — Python 3.13+)
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/python -m uvicorn backend.main:app --reload --port 8000

# Backend (Windows — Python 3.12+)
./.venv312/Scripts/python.exe -m pip install -r backend/requirements.txt
./.venv312/Scripts/python.exe -m uvicorn backend.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npx prisma migrate dev && npm run dev
```

`.venv312` is the canonical Windows environment. On macOS, use `.venv` with system Python 3.13+.

## Directory Structure (Top Level)

```
pyfinagent/
+-- backend/
|   +-- agents/         # Orchestrator, debate, risk_debate, bias, trace, memory, skills/
|   +-- api/            # FastAPI routes (analysis, reports, signals, portfolio, backtest, etc.)
|   +-- backtest/       # Walk-forward ML engine, quant optimizer, result persistence, analytics
|   +-- config/         # Pydantic settings, skill-loaded prompts
|   +-- db/             # BigQuery client (88-column ML schema)
|   +-- services/       # Paper trader, outcome tracker, autonomous loop, perf metrics
|   +-- slack_bot/      # Slack bot (commands, scheduler, formatters)
|   +-- tasks/          # Background task runners (analysis)
|   +-- tools/          # 16 data tools (yfinance, SEC, options, patents, FRED, Monte Carlo, etc.)
|   +-- main.py         # FastAPI app + router registration
+-- frontend/
|   +-- src/app/        # 10 pages (analyze, signals, reports, backtest, paper-trading, etc.)
|   +-- src/components/ # DebateView, RiskDashboard, SignalCards, OptimizerInsights, OptimizerProgressChart, etc.
|   +-- src/lib/        # types.ts, api.ts, auth.ts, icons.ts
+-- docs/ARCHITECTURE.md  # Full architecture reference (agent registry, API endpoints, BQ schemas)
+-- UX-AGENTS.md          # Frontend UX conventions and component specs
+-- trading_agent.md      # Autonomous trading optimization (Karpathy autoresearch pattern)
+-- PLAN.md               # Ford's master optimization plan (Phase 0-4)
+-- PHASE0_FINDINGS.md    # Formula validation findings with academic citations
+-- CHANGELOG.md          # Version history (v1.0 -> v5.12)
```

## 15-Step Analysis Pipeline

| Step | Name | Agent/Tool |
|------|------|-----------|
| 0 | Market Intel | Alpha Vantage + yfinance (parallel) |
| 1 | Ingestion | Cloud Function -> BigQuery |
| 2 | Quant Data | Cloud Function + yfinance merge |
| 3 | Document Analysis | RAG Agent (Vertex AI Search on 10-K/10-Q) |
| 4 | Sentiment | Market Agent (Google Search Grounding) |
| 5 | Competitors | Competitor Agent (Google Search Grounding) |
| 6 | Data Enrichment | 12 tools in parallel (sector-routed) |
| 6b | Info-Gap Detection | AlphaQuanter ReAct: assess -> retry -> quality score |
| 7 | Enrichment Analysis | 12 LLM agents (one per tool) |
| 8 | Agent Debate | Bull<->Bear (N rounds) + Devil's Advocate + Moderator |
| 9 | Enhanced Macro | FRED + Alpha Vantage combined |
| 10 | Deep Dive | Contradiction probing via RAG |
| 11 | Synthesis | All outputs -> structured JSON (reflection loop with Critic) |
| 12 | Critic + Bias Audit | Hallucination check, bias detection, knowledge conflicts |
| 12c | Risk Assessment | Aggressive/Conservative/Neutral analysts + Risk Judge |

**Design patterns**: Reflection Loop (Synthesis<->Critic, max 2 iterations), Session Memory, Quality Gates, Sector Routing.

## Key Conventions

- **Agent communication**: All inter-agent data is structured JSON. See `backend/config/prompts.py` for schemas.
- **Signal format**: `{ "ticker": "AAPL", "signal": "BULLISH", "summary": "...", "data": {...} }`
- **DecisionTrace**: Every LLM call produces a trace (agent name, timestamp, signal, confidence, evidence, reasoning). Defined in `backend/agents/trace.py`.
- **Glass Box**: Every agent I/O visible in UI. Debate shows bull/bear arguments. Bias flags surfaced prominently.
- **Anti-hallucination**: Typed Fact Ledger anchors all agents to quant data. Critic uses Chain-of-Verification. Structured output via Pydantic schemas (`backend/agents/schemas.py`).
- **Skills system**: Agent prompts in `backend/agents/skills/*.md`, loaded via `load_skill()` + `format_skill()` with `{{variable}}` placeholders. SkillOptimizer runs autonomous prompt experiments.
- **Cost controls**: Per-agent output token limits (1024-4096), Lite Mode, budget warnings, prompt truncation.

## Scoring System

**5-Pillar Weighted Matrix** (0-10 scale):
| Pillar | Weight | Measures |
|--------|--------|---------|
| Corporate Quality | 35% | Moat, management, financial health |
| Industry Position | 20% | Competitive landscape, market share |
| Valuation | 20% | P/E, PEG, FCF yield, margin of safety |
| Sentiment | 15% | News/social, insider, options flow |
| Governance | 10% | Compensation, board independence |

**Final Score** = Sum(pillar x weight) -> 0.0-10.0

## Walk-Forward Backtest Engine

Zero-LLM-cost ML backtest system. Trains GradientBoosting on ~49 features with walk-forward expanding windows.

**Data**: 299K price rows, 149 tickers, 2018-2025. 4,412 macro data points (FRED 7-series). ~24 walk-forward windows.

**6 Strategies** (optimizer rotates between these):

| Strategy | Research Basis |
|----------|----------------|
| Triple Barrier (vol-adjusted) | López de Prado AFML Ch. 3, Almgren & Chriss (2000) tx cost |
| Quality Momentum | Asness et al. (2019) QMJ, Novy-Marx (2013) |
| Mean Reversion (2-stage) | Lo & MacKinlay (1990), uses `mr_holding_days` validation |
| Factor Model (5-factor) | Fama & French (2015), Carhart (1997), Jegadeesh & Titman (1993) |
| Meta-Label (2-stage) | López de Prado AFML Ch. 3.6, 3-fold CV secondary model |
| Blend (weighted vote) | Dietterich (2000) ensemble methods |

**Key techniques**: Sample weights (AFML Ch. 4), fractional differentiation (Ch. 5), DSR ≥ 0.95 guard (Bailey & LdP 2014), inverse-vol sizing (AQR), turbulence dampening (FinRL), Amihud liquidity filter.

**Autoresearch pattern** (Karpathy): autonomous loop — baseline → propose → measure → keep/discard → log. See `trading_agent.md`.

## Environment Variables

Backend: `backend/.env` -- GCP_PROJECT_ID, GEMINI_MODEL, API keys (AV, FRED, API_NINJAS), AUTH_SECRET, ALLOWED_EMAILS, debate/cost config.
Frontend: `frontend/.env.local` -- AUTH_SECRET, AUTH_GOOGLE_ID/SECRET, ALLOWED_EMAILS, NEXT_PUBLIC_API_BASE.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full env var reference.

## Authentication

NextAuth.js v5 (Google SSO + Passkey/WebAuthn). Backend validates JWE tokens via HKDF + AES-256-CBC. Email whitelist enforced on both sides. See `backend/api/auth.py` and `frontend/src/lib/auth.ts`.

## Security

- OWASP headers on all responses. CORS allows localhost + Tailscale.
- SEC EDGAR requires custom User-Agent (`FirstName LastName email@domain.com`).
- Secrets: `backend/.env` locally, Google Cloud Secret Manager in production.
- Input validation on all API endpoints. Ticker symbols sanitized. No raw user input to LLM prompts.

## Full References

| Document | Content |
|----------|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Complete agent registry, API endpoints (70+ routes), BQ schemas (88 columns), data tools, research foundations |
| [UX-AGENTS.md](UX-AGENTS.md) | Frontend component specs, design tokens, Phosphor icon conventions |
| [trading_agent.md](trading_agent.md) | Autonomous trading optimization, 3-loop architecture, MDA->Agent bridge |
| [CHANGELOG.md](CHANGELOG.md) | Version history with detailed migration notes |
