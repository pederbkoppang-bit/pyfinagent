# pyfinAgent — Complete System Architecture

_Last updated: 2026-04-06 18:16 UTC by Ford_

Three interconnected systems, four layers, 40+ agents.

---

## System Map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           OPENCLAW (Runtime Host)                           │
│  Gateway: ws://127.0.0.1:18789 · Mac Mini · Europe/Oslo                    │
│  Agent: main (Ford 🔧) · Model: claude-sonnet-4-20250514                   │
│  Workspace: ~/.openclaw/workspace                                          │
│                                                                             │
│  ┌─── Cron Jobs ──────────────────────────────────────────────────────┐    │
│  │ ✅ Heartbeat (30min, 07-23)     → main session                     │    │
│  │ ✅ Gateway Watchdog (5min)      → isolated session                 │    │
│  │ ✅ Work Driver (15min)          → main session                     │    │
│  │ ✅ Paper Trading Daily (16:30)  → isolated session                 │    │
│  │ ✅ Morning Status Report (7am)  → isolated session                 │    │
│  │ ✅ Evening Harness Summary (6pm)→ isolated session                 │    │
│  │ ✅ Slack Health Check (8/12/18) → isolated session                 │    │
│  │ ✅ Daily Memory Flush (23:00)   → main session                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─── Workspace Files ────────────────────────────────────────────────┐    │
│  │ AGENTS.md    — Operating instructions for Ford                     │    │
│  │ IDENTITY.md  — Ford's identity (🔧, quant research partner)       │    │
│  │ SOUL.md      — Personality, boundaries, continuity                 │    │
│  │ USER.md      — Peder's profile, preferences, contact              │    │
│  │ TOOLS.md     — Environment, SSH, services, key paths              │    │
│  │ BOOTSTRAP.md — Gateway startup checklist                           │    │
│  │ HEARTBEAT.md — Heartbeat protocol + master plan status             │    │
│  │ MEMORY.md    — Long-term curated memory                            │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─── Channels ───────────────────────────────────────────────────────┐    │
│  │ webchat     — OpenClaw Control UI                                  │    │
│  │ slack       — #ford-approvals (C0ANTGNNK8D), socket mode           │    │
│  │ iMessage    — +4794810537 (escalation fallback)                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌───────────────────────┐ ┌─────────────────┐ ┌──────────────────┐
│  LAYER 1: Analysis    │ │ LAYER 2: MAS    │ │ LAYER 3: Harness │
│  Pipeline (Gemini)    │ │ (Anthropic)     │ │ (Anthropic)      │
└───────────────────────┘ └─────────────────┘ └──────────────────┘
```

---

## Layer 1: Analysis Pipeline (Gemini)

**Purpose:** Per-ticker 15-step enrichment → debate → synthesis → signal generation
**Models:** Gemini 2.0 Flash (Vertex AI), fallback: GitHub Models, OpenAI
**Entry point:** `backend/agents/orchestrator.py` (1477 lines)

### Pipeline Flow

```
Screen (yfinance)
    │
    ▼
┌─── 15 Enrichment Agents (parallel) ───────────────────────────────┐
│                                                                    │
│  market_agent          — Market intelligence + macro context       │
│  enhanced_macro_agent  — FRED data, macro trends                   │
│  quant_model_agent     — Statistical model features                │
│  rag_agent             — RAG + Google Search Grounding              │
│  insider_agent         — SEC insider transactions                   │
│  options_agent         — Options flow analysis                      │
│  social_sentiment_agent— Reddit, Twitter, StockTwits               │
│  alt_data_agent        — Google Trends, alternative data            │
│  earnings_tone_agent   — Earnings call NLP                          │
│  anomaly_agent         — Statistical anomaly detection              │
│  nlp_sentiment_agent   — News sentiment NLP                         │
│  patent_agent          — Patent tracker                             │
│  sector_analysis_agent — Sector rotation + relative strength        │
│  sector_catalyst_agent — Sector-specific catalysts                  │
│  supply_chain_agent    — Supply chain risk analysis                  │
│                                                                    │
│  Sector-aware: SECTOR_SKIP_MAP skips irrelevant tools              │
└────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── Debate System (3-5 rounds) ─────────────────────────────────────┐
│  bull_agent            — Bullish thesis                             │
│  bear_agent            — Bearish thesis                             │
│  devils_advocate_agent — Challenges consensus                       │
│  moderator_agent       — Synthesizes debate → consensus             │
│  competitor_agent      — Peer comparison                            │
└────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── Risk Assessment ───────────────────────────────────────────────┐
│  aggressive_analyst    — High-risk perspective                     │
│  conservative_analyst  — Risk-averse perspective                   │
│  neutral_analyst       — Balanced view                             │
│  risk_judge            — Final risk verdict                        │
└────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─── Synthesis (Evaluator-Optimizer pattern) ────────────────────────┐
│  synthesis_agent       — Produces final signal                     │
│  critic_agent          — Reviews synthesis, may revise (max 2)     │
│  deep_dive_agent       — Extended analysis on request              │
│  scenario_agent        — Scenario modeling                         │
│  info_gap_agent        — Identifies missing information            │
└────────────────────────────────────────────────────────────────────┘
    │
    ▼
Signal → BigQuery → Frontend → Slack notifications
```

**Total Layer 1 agents: 28** (skill .md files in `backend/agents/skills/`)

### Support Modules

| Module | Purpose |
|--------|---------|
| `memory.py` | BM25 situation→lesson retrieval (learns from outcomes) |
| `trace.py` | `DecisionTrace` + `AnalysisContext` (XAI audit trail) |
| `bias_detector.py` | Tech/confirmation/recency bias detection |
| `conflict_detector.py` | Parametric vs real-time knowledge conflicts |
| `cost_tracker.py` | Per-agent token/cost tracking (28 models × 4 providers) |
| `compaction.py` | Context compression for constrained models |
| `evidence_engine.py` | Evidence strength scoring |
| `info_gap.py` | Missing information detection |
| `skill_optimizer.py` | Autoresearch-style prompt optimization |
| `llm_client.py` | Multi-provider router (Gemini, Claude, OpenAI, GitHub) |
| `schemas.py` | Pydantic output schemas (Gemini structured output) |

---

## Layer 2: MAS Orchestrator (Anthropic)

**Purpose:** Slack/iMessage routing, Q&A, research — Anthropic multi-agent pattern
**Models:** Claude Opus 4.6 + Claude Sonnet 4.6
**Entry point:** `backend/agents/multi_agent_orchestrator.py`
**Config:** `backend/agents/agent_definitions.py`

### Flow

```
User (Slack / iMessage / webchat)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Communication Agent (claude-sonnet-4-6)    │
│  • Classifies query → 3 tiers              │
│  • Routes to appropriate agent(s)           │
│  • max_tokens: 500                          │
└──────────────────┬──────────────────────────┘
         ┌─────────┼─────────┐
         ▼         ▼         ▼
     TRIVIAL    SIMPLE    COMPLEX
     (local)   (1 agent) (2-3 parallel)
         │         │         │
         ▼         ▼         ▼
  Direct resp  ┌────────────────────────────────────────┐
  (0 tokens)   │                                        │
               ▼                                        ▼
┌──────────────────────┐  ┌──────────┐  ┌──────────────────┐
│ Ford (Main Agent)    │  │ Analyst  │  │ Researcher       │
│ claude-opus-4-6      │  │ (Q&A)   │  │ claude-sonnet-4-6│
│ max: 1500 tok        │  │ opus-4-6│  │ max: 3000 tok    │
│ Orchestrates, plans  │  │ max:2500│  │ Literature, arXiv │
│ Triggers harness     │  │ Quant   │  │ Papers, evidence  │
│ Delegates to QA/Res  │  │ analysis│  │ RESEARCH.md       │
└──────────┬───────────┘  └─────────┘  └──────────────────┘
           │
           ▼  "More research needed?" (max 3 rounds)
┌────────────────────────────┐
│ Quality Gate (sonnet-4-6)  │
│ 4-criterion rubric:        │
│  Accuracy (0.0-1.0)        │
│  Completeness (0.0-1.0)    │
│  Groundedness (0.0-1.0)    │
│  Conciseness (0.0-1.0)     │
│ FAIL if any <0.6 or avg<0.7│
│ 3 few-shot calibration     │
└──────────┬─────────────────┘
           ▼
┌────────────────────────────┐
│ CitationAgent (sonnet-4-6) │
│ Adds [1] [2] source markers│
└──────────┬─────────────────┘
           ▼
       Response → User
```

**Total Layer 2 agents: 6** (Communication, Ford, Q&A, Researcher, Quality Gate, Citation)

### MAS Features

| Feature | Implementation |
|---------|---------------|
| Interleaved thinking | 2048 tok/turn budget per subagent |
| Scoring rubric | 4-criterion 0.0-1.0 with 3 few-shot examples |
| Observation masking | ACON-inspired, 60% context window trigger |
| Parallel tools | Multiple harness reads in single turn |
| Event bus | `mas_events.py` → SSE at `/api/mas/events` |
| Context resets | Fresh context per subagent turn |

---

## Layer 3: Harness (Autonomous Loop)

**Purpose:** Planner → Generator → Evaluator optimization cycles
**Models:** Claude (via agent system)
**Entry points:** `run_harness.py`, `backend/autonomous_harness.py`

### Flow

```
┌────────────────────────────────────────────────────────┐
│                 Autonomous Harness Loop                  │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   Planner    │───▶│  Generator   │───▶│  Evaluator │ │
│  │  (LLM-based) │    │  (executes)  │    │  (LLM-based│ │
│  │              │◀───│              │◀───│   review)  │ │
│  │ planner_     │    │ run_harness  │    │ evaluator_ │ │
│  │ agent.py     │    │ .py          │    │ agent.py   │ │
│  └──────────────┘    └──────────────┘    └────────────┘ │
│         │                                       │        │
│         ▼                                       ▼        │
│  Spot Checks: CostStress, RegimeShift, ParamSweep      │
│  Results → feedback → Planner learns                     │
└────────────────────────────────────────────────────────┘
          │
          ▼
  handoff/contract.md, experiment_results.md, evaluator_critique.md
  quant_results.tsv (Sharpe History chart)
```

### Harness Agents

| Agent | File | Purpose |
|-------|------|---------|
| Planner | `planner_agent.py` | Proposes parameter changes, strategy modifications |
| Evaluator | `evaluator_agent.py` | Reviews proposals, runs spot checks, PASS/FAIL |
| Meta Coordinator | `meta_coordinator.py` | Cross-loop sequencing (Quant→Skill→Perf) |
| Skill Optimizer | `skill_optimizer.py` | Autoresearch-style prompt optimization |

---

## Layer 4: Services & Infrastructure

### Backend Services (`backend/services/`)

| Service | File | Purpose |
|---------|------|---------|
| Paper Trader | `paper_trader.py` | Autonomous daily: Screen→Analyze→Decide→Trade |
| Autonomous Loop | `autonomous_loop.py` | Market-hours trading scheduler |
| Portfolio Manager | `portfolio_manager.py` | Position tracking, P&L |
| Outcome Tracker | `outcome_tracker.py` | Signal→outcome evaluation |
| Ticket Queue | `ticket_queue_processor.py` | Slack ticket processing (30s batch) |
| SLA Monitor | `sla_monitor.py` | Response time monitoring |
| Stuck Task Reaper | `stuck_task_reaper.py` | 15-minute timeout on hung tickets |
| Queue Notifications | `queue_notification.py` | Position updates |
| Response Delivery | `response_delivery.py` | Agent response routing |

### MCP Servers (`backend/agents/mcp_servers/`)

| Server | Purpose |
|--------|---------|
| `data_server.py` | Market data access (yfinance, FRED, BQ) |
| `backtest_server.py` | Backtest execution + results |
| `signals_server.py` | Signal generation + portfolio |

### Slack Bot (`backend/slack_bot/`)

| Module | Purpose |
|--------|---------|
| `app.py` | Entry point (AsyncApp, Socket Mode) |
| `assistant_lifecycle.py` | Slack AI Agent lifecycle (AsyncAssistant) |
| `streaming_integration.py` | Real MAS orchestrator → word-by-word streaming |
| `app_home.py` | App Home dashboard (live data from all 3 systems) |
| `commands.py` | Slash commands |
| `governance.py` | Audit logging, rate limiting, HITL |
| `scheduler.py` | Morning digest, proactive alerts |

### Frontend (`frontend/src/app/`)

| Route | Purpose |
|-------|---------|
| `/` | Home dashboard |
| `/analyze` | Deep analysis trigger |
| `/signals` | Signal overview |
| `/backtest` | Backtest results + Sharpe History |
| `/agents` | **MAS Dashboard** (live SSE events, run history, agent map) |
| `/paper-trading` | Paper trading portfolio |
| `/portfolio` | Portfolio management |
| `/performance` | Performance metrics |
| `/reports` | Analysis reports |
| `/settings` | System settings |

---

## Agent Inventory (Complete)

### By Layer

| Layer | Count | Provider | Purpose |
|-------|-------|----------|---------|
| 1. Analysis Pipeline | 28 | Gemini (Vertex AI) | Per-ticker enrichment, debate, synthesis |
| 2. MAS Orchestrator | 6 | Claude (Anthropic) | Routing, Q&A, research, quality control |
| 3. Harness | 4 | Claude (Anthropic) | Autonomous optimization cycles |
| **Total** | **38** | | |

### By Model

| Model | Agents | Purpose |
|-------|--------|---------|
| `gemini-2.0-flash` | 28 (Layer 1) | Analysis pipeline |
| `claude-opus-4-6` | Ford, Q&A Analyst | Orchestration, quantitative reasoning |
| `claude-sonnet-4-6` | Communication, Researcher, QG, Citation | Routing, research, quality |
| `claude-sonnet-4-20250514` | OpenClaw main agent | Gateway sessions, heartbeats |
| `claude-haiku-4-5` | OpenClaw cron jobs | Lightweight monitoring |

---

## Data Flow

```
Market Data (yfinance, FRED, BQ)
    │
    ▼
Layer 1: Orchestrator (28 agents) → Analysis Report → BQ
    │                                                  │
    ▼                                                  ▼
Layer 3: Harness (Planner ↔ Evaluator)          Frontend (10 routes)
    │                                                  │
    ▼                                                  ▼
Layer 2: MAS (Slack Q&A about results)          User (browser)
    │
    ▼
Slack / iMessage / OpenClaw webchat
```

---

## File Map

```
pyfinagent/
├── ARCHITECTURE.md          ← This file
├── PLAN.md                  ← Master plan (5 phases)
├── RESEARCH.md              ← Literature review
├── AGENTS.md                ← (legacy, not used by OpenClaw)
├── UX-AGENTS.md             ← Frontend component specs
│
├── backend/
│   ├── main.py              ← FastAPI app (port 8000)
│   ├── agents/
│   │   ├── orchestrator.py  ← Layer 1: 15-step pipeline (1477 lines)
│   │   ├── agent_definitions.py ← Layer 2: MAS config (4 agents)
│   │   ├── multi_agent_orchestrator.py ← Layer 2: MAS orchestrator
│   │   ├── planner_agent.py ← Layer 3: LLM planner
│   │   ├── evaluator_agent.py ← Layer 3: LLM evaluator
│   │   ├── meta_coordinator.py ← Layer 3: Cross-loop sequencing
│   │   ├── mas_events.py    ← Event bus (SSE)
│   │   ├── cost_tracker.py  ← Token/cost tracking
│   │   ├── llm_client.py    ← Multi-provider LLM router
│   │   ├── memory.py        ← BM25 situation memory
│   │   ├── trace.py         ← Decision trace (XAI)
│   │   ├── skills/          ← 28 agent skill prompts (.md)
│   │   └── mcp_servers/     ← 3 MCP servers
│   ├── api/
│   │   └── mas_events.py    ← /api/mas/* endpoints
│   ├── services/            ← 9 background services
│   ├── slack_bot/           ← 7 modules (Socket Mode)
│   └── backtest/            ← Engine + optimizer + experiments
│
├── frontend/
│   └── src/app/             ← 10 Next.js routes
│
├── handoff/                 ← Harness artifacts
├── run_harness.py           ← Harness entry point
└── scripts/                 ← Utilities

~/.openclaw/
├── openclaw.json            ← Gateway config (channels, agents, cron)
├── workspace/
│   ├── AGENTS.md            ← OpenClaw operating instructions (Ford)
│   ├── IDENTITY.md          ← Ford 🔧
│   ├── SOUL.md              ← Personality
│   ├── USER.md              ← Peder
│   ├── TOOLS.md             ← Environment
│   ├── BOOTSTRAP.md         ← Startup checklist
│   ├── HEARTBEAT.md         ← Heartbeat protocol
│   ├── MEMORY.md            ← Long-term memory
│   └── pyfinagent/          ← ← The project
└── agents/
    └── main/                ← Ford agent (124 sessions)
```

---

## Sync Points (OpenClaw ↔ pyfinAgent ↔ Slack)

| What | OpenClaw | pyfinAgent | Slack |
|------|----------|------------|-------|
| Agent models | openclaw.json (defaults) | `AGENT_CONFIGS` (runtime) | App Home dropdowns |
| Cron jobs | `openclaw cron list` | — | App Home cron section |
| Health | gateway status | `/api/health` | App Home health |
| Events | — | `mas_events.py` (SSE) | App Home recent events |
| Costs | — | `cost_tracker.py` | App Home cost tracker |
| Sessions | `sessions list` | — | App Home session count |
| Model changes | — | `POST /api/mas/agents/{type}/model` | Dropdown → runtime |
| Dashboard | — | `GET /api/mas/dashboard` | App Home + frontend `/agents` |

---

## References

- [Anthropic: Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Anthropic: Harness Design for Long-Running Apps](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138)
- Bailey & López de Prado (2014): Deflated Sharpe Ratio
- Lo (2002): Serial correlation correction for Sharpe ratio
