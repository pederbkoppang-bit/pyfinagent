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

| Server | Purpose | Phase |
|--------|---------|-------|
| `data_server.py` | Read-only market data (yfinance, FRED, BQ) | 3.0 |
| `backtest_server.py` | Backtest execution + results | 3.0 |
| `signals_server.py` | Signal generation + validation + publish | 3.0 / 3.7.2 |
| `risk_server.py` | Risk gate: kill_switch, PBO veto, 6 tools | 3.7.3 |

Cross-cutting infra (not MCP servers but part of the MCP architecture):
- `backend/agents/mcp_capabilities.py` -- HMAC-SHA256 capability tokens (30-min TTL per NIST SP 800-63B-4), 6 roles mapped to fixed scope sets, PII scrub on inbound args (phase-3.7.7)
- `backend/services/mcp_health_cron.py` -- weekly supply-chain health monitor: stale repo detection, license audit, CVE scan (phase-3.5.7)
- `.mcp.json` -- external MCP server registry (Slack, Alpaca), version-pinned for supply-chain hardening (phase-3.7.6 / phase-3.0)

Detailed reference:
- `docs/MCP_ARCHITECTURE.md` -- server inventory, transport choice, phase lineage, ADR cross-links
- `docs/MCP_SECURITY.md` -- threat model, capability tokens, PII scrub, supply-chain pinning, rate-limit deferral rationale

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

## Anthropic Files API — ZDR status (phase-4.14.16)

pyfinAgent uses the Anthropic beta Files API (header
`anthropic-beta: files-api-2025-04-14`) for SEC filings >32 MB via
`backend/tools/sec_insider.py::upload_large_filing_to_files_api`.
Uploaded `file_id`s are re-referenced across downstream pipeline calls
so the filing bytes are only transmitted once.

**ZDR status:** the Files API is NOT eligible for zero-data-retention
as of 2026-04. Do NOT upload filings containing customer PII through
this pathway. Re-evaluate when Anthropic changes ZDR eligibility.

## Research Gate Discipline (phase-4.16)

MADR-style record of the research-gate rules every MAS cycle
follows. This is the REFERENCE doc; the how-to lives in
`.claude/rules/research-gate.md`, and the agent-facing rules live
in `.claude/agents/researcher.md`. Cross-link only -- do not
duplicate.

### Context

Phase-4.16.1 raised the research-gate floor from >=3 sources to
>=5 sources read in full + a mandatory last-2-year scan, after
Peder flagged cycles (e.g. researcher_64) that cited 3 URLs in a
footer but fetched only 1 in full. Anthropic's built-multi-agent-research-system
post confirms "agents consistently chose SEO-optimized content
farms over authoritative but less highly-ranked sources" -- the
enforced floor is the countermeasure.

### Decision

Every `researcher` spawn (any tier) must:

1. Fetch and read IN FULL at least **5** authoritative sources via
   `WebFetch`. Tier (simple/moderate/complex) controls DEPTH and
   LENGTH of the brief, never the source floor.
2. Perform an explicit **last-2-year scan** (e.g. 2024-2026 at
   time of writing). Report outcome even when "no new findings".
3. Emit the JSON envelope unconditionally with
   `external_sources_read_in_full`, `snippet_only_sources`,
   `recency_scan_performed`, `gate_passed`.
4. Return `gate_passed: false` when fewer than 5 sources were
   fetched in full, rather than padding the brief.

### Consequences

Q/A gates PASS/CONDITIONAL verdicts on the JSON envelope. A
brief that lists 5 URLs in a footer but only fetched 1 is a
CONDITIONAL (auditable). The supplementary-researcher pattern
(cycle-2 with a fresh researcher spawn) is the documented repair
when a brief under-fetches.

### Handoff folder convention (phase-4.16.2)

- `handoff/current/` -- the **currently-in-flight** step's files
  plus `_templates/` (canonical MD scaffolds). No done-step files.
- `handoff/archive/phase-<sid>/` -- completed-step snapshots.
  Populated by `.claude/hooks/archive-handoff.sh` on masterplan
  status flip. Idempotent with `-v{n}` suffix.
- `handoff/audit/` -- append-only JSONL audit streams from hooks
  (pre_tool_use, config_change, instructions_loaded,
  prompt_leak_redteam).
- `handoff/logs/` -- runtime process logs (gitignored).

Backfill + verifier at `scripts/housekeeping/`. The verifier is
the immutable criterion for phase-4.16.2.

### Confirmation

- `grep -q "Research Gate" ARCHITECTURE.md` -- this section anchors
  the reference.
- `grep -q "5 sources" .claude/rules/*.md` -- the how-to file
  carries the agent-visible clause.
- `python scripts/housekeeping/verify_handoff_layout.py` exits 0.

### Cross-references

- `.claude/agents/researcher.md` (agent prompt)
- `.claude/rules/research-gate.md` (how-to guide)
- `CLAUDE.md` (authoritative cycle protocol; cross-links here)
- `docs/runbooks/per-step-protocol.md` (operator runbook)
