# CLAUDE.md — pyfinAgent Project Context

## Quick Start

```bash
# Backend (always activate venv first)
cd pyfinagent
source .venv/bin/activate
python -m uvicorn backend.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev  # port 3000

# Harness (autonomous optimization)
source .venv/bin/activate
python scripts/harness/run_harness.py [--cycles N] [--iterations-per-cycle N] [--dry-run]

# Slack bot (standalone process)
source .venv/bin/activate
python -m backend.slack_bot.app

# Verify syntax after code changes
python -c "import ast; ast.parse(open('path/to/file.py').read())"
```

## Critical Rules

- **Always `source .venv/bin/activate`** before running Python
- **Always call `cache.preload_macro()`** or backtests hang after ~40min
- **Kill parent AND child workers** when restarting backend (zombie prevention)
- **Backend (8000) + Frontend (3000) must always be running and in sync**
- **No emojis in frontend** — use Phosphor Icons only
- **Every backtest result** → save to `backend/backtest/experiments/results/` + append to `quant_results.tsv`
- **BQ timeout: 30s** on all fallback queries
- **BigQuery MCP is available** — see "BigQuery Access (MCP)" section below. Use it for schema inspection, data validation, and read-only analytics before touching Python BQ clients.
- **LLM API costs** require Peder's explicit approval
- **Always read `.claude/masterplan.json`** before starting work — it's the machine-readable task tracker
- **Use `/masterplan`** to see current state and next actionable step
- **Never edit verification criteria** in masterplan.json — they are immutable
- **Research Gate is mandatory** — no step proceeds to GENERATE without deep research (see PLAN.md lines 44-83)
- **Read `.claude/context/`** for project memory: project.md, mas-architecture.md, research-gate.md, owner.md
- **NEVER manually update CHANGELOG.md** — the PostToolUse hook does it automatically on every commit. Skip changelog tasks entirely.
- **ALWAYS append to `handoff/harness_log.md`** after completing a masterplan step — use the cycle format (see existing entries). This feeds the Harness tab on the backtest page.
- **ALWAYS work on main branch** — run `git checkout main && git pull origin main` at startup. Push directly to main, never create feature branches or PRs.

## Architecture (see ARCHITECTURE.md for full details)

4 layers, 38 agents:
1. **Analysis Pipeline** (28 Gemini agents) — `backend/agents/orchestrator.py`
2. **MAS Orchestrator** (6 Claude agents) — `backend/agents/multi_agent_orchestrator.py`
3. **Harness** (4 Claude agents) — `scripts/harness/run_harness.py`, `backend/autonomous_harness.py`
4. **Services** — Paper trading, ticket queue, SLA monitor, Slack bot

## Stack

- **Backend:** FastAPI + Python 3.14, port 8000
- **Frontend:** Next.js 15 + React 19 + TypeScript 5.6 + Tailwind, port 3000
- **AI:** Vertex AI (Gemini), Anthropic (Claude Opus/Sonnet), multi-provider via `llm_client.py`
- **Storage:** BigQuery, GCS
- **Auth:** NextAuth.js v5 (Google SSO + Passkey/WebAuthn)

## Key Files

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app entry |
| `backend/agents/orchestrator.py` | Layer 1: 15-step pipeline (1477 lines) |
| `backend/agents/agent_definitions.py` | Layer 2: MAS agent configs |
| `backend/agents/multi_agent_orchestrator.py` | Layer 2: MAS orchestrator |
| `backend/agents/planner_agent.py` | Layer 3: LLM planner |
| `backend/agents/evaluator_agent.py` | Layer 3: LLM evaluator |
| `backend/agents/skills/*.md` | 28 analysis agent prompts |
| `backend/backtest/backtest_engine.py` | Backtest engine (1167 lines) |
| `backend/backtest/quant_optimizer.py` | Quant optimizer |
| `backend/backtest/experiments/optimizer_best.json` | Current best parameters |
| `backend/slack_bot/app.py` | Slack bot entry (Socket Mode) |
| `frontend/src/lib/api.ts` | API client (Bearer auth, 30s timeout) |
| `frontend/src/lib/types.ts` | All TypeScript interfaces |
| `frontend/src/lib/icons.ts` | Centralized Phosphor icon exports |

## BigQuery Access (MCP)

The harness environment injects a BigQuery MCP server with **read AND write**
access to project **`sunny-might-477607-p8`**. Prefer these tools over
spinning up a Python `bigquery.Client` for ad-hoc inspection, validation, and
analytics — they're faster, require no auth plumbing, and leave no local state.

**Project:** `sunny-might-477607-p8`

**Datasets:**
| Dataset | Location | Purpose |
|---|---|---|
| `pyfinagent_data` | US | Primary prod data (signals, prices, fundamentals, macro) |
| `pyfinagent_staging` | US | Staging / pre-prod |
| `pyfinagent_hdw` | US | Historical data warehouse |
| `pyfinagent_pms` | US | Portfolio management / paper trading |
| `financial_reports` | us-central1 | Financial filings |
| `all_billing_data` | EU | GCP billing export |

**Available MCP tools** (names are `mcp__<server-id>__<tool>`; discover via `ToolSearch` with query `bigquery`):
- `list_dataset_ids` — enumerate datasets in a project
- `list_table_ids` — enumerate tables in a dataset
- `get_dataset_info` / `get_table_info` — schema + metadata
- `execute_sql_readonly` — SELECT only, safe default
- `execute_sql` — full DML/DDL (INSERT, UPDATE, DELETE, MERGE, CREATE, DROP)

**Rules:**
1. **Default to `execute_sql_readonly`.** Only use `execute_sql` when the task
   explicitly requires a mutation, and show the SQL in the session log first.
2. **Always bound queries.** Add `LIMIT` and partition/date filters on
   `historical_*` tables or costs balloon fast.
3. **Obey the 30s timeout rule** from Critical Rules above — if a query risks
   exceeding it, add filters or sample instead.
4. **Never `DROP` or unqualified `DELETE`** without explicit owner approval
   (see `.claude/context/owner.md`).
5. **Migration scripts still live in** `scripts/migrations/*.py` and use the
   Python `google-cloud-bigquery` client for schema changes that need to be
   version-controlled and re-runnable. Don't replace those with ad-hoc MCP
   calls — use MCP for *inspection*, migrations for *change*.
6. **If MCP tools aren't present** in a given session (e.g. the server didn't
   attach), fall back to `bq` CLI or the Python client with `GCP_PROJECT_ID`
   from `backend/.env`.

**Typical uses during autonomous runs:**
- Sanity-check that a backtest's input tables have fresh data before running
- Verify harness learning logs are being written (`pyfinagent_data.harness_learning_log`)
- Spot-check signal outputs vs. expectations
- Validate migration outcomes without a full Python round-trip

## Testing

```bash
# Backend syntax check
python -c "import ast; ast.parse(open('backend/path/file.py').read())"

# Frontend build check
cd frontend && npm run build

# Run harness with dry-run
python scripts/harness/run_harness.py --dry-run --cycles 1
```

## Frontend Conventions

- Read `.claude/rules/frontend.md` + `frontend-layout.md` before ANY frontend work
- Page shell: `<div className="flex h-screen overflow-hidden"><Sidebar /><main>...</main></div>`
- Icons: import from `@/lib/icons.ts`, never `@phosphor-icons/react` directly
- Charts: Recharts only, dark theme (`#0f172a` bg)
- Scrollable containers: always `scrollbar-thin`
- Every component needs error/loading/empty states

## Backend Conventions

- Read `.claude/rules/backend-agents.md` for agent pipeline rules
- Output limits: Enrichment 1024, Debate 1536, Synthesis 4096 tokens
- Structured output via Gemini JSON schema enforcement
- Google Search Grounding is Gemini-only (degrades on Claude/OpenAI)
- Agent memories persisted to BQ, loaded on startup via BM25

## Harness Protocol

Every plan step follows: RESEARCH → PLAN → GENERATE → EVALUATE → LOG
- RESEARCH gate: ≥3 sources, ≥10 URLs, read full papers not abstracts
- PLAN: `handoff/contract.md` with success criteria
- GENERATE: Do the work, `handoff/experiment_results.md`
- EVALUATE: Independent verification, `handoff/evaluator_critique.md`
- LOG: Update PLAN.md, RESEARCH.md, memory, Slack

## Git

- Commit early, commit often with descriptive messages
- 84 commits ahead of origin — push needs Peder's approval
- GitHub user: pederbkoppang-bit
