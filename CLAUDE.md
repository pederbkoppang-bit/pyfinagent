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

- **🔴 BEFORE EVERY MASTERPLAN STEP**: read and follow `.claude/agents/per-step-protocol.md` end-to-end. The five handoff files (contract.md, experiment_results.md, evaluator_critique.md, harness_log.md append, masterplan.json status flip) are NON-SKIPPABLE. Always spawn BOTH qa-evaluator AND harness-verifier IN PARALLEL (single message, two Agent tool calls) — never just one. Self-evaluation by the orchestrator is forbidden.
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

## Harness Protocol (MANDATORY — NOT SKIPPABLE)

Every masterplan step (`phase-X` → `phase-X.Y`) MUST follow the full
loop below. The workflow is load-bearing for phase-4 go-live — skipping
phases or files is a breach of the contract. This applies equally when
execution is manual (Claude spawns sub-agents directly) and when
`scripts/harness/run_harness.py` is the driver.

### The five-file protocol

Every step produces, in order, exactly these artifacts:

| Phase | File (under `handoff/current/`) | Must contain |
|-------|-------------------------------|--------------|
| RESEARCH | (no file of its own; research feeds the contract) | ≥3 sources, ≥10 URLs, cite per claim, read full papers not abstracts |
| PLAN | `contract.md` | Step id, research-gate summary, hypothesis, immutable success criteria copied verbatim from `.claude/masterplan.json`, plan steps, references |
| GENERATE | `experiment_results.md` | What was built/changed + file list + verbatim verification command output + artifact shape |
| EVALUATE | `evaluator_critique.md` | **Both** qa-evaluator AND harness-verifier verdicts (spawned IN PARALLEL, not sequentially), each quoting their reasoning + violated_criteria + checks_run. Never one without the other. |
| LOG | appended block in `handoff/harness_log.md` | `## Cycle N -- YYYY-MM-DD -- phase=X.Y result=PASS/CONDITIONAL/FAIL` header + summary |

These files must exist and be up-to-date BEFORE marking the step
`status: done` in `.claude/masterplan.json`. The
`archive-handoff` PostToolUse hook moves them to
`handoff/archive/phase-X.Y/` on step transition; never delete them.

### Dual-evaluator rule

`EVALUATE` always means both:

1. **qa-evaluator** — reviews the contract, code, and artifacts;
   returns PASS / CONDITIONAL / FAIL with cited violations.
2. **harness-verifier** — runs the immutable verification command
   from `.claude/masterplan.json` itself, reproduces the result,
   returns PASS / FAIL with numbers.

Both are spawned in ONE parallel block (a single message with two
`Agent` tool calls), never sequentially. If either returns FAIL, the
step does not pass. If qa-evaluator returns CONDITIONAL, the
blockers must be addressed in the same cycle before `status: done`.

### Research gate

Before PLAN is written, at least one of `researcher` or `Explore`
subagents must run. For greenfield architecture or new data sources
use `researcher` (external literature, URLs). For code-audit work
use `Explore` (internal file:line references). For steps with
external + internal scope run both in parallel.

### Scheduled harness

`scripts/harness/run_harness.py` owns phase-2 step 2.12 parameter
optimization. Run it at least once per session with a short
`--cycles 1 --iterations-per-cycle 10`. It writes the same five
files (the `archive-handoff` hook handles the rotation).

### Failure discipline

- F1 (retry loop): `consecutive_fails` counter, revert-not-restart,
  certified_fallback escalation after 3 consecutive FAILs.
- F2 (research-on-demand): planner emits `research_needed` flag
  with a 4-key brief (objective / output_format / tool_scope /
  task_boundaries). The harness reads this and re-spawns research
  before attempting GENERATE again.

### Never do

- Mark a step done without all five files.
- Run only one evaluator (qa OR harness-verifier).
- Amend a step's immutable verification criteria.
- Skip `harness_log.md` append (it feeds the Harness tab on the
  backtest page and the next cycle's resume detection).

## Git

- Commit early, commit often with descriptive messages
- 84 commits ahead of origin — push needs Peder's approval
- GitHub user: pederbkoppang-bit
