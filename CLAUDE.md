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

- **🔴 MAS HARNESS LOOP — NON-NEGOTIABLE FOR EVERY MASTERPLAN STEP.** This project is a long-running autonomous application; the canonical reference is Anthropic's ["Harness Design for Long-Running Apps"](https://www.anthropic.com/engineering/harness-design-long-running-apps). Every step follows the three-phase cycle `Plan → Generate → Evaluate` with file-based handoffs as durable state. Read `docs/runbooks/per-step-protocol.md` end-to-end before starting ANY step. **The MAS is exactly 3 agents: Main (this session) + Researcher + Q/A.** Researcher absorbs the old `Explore` subagent's role (internal code exploration) AND external literature research in one session. Q/A absorbs the old `harness-verifier` role (deterministic reproduction) AND LLM judgment in one session. The five handoff artifacts (`handoff/current/contract.md`, `handoff/current/experiment_results.md`, `handoff/current/evaluator_critique.md`, `handoff/harness_log.md` append, `.claude/masterplan.json` status flip) are NON-SKIPPABLE. **Spawn `researcher` BEFORE every contract (the research gate)** and spawn `qa` ONCE after every GENERATE. **Self-evaluation by the orchestrator is forbidden** (Anthropic: "agents tend to confidently praise their own work"). Periodically stress-test the scaffolding — as models improve, any assumption "the model can't do X" is worth re-running without the harness to prune dead weight.
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
- **Agent definition changes require session restart.** `.claude/agents/*.md` files are snapshotted by the Agent-tool loader at session start. Adding/merging/renaming agents mid-session won't make them dispatchable until you `/clear` or restart Claude Code. When you edit agent files, note in the handoff that the next session cycle must verify the new roster is live.
- **Separation of duties on agent edits.** The same Claude Code session should not both author an agent `.md` change AND self-evaluate work that depends on it. For substantive edits to `.claude/agents/`, leave a note in `handoff/harness_log.md` requesting Peder review before the next step depends on the change.

## Architecture (see ARCHITECTURE.md for full details)

4 layers:
1. **Analysis Pipeline** (28 Gemini agents) — `backend/agents/orchestrator.py`
2. **MAS Orchestrator** (in-app Claude agents — domain orchestration) — `backend/agents/multi_agent_orchestrator.py`
3. **Harness MAS (exactly 3 agents)** — Main (this Claude Code session) + Researcher (`.claude/agents/researcher.md`) + Q/A (`.claude/agents/qa.md`). Driven by `scripts/harness/run_harness.py` + `backend/autonomous_harness.py`. Researcher absorbs the old `Explore` role; Q/A absorbs the old `harness-verifier` role. No separate Explore. No separate harness-verifier. Don't re-split.
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

The BigQuery MCP server is **harness-injected** (not pinned in `.mcp.json`).
The harness environment injects it with **read AND write**
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

Canonical reference: https://www.anthropic.com/engineering/harness-design-long-running-apps
plus "How We Built Our Multi-Agent Research System" and "Building
Effective Agents." Project implementation: `.claude/agents/per-step-protocol.md`.

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
| EVALUATE | `evaluator_critique.md` | **Q/A verdict** (single agent, merged qa-evaluator + harness-verifier). Must include deterministic checks_run + LLM judgment + violated_criteria + verdict (PASS / CONDITIONAL / FAIL). |
| LOG | appended block in `handoff/harness_log.md` | `## Cycle N -- YYYY-MM-DD -- phase=X.Y result=PASS/CONDITIONAL/FAIL` header + summary |

These files must exist and be up-to-date BEFORE marking the step
`status: done` in `.claude/masterplan.json`. The
`archive-handoff` PostToolUse hook moves them to
`handoff/archive/phase-X.Y/` on step transition; never delete them.

### Single-Q/A rule (was: dual-evaluator)

`EVALUATE` is handled by the merged **Q/A** agent
(`.claude/agents/qa.md`). Q/A runs deterministic-first:
1. Syntax + file-existence + immutable verification command exit code
2. Reads existing `evaluator_critique.md` + `experiment_results.md`
3. Optional harness dry-run if within 55s budget
4. LLM judgment covers contract alignment, mutation-resistance,
   anti-rubber-stamp, scope honesty, research-gate compliance

Returns `{ok, verdict, violated_criteria, violation_details,
certified_fallback, checks_run}`.

**If `ok: false` / verdict is CONDITIONAL or FAIL — the canonical
cycle-2 flow (per Anthropic's
[multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
and [harness design](https://www.anthropic.com/engineering/harness-design-long-running-apps)):**

1. Main reads the critique's violated_criteria + violation_details.
2. Main fixes the blockers **and updates the handoff files**
   (`experiment_results.md`, `evaluator_critique.md` appended
   Follow-up section, plus any code/doc the critique flagged).
3. Main spawns a **fresh** Q/A. The fresh Q/A reads the updated
   files — evidence has changed, so the new verdict reflects the
   fix, not a different opinion on the same evidence.

This is NOT "second-opinion-shopping". The documented pattern is
file-based communication between fresh instances per phase:
> "Communication was handled via files: one agent would write a
> file, another agent would read it and respond either within
> that file or with a new file"
> — Anthropic, *Harness design for long-running apps*

> "The LeadResearcher synthesizes these results and decides
> whether more research is needed—if so, it can create additional
> subagents"
> — Anthropic, *How we built our multi-agent research system*

**What IS forbidden**: spawning a fresh Q/A to overturn a verdict
on **unchanged evidence** (no file updates, no fix applied) hoping
for a different answer. That's second-opinion-shopping and
compromises evaluator independence.

**Historical note on `SendMessage`**: earlier iterations of this
rule prescribed `SendMessage` back to the same Q/A instance.
Subagent tool definitions did not include `SendMessage` in their
tool lists until 2026-04-18, and even with the tool, Anthropic's
documented subagent lifecycle is single-turn synchronous
(one-shot). Dormant agents don't auto-replay on inbox delivery.
The file-based fresh-respawn pattern is both the documented path
and the empirically reliable one.

### Research gate (MUST-BE-USED)

Before PLAN is written, the `researcher` subagent must run. It now
covers BOTH external literature AND internal code-audit in one
session (the old `Explore` subagent has been merged in). Pass the
caller the effort tier (`simple` / `moderate` / `complex`). If
`gate_passed: false`, do not proceed to PLAN. **Main drifts on this
under time pressure** — auto-memory `feedback_research_gate.md` and
phase-4.10 audit document 7-of-9-cycle slips. Enforcement layers:
- `InstructionsLoaded` hook reloads this rule every session
- Researcher description uses "MUST BE USED" phrasing so auto-
  delegation fires proactively
- Q/A's LLM-judgment leg checks for researcher output in the
  contract's references section

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
- Skip RESEARCH because "we've been here before" — if the step is
  new, tier can be `simple` but the phase can't be skipped.
- Re-split agents: reintroducing `Explore` or `harness-verifier` as
  separate files after they've been merged is the old pattern.
- Amend a step's immutable verification criteria.
- Skip `harness_log.md` append (it feeds the Harness tab on the
  backtest page and the next cycle's resume detection).
- Self-evaluate (Main reporting PASS without spawning Q/A).
- Second-opinion-shop after CONDITIONAL on **unchanged evidence**
  — spawning a fresh Q/A without fixing the flagged blockers and
  updating the handoff files is forbidden (that's verdict-shopping).
  Conversely, spawning a fresh Q/A AFTER fixing blockers and
  updating the files IS the documented pattern — the new verdict
  reflects the fix, not a different opinion. See the "canonical
  cycle-2 flow" block above.

### Stress-test doctrine (Anthropic)

"Every component in a harness encodes an assumption about what the
model can't do on its own, and those assumptions are worth stress
testing." On each new Claude model release, re-run a representative
step WITHOUT the harness (no subagents, no handoff files) and compare
the output to the harness-produced result. If the model now does X on
its own, remove the scaffolding for X. Stale scaffolding is dead
weight — prune it.

## Git

- Commit early, commit often with descriptive messages
- 84 commits ahead of origin — push needs Peder's approval
- GitHub user: pederbkoppang-bit
