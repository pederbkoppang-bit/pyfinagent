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

- **🔴 MAS HARNESS LOOP — NON-NEGOTIABLE FOR EVERY MASTERPLAN STEP.** This project is a long-running autonomous application; the canonical reference is Anthropic's ["Harness Design for Long-Running Apps"](https://www.anthropic.com/engineering/harness-design-long-running-apps). Every step follows the three-phase cycle `Plan → Generate → Evaluate` with file-based handoffs as durable state. Read `docs/runbooks/per-step-protocol.md` end-to-end before starting ANY step. **The Harness MAS layer (Layer 3) is exactly 3 agents: Main (this session) + Researcher + Q/A.** The broader dev MAS (Layer-2 in-app orchestrator agents in `backend/agents/agent_definitions.py` + Layer-4 meta-evolution services like `backend/meta_evolution/directive_review.py`) has additional members; `backend/agents/_inventory.json` is the canonical roster. Don't conflate the two — names like "Researcher" or "Ford" in Layer-2 are distinct from the Layer-3 Claude Code subagents with the same root labels. Researcher absorbs the old `Explore` subagent's role (internal code exploration) AND external literature research in one session. Q/A absorbs the old `harness-verifier` role (deterministic reproduction) AND LLM judgment in one session. The five handoff artifacts (`handoff/current/contract.md`, `handoff/current/experiment_results.md`, `handoff/current/evaluator_critique.md`, `handoff/harness_log.md` append, `.claude/masterplan.json` status flip) are NON-SKIPPABLE. **Spawn `researcher` BEFORE every contract (the research gate)** and spawn `qa` ONCE after every GENERATE. **Self-evaluation by the orchestrator is forbidden** (Anthropic: "agents tend to confidently praise their own work"). Periodically stress-test the scaffolding — as models improve, any assumption "the model can't do X" is worth re-running without the harness to prune dead weight.
- **Always `source .venv/bin/activate`** before running Python
- **Always call `cache.preload_macro()`** or backtests hang after ~40min
- **Kill parent AND child workers** when restarting backend (zombie prevention)
- **Backend (8000) + Frontend (3000) must always be running and in sync**
- **No emojis in frontend** — use Phosphor Icons only
- **Every backtest result** → save to `backend/backtest/experiments/results/` + append to `quant_results.tsv`
- **BQ timeout: 30s** on all fallback queries
- **BigQuery MCP is available** — see "BigQuery Access (MCP)" section below. Use it for schema inspection, data validation, and read-only analytics before touching Python BQ clients.
- **LLM API costs** require Peder's explicit approval
- **UI verification via Playwright MCP (goal-post-away-review, 2026-06-10)** -- whenever
  the operator pastes a UI screenshot, or any step makes a claim about the UI, ALWAYS
  verify against the RUNNING app via the Playwright MCP (browser_navigate +
  browser_snapshot / browser_take_screenshot) behind the NextAuth wall. Code reading
  alone is not UI evidence. Every UI-touching live_check includes a Playwright capture.
- **Always read `.claude/masterplan.json`** before starting work — it's the machine-readable task tracker
- **Use `/masterplan`** to see current state and next actionable step
- **Never edit verification criteria** in masterplan.json — they are immutable
- **Research Gate is mandatory** — no step proceeds to GENERATE without deep research (see PLAN.md lines 44-83)
- **Read `.claude/context/`** for project memory: project.md, mas-architecture.md, research-gate.md, owner.md
- **NEVER manually update CHANGELOG.md** — the PostToolUse hook does it automatically on every commit, including a semver-aware version bump per Conventional Commits (`feat:` → minor, `fix:`/`bug:`/`perf:` → patch, `BREAKING CHANGE`/`feat!:` → major, `phase-X.0:` → minor, `phase-X.Y:` → patch, `chore:`/`docs:`/`refactor:`/`test:`/`style:`/`ci:`/`build:` → no version row). See `.claude/hooks/post-commit-changelog.sh::classify_commit`. Skip changelog tasks entirely.
- **Commit message convention** — every meaningful commit MUST start with a Conventional Commits prefix so the classifier picks the right bump. Examples: `feat: add new endpoint`, `fix(scheduler): null guard`, `phase-23.7.1: research brief`, `chore: housekeeping`. Bare unprefixed subjects default to patch (safe fallback).
- **Per-step auto-push** — when a masterplan step status flips to `done` in `.claude/masterplan.json`, the `auto-commit-and-push.sh` PostToolUse hook stages all changes, commits with the step's name as subject, invokes the changelog hook, and pushes to `origin/main`. No manual `git push` per step. Push failures log to `handoff/logs/auto-push.log` and exit 0 (do not break the masterplan Write). Re-run `git push origin main` manually if the log shows a failure.
- **`verification.live_check` gate (phase-23.8.1 / audit R-1)** — masterplan steps may set an optional `verification.live_check` field (any non-empty string describing the required evidence shape — e.g. `"curl output from /api/paper-trading/portfolio showing post-fix sector_breakdown"` or `"BQ row from paper_trades.signals with lite_path=true after the next autonomous cycle"`). When set, the auto-commit-and-push hook calls `.claude/hooks/lib/live_check_gate.py` which checks for `handoff/current/live_check_<step_id>.md`. If the file is absent the hook logs a WARN line to `handoff/logs/auto-push.log` and **skips the auto-push** for that step — the commit and changelog still happen, but the push to `origin/main` is held. Operator workflow: create the file with verbatim live-system evidence (curl output, BQ query result, screenshot path) and re-trigger by re-editing the masterplan (any no-op edit), OR run `git push origin main` manually. The gate is fail-open: any helper error → proceed as today, consistent with the hook's discipline of never breaking the masterplan Write. Design grounded in Anthropic harness-design's file-based handoff pattern; see `docs/audits/dev-mas-2026-05-11/04-remediation.md` R-1 for the audit basis. The gate exists to convert "the agent claimed PASS" into "an artifact exists that an operator can audit"; it directly attacks the VERIFICATION_DEFECT systemic pattern surfaced by the dev-MAS audit.
- **ALWAYS append to `handoff/harness_log.md`** after completing a masterplan step — use the cycle format (see existing entries). This feeds the Harness tab on the backtest page. The append should happen BEFORE the status flip so it's included in the auto-commit.
- **ALWAYS work on main branch** — run `git checkout main && git pull origin main` at startup. Push directly to main, never create feature branches or PRs.
- **Agent definition changes require session restart.** `.claude/agents/*.md` files are snapshotted by the Agent-tool loader at session start. Adding/merging/renaming agents mid-session won't make them dispatchable until you `/clear` or restart Claude Code. When you edit agent files, note in the handoff that the next session cycle must verify the new roster is live. **Verification path** (phase-23.3.0): run `scripts/qa/verify_qa_roster_live.sh` after restart — the script checks on-disk state + origin/main commit visibility and embeds the literal self-disclosure prompt to send a fresh Q/A subagent to confirm the new section is in its snapshot. The retry-on-FAIL doctrine (`docs/runbooks/per-step-protocol.md` §4 "Retry-on-FAIL loop") describes what to do if the new section is NOT in the snapshot.
- **Separation of duties on agent edits.** The same Claude Code session should not both author an agent `.md` change AND self-evaluate work that depends on it. For substantive edits to `.claude/agents/`, leave a note in `handoff/harness_log.md` requesting Peder review before the next step depends on the change.
- **Fable 5 policy (updated 2026-07-09, goal-phase67-fable-window; history: adopted phase-59.1 2026-06-11, reverted 2026-07-08 burn-down day)** -- Claude Fable 5 (`claude-fable-5`; Claude Code alias `fable`, requires v2.1.170+) is $10/$50 per Mtok (2x Opus 4.8) via the metered API and draws Max USAGE CREDITS outside free windows -- so the standing doctrine is: **Fable only where it runs free (the Claude Code Max rail), only on dev-harness roles, only during free windows, always with a scheduled revert.** Current state (REVERTED 2026-07-13 via masterplan step 67.4): the free Fable 5 window on Max ended 2026-07-12; no `FABLE PERMANENT: AUTHORIZE` was recorded in `handoff/harness_log.md`, so the scheduled revert executed on schedule -- the Layer-3 Researcher + Q/A frontmatter is back to `model: opus` (steady state). Every phase-67 artifact upgrade (qa lint+runtime-smoke gates, code-review consumer-contract-break heuristic, researcher WRITE-FIRST discipline) was retained across the revert. Any future Fable repin requires a fresh free-window confirmation AND a new scheduled revert step. ALL Layer-2/in-app pins stay OFF Fable (2026-07-08 revert holds: `mas_main`/`autoresearch_strategic`/ticket agents on `claude-opus-4-8`); PER-TICKER/METERED ROLES STAY OFF FABLE (`mas_qa`, `mas_communication`, `mas_research`, `autoresearch_fast/smart`, all `gemini_*`) -- do not repin any in-app role without a fresh cost analysis. Fable's effort table is low->max; doc baseline `high` ("often exceeds xhigh performance on prior models"); gate roles deliberately over-spec at `max` -- NOTE `xhigh` silently downgrades to `high` for non-Opus-4.8/4.7 models (llm_client.py:1507-1512), so Fable roles must use `max`, never `xhigh`. Classifier note: Fable silently falls back to Opus 4.8 on cyber/bio/distillation prompts (finance unaffected); never instruct an agent to echo its reasoning as output text (reasoning_extraction refusal). `EFFORT_SUPPORTED_MODELS` + `MODEL_EFFORT_FALLBACK` carry `claude-fable-5` entries -- REQUIRED, else `llm_client.py` silently drops the effort param. Session switching: `/model fable`. Agent-file pins take effect at the NEXT session start (roster snapshot); verify with `scripts/qa/verify_qa_roster_live.sh`. STALL WATCH: two Fable Q/A spawns stalled mid-evaluation 2026-07-09 (Cycle-76 addendum) -- if a fresh-session Fable Q/A stalls, revert the qa.md pin immediately; Opus is the reliable evaluator.
- **Effort policy (Layer-3 harness MAS — operator override, phase-29.2 2026-05-18; model upgrade 2026-05-28)** — canonical source: https://platform.claude.com/docs/en/build-with-claude/effort. Anthropic's *recommended* baseline is Opus 4.8/4.7 → `xhigh`, Sonnet 4.6 → `medium` (the 4.8 effort doc explicitly says: "The guidance for Claude Opus 4.7 above also applies to Claude Opus 4.8. Start with `xhigh` for coding and agentic use cases"; default is `high` on all surfaces incl. Claude Code, so explicit set is required for `xhigh`/`max`). This project deliberately runs over-spec because the owner is on a **Max subscription** (flat-fee, no per-token ceiling on Claude Code first-party usage) and because both Researcher and Q/A are **rare-event roles** (fire once per masterplan step, not per ticker), so token cost is contained regardless of effort. Audit-basis: overnight prompt 2026-05-18 + `handoff/archive/phase-29.2/research_brief.md` (17-pt GPQA Diamond gap + 79-Elo GDPval-AA gap of Opus over Sonnet 4.6 — quality-depth matters on research-synthesis and evaluator-gate work). 2026-05-28 model bump (4-7 → 4-8): same $5/$25 pricing, same flagship Opus tier, strict improvement on agentic coding + honesty + financial-analysis benchmarks per Anthropic news (https://www.anthropic.com/news, "Introducing Claude Opus 4.8"); xhigh remains accepted on 4.8.
  - **Main (this Claude Code session)** — `.claude/settings.json:effortLevel = xhigh` (Opus 4.8 as of 2026-05-28, was 4.7 — recommended starting point per doc). Switch sessions to 4.8 via `/model claude-opus-4-8`.
  - **Q/A subagent** — `.claude/agents/qa.md` frontmatter `model: opus` (alias auto-resolves to latest Opus = 4.8) + `effort: max`. Codified permanent in phase-29.2 (was xhigh pre-23.2.2; raised in phase-23.2.2 with intent-to-revert that was never executed across 30+ cycles; now made permanent).
  - **Researcher subagent** — `.claude/agents/researcher.md` frontmatter `model: opus` (alias → 4.8) + `effort: max`. **Operator override of Anthropic baseline** (Sonnet 4.6 / medium): research depth on Opus outperforms Sonnet 4.6 on GPQA-adjacent tasks (multi-source synthesis, analytical judgment, code audit). Subagent frontmatter `model: opus` is officially supported per https://code.claude.com/docs/en/sub-agents + https://code.claude.com/docs/en/model-config. Max plan auto-includes Opus 1M context, mitigating GitHub issue #51060.
  - **Layer-2 in-app MAS** (`backend/config/model_tiers.py::EFFORT_DEFAULTS`) is a **separate system** from Layer-3. `mas_main` runs at `xhigh` Opus 4.8; `mas_qa` historically `high` (cost-balanced because `mas_qa` fires per ticker analysis, NOT per dev cycle). 2026-05-28: `mas_main`, `mas_qa`, `autoresearch_strategic` model pins bumped 4-7 → 4-8 with owner sign-off in the same upgrade pass — same $5/$25 pricing means the original rate-limit/cost rationale for separate audit (per-ticker firing) didn't change between versions.
  - Per-model fallback table at `backend/config/model_tiers.py::MODEL_EFFORT_FALLBACK` (Opus 4.8/4.7 → xhigh; Sonnet 4.6 → medium). Roles in `EFFORT_DEFAULTS` override the model fallback.
  - When adding a new Claude agent, default to the Anthropic-recommended pairing for that model family unless this project's Max-subscription + rare-event rationale applies. Cite this CLAUDE.md section in the PR if you deviate.
- **MCP `alwaysLoad` discipline (phase-29.0-F8 / phase-40.2, Claude Code v2.1.121+)** -- `.mcp.json` per-server `alwaysLoad: true|false` controls whether that server's tools skip tool-search deferral at session start. The real adoption lives in `.mcp.json:44,55,66,77`:
  - `pyfinagent-data` -- `alwaysLoad: true` (constant BQ inspection)
  - `pyfinagent-risk` -- `alwaysLoad: true` (kill-switch + PBO gates)
  - `pyfinagent-backtest` -- `alwaysLoad: false` (rare invocation; tool-search deferral OK)
  - `pyfinagent-signals` -- `alwaysLoad: false` (1887 lines; startup cost matters)
  - `playwright` -- `alwaysLoad: false` (phase-59.2: fires ~1-3x per UI-touching step, not every turn; ~22 tool defs; alwaysLoad blocks session startup until the npx server connects [5s cap] -- pure tax for an episodic server. Pinned `@playwright/mcp@0.0.76` as of 2026-06-11; NOTE: editing `.mcp.json` mid-session does NOT respawn a connected stdio server -- reconnect via `/mcp` or disclose the capture-time version in the live_check. The BINDING UI-verification rule lives in `.claude/agents/qa.md` §1c + the Critical-Rules Playwright bullet above.)
  - External MCP servers (alpaca, bigquery, paper-search-mcp) omit the key (default false).
  - The **Figma MCP** is a claude.ai SESSION CONNECTOR (`mcp__claude_ai_Figma__*`), NOT pinned in `.mcp.json` -- it is absent in headless/cron runs, so it is ADVISORY-ONLY (design work) and never verification-load-bearing. Workflow: `.claude/rules/frontend.md` "Figma MCP workflow".
  The phase-40.2 statusMessage cross-reference in `.claude/settings.json` config-change-audit hook is a discoverability pointer so future readers can find the `.mcp.json` adoption from the settings file (Claude Code's strict schema validation forbids `_doc_*` top-level keys; `statusMessage` accepts any string).
- **Hook `continueOnBlock` (phase-40.2, Claude Code v2.1.139+)** -- a per-hook-entry child key valid only on `prompt`-type hook entries inside `PostToolUse`. When set, it feeds the hook's rejection reason back to Claude and continues the turn instead of halting. pyfinagent currently uses only `command`-type hooks (schema does not accept `continueOnBlock` on command type), so the phase-40.2 adoption is a discoverability cross-reference in the config-change-audit statusMessage. When a future `prompt`-type hook is added (e.g. for `feedback_auto_commit_hook_stalls.md` mitigation), set `continueOnBlock: true` on it.
- **Hook-level `effort.level` visibility (phase-40.2, Claude Code v2.1.141+)** -- hooks receive the active effort tier via the `effort.level` JSON input field AND the `$CLAUDE_EFFORT` env var. This reflects the ACTUAL level after any model-downgrade fallback. Distinct from the top-level `effortLevel` settings.json key (xhigh on this project) which is the persistent-session default. Hooks that need to gate on effort (e.g. skip expensive checks at `low`) should read `$CLAUDE_EFFORT`.

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

The BigQuery MCP server is **pinned in `.mcp.json`** as of phase-23.2.21.
Package: `mcp-server-bigquery==0.3.2` (LucasHild, Feb 2026), launched via
`uvx`, mirroring the alpaca MCP shape. Authenticates via the user's
**Application Default Credentials** (`~/.config/gcloud/application_default_credentials.json`)
— no per-session OAuth, no env vars. Project + location are passed as
CLI args (`--project sunny-might-477607-p8 --location US`). Prefer these
tools over spinning up a Python `bigquery.Client` for ad-hoc inspection,
validation, and analytics — they leave no local state.

**Project:** `sunny-might-477607-p8`

**Datasets:**
| Dataset | Location | Purpose |
|---|---|---|
| `pyfinagent_data` | US | Primary prod data (signals, prices, fundamentals, macro, llm_call_log, strategy_decisions) |
| `pyfinagent_staging` | US | Staging / pre-prod |
| `pyfinagent_hdw` | US | Historical data warehouse |
| `pyfinagent_pms` | US | Active holdings view + alpha velocity samples + directive versions + portfolio_status_snapshot + portfolio_transactions + strategy deployments (legacy portfolio-management tables; NOT the paper-trading tables) |
| `financial_reports` | us-central1 | Financial filings AND **paper trading tables** (`paper_trades`, `paper_positions`, `paper_portfolio`). The paper-trading tables live HERE, not in `pyfinagent_pms`, per `backend/db/bigquery_client.py:486` (`_pt_table()` uses `settings.bq_dataset_reports = "financial_reports"`). Discovered during phase-23.2.2 reconciliation; doc clarified 2026-05-16. |
| `all_billing_data` | EU | GCP billing export |

**Available MCP tools** (names are `mcp__bigquery__<tool>`; discover via `ToolSearch` with query `bigquery`):
- `mcp__bigquery__list-tables` — enumerate tables in the configured project/dataset
- `mcp__bigquery__describe-table` — return schema + metadata for a table
- `mcp__bigquery__execute-query` — arbitrary SQL (read AND write — no separate
  readonly variant on this package). Denied by default in `.claude/settings.json`
  so write-class queries require explicit user approval per call.

**Rules:**
1. **Default to `list-tables` / `describe-table` for inspection.** Only reach for
   `execute-query` when SQL is truly required. Each `execute-query` call is gated
   by an approval prompt (deny rule in `.claude/settings.json`).
2. **Always bound queries.** Add `LIMIT` and partition/date filters on
   `historical_*` tables or costs balloon fast.
3. **Obey the 30s timeout rule** from Critical Rules above — if a query risks
   exceeding it, add filters or sample instead. The pinned MCP accepts
   `--timeout` if a different ceiling is needed.
4. **Never `DROP` or unqualified `DELETE`** without explicit owner approval
   (see `.claude/context/owner.md`). The deny rule on `execute-query`
   already forces a prompt; treat that prompt as a real gate, not a rubber-stamp.
5. **Migration scripts still live in** `scripts/migrations/*.py` and use the
   Python `google-cloud-bigquery` client for schema changes that need to be
   version-controlled and re-runnable. Don't replace those with ad-hoc MCP
   calls — use MCP for *inspection*, migrations for *change*.
6. **If MCP tools aren't present** in a given session (e.g. the server failed
   to attach, or the user hasn't restarted Claude Code since the pin), fall
   back to the Python client with `GCP_PROJECT_ID` from `backend/.env`. ADC
   on the user's Mac covers both paths.
7. **Smoke test** lives at `scripts/mcp_servers/smoke_test_bigquery_mcp.py`.
   Run it after upgrading the pinned version or if the server stops attaching.

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

**Launch — Workflow structured-output is FIRST-CLASS; Agent-tool is the
fallback (phase-71.1).** The primary unattended launch for Q/A (and the
Researcher gate) is the checked-in `.claude/workflows/qa-verdict.js`
Workflow script, run via the Workflow tool with `args={step_id,
criteria[], verification_command, evidence}`. It runs the role as
`agent(prompt, {schema, agentType:'general-purpose', model:'opus',
effort:'max'})` and **the verdict is the captured return value** —
structured-outputs GA (constrained decoding) guarantees the shape, so it
does NOT depend on a subagent file-write flush. This is the stall-immune
path (the Agent-tool subagent end-flush stalled 6× on 2026-07-11,
intermittent + model-agnostic — auto-memory
`feedback_workflow_qa_when_subagents_stall`); it runs $0 on the Opus Max
rail. The Agent-tool `qa`/`researcher` subagents are the documented
**fallback** (and the worktree CI path). **Main MUST transcribe the
returned verdict VERBATIM** into `handoff/current/evaluator_critique.md`
(no editorial edits, no paraphrase) — Main records the verdict, never
authors it, so the no-self-eval guarantee stays airtight. An
errored/empty return is **NO VERDICT, never PASS** → fall back to the
Agent-tool path. The Q/A returns a verdict and STOPS; it never loops
fix→re-grade internally (Main owns the fix + spawns a FRESH Q/A on
changed evidence — the cycle-2 flow below). Single-Q/A-per-step and the
exactly-3-agents doctrine are unchanged: the Workflow path is a launch
mechanism, not a fourth agent. The Workflow launch has the Q/A **read
`qa.md` from disk at runtime**, so a `qa.md` edit is live immediately on
this path; only the Agent-tool roster snapshots at session start.

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
  **3rd-CONDITIONAL auto-FAIL:** if a single step-id accumulates 3+
  consecutive CONDITIONAL verdicts without an intervening PASS or
  FAIL, the next Q/A pass MUST return FAIL (not another CONDITIONAL).
  Q/A reads `handoff/harness_log.md` to count prior CONDITIONALs for
  that step-id. Counter resets on PASS, FAIL, or new step-id. This
  prevents the harness from logging instead of correcting (see
  `docs/runbooks/per-step-protocol.md` §4 EVALUATE for full text).
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
