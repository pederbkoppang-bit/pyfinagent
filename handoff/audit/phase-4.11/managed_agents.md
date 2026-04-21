# Managed Agents Deep Audit (phase-4.11.1)

## URL coverage

All 9 canonical Managed Agents pages fetched in full via WebFetch. No
linked `/v1/agents`, `/v1/sessions`, `/v1/environments` reference pages
were available as standalone targets beyond what is inlined in the
guides; every endpoint surface used in the guides is documented below.

- overview, quickstart, agent-setup, sessions, skills, tools, memory,
  files, vaults — all retrieved 2026-04-18.
- Features beta-gated (`managed-agents-2026-04-01` header on every
  request). Memory, outcomes, multi-agent are Research Preview and
  require a separate access form.

## Per-page digests

### overview
Managed Agents is explicitly framed as the **opposite product surface**
to the Messages API. Messages = "direct model prompting, custom agent
loops"; Managed Agents = "pre-built, configurable agent harness that
runs in managed infrastructure". Core objects: `agent` (persona +
tools + skills, versioned), `environment` (cloud container template),
`session` (running instance), `events` (SSE). Runs Claude 4.5+ only.
Rate limits: 60/min create, 600/min read per org, plus tier spend
limits.

### quickstart
Installs `ant` CLI + SDKs (Python/TS/Go/Java/C#/Ruby/PHP). Flow:
`POST /v1/agents` → `POST /v1/environments` → `POST /v1/sessions` →
`POST /v1/sessions/{id}/events` + SSE stream at
`/v1/sessions/{id}/stream`. `agent_toolset_20260401` enables the
full built-in toolset. Session stays `idle` until a user event; agent
autonomously tool-calls until it emits `session.status_idle`.

### agent-setup
Agents are **versioned resources**. Fields: `name`, `model`, `system`,
`tools`, `mcp_servers`, `skills`, `callable_agents` (multi-agent, RP),
`description`, `metadata`. Updates generate new versions with
optimistic concurrency (`version` argument). Lifecycle: update → new
version; list versions; archive (read-only; existing sessions keep
running). Agents can be pinned per-session by passing
`{type:"agent", id, version}`.

### sessions
A session requires `agent` + `environment_id`. Statuses:
`idle | running | rescheduling | terminated`. Sessions are
**stateful** — history persisted server-side, container mounted,
retrievable and listable. Event delivery: POST user events, open
SSE stream. Archive preserves history and blocks new events; delete
tears down container + events. Files, memory stores, environments,
and agents are independent and survive session deletion. Supports
`vault_ids[]` and `resources[]` (files, memory_stores, GitHub repos).

### skills
**Same SKILL.md model as Claude Code**: filesystem-based, progressive
disclosure, attached to the agent. Two flavors: `anthropic` pre-built
(e.g., `xlsx`, `pptx`, `docx`, `pdf`) and `custom` org-authored with
versioning (`latest` or pinned). Cap: 20 skills per session. Skills
are invoked automatically when relevant; they do not consume context
until needed.

### tools
Built-in toolset (`agent_toolset_20260401`): `bash`, `read`, `write`,
`edit`, `glob`, `grep`, `web_fetch`, `web_search` — a 1:1 subset of
Claude Code's harness. Per-tool enable/disable via `configs[]`;
`default_config.enabled:false` for whitelist mode. Custom tools are
client-executed (Messages-API-equivalent tool-use contract) and MCP
servers attach at agent level.

### memory
**Research Preview.** Memory stores (`memstore_...`) are workspace-
scoped collections of ≤100KB text "memories" mounted per session via
`resources[].memory_store`. Up to 8 stores/session, `read_only` or
`read_write`. Agent gets `memory_{list,search,read,write,edit,delete}`
tools automatically. Every mutation creates an immutable `memver_...`
with full audit trail, optimistic concurrency via
`content_sha256`/`not_exists` preconditions, and a `redact` endpoint
for PII/secret scrubbing that keeps the audit record but nukes the
content. This is the first-class replacement for our BM25-over-BQ
long-term memory.

### files
Upload via Files API → mount at `resources[].file` with arbitrary
`mount_path` (read-only inside container, absolute paths). Up to 100
files/session. Files are resources independent of session lifecycle.
Session-scoped listing via `files.list(scope_id=sesn_...)` lets you
retrieve artifacts the agent produced. Copies into session don't
count against storage limits.

### vaults
Per-end-user credential primitive. Workspace-scoped. Holds up to 20
`credential` objects, each bound immutably to a single
`mcp_server_url`. Two auth types: `mcp_oauth` (Anthropic handles
refresh when you register `refresh.token_endpoint` + client auth
style) and `static_bearer`. Secret fields write-only, never returned.
`vault_ids[]` passed at session creation; mid-session rotation
propagates without restart. Only useful for **MCP-server auth** — not
a general secret manager (cannot inject arbitrary env vars into the
container, cannot hold non-MCP keys).

## pyfinAgent fit analysis

**1. Is it a different product surface?**
Yes. Managed Agents is a fully **server-hosted, stateful container
harness** — Anthropic runs the agent loop, the sandbox, the tool
execution, and persists event history. The Messages API we rely on
(`llm_client.py`, all 28 Gemini agents via Vertex, our MAS
orchestrator) is not replaced — Managed Agents only hosts Claude
models (4.5+) and does not support Gemini, so Layer 1 stays on
Messages/Vertex regardless.

**2. Would Layer-2 MAS or the harness cycle benefit from migration?**
*Layer-2 MAS* (`multi_agent_orchestrator.py`): mixed. Managed Agents
would give us free sandboxed bash/file tools, SSE streaming, and
server-side conversation state — but we already run these agents in
our own FastAPI process and need tight integration with BQ, paper
trader, ticket queue. Migration cost is high for modest gain.
*Harness cycle* (`scripts/harness/run_harness.py`,
`autonomous_harness.py`): **potentially high-value**. The harness is
long-running, tool-heavy, already follows Plan→Generate→Evaluate with
Claude Opus. Managed Agents natively supports: durable sessions,
resume semantics, event log = `handoff/`-equivalent, SSE streaming to
the frontend Harness tab, vault for MCP auth, memory stores for cross-
cycle learnings (replacing our `pyfinagent_data.harness_learning_log`
BQ table). The dual-evaluator pattern maps cleanly onto
`callable_agents` (multi-agent RP).

**3. Cost / retention / residency.**
No public pricing table on these pages. The container compute is
billed in addition to model inference. Rate-limited 60 create / 600
read per min per org. Data residency not discussed — assume US-only
until Anthropic documents otherwise; a blocker for any
EU-residency-sensitive data (our GCP billing export is EU; none of
our prod data is). Session archive preserves history indefinitely;
delete is hard. Memory versions accumulate forever until explicitly
deleted or redacted.

**4. Vaults vs. GCP Secret Manager / env vars.**
**Vaults are narrowly scoped to MCP-server auth.** They do not
replace Secret Manager for GCP service accounts, Slack signing
secrets, NextAuth keys, Anthropic/Gemini API keys, etc. If we ever
add user-authorized MCP servers (e.g., per-user Slack, Linear,
GitHub OAuth for the Slack bot), vaults would be the right tool and
would eliminate us writing a per-user OAuth token store. For our
current single-tenant admin-only app, no immediate relevance.

**5. Does it solve our file-based handoff problem?**
Partially, and worth serious thought for phase-4.11+. Our
`handoff/current/{contract,experiment_results,evaluator_critique}.md`
+ `harness_log.md` is essentially a hand-rolled implementation of
what Managed Agents gives natively as: session event history +
memory store + session `resources[]`. The five-file protocol is load-
bearing precisely because the Messages API has no server-side
session. If we move the harness loop onto Managed Agents, three of
the five files become server-side primitives; we'd keep
`contract.md` (human-readable plan) and `harness_log.md` (cross-cycle
summary, which maps to a memory store). **But** the protocol's
real value is Anthropic's "harness design" discipline (immutable
success criteria, dual evaluator, research gate) — which is
orthogonal to where state lives. Moving to Managed Agents would not
remove the discipline, only the file plumbing.

## MUST FIX
None. This is greenfield. Our current harness is conformant with the
Anthropic harness-design doctrine; it just uses a different storage
substrate.

## NICE TO HAVE / adoption evaluation

Ranked by ROI:

1. **Pilot the harness cycle on Managed Agents (phase-4.12 candidate).**
   Single-agent first: port `run_harness.py` GENERATE phase to a
   Managed Agent session with `agent_toolset_20260401`, attach a
   memory store in place of `harness_learning_log`, keep
   qa-evaluator and harness-verifier as local subagents until
   `callable_agents` leaves Research Preview. Expected win: kill
   zombie-worker problems, free SSE stream for the Harness tab, and
   get audited memory versioning for free. Request memory + multi-
   agent RP access via the form linked in the overview page.

2. **Adopt Anthropic pre-built skills** (xlsx/pptx/docx/pdf) for the
   Slack bot and investor-report flow. Replaces any hand-rolled
   openpyxl/python-pptx paths. Zero migration cost — just attach
   them to the agent config.

3. **Defer vaults** until we add user-facing MCP integrations.
   Current single-tenant model doesn't need them; GCP Secret Manager
   continues to cover service-account secrets.

4. **Do NOT migrate Layer-1 or Layer-2 yet.** Layer 1 is Gemini-bound;
   Managed Agents is Claude-only. Layer 2 has too much local
   orchestration (paper trader, ticket queue) to justify the
   container round-trip cost per turn.

5. **Stress-test doctrine check.** Per CLAUDE.md, "every harness
   component encodes an assumption about what the model can't do" —
   Managed Agents is Anthropic's own answer to the same question,
   so it is worth re-running a representative harness step via a
   Managed Agent session (no local five-file plumbing) and comparing
   the output quality/cost to our current run. That experiment is a
   direct test of whether our scaffolding is still load-bearing.

## References

- https://platform.claude.com/docs/en/managed-agents/overview
- https://platform.claude.com/docs/en/managed-agents/quickstart
- https://platform.claude.com/docs/en/managed-agents/agent-setup
- https://platform.claude.com/docs/en/managed-agents/sessions
- https://platform.claude.com/docs/en/managed-agents/skills
- https://platform.claude.com/docs/en/managed-agents/tools
- https://platform.claude.com/docs/en/managed-agents/memory
- https://platform.claude.com/docs/en/managed-agents/files
- https://platform.claude.com/docs/en/managed-agents/vaults
- https://www.anthropic.com/engineering/harness-design-long-running-apps
  (project canonical harness doctrine)
- `/Users/ford/.openclaw/workspace/pyfinagent/CLAUDE.md` — harness
  protocol, research-gate, stress-test doctrine
- `/Users/ford/.openclaw/workspace/pyfinagent/scripts/harness/run_harness.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/autonomous_harness.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/multi_agent_orchestrator.py`
