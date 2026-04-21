# Managed Agents Sub-pages Audit (phase-4.12.2)

Deep read of 9 sub-pages the phase-4.11 sweep didn't cover as standalone
pages. All URLs returned HTTP 200; the docs site is a Mintlify SPA so
WebFetch sometimes saw the "Not Found – Loading..." shell and had to be
retried against the canonical slug.

## URL coverage

| # | Requested URL | Status | Canonical URL used |
|---|---|---|---|
| 1 | `/managed-agents/mcp-connector` | CHECKED | same |
| 2 | `/managed-agents/permission-policies` | CHECKED | same |
| 3 | `/managed-agents/claude-api-skill` | FAILED (no such page) | does not exist; coverage picked up from `/managed-agents/skills` + `/managed-agents/anthropic-skills` (both SPA 404) |
| 4 | `/managed-agents/cloud-environment-setup` | CHECKED (via `/managed-agents/environments`) | `/managed-agents/environments` |
| 5 | `/managed-agents/container-reference` | CHECKED (via `/managed-agents/cloud-containers`) | `/managed-agents/cloud-containers` |
| 6 | `/managed-agents/session-event-stream` | CHECKED (via `/managed-agents/events-and-streaming`) | `/managed-agents/events-and-streaming` |
| 7 | `/managed-agents/outcomes` | CHECKED (via `/managed-agents/define-outcomes`) | `/managed-agents/define-outcomes` |
| 8 | `/managed-agents/github-access` | CHECKED | same |
| 9 | `/managed-agents/multi-agent` | CHECKED | same |

**Key finding up top:** there is no "Claude API skill" sub-page. "Skills"
on Managed Agents means reusable filesystem expertise packs (pre-built
Anthropic `xlsx`/`pptx`/`docx`/`pdf` plus org-custom skills capped at 20
per session). Nested Messages-API calls from inside a session are not
exposed as a skill — the agent *is* the Claude call, and further
sub-agent dispatch is done via `callable_agents` (multiagent).

## Per-page digests

### 1. MCP connector (managed-agents variant)
- Two-step wiring: `mcp_servers` on **agent** (name + URL, no secrets),
  `vault_ids` on **session** (per-session credentials via pre-registered
  vaults).
- Transport: **remote MCP servers only, streamable HTTP**. No stdio, no
  local process spawning. Custom FastMCP servers are supported iff they
  expose an HTTPS endpoint implementing the streamable-HTTP transport.
- Auth failure is non-fatal: session still starts; a `session.error` event
  emits and retries happen on the next `idle → running` transition.
- Default permission policy for MCP tools: `always_ask` (see #2).
- Vs. the standalone `/agents-and-tools/mcp-connector`: the managed
  variant adds the *agent / session / vault* split — the agent definition
  is reusable, secrets live in vaults, and runtime approval is the
  session's responsibility. The standalone connector is a single-call
  pattern with no server-side vault.

### 2. Permission policies
- **Only two types**: `always_allow` and `always_ask`. No RBAC, no ABAC,
  no per-argument rules, no per-user scoping.
- Granularity: toolset-level `default_config.permission_policy`, with
  per-tool overrides via `configs[].name` (e.g. override `bash` to
  `always_ask` while keeping everything else `always_allow`).
- Default for the agent toolset = `always_allow`; default for every MCP
  toolset = `always_ask`.
- Enforcement is **on the Anthropic platform** — when a tool hits
  `always_ask`, the session goes to `status_idle` with
  `stop_reason.requires_action.event_ids[]`; client sends
  `user.tool_confirmation` with `result: "allow" | "deny"` (+ optional
  `deny_message`).
- **Custom tools are not governed by permission policies** — your
  application receives `agent.custom_tool_use` and fully owns the
  execute-or-reject decision before returning `user.custom_tool_result`.

### 3. (No "Claude API skill" page)
- The docs surface `xlsx`, `pptx`, `docx`, `pdf` as Anthropic-built
  skills; "custom" skills are org-authored markdown+asset bundles
  uploaded via the Files API and referenced by `skill_id`.
- Limit: 20 skills per session (sum across all agents in a multiagent
  session).
- **No documented mechanism for a managed agent to issue a nested
  `/v1/messages` Claude API call as a "skill".** Nested LLM calls are
  done by calling a sibling managed agent (multiagent / callable_agents,
  see #9).

### 4. Cloud environment setup
- `POST /v1/environments` with `config.type: "cloud"`. Name unique per
  org+workspace.
- **Packages pre-install** supported for `apt`, `cargo`, `gem`, `go`,
  `npm`, `pip`. Version-pinnable; cached across sessions sharing the env.
- **Networking** (critical for pyfinagent pilot): `unrestricted`
  (default, minus an Anthropic safety blocklist) or `limited`
  + `allowed_hosts` (HTTPS only) + `allow_mcp_servers` bool
  + `allow_package_managers` bool.
- **No VPC peering, no private inbound.** Egress is the only lever. To
  reach our FastAPI on port 8000 we'd need a public HTTPS endpoint on an
  allowlisted host, or run everything behind a CloudFront/Cloud-Run style
  proxy. No mention of GCP IAM / service-account injection — credentials
  flow through vaults (MCP) or `authorization_token` fields (GitHub).
- Envs are archivable but **not versioned**; each session gets an
  isolated container (no shared FS state between sessions).

### 5. Container reference
- **Ubuntu 22.04 LTS / x86_64**, up to **8 GB RAM, 10 GB disk**.
- Network disabled by default (turn on via env `networking` config).
- Pre-installed: Python 3.12+ (pip, uv), Node 20+ (npm/yarn/pnpm), Go
  1.22+, Rust 1.77+, Java 21+, Ruby 3.3+, PHP 8.3+, GCC 13+.
- Utilities: `git`, `curl`, `jq`, `tar/zip`, `ssh/scp`, `tmux`, `make`,
  `cmake`, `ripgrep`, `tree`, `htop`, `vim`, `nano`, SQLite, `psql`,
  `redis-cli`. Docker listed as "limited availability".
- **Not pre-installed (matters for us):** `gcloud`, `bq`, `gh`. These
  would need to be added via `apt` packages or a pip/npm substitute
  (`google-cloud-bigquery` Python SDK is pip-installable).
- Output convention for outcome-driven sessions: `/mnt/session/outputs/`
  → auto-exposed via Files API scoped to the session.

### 6. Session event stream (events-and-streaming)
- **SSE** at `/v1/sessions/:id/stream`; threads at
  `/v1/sessions/:id/threads/:id/stream`; all events also listable via
  `/events` for replay.
- User events: `user.message`, `user.interrupt`,
  `user.custom_tool_result`, `user.tool_confirmation`,
  `user.define_outcome`. Agent events: `agent.message`,
  `agent.thinking`, `agent.tool_use`/`tool_result`, `agent.mcp_tool_use`,
  `agent.custom_tool_use`, and in multiagent, `agent.thread_message_sent`
  / `agent.thread_message_received`. Session events: `session.created`,
  `session.status_running`, `session.status_idle` (with `stop_reason`),
  `session.error`, `session.thread_created`, `session.thread_idle`.
- Reconnect = reopen the SSE; full event history is persisted
  server-side and re-fetchable; auth = `x-api-key` header.
- **The frontend should NOT consume this stream directly** — API key
  exposure. The proxy pattern (backend relays SSE to the Harness tab) is
  the right architecture.

### 7. Define outcomes (research preview)
- Requires `managed-agents-2026-04-01-research-preview` beta header.
- Send `user.define_outcome` with `description`, `rubric` (inline text
  or Files-API `file_id`), and `max_iterations` (default 3, max 20).
- A **separate grader** (own context window) scores per-criterion against
  the rubric and returns one of: `satisfied`, `needs_revision`,
  `max_iterations_reached`, `failed`, `interrupted`.
- Events: `span.outcome_evaluation_start` / `_ongoing` / `_end`
  (with per-criterion `explanation` + usage). Grader reasoning is
  **opaque** — "you see that it's working, not what it's thinking".
- Only one outcome at a time; chain by sending a new `user.define_outcome`
  after the terminal event.

### 8. Accessing GitHub
- **Mount pattern**: `resources: [{type: "github_repository", url,
  mount_path, authorization_token}]` on session create.
- PAT-based (fine-grained PAT recommended). Scopes: `repo` for private
  clone / PR / issue create. No GitHub App flow documented.
- Repos are cached across sessions. **Multiple repos** supported. Token
  is rotatable via `PATCH /v1/sessions/:id/resources/:id`.
- PR creation is not a filesystem push — it goes through the **GitHub
  MCP server** (`https://api.githubcopilot.com/mcp/`) which the agent
  calls as a tool. Raw `git push` from inside the container is not the
  documented path.

### 9. Multiagent sessions (research preview)
- `callable_agents: [{type: "agent", id, version}]` on the orchestrator
  agent. **Only one level of delegation** — sub-agents cannot spawn their
  own sub-agents.
- **Shared container + filesystem**, but each sub-agent runs in its own
  **thread**: isolated context window, own model / system prompt /
  tools / MCP servers / skills.
- Threads are persistent: you can send follow-ups and state is retained.
- Parallel execution is explicitly supported ("Agents can act in parallel
  with their own isolated context, which helps improve output quality
  and time to completion").
- Session stream shows aggregated activity + `agent.thread_message_*` and
  `session.thread_created/idle`; drill into a specific thread via
  `/threads/:id/stream`.
- `session_thread_id` must be echoed on tool-confirmation /
  custom-tool-result replies when the request originated in a subagent.

## Harness-pilot fit analysis

Mapping our five-phase loop (RESEARCH / PLAN / GENERATE / EVALUATE / LOG)
onto managed agents:

| Our construct | Managed-agents construct | Fit |
|---|---|---|
| `scripts/harness/run_harness.py` driver | Outer application holding the API key, sending user events, rotating cycles | GOOD — standard pattern |
| `handoff/current/*.md` files | Session filesystem (persistent per-session) + Files API export on idle | GOOD — `/mnt/session/outputs/` matches `archive-handoff` semantics |
| Immutable verification criteria | `user.define_outcome` + markdown rubric + grader | STRONG — grader literally returns `satisfied / needs_revision / max_iterations_reached / failed`, which maps cleanly to our PASS/CONDITIONAL/FAIL contract |
| Dual evaluator (qa + harness-verifier IN PARALLEL) | Orchestrator with two `callable_agents`, spawned as threads | **STRONG** — multiagent explicitly supports parallel threads, each with isolated context; this is the direct analogue of our "two Agent tool calls in one message" rule |
| `.claude/agents/*.md` sub-agent definitions | Agent objects created once and referenced by id+version | GOOD — version pinning in `callable_agents` matches our "immutable criteria" discipline |
| MCP servers (FastMCP, in-process) | Remote MCP via streamable HTTP + vault-backed creds | **PARTIAL BLOCKER** — our MCP servers are currently in-process FastMCP; they'd need to be exposed behind HTTPS endpoints to be attachable |
| HMAC capability tokens in `backend/agents/mcp_capabilities.py` | `permission_policy: always_allow / always_ask` + vault credentials | **MISMATCH** — only two policies, no role-scope bindings. Our HMAC capability model is strictly more expressive; on managed agents we'd downgrade to coarse `always_ask` + app-side enforcement of custom tools |
| BigQuery MCP (sunny-might-477607-p8) | Remote MCP possible; requires public HTTPS + GCP creds in a vault | WORKABLE but needs wrapper deployment |
| FastAPI backend on `:8000` (localhost) | `networking: limited` with `allowed_hosts: ["api.pyfinagent.example"]` | **BLOCKER as-is** — needs public HTTPS + auth layer before a managed container can reach it |
| GitHub commit-and-push from inside cycle | `resources: github_repository` + GitHub MCP tools | GOOD — PR flow matches our `git commit && git push origin main` |

### Blockers
1. **Permission policy is too coarse.** `mcp_capabilities.py` enforces
   per-role, per-scope, HMAC-bound capabilities. Managed Agents exposes
   only `always_allow` / `always_ask`. Fine-grained enforcement would
   have to move to the host app handling `user.tool_confirmation` or into
   the MCP server itself.
2. **FastAPI on `:8000` is unreachable.** No VPC peering. Requires a
   public HTTPS endpoint before a managed container can call it —
   non-trivial for a local-dev backend.
3. **In-process FastMCP servers are unreachable.** Managed Agents only
   connects to **remote** streamable-HTTP MCP. We'd have to deploy (e.g.
   Cloud Run) at least a shim for each server we want to expose.
4. **No `gcloud`/`bq`/`gh` pre-installed.** Not a hard blocker — pip/apt
   install works — but `gcloud` auth would still need service-account
   JSON injected somehow (vaults are MCP-only; GitHub token field is
   resource-type-specific; no generic env-secret documented).

### Wins
1. **Multiagent + callable_agents is a direct fit for our dual-evaluator
   rule.** Two threads, parallel, isolated context, each with its own
   system prompt — this is what we're already doing manually with two
   `Agent` tool calls. The `session_thread_id` echo pattern handles
   evaluator responses cleanly.
2. **`define_outcome` + rubric could replace our contract.md immutable
   success criteria** verbatim. The grader is a separate context window
   — matches Anthropic's own "agents tend to confidently praise their
   own work" guidance that we cite in CLAUDE.md.
3. **Session filesystem + Files API export** replaces most of the
   `handoff/current/*.md` mechanics and eliminates the
   `archive-handoff` PostToolUse hook.
4. **Thread-level event streams** give the Harness tab a richer feed
   than our current `harness_log.md` append-only.

## Findings

1. `claude-api-skill` is not a real page; skills = filesystem packs, not
   nested LLM calls. Nested inference = multiagent / `callable_agents`.
2. Permission policies are binary and toolset-scoped. Our HMAC capability
   token model (`backend/agents/mcp_capabilities.py`) is strictly more
   expressive and cannot be mapped 1-to-1.
3. Multiagent (Research Preview) is the single best-fit feature for our
   harness — it formalises exactly the "spawn qa-evaluator and
   harness-verifier IN PARALLEL under one lead" rule in CLAUDE.md.
4. `define_outcome` (Research Preview) provides a grader with its own
   context — a structural defense against the self-evaluation
   anti-pattern we already ban.
5. Container is capable (8 GB / 10 GB, Ubuntu 22.04) but does **not**
   include `gcloud`/`bq`/`gh`; networking defaults to disabled; no
   documented VPC peering; egress-only via `allowed_hosts`.
6. MCP support is remote-HTTPS only — our in-process FastMCP servers
   would need to be deployed publicly first.
7. GitHub integration (mount + MCP) covers our commit/push cycle
   cleanly; no GitHub App flow shown (PAT only).
8. SSE streams are replayable and per-thread — good frontend fit via a
   backend proxy (never exposing the API key to the Harness tab).

## Pilot GO/NO-GO recommendation

**CONDITIONAL GO on access-form submission — pilot scope must be narrow.**

*Submit the access form now*, but scope the pilot to:
- **One** masterplan step (phase-4.6 or similar parameter-optimization
  step — small, well-rubriced, no FastAPI backend call-outs).
- Orchestrator = managed agent with `callable_agents: [qa-evaluator,
  harness-verifier]`.
- Outcome = rubric copied verbatim from the step's `contract.md` success
  criteria.
- No custom MCP for the pilot; use only `agent_toolset_20260401` + the
  built-in GitHub MCP. Defer BigQuery / FastAPI reachability until the
  next pilot.
- Network = `limited` with `allowed_hosts: ["api.github.com",
  "api.anthropic.com"]` only.

**NO-GO blockers if not resolved before a broader rollout:**
- Moving HMAC-capability enforcement out of `mcp_capabilities.py` into
  the host app's `user.tool_confirmation` loop, or shifting it into each
  (now public) MCP server.
- Deploying a public HTTPS shim for the FastMCP servers we rely on in
  the full harness (BigQuery, paper-trading, ticket-queue).
- Producing a secrets-injection plan for `gcloud` service-account JSON
  (current docs don't document generic env secrets — only vaults for
  MCP auth and `authorization_token` for GitHub resources).

Net: multiagent + outcomes are compelling enough to justify submitting
the form; the pilot is a testbed to answer the three blocker questions
above, not a migration path for the existing harness in its current
shape.

## References

- https://platform.claude.com/docs/en/managed-agents/overview
- https://platform.claude.com/docs/en/managed-agents/mcp-connector
- https://platform.claude.com/docs/en/managed-agents/permission-policies
- https://platform.claude.com/docs/en/managed-agents/skills
- https://platform.claude.com/docs/en/managed-agents/environments
- https://platform.claude.com/docs/en/managed-agents/cloud-containers
- https://platform.claude.com/docs/en/managed-agents/events-and-streaming
- https://platform.claude.com/docs/en/managed-agents/define-outcomes
- https://platform.claude.com/docs/en/managed-agents/github-access
- https://platform.claude.com/docs/en/managed-agents/multi-agent
- https://platform.claude.com/docs/en/agents-and-tools/mcp-connector (for contrast)
- pyfinagent: `backend/agents/mcp_capabilities.py` (HMAC capability model)
- pyfinagent: `CLAUDE.md` (dual-evaluator rule, self-eval ban, research gate)
- pyfinagent: `handoff/audit/phase-4.11/managed_agents.md` (prior sweep)
