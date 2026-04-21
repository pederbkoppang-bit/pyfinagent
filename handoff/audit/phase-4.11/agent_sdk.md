# Agent SDK Deep Audit (phase-4.11.0)

Date: 2026-04-18. Scope: all 29 Agent SDK documentation pages under
https://code.claude.com/docs/en/agent-sdk/, audited against pyfinAgent
(`/Users/ford/.openclaw/workspace/pyfinagent`). No source changes made.

## URL coverage

| # | URL (stem under /en/agent-sdk/) | Status | Notes |
|---|---|---|---|
| 1 | `overview` | CHECKED | Full page read. |
| 2 | `quickstart` | CHECKED | Install, Opus 4.7 troubleshooting. |
| 3 | `agent-loop` | CHECKED | Turns, messages, compaction, effort, budget. |
| 4 | `claude-code-features` | CHECKED | `settingSources`, CLAUDE.md, skills, hooks. |
| 5 | `cost-tracking` | CHECKED | `total_cost_usd` warning, `modelUsage`, cache fields. |
| 6 | `custom-tools` | CHECKED | `@tool`, `create_sdk_mcp_server`, annotations. |
| 7 | `file-checkpointing` | CHECKED | `enable_file_checkpointing`, `rewind_files`. |
| 8 | `hooks` | CHECKED | 17+ events, filesystem vs programmatic. |
| 9 | `hosting` | CHECKED | Sandbox providers, 4 deployment patterns. |
| 10 | `mcp` | CHECKED | stdio/http/sse, `mcp__<srv>__<tool>`. |
| 11 | `migration-guide` | CHECKED | `ClaudeCodeOptions`→`ClaudeAgentOptions`. |
| 12 | `modifying-system-prompts` | CHECKED | preset/append, `excludeDynamicSections`. |
| 13 | `observability` | CHECKED | OTel, `CLAUDE_CODE_ENABLE_TELEMETRY=1`, span names. |
| 14 | `permissions` | CHECKED | 6 modes, evaluation order. |
| 15 | `plugins` | CHECKED | `.claude-plugin/plugin.json`. |
| 16 | `python` | CHECKED | Reference. |
| 17 | `secure-deployment` | CHECKED | Threat model, gVisor/Firecracker, proxy pattern. |
| 18 | `sessions` | CHECKED | resume/continue/fork, `~/.claude/projects/`. |
| 19 | `skills` | CHECKED | `.claude/skills/*/SKILL.md`. |
| 20 | `slash-commands` | CHECKED | `.claude/commands/*.md` (legacy). |
| 21 | `streaming-output` | CHECKED | `includePartialMessages`, `StreamEvent`. |
| 22 | `streaming-vs-single-mode` | CHECKED | Input modes. |
| 23 | `structured-outputs` | CHECKED | `output_format={"type":"json_schema", ...}`. |
| 24 | `subagents` | CHECKED | `AgentDefinition`, Agent tool, resume. |
| 25 | `todo-tracking` | CHECKED | `TodoWrite` tool. |
| 26 | `tool-search` | CHECKED | `ENABLE_TOOL_SEARCH=auto:N`. |
| 27 | `typescript` | CHECKED | Reference. |
| 28 | `typescript-v2-preview` | CHECKED | `unstable_v2_*` API. |
| 29 | `user-input` | CHECKED | `canUseTool`, `AskUserQuestion`. |

No redirects, no 404s.

## Per-page digests

### Core loop + lifecycle

- **overview**: `query()` + `ClaudeAgentOptions` wrap Claude Code's loop. Six tabs
  (tools / hooks / subagents / MCP / permissions / sessions) show feature
  surface. Opus 4.7 needs SDK >=0.2.111.
- **quickstart**: `pip install claude-agent-sdk`, `ANTHROPIC_API_KEY`, one-shot
  bug-fix example. Permission modes: `acceptEdits` / `dontAsk` / `bypassPermissions` / `default` / `auto`.
- **agent-loop**: Five message types (`SystemMessage`, `AssistantMessage`,
  `UserMessage`, `StreamEvent`, `ResultMessage`). `max_turns`, `max_budget_usd`,
  `effort` (low/medium/high/xhigh/max). Automatic prompt caching + compaction
  emits `compact_boundary`. Result subtypes: `success`, `error_max_turns`,
  `error_max_budget_usd`, `error_during_execution`, `error_max_structured_output_retries`.
- **claude-code-features**: `settingSources=["user","project","local"]` controls
  CLAUDE.md + `.claude/rules/*.md` + skills + hooks + `settings.json`. Managed
  policy + `~/.claude.json` + auto-memory load regardless.
- **streaming-vs-single-mode** / **streaming-output**: streaming is default
  and preferred. `include_partial_messages=True` emits raw
  `content_block_delta` events.

### Extension

- **custom-tools**: `@tool(name, desc, schema)` + `create_sdk_mcp_server()`
  wrap a function in an in-process MCP server, name becomes
  `mcp__<server>__<tool>`. `readOnlyHint` unlocks parallel tool calls.
  Return `{"is_error": True}` to keep the loop alive on tool failure.
- **mcp**: stdio / http / sse / in-process SDK servers. `.mcp.json` loaded via
  `settingSources`. Init message carries `mcp_servers: [{name, status}]` for
  connection errors.
- **tool-search**: `ENABLE_TOOL_SEARCH` = `true` / `auto` / `auto:N` / `false`.
  Withholds tool schemas until the model calls the search tool. Requires Sonnet/
  Opus 4+. Cap of 10k tools.
- **subagents**: `agents={name: AgentDefinition(description, prompt, tools,
  model, skills, mcpServers)}`. Fresh context; only final message returns.
  `Agent` tool must be in `allowedTools`. Filesystem equivalent in
  `.claude/agents/*.md`. Subagents cannot spawn subagents.
- **skills**: `.claude/skills/<name>/SKILL.md` with YAML frontmatter;
  description controls invocation. `"Skill"` must be in `allowedTools` when
  an explicit list is used.
- **slash-commands**: `.claude/commands/*.md` (legacy; prefer skills).
  Arguments via `$1`, `$ARGUMENTS`. Bash injection via `` !`cmd` ``.
- **plugins**: `plugins: [{type: "local", path: ...}]` loads
  `.claude-plugin/plugin.json` bundles (skills + agents + hooks + mcp).
- **modifying-system-prompts**: Default is minimal. `system_prompt={"type":
  "preset", "preset": "claude_code"}` restores Claude Code behavior; `append`
  adds project-specific text; `exclude_dynamic_sections=True` moves cwd / git
  / date out of system prompt so prompt cache hits across hosts.
- **todo-tracking**: `TodoWrite` built-in tool; inspect `tool_use` blocks to
  render progress.
- **user-input**: `can_use_tool` callback returns `PermissionResultAllow` /
  `Deny`. `AskUserQuestion` tool for multiple-choice clarifying questions.
  Python needs a dummy `PreToolUse` hook to keep stream open.
- **structured-outputs**: `output_format={"type":"json_schema", "schema": ...}`
  produces validated JSON in `result.structured_output`. Retries on mismatch;
  returns `error_max_structured_output_retries` on failure.

### State / control

- **sessions**: Files at `~/.claude/projects/<encoded-cwd>/<session_id>.jsonl`.
  `resume`, `continue_conversation`, `fork_session`. `list_sessions`,
  `get_session_messages`, `rename_session`, `tag_session`. Cross-host: move
  the `.jsonl` or re-hydrate state yourself.
- **file-checkpointing**: `enable_file_checkpointing=True` +
  `extra_args={"replay-user-messages": None}`. `rewind_files(uuid)` restores
  Write/Edit/NotebookEdit changes; Bash writes not tracked.
- **hooks**: 17 events (Python supports 9; TS adds `SessionStart`, `SessionEnd`,
  `TeammateIdle`, `TaskCompleted`, `ConfigChange`, `WorktreeCreate`,
  `WorktreeRemove`, `Setup`, `Notification` variants). Return
  `{"hookSpecificOutput":{"permissionDecision":"deny"|"allow"|"ask",
  "updatedInput":...}}`. Async side-effect mode via `{"async_": True}`.
- **permissions**: 5-step evaluation order: hooks → deny rules → mode → allow
  rules → `canUseTool`. Modes: `default`, `dontAsk`, `acceptEdits`, `plan`,
  `bypassPermissions`, `auto` (TS). Subagents inherit `bypassPermissions`
  silently — security hazard.
- **cost-tracking**: `total_cost_usd` is client-side estimate (warning on
  every page). `modelUsage[model].{costUSD, inputTokens, outputTokens,
  cacheReadInputTokens, cacheCreationInputTokens}`. Parallel tool calls
  produce duplicate ids → dedupe.
- **observability**: OTel export via env vars, `CLAUDE_CODE_ENABLE_TELEMETRY=1`
  + `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`. Spans: `claude_code.interaction`,
  `claude_code.llm_request`, `claude_code.tool` (children
  `blocked_on_user`, `execution`), `claude_code.hook`. Do NOT use
  `console` exporter (collides with SDK stdout).

### Deploy / migrate

- **hosting**: 4 patterns (ephemeral, long-running, hybrid, single multi-agent
  container). Sandbox providers: Modal, Cloudflare, Daytona, E2B, Fly, Vercel.
- **secure-deployment**: Threat model = prompt injection. Layers:
  sandbox-runtime → Docker → gVisor → Firecracker. `ANTHROPIC_BASE_URL` for
  sampling proxy; `HTTP_PROXY` for TLS-terminated egress. Least-privilege
  mount list (exclude `.env`, `~/.aws/`, `~/.ssh`).
- **migration-guide**: Rename `claude-code-sdk` → `claude-agent-sdk`;
  `ClaudeCodeOptions` → `ClaudeAgentOptions`. Settings sources and system
  prompt no longer defaulted — opt in explicitly.
- **python** / **typescript** / **typescript-v2-preview**: SDK reference; V2
  TS preview replaces async generator with `createSession`/`send`/`stream`.

## pyfinAgent vs Agent SDK

Evidence rooted in file line numbers.

### Zero SDK adoption

`grep claude_agent_sdk -r .` returns only
`handoff/audit/phase-4.10/agent_teams.md` (a previous audit). No import in
backend/, scripts/, or frontend/. Anthropic client is the raw SDK at
`backend/agents/llm_client.py`, `backend/agents/planner_agent.py`,
`backend/agents/planner_enhanced.py`,
`backend/agents/multi_agent_orchestrator.py`,
`backend/services/autonomous_loop.py`,
`backend/services/ticket_queue_processor.py`.

### Agent loop

- `scripts/harness/run_harness.py:1-100` implements Plan→Generate→Evaluate
  manually using `BacktestEngine` + `QuantStrategyOptimizer` + file-based
  handoffs.
- `backend/agents/multi_agent_orchestrator.py:35-80` runs a custom "Ford"
  research loop with `MAX_RESEARCH_ITERATIONS=3`, `MAX_TOOL_TURNS=5`, tool
  list in `AGENT_TOOLS` (lines 72+).
- SDK equivalent: `query()` with `max_turns`, `max_budget_usd`, built-in
  tool loop, automatic compaction. pyfinAgent reinvents the outer loop but
  retains a proprietary inner loop (quant optimization + MAS). The outer
  loop is exactly what the SDK gives you for free.

### Session management

- No analogue in pyfinAgent. Each cycle is independent; state flows through
  `handoff/current/*.md` files and `handoff/harness_log.md`.
- SDK `sessions` gives resume / fork / list, keyed by `cwd`. Could replace
  the handoff directory for intra-cycle continuation, but the handoff files
  provide human-readable audit that `jsonl` transcripts do not.

### Cost tracking

- `backend/agents/cost_tracker.py:17-69` hard-codes a `MODEL_PRICING` dict
  (per-model USD/1M tokens) and computes cost from `usage_metadata` off
  Vertex AI responses. Cache pricing at `:130-138` applies Anthropic's 90%
  read discount.
- SDK equivalent: `ResultMessage.total_cost_usd` + `modelUsage` per-model
  breakdown, with the explicit warning that values are client-side estimates
  (cost-tracking doc). pyfinAgent's price table is more complete (28 models
  across 4 providers) than SDK's bundled prices, and the Gemini-first stack
  needs custom pricing anyway. The SDK only knows Claude models.

### Observability

- `handoff/harness_log.md` is the canonical run log (see CLAUDE.md "Harness
  Protocol" rule). No OTel.
- Frontend backtest Harness tab reads that file directly.
- SDK `observability` (OTel) would give per-tool-call spans and centralized
  metrics, but would require a collector; orthogonal to the narrative log
  the team reads.

### Permissions / capability tokens

- `backend/agents/mcp_capabilities.py:1-268` implements HMAC-SHA256 capability
  tokens binding `(session_id, role, scopes, expires_at)` with `TTL=1800s`.
  `ROLE_SCOPES` (lines 55-68) defines 6 roles (researcher, strategy, risk,
  evaluator, orchestrator, paper_trader). `enforce(required_scope)` decorator
  wraps MCP tool entry points. `scrub_args()` regex-redacts PII before tool
  body.
- SDK equivalent: `permissionMode` + `allowedTools` / `disallowedTools` +
  `canUseTool` callback + `PreToolUse` hooks. Evaluation order:
  hooks → deny → mode → allow → `canUseTool` (permissions doc).
- pyfinAgent's system is STRONGER: signed capability tokens survive
  compromise of the agent process; SDK permissions are enforced only within
  the CLI child. For multi-agent orchestration where one agent's token
  cannot be reused by another, the HMAC layer is load-bearing.

### MCP

- pyfinAgent has 4 MCP servers: `backend/agents/mcp_servers/` contains
  `backtest_server.py`, `data_server.py`, `risk_server.py`,
  `signals_server.py`. All in-process FastMCP, enforced by
  `mcp_capabilities.enforce()`.
- SDK custom-tools doc describes exactly this pattern with `@tool` +
  `create_sdk_mcp_server`. Semantically identical; pyfinAgent predates the
  adoption curve.
- BigQuery MCP is available in the Claude Code harness itself (CLAUDE.md
  "BigQuery Access (MCP)" section) — that is already SDK-like.

### Structured outputs

- `backend/agents/schemas.py` defines Pydantic output schemas for Gemini
  structured output (rule file `.claude/rules/backend-agents.md`: "Structured
  output via Gemini JSON schema enforcement").
- SDK `structured-outputs` would automate schema conversion + retry. Two
  blockers: (a) Gemini is the primary backend, SDK is Claude-only;
  (b) retry on schema mismatch adds latency the quant pipeline doesn't
  tolerate.

### Tool search

- pyfinAgent has 28 skill prompts in `backend/agents/skills/*.md` plus
  6-agent MAS. Loaded on-demand by `load_skill()` (rule file lines 22-27).
- SDK `ENABLE_TOOL_SEARCH=auto:N` solves the same context-bloat problem but
  only for MCP tool schemas, not markdown skill content. pyfinAgent already
  avoids schema preload by using Gemini's tool schemas (bounded) and
  markdown skills (not schemas). Low value-add.

### Subagents

- `backend/agents/agent_definitions.py` (via
  `multi_agent_orchestrator.py:42-53`) defines 6 MAS agents programmatically
  — functionally the same as SDK's `AgentDefinition`.
- Harness sub-agents (`qa-evaluator`, `harness-verifier`, `researcher`,
  `per-step-protocol`) live in `.claude/agents/*.md` — already the filesystem
  path the SDK expects.

### Hooks

- pyfinAgent has `.claude/hooks/` (listed by dir scan) plus PostToolUse
  behavior described in CLAUDE.md ("archive-handoff hook", auto-changelog
  hook on commits — see recent git log).
- SDK `hooks` doc covers every one of these patterns. The hook infrastructure
  is already idiomatic SDK usage.

### File checkpointing

- pyfinAgent has `.claude/settings.json.bak-harness-ABCD` and
  `.claude/masterplan.json.bak-phase4.5` — manual backups. No automated
  rewind.
- SDK `enable_file_checkpointing=True` + `rewind_files(uuid)` would replace
  hand-rolled backup files. Could apply to `optimizer_best.json` safety.

### Slash commands / skills

- Already using `/masterplan` slash command (in available skills list).
- `.claude/skills/` populated (dir listing confirms it exists). Fully
  idiomatic.

### Plugins

- No plugin usage in pyfinAgent. Not needed for a single-project deployment.

## Findings table

| Aspect | Status | Evidence | Notes |
|---|---|---|---|
| Agent SDK import | ABSENT | grep -r claude_agent_sdk → 0 hits (excl. prior audit) | Pure custom orchestration |
| Anthropic raw SDK | YES | `backend/agents/llm_client.py` + 5 others | Baseline; pre-agent-SDK |
| Agent loop | CUSTOM | `scripts/harness/run_harness.py`; `multi_agent_orchestrator.py:58-63` | Three-phase loop matches SDK pattern but hand-rolled |
| Max-turn / budget | CUSTOM | `MAX_RESEARCH_ITERATIONS=3`, `MAX_TOOL_TURNS=5`; harness `MAX_CONSECUTIVE_FAIL=3` | SDK has `max_turns`/`max_budget_usd` primitives |
| Sessions / resume | ABSENT | No session API; handoff files drive continuity | Intentional — human-readable audit > jsonl |
| Cost tracking | CUSTOM | `backend/agents/cost_tracker.py:17-240` | 28-model, 4-provider table; SDK knows only Claude |
| Prompt caching | YES | `cost_tracker.py:130-138` applies 90% discount | Native, correctly priced |
| Observability | CUSTOM | `handoff/harness_log.md` append-only markdown | No OTel; frontend reads file |
| Permissions | CUSTOM+STRONGER | `backend/agents/mcp_capabilities.py:96-165` (HMAC tokens, TTL 1800s) | Survives agent compromise; SDK modes don't |
| Hooks | IDIOMATIC | `.claude/hooks/` dir + CLAUDE.md references | Already using Claude Code hooks |
| MCP servers | IDIOMATIC | `backend/agents/mcp_servers/` × 4 | FastMCP in-process, matches SDK custom-tools pattern |
| Subagents (MAS) | CUSTOM | `multi_agent_orchestrator.py` Ford loop | 6 agents; SDK `AgentDefinition` ≈ same shape |
| Subagents (harness) | IDIOMATIC | `.claude/agents/*.md` | qa-evaluator, harness-verifier, researcher, per-step-protocol |
| Skills | IDIOMATIC | `backend/agents/skills/*.md` + `.claude/skills/` | 28 Layer-1 prompts + Claude Code skills |
| Slash commands | IDIOMATIC | `/masterplan` referenced in CLAUDE.md | Already using |
| Structured output | CUSTOM | `backend/agents/schemas.py` Pydantic + Gemini JSON schema | Gemini-specific; SDK won't help |
| Tool search | ABSENT | No `ENABLE_TOOL_SEARCH` | Not needed; skills are markdown not schemas |
| File checkpointing | MANUAL | `.claude/*.bak-*` files | Hand-rolled; SDK `rewind_files` would be cleaner |
| Streaming output | N/A | Backend uses orchestrator-direct calls | Frontend streams via SSE (different layer) |
| Plugins | ABSENT | No `plugin.json` anywhere | Not needed |
| System prompt preset | ABSENT | Custom system prompts per agent | `.claude/context/` replaces preset mechanism |

## MUST FIX

None. pyfinAgent's current architecture is not broken; it is deliberately
divergent for load-bearing reasons (Gemini primary, HMAC security,
markdown-first audit trail).

The only items that come close to "must fix" are risk surfaces:

1. **Subagent permission inheritance.** SDK permissions doc warns:
   "When the parent uses `bypassPermissions`, `acceptEdits`, or `auto`, all
   subagents inherit that mode and it cannot be overridden per subagent."
   pyfinAgent's capability-token system DOES prevent this, but the
   `.claude/settings.json` Claude Code harness itself (the one running the
   masterplan) may grant sub-agents more scope than intended. Worth
   auditing `less-permission-prompts` skill output against the 7-of-9
   slippage recorded in user memory `feedback_research_gate.md`.

2. **Cost estimation drift.** `cost_tracker.py:17-69` pricing is dated
   "June 2026" but today is 2026-04-18 — ahead of itself, which suggests
   pricing has not been revalidated since the table was written. SDK's
   cost-tracking page explicitly warns client-side estimates drift; same
   warning applies to pyfinAgent. Add a freshness check.

## NICE TO HAVE

1. **File checkpointing** (`enable_file_checkpointing=True`) for
   `optimizer_best.json` + masterplan state. Replaces the manual
   `.bak-phase4.5` / `.bak-harness-ABCD` backup files with programmatic
   rewind. ~1 hour integration if running inside Claude Code harness.

2. **OpenTelemetry export** of the harness. Complements — does not replace —
   `harness_log.md`. Would let the frontend Harness tab query Honeycomb /
   Grafana for per-tool-call latency instead of just the markdown summary.
   Turn on with `CLAUDE_CODE_ENABLE_TELEMETRY=1` in the Claude Code harness
   env.

3. **`excludeDynamicSections=True`** for the Claude Code harness processing
   pyfinAgent masterplan steps. Each phase-4.X cycle currently pays full
   prompt-cache cost because cwd/git-status reshape the system prompt
   per-run. This flag moves those into the user message, letting prompt
   cache hit across cycles. Estimated ~10-30% token cost reduction on
   harness cycles.

4. **Tool search (`ENABLE_TOOL_SEARCH=auto:10`)** on the Claude Code harness
   invocation. With 28 skill prompts + 6 MAS agents + 4 MCP servers, the
   tool-schema surface exceeds 30-50 tools where SDK docs report accuracy
   degradation. Auto-activation when schemas exceed 10% of context costs
   nothing when under threshold.

5. **Plugin-packaged harness protocol.** Move `.claude/agents/*.md` + hooks
   + per-step-protocol into a versioned `pyfinagent-harness` plugin
   (`.claude-plugin/plugin.json`). Reusable across sister projects; version
   upgrades are a single `path` swap.

## Strategic: adopt-or-not

**Keep custom. Selectively adopt SDK primitives at the harness boundary.**

Reasons:

1. **Gemini is the primary inference backend.** CLAUDE.md Architecture
   section: 28 Gemini agents in Layer 1. Agent SDK is Claude-only for
   sampling (supports Bedrock/Vertex but only for Claude models). Full
   adoption means rewriting the Layer-1 pipeline in Claude, multiplying
   cost by ~10x (Claude Opus is the most expensive Claude model, Gemini
   2.0 Flash is the cheapest production model). No.

2. **The capability-token security boundary is stronger than SDK
   permissions.** `mcp_capabilities.py` binds signed tokens to sessions;
   SDK modes are advisory and inherited by subagents. For a system that
   decides trades, the HMAC layer is not replaceable by config.

3. **The handoff-file audit trail is load-bearing.** CLAUDE.md "Harness
   Protocol" lists five non-skippable markdown files per cycle, human-read
   by Peder and the two evaluators. SDK `sessions` produce jsonl transcripts
   that are machine-parseable but not suited to the "cite reason, fill
   violated_criteria, pass/conditional/fail" structured review flow the
   project depends on.

4. **Where the SDK genuinely helps is at the OUTER harness boundary** —
   the Claude Code process that runs the per-step protocol. That process
   IS an Agent SDK agent (implicitly, because it's Claude Code). What
   pyfinAgent does NOT do is configure it optimally:
   - No `excludeDynamicSections` → cache misses every run
   - No OTel export → harness cycle metrics are markdown-only
   - Manual state backups → no `rewind_files`
   - All-tools-loaded → risk of accuracy degradation once skill count
     climbs further

5. **The two `anthropic` import sites in Layer 2/3** (`planner_agent.py`,
   `planner_enhanced.py`, `evaluator_agent.py`, `multi_agent_orchestrator.py`)
   COULD migrate to Agent SDK for the benefit of built-in tools (Read/Grep/
   Bash for harness-state inspection), sessions for planner/evaluator
   continuity across phases, and structured outputs for the evaluator's
   PASS/CONDITIONAL/FAIL verdict schema. This is an additive, not
   replacing, migration and is the single highest-ROI SDK integration
   target. Estimate: ~2-3 days to migrate planner + evaluator,
   self-contained, reversible.

Bottom line: pyfinAgent is a hybrid system where the correct
answer is hybrid adoption. Keep Gemini Layer 1, keep HMAC capability
tokens, keep markdown handoff files. Migrate the Claude-based
planner/evaluator (Layer 3) to the Agent SDK and tune the outer Claude Code
harness with the 4 "nice to have" flags above. Do NOT attempt a wholesale
migration.

## References

- https://code.claude.com/docs/en/agent-sdk/overview
- https://code.claude.com/docs/en/agent-sdk/quickstart
- https://code.claude.com/docs/en/agent-sdk/agent-loop
- https://code.claude.com/docs/en/agent-sdk/claude-code-features
- https://code.claude.com/docs/en/agent-sdk/cost-tracking
- https://code.claude.com/docs/en/agent-sdk/custom-tools
- https://code.claude.com/docs/en/agent-sdk/file-checkpointing
- https://code.claude.com/docs/en/agent-sdk/hooks
- https://code.claude.com/docs/en/agent-sdk/hosting
- https://code.claude.com/docs/en/agent-sdk/mcp
- https://code.claude.com/docs/en/agent-sdk/migration-guide
- https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts
- https://code.claude.com/docs/en/agent-sdk/observability
- https://code.claude.com/docs/en/agent-sdk/permissions
- https://code.claude.com/docs/en/agent-sdk/plugins
- https://code.claude.com/docs/en/agent-sdk/python
- https://code.claude.com/docs/en/agent-sdk/secure-deployment
- https://code.claude.com/docs/en/agent-sdk/sessions
- https://code.claude.com/docs/en/agent-sdk/skills
- https://code.claude.com/docs/en/agent-sdk/slash-commands
- https://code.claude.com/docs/en/agent-sdk/streaming-output
- https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode
- https://code.claude.com/docs/en/agent-sdk/structured-outputs
- https://code.claude.com/docs/en/agent-sdk/subagents
- https://code.claude.com/docs/en/agent-sdk/todo-tracking
- https://code.claude.com/docs/en/agent-sdk/tool-search
- https://code.claude.com/docs/en/agent-sdk/typescript
- https://code.claude.com/docs/en/agent-sdk/typescript-v2-preview
- https://code.claude.com/docs/en/agent-sdk/user-input

Internal evidence:
- `/Users/ford/.openclaw/workspace/pyfinagent/CLAUDE.md`
- `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/backend-agents.md`
- `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/security.md`
- `/Users/ford/.openclaw/workspace/pyfinagent/scripts/harness/run_harness.py:1-100`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/multi_agent_orchestrator.py:1-80`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/cost_tracker.py:1-240`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_capabilities.py:1-268`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_servers/` (4 FastMCP servers)
