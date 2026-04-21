# Claude Documentation Alignment -- Consolidated Report (phase-4.10)

Produced: 2026-04-18. Scope: audit-only. No source modified.
Sources: `handoff/audit/phase-4.10/{extended_thinking,adaptive_thinking,sub_agents,agent_teams,mcp,platform_overview}.md`.

## Executive summary

pyfinAgent is well-positioned on most doc-aligned primitives
(MCP config shape, sub-agent frontmatter, tool-use loop with
parallel execution, capability-token guardrails beyond what
Claude Code offers). The gaps cluster in three places:

1. **Opus 4.7 readiness** -- thinking API shape, pricing table,
   and cache threshold are all on Opus-4.6 assumptions; a blind
   model upgrade would 400 on every Claude call and
   under-report cost by ~50-187x.
2. **Runtime safety on external MCP** -- the Alpaca write path
   relies on a single env var; `enableAllProjectMcpServers:
   true` silently overrides the allowlist.
3. **Unexercised platform features** that Anthropic has shipped
   for exactly our workload: prompt caching on MAS hot path,
   Batches API for overnight harness, Files API + Citations for
   SEC filings, Priority tier for live paper-trader.

## Per-topic findings table

| Topic | Correct | Missing | Incorrect | Recommended |
|-------|---------|---------|-----------|-------------|
| Extended thinking (4.10.0) | `temperature=1` forcing (`llm_client.py:628`); Layer-1 signal pipeline correctly off | `adaptive` mode, `effort`, `display`, interleaved-beta header, judge-side Claude path | Deprecated `{type:"enabled"}` on every Claude call; `isinstance(model, GeminiClient)` gate in `orchestrator.py:396` skips Claude judges | Thinking for harness agents (researcher / qa-evaluator / harness-verifier) + planner_agent |
| Adaptive thinking (4.10.1) | Cost ceiling via `cost_tracker` enforced (`orchestrator.py:1485`); interleaved tool-loop shape | Adaptive anywhere; `output_config.effort`; complexity-aware model router | Static `_BUILD_TIER` in `model_tiers.py:42-59` + hardcoded `claude-opus-4-6`; `claude-sonnet-4-20250514` legacy ID in `autonomous_loop.py:438` | `AgentConfig.effort` field; downshift Haiku/Sonnet/Opus by `QueryComplexity` |
| Sub-agents (4.10.2) | Frontmatter schema compliant on all 3 agents; tools least-privilege; no duplicated responsibilities | "Use proactively" / "MUST BE USED" trigger phrasing in descriptions; `SubagentStop` hook for dual-evaluator rule | `per-step-protocol.md` located in agent dir (loader ignores it, but confusing) | `permissionMode: plan` on reviewers; last-reviewed-date stamps; stress-test re-run record for Opus 4.7 |
| Agent teams (4.10.3) | Subagent definitions (3 files) are valid teammate candidates | Actual team spawn anywhere | `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` + `TeammateIdle` hook enabled but unreachable (no team ever created) | Pilot a team for phase-4.10-style research gates OR strip the dead flag + hook |
| MCP (4.10.4) | `.mcp.json` schema + `${VAR}` expansion; capability tokens (HMAC) + PII scrub + debounce + output cap -- stronger than built-in; scoped audit tooling (`mcp_inventory.py`, risk_score, storm_regression) | `mcp__alpaca__place_order` / `cancel_order` deny-list; per-tool `mcp__*` allowlist; BigQuery MCP config | `enableAllProjectMcpServers: true` + `enabledMcpjsonServers: ["slack"]` contradiction (allowlist is dead); `backend/mcp/*.py` stubs flagged but kept | Stdio-wrap FastMCP `data_server`/`signals_server` for Claude Code visibility; `managed-mcp.json` for go-live; `oauth.scopes` pinning |
| Platform overview (4.10.5) | Tool-use loop with parallel ThreadPoolExecutor execution; cache-aware cost math (10% read rate); system-prompt cache block shape | Batches API; Files API; Citations; Vision/PDF; Priority Tier (`service_tier`); Admin API / Usage & Cost API; SSE streaming for Claude; `strict: true` on tool schemas; `request_id` logging | `cost_tracker.MODEL_PRICING` has no Opus-4-7/4-6 or Haiku-4-5 entries -- falls back to $0.10/$0.40 default (50-187x under-report); `GITHUB_MODELS_CATALOG` still advertises retired Sonnet 3.5/3.7; system prompt <4096 tokens so Opus-4-7 prompt cache never triggers; retry loops ignore `retry-after` header | Cache `AGENT_TOOLS` + per-agent system prompt in MAS hot path; Batches for overnight harness + gauntlet; Files API + Citations for SEC 10-K/10-Q; Priority tier for paper-trader/slack; workspace split for attribution |

## MUST FIX -- correctness & safety (ordered by blast radius)

1. **Stale pricing table under-reports Claude spend 50-187x.**
   `backend/agents/cost_tracker.py MODEL_PRICING` has no
   `claude-opus-4-7`, `claude-opus-4-6`, `claude-haiku-4-5`
   entries; every Claude call through `make_client ->
   ClaudeClient` falls back to `_DEFAULT_PRICING = (0.10,
   0.40)`. Every autonomous budget check is wrong. Blocking
   for the budget dashboard & kill-switch.

2. **Resolve `.claude/settings.local.json` MCP contradiction
   and add Alpaca write-tool deny-list.** `enableAllProject
   McpServers: true` supersedes `enabledMcpjsonServers:
   ["slack"]` (allowlist dead). Combined with
   `defaultMode: bypassPermissions`, one env flip of
   `ALPACA_PAPER_TRADE=true` equals live-order execution with
   no Claude Code guard. Add `mcp__alpaca__place_order`,
   `mcp__alpaca__cancel_order`, `mcp__alpaca__replace_order`
   to `permissions.deny` and pick ONE allowlist pattern.

3. **Opus 4.7 thinking-API correctness.** `backend/agents/
   llm_client.py:622-628` and `multi_agent_orchestrator.py:
   944-954` emit `{"type":"enabled","budget_tokens":N}`
   unconditionally. Opus 4.7 **rejects this with 400**. Add
   an `"adaptive"` branch (forward `output_config.effort`,
   `thinking.display`) and gate manual-vs-adaptive on model
   ID. Without this, any model upgrade breaks every MAS call.

4. **Drop Gemini-only thinking gate on judges.**
   `orchestrator.py:396` and `risk_debate.py:55` gate
   `thinking` injection on `isinstance(model, GeminiClient)`
   -- switching any judge to Claude silently disables
   extended thinking despite `ENABLE_THINKING=true`.
   Each provider client should decide thinking support.

5. **Respect `retry-after` header on 429.** `debate.py:
   67-69`, `risk_debate.py:51`, `ticket_queue_processor.py:
   348-354`, `tools/alt_data.py:88-90` all use hardcoded
   exponential backoff (5/10/20s) that ignores the header.
   Anthropic ban risk under sustained load. Also log
   `response._request_id` on failure for actionable support.

6. **Prompt-cache threshold miss on Opus 4.7.** `ClaudeClient`
   wraps a ~20-char system prompt in `cache_control` --
   under the 4096-token min for Opus 4.7 / Haiku 4.5, so the
   cache silently never creates an entry. Either move skill
   prompts into the cached system block (reaches threshold)
   or set expectations against Sonnet 4.5 (1024 min).

7. **Retire deprecated model IDs.**
   `backend/services/autonomous_loop.py:438` and
   `backend/slack_bot/mcp_tools.py:73` still use
   `claude-sonnet-4-20250514`; `GITHUB_MODELS_CATALOG` in
   `llm_client.py:63-67` advertises Sonnet 3.5/3.7 without
   deprecation notes. Haiku 3 retires 2026-04-19 (tomorrow),
   Sonnet 4 / Opus 4 retire 2026-06-15. Migrate to
   `claude-sonnet-4-6` / `claude-opus-4-7`.

8. **Prune `backend/mcp/*.py` legacy stubs.** Flagged
   `stub: true` by `scripts/audit/mcp_inventory.py`; duplicate
   the canonical `backend/agents/mcp_servers/` implementations.

9. **Resolve BigQuery MCP doc drift.** CLAUDE.md lines 84-123
   document `execute_sql` (full DML/DDL) as available but
   no config file declares the server. Either pin in
   `.mcp.json` with scoped OAuth + deny on mutating tool, OR
   update CLAUDE.md to "harness-injected, may be absent".
   Add `mcp__bigquery__execute_sql` to `permissions.deny` as
   default.

10. **Strengthen sub-agent descriptions for auto-delegation.**
    None of the three `.claude/agents/*.md` contain
    "use proactively" / "MUST BE USED" trigger phrasing the
    docs explicitly recommend. Works today because the
    runbook spawns them manually; brittle otherwise.

## NICE TO HAVE -- cost, latency, observability, features (ranked by ROI)

1. **Prompt caching across MAS hot path.** Wrap `AGENT_TOOLS`
   (7 definitions, ~300 tokens of stable schema JSON) and
   each agent's `system=agent_config.system_prompt` in
   `cache_control: {type: "ephemeral"}` in
   `multi_agent_orchestrator.py:944-954`. Expected ~90%
   reduction on input costs for tool-loop turns 2..N and ~85%
   latency reduction on cache hits. Biggest single cost win.

2. **Batches API for overnight workloads.** `/v1/messages/
   batches` gives a 50% discount; max 100k requests/batch,
   256MB. Apply to `scripts/harness/run_harness.py` cycles,
   `backend/backtest/gauntlet/regimes.py` regime sweeps, the
   Layer-1 28-agent analysis pipeline, and any nightly
   research scans. Halves that bill.

3. **Files API + Citations for SEC filings.** Upload every
   10-K/10-Q once (`anthropic-beta: files-api-2025-04-14`),
   reuse `file_id` across `deep_dive_agent`, `insider_agent`,
   `sec_insider.py`, `earnings_tone.py`. Enable
   `citations: {enabled: true}` -- `cited_text` is free (no
   output-token cost). Massive ingestion-cost reduction.

4. **Native PDF document blocks** replace custom text
   extraction in `earnings_tone.py` / `sec_insider.py`.
   Claude sees charts + tables + page layout; materially
   better signal on earnings decks.

5. **Priority Tier (`service_tier="auto"`) for live paper-
   trader + slack-bot.** `claude-opus-4-7` commit + 99.5%
   uptime target protects MAS from 529 overloaded during
   market hours -- the biggest availability risk today.

6. **Complexity-aware model router** (pyfinAgent-native
   adaptive). Replace static `_BUILD_TIER` in
   `model_tiers.py` with: TRIVIAL -> Haiku 4.5,
   SIMPLE -> Sonnet 4.6 + `effort=low`, MODERATE -> Sonnet
   4.6 + `effort=medium`, COMPLEX -> Opus 4.7 adaptive +
   `effort=high`. Complements Anthropic's adaptive (which
   flexes budget within one model).

7. **Extend adaptive thinking to harness agents** (researcher,
   qa-evaluator, harness-verifier) with `effort: xhigh` on
   Opus 4.7 -- latency-tolerant, reasoning-heavy, exactly
   the doc's recommended use case.

8. **SSE streaming for long MAS turns.**
   `client.messages.stream(...).get_final_message()` removes
   the 10-min non-streaming timeout risk.

9. **Admin API + Usage & Cost API** as server-side truth for
   the budget dashboard (replace `cost_tracker.summarize()`).
   Split workloads into separate workspaces (harness vs MAS
   vs paper-trader) for clean attribution.

10. **`strict: true` on the 7 MAS tool definitions.** Cheap
    schema-conformance insurance (docs recommend it).

11. **`SubagentStop` hook on qa-evaluator / harness-verifier**
    refusing `ok: true` unless the sibling evaluator also
    ran. Enforces the dual-evaluator rule at the permission
    layer (currently runbook-only; violated on cycles 79-85
    per auto-memory).

12. **`permissionMode: plan` on both reviewers** as belt-and-
    suspenders over the existing read-only tool allowlist.

13. **Agent-teams pilot for phase-4.10-style research gates.**
    Adversarial-hypothesis + multi-angle research gates map
    directly onto the doc's "competing hypotheses" use case.
    Existing subagent definitions are already valid teammate
    roles. OR strip the dead
    `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` flag per
    stress-test doctrine.

14. **Stdio-wrap FastMCP servers for Claude Code visibility.**
    `data_server` / `signals_server` already have the
    guardrails; add a stdio transport and a `.mcp.json` entry
    so the orchestrating Claude Code session can query
    signals / run mini-backtests / call the harness verifier
    without shelling to Python.

15. **Managed MCP config for go-live** (`/etc/claude-code/
    managed-mcp.json`) enforces `allowedMcpServers` at the
    host level; blocks ad-hoc server additions mid-session.

16. **Relocate `per-step-protocol.md`** out of
    `.claude/agents/` into `.claude/docs/` or
    `.claude/runbooks/` -- today it sits among agent files
    and only the YAML-frontmatter loader disambiguates.

## Implementation sizing (for the action-list approval)

- Items 1-9 of MUST FIX are each 1-2 hour changes except
  items 1, 2, 3 which each span multiple files and deserve
  their own phase-4.10.x cycle under the full harness loop.
- Items 10 (sub-agent descriptions) is a 15-minute text
  change, but worth its own cycle for research-gate
  discipline.
- NICE TO HAVE items 1, 2, 3, 5 are the highest cost/latency
  ROI. They deserve dedicated phase-4.11 steps (call it
  "Platform-feature adoption").
- Items 13, 14, 15 are medium-effort architecture decisions
  that probably want their own proposal docs first.

## References

All per-topic findings files in
`handoff/audit/phase-4.10/`:
- `extended_thinking.md`
- `adaptive_thinking.md`
- `sub_agents.md`
- `agent_teams.md`
- `mcp.md`
- `platform_overview.md`

Documentation accessed 2026-04-18 across platform.claude.com
and code.claude.com.
