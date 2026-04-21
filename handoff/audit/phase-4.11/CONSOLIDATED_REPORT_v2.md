# Claude Docs Alignment -- Consolidated v2 (phase-4.11)

Produced: 2026-04-18. Scope: audit-only. Merges phase-4.10 (6 topic
audits, ~14-20 pages) + phase-4.11 (8 deep audits, ~150 pages across
the full doc tree on both platform.claude.com and code.claude.com).

All 8 phase-4.11 audit files in `handoff/audit/phase-4.11/`:
`agent_sdk.md`, `managed_agents.md`, `tool_use_primitives.md`,
`skills_prompting_context.md`, `claude_code_core.md`,
`claude_code_surfaces.md`, `api_and_sdks.md`, `misc_and_admin.md`.

## Executive summary

pyfinAgent's architecture is defensibly custom: Gemini Layer-1 can't
migrate (Agent SDK is Claude-only); HMAC capability tokens are
STRONGER than SDK permissions; file-based harness handoff is
human-audit-first and load-bearing for dual-evaluator discipline.
The correct adoption strategy is **hybrid**: keep the custom
backbone, selectively adopt SDK primitives at the harness boundary,
pilot Managed Agents for one harness cycle, fix concrete Opus 4.7
breakages, and claim the cost/latency wins from unexercised
platform features.

Three safety/correctness issues are BLOCKING for phase-4 go-live:
1. Cost-tracker pricing under-reports Opus spend ~50-187x.
2. MCP safety: Alpaca write tools can execute with one env flip.
3. `defaultMode: bypassPermissions` on a non-containerized dev host.

Three hidden Opus 4.7 breakages will 400 on the next model upgrade:
4. `thinking: {type: "enabled", budget_tokens}` rejected.
5. `temperature`/`top_p`/`top_k` non-default rejected.
6. Retired model IDs in `GITHUB_MODELS_CATALOG` will 400 on request.

## Per-topic findings summary (14 audits, ~170 pages total)

| Topic | Audit file | Top correct | Top missing | Top incorrect |
|-------|------------|-------------|-------------|---------------|
| Extended thinking | 4.10/extended_thinking.md | `temperature=1` forcing | adaptive, effort, display, interleaved-beta header | `{type:"enabled"}` universal |
| Adaptive thinking | 4.10/adaptive_thinking.md | Cost ceiling | adaptive anywhere; complexity-aware router | Gemini-only gate on judges |
| Sub-agents | 4.10/sub_agents.md | Frontmatter compliant, least-privilege | "use proactively" trigger phrasing | `per-step-protocol.md` in agent dir |
| Agent teams | 4.10/agent_teams.md | Subagent defs are valid teammate roles | Actual team spawn | `EXPERIMENTAL_AGENT_TEAMS=1` + `TeammateIdle` enabled but unreachable |
| MCP | 4.10/mcp.md | `.mcp.json` schema; capability tokens; audit tooling | Alpaca write-tool deny; BigQuery MCP config | `enableAllProjectMcpServers:true` + allowlist contradiction |
| Platform overview | 4.10/platform_overview.md | Tool-use loop; cache-aware cost math | Batches, Files, Citations, Priority Tier, Admin API, streaming, `strict:true`, request_id | Pricing table stale; retry loops ignore `retry-after` |
| Agent SDK (29 pgs) | 4.11/agent_sdk.md | Hooks + MCP + skills + subagents ALL idiomatic | SDK imports; `excludeDynamicSections`; OTel; file-checkpointing | Zero SDK adoption; `mas_main` hardcoded to legacy Opus IDs |
| Managed Agents (9 pgs) | 4.11/managed_agents.md | Own custom harness works | Any Managed Agents use | N/A (greenfield) |
| Tool-use primitives (14 pgs) | 4.11/tool_use_primitives.md | Tool-use loop | Advisor, PTC, Memory tool, Web-search, strict, cache_control on AGENT_TOOLS | Dead `mas_events.py` scaffolding around read-tools |
| Skills + prompting + context (14 pgs) | 4.11/skills_prompting_context.md | Length discipline on 28 skills | XML tags, prefill, multishot, frontmatter, YAML | `thinking.budget_tokens` silently broken on 4.7; cache block < 4096 tok |
| Claude Code core (26 pgs) | 4.11/claude_code_core.md | Hooks on commit + masterplan; sub-agents idiomatic | PreToolUse for dangerous cmds; sandboxing; plugin packaging; InstructionsLoaded hook | `bypassPermissions` default on dev host; stray `*.bak-*` in `.claude/` |
| Claude Code surfaces (33 pgs) | 4.11/claude_code_surfaces.md | `claude.yml`, `claude-code-review.yml` plugin-aligned; custom Slack is irreplaceable | Model pin on claude.yml; surface mapping for cron_budget.yaml | cron_budget.yaml aspirational; "15/day" is self-imposed, not platform limit |
| API + SDKs (23 pgs) | 4.11/api_and_sdks.md | System-as-top-level; `temperature=0` compatible | retry-after, rate-limit headers, request_id, anthropic-beta, structured outputs, Batches, streaming, count_tokens | SDK 9 versions behind; exception catching by string match |
| Models / pricing / admin (34 pgs) | 4.11/misc_and_admin.md | `_LIVE_TIER` sentinel guards | Admin/Usage-Cost API; workspace separation; Voyage embeddings; vision | `MODEL_PRICING` missing all Opus-4-x entries; Opus 4.7 tokenizer change unprepared |

## MUST FIX -- unified list (ordered by blast radius)

Consolidates overlapping items from phase-4.10 + phase-4.11. Each has
concrete file:line anchors.

### Correctness / safety (blocks phase-4 go-live)

1. **Cost-tracker pricing under-reports Opus spend ~50-187x.**
   `backend/agents/cost_tracker.py:17-28 MODEL_PRICING` has no
   entries for `claude-opus-4-7`, `claude-opus-4-6`,
   `claude-opus-4-5`, `claude-opus-4-1`, `claude-sonnet-4-5`,
   `claude-haiku-4-5`. `model_tiers.py:47` resolves `mas_main` +
   `mas_qa` to Opus IDs, so every MAS call falls back to
   `_DEFAULT_PRICING = (0.10, 0.40)`. Blocks budget dashboard +
   kill-switch.

2. **MCP safety: Alpaca write-tools unguarded.**
   `.claude/settings.local.json` `enableAllProjectMcpServers:true`
   silently overrides `enabledMcpjsonServers:["slack"]` (allowlist
   dead); combined with `defaultMode: bypassPermissions`, one env
   flip of `ALPACA_PAPER_TRADE=true` = live orders with no guard.
   Add `mcp__alpaca__place_order`, `mcp__alpaca__cancel_order`,
   `mcp__alpaca__replace_order` to `permissions.deny`; pick one
   allowlist pattern.

3. **Permission mode out of envelope.** `defaultMode:
   bypassPermissions` on a non-containerized dev host is outside
   the documented safety envelope (permission-modes.md). Switch
   short-term to `acceptEdits`, long-term to `auto` with
   `autoMode.environment` tuned for GitHub + BigQuery + our buckets.

4. **Sandboxing not enabled.** macOS Seatbelt is free on Darwin
   25.4.0; combined with `acceptEdits`/`auto` it replaces bypass
   mode safely with OS-level write + network enforcement. Block
   provided in `claude_code_core.md` §4.

5. **Opus 4.7 thinking-API 400.** `backend/agents/llm_client.py:
   622-628` and `backend/agents/multi_agent_orchestrator.py:
   944-954` emit `{"type":"enabled","budget_tokens":N}` -- Opus
   4.7 rejects with 400. Add `"adaptive"` branch, forward
   `output_config.effort` + `thinking.display`, gate manual-vs-
   adaptive on model ID.

6. **Opus 4.7 sampling-param 400.** Setting `temperature`/`top_p`/
   `top_k` to non-default also returns 400 on Opus 4.7. We pass
   `temperature=0` in multiple places -- must omit on 4.7 callsites.

7. **Retired models in `GITHUB_MODELS_CATALOG`.** `backend/agents/
   llm_client.py:63-67` still advertises `claude-3-5-sonnet-
   20241022`, `claude-3-5-haiku-20241022`, `claude-3-7-sonnet-
   20250219` -- all retired 2026-02-19, requests now 400. Remove +
   add current Opus 4.7 / Sonnet 4.6 / Haiku 4.5 / legacy 4-x.

8. **Stale snapshot ID.** `backend/services/autonomous_loop.py:438`
   uses `claude-sonnet-4-20250514`; everything else is on
   `claude-sonnet-4-6`. Fix. Add pre-commit grep to prevent
   dated-snapshot regressions.

9. **Retry loops ignore `retry-after` header.** `backend/agents/
   debate.py:67-69`, `risk_debate.py:51`, `orchestrator.py:387-440`,
   `services/ticket_queue_processor.py:348-354`, `tools/
   alt_data.py:88-90` all use hardcoded 5s×2^n exponential backoff.
   429 `retry-after` never read. Ban risk under sustained load.

10. **Exception classes caught by string match.** Every retry loop
    catches bare `Exception` and substring-matches `"ratelimit"`/
    `"overload"`/`"unavailable"` in class name. `anthropic.
    RateLimitError`, `APIStatusError`, `APIConnectionError` NEVER
    imported anywhere. Fragile; misses some classes entirely.
    Migration to SDK `max_retries` (which respects `retry-after`)
    deletes ~120 LOC + fixes #9 simultaneously.

11. **No `request_id` logging.** `message._request_id` ignored on
    success + failure paths. Support escalations harder than
    necessary. Add to `cost_tracker.record()` + every `logger.
    error("Agent ... failed ...")` path.

12. **Gemini-only thinking gate on judges.** `backend/agents/
    orchestrator.py:396` and `risk_debate.py:55` gate `thinking`
    injection on `isinstance(model, GeminiClient)`. Switching any
    judge to Claude silently disables extended thinking despite
    `ENABLE_THINKING=true`. Drop the gate; delegate to client.

13. **Prompt cache threshold miss.** `ClaudeClient` wraps a ~20-char
    system prompt in `cache_control`; under Opus 4.7/Haiku 4.5's
    4,096-token minimum, cache silently never creates an entry.
    Move skill body + schema + FACT_LEDGER into the cached system
    block; add `ttl: "1h"` for harness cycle time.

14. **BigQuery MCP doc drift.** CLAUDE.md lines 84-123 document
    `execute_sql` (full DML/DDL) as available; no config file
    declares the server. Either pin in `.mcp.json` with scoped
    OAuth + deny on mutating tool, or update CLAUDE.md to
    "harness-injected, may be absent". Add `mcp__bigquery__
    execute_sql` to `permissions.deny` by default.

15. **Prune dead MCP stubs.** `backend/mcp/*.py` flagged `stub:true`
    in inventory -- duplicates canonical `backend/agents/
    mcp_servers/`.

16. **Sub-agent descriptions need trigger phrasing.**
    `.claude/agents/{qa-evaluator,harness-verifier,researcher}.md`
    descriptions are declarative ("Independent QA evaluator for...")
    not action-oriented ("use proactively after any GENERATE step,
    immediately before marking a masterplan step done"). Weak for
    auto-delegation.

17. **`claude.yml` under-pinned.** `.github/workflows/claude.yml`
    defaults to Sonnet + broad tool access. Pin
    `claude_args: '--model claude-opus-4-7 --allowed-tools
    "Bash(gh pr *),Read,Edit"'`.

18. **Anthropic SDK 9 versions behind.** `backend/requirements.txt:
    37` pins `anthropic==0.87.0`; latest is 0.96.0. The CVE cited
    in the pin comment is fixed in 0.87 -- nothing blocks bumping.

19. **`cron_budget.yaml` aspirational without surface mapping.**
    15-slot cap is self-imposed, NOT an Anthropic limit. Slots
    have no surface mapping (routines vs desktop vs /loop). Phase
    10.7 Meta-Evolution can't wire without this decision. Also:
    routines have NO permission-mode picker, so trading-ops slots
    1-5 need an external approval gate if migrated.

20. **Haiku 3 retires tomorrow (2026-04-19).** Grep confirms zero
    hits in codebase; add CI assertion
    `assert "claude-3-haiku-20240307" not in source_tree` to
    prevent future regression.

### High-value hardening (strongly recommended before go-live)

21. **Missing `PreToolUse` hooks** for `rm -rf`, `git push
    --force`, `git reset --hard`, `mcp__.*__execute_sql`. CLAUDE.md
    prose cannot enforce these.

22. **Missing `ConfigChange` + `InstructionsLoaded` hooks.** The
    latter would have caught the research-gate miss on 7 of 9
    phase-4.8 cycles (per auto-memory `feedback_research_gate.md`).

23. **Remove stray `.claude/*.bak-*` files.** `.claude/` is scanned
    at session start; `*.bak-phase4.5`, `*.bak-harness-ABCD`
    pollute the layout.

## NICE TO HAVE -- ranked by cost/latency/observability ROI

Grouped by implementation cluster. Each phase-4.12+ candidate.

### Cluster A -- Prompt caching + Batches (biggest $$ wins)

A1. **Cache AGENT_TOOLS + per-agent system in MAS hot path.**
    `multi_agent_orchestrator.py:944-954` calls `client.messages.
    create` with raw `system=agent_config.system_prompt` and raw
    `tools=AGENT_TOOLS`. Wrap in `cache_control:{type:ephemeral,
    ttl:"1h"}`. Expected ~90% input cost reduction on tool-loop
    turns 2..N, ~85% latency cut on cache hits.

A2. **Batches API for overnight workloads.** `/v1/messages/
    batches` = 50% discount; max 100k/batch, 256MB. Apply to:
    `scripts/harness/run_harness.py` cycles, `backend/backtest/
    gauntlet/regimes.py`, `backend/services/autonomous_loop.
    _run_claude_analysis()` nightly ticker sweep, Layer-1 28-agent
    pipeline. Halves that bill.

A3. **`anthropic-beta: extended-cache-ttl-2025-04-11`** on
    `ClaudeClient` + `ttl:"1h"`. Expected cache-hit rate 30% (5m
    window) -> 60-80%.

A4. **`anthropic-beta: interleaved-thinking-2025-05-14`** on MAS
    tool loops. Opus 4.6/4.5; Opus 4.7 auto-includes.

### Cluster B -- Tool primitives that reshape MAS

B1. **Advisor tool (`advisor_20260301`).** Sonnet-4.6 executor +
    Opus-4.7 advisor mid-generation. ~60% Opus output tokens
    -> Sonnet on MAS tool loop. Highest single cost lever if B1+A1
    combined.

B2. **Programmatic tool calling.** Wrap 7 `AGENT_TOOLS` with
    `allowed_callers: ["code_execution_20260120"]` -- Ford's 7
    sequential reads collapse to 1 async-gather; intermediate
    results never enter context. Expect ~80% context-window
    reduction on planning turn.

B3. **Memory tool (`memory_20250818`).** Unify `.claude/agent-
    memory/researcher/*.md` (5 files in-repo) + user auto-memory
    (outside repo) behind one trained-in schema. Native
    compaction integration for 28-cycle harness runs + built-in
    "check memory first" system prompt.

B4. **Web search + web fetch (2026-02-09).** Replace `researcher`
    harness subagent's manual URL-paste with native
    `web_search_20260209` + `web_fetch_20260209`. Citations emitted
    natively in `web_search_result_location` shape -- matches
    research-gate "cite per claim" rule.

B5. **`strict: true` + `cache_control` on AGENT_TOOLS.** 15-min
    change; closes schema-drift footgun + caches 7-tool prefix
    across harness cycles.

### Cluster C -- Platform features we haven't touched

C1. **Files API + Citations for SEC filings.** Upload every
    10-K/10-Q once (`anthropic-beta: files-api-2025-04-14`), reuse
    `file_id` across `deep_dive_agent`, `insider_agent`,
    `sec_insider.py`, `earnings_tone.py`. Enable
    `citations:{enabled:true}` -- `cited_text` is free (no
    output-token cost).

C2. **Native PDF document blocks.** Replace custom text extraction
    in `earnings_tone.py` / `sec_insider.py`; Claude sees charts +
    tables + page layout.

C3. **Vision on Opus 4.7 (2576px / 3.75MP, 1:1 coord).** Earnings-
    call slide decks, analyst chart OCR, SEC filing figure
    extraction.

C4. **Priority Tier (`service_tier="auto"`)** for live paper-
    trader + slack-bot. Protects from 529 overloaded during market
    hours -- biggest availability risk today.

C5. **Structured outputs (`output_config.format`).** Replace
    `text.find('{')/rfind('}')` JSON parse-by-luck in
    `planner_agent.py:136`, `paper_trader`, all MAS responders.
    Opus/Sonnet 4.6+ native.

C6. **Streaming synthesis.** `client.messages.stream(...).
    get_final_message()` on the 4096-token synthesis call --
    removes 10-min non-streaming timeout risk.

C7. **`count_tokens` endpoint** replaces hand-rolled char-count
    approximations in `llm_client.py:100-146`.

C8. **Monitor tool (W15).** Background watcher streams events as
    transcript messages -- replaces `run_harness.py` polling
    loops + backtest status `while True; sleep 2; curl /status`.

### Cluster D -- Admin / observability

D1. **Admin API + Usage & Cost API integration.** Provision
    `ANTHROPIC_ADMIN_KEY`; daily cron to BQ table
    `pyfinagent_data.anthropic_usage_report`; frontend
    reconciliation vs `cost_tracker` (catches future pricing drift
    like #1 above).

D2. **Workspace separation.** `pyfinagent-prod`, `pyfinagent-
    harness`, `pyfinagent-backtest`; per-workspace keys + spend
    caps; prompt-cache isolation (2026-02-05 feature).

D3. **OTel export** of the Claude Code harness
    (`CLAUDE_CODE_ENABLE_TELEMETRY=1`). Complements (does not
    replace) `harness_log.md`. Per-tool-call spans, Honeycomb /
    Grafana queries for latency.

D4. **`metadata={"user_id": ...}`** on Slack-ticket calls for
    abuse-detection attribution.

### Cluster E -- Agent SDK selective adoption

E1. **Port `planner_agent.py` + `evaluator_agent.py` to Agent
    SDK.** ~2-3 day reversible migration. Gains: built-in Read/
    Grep/Bash for harness-state inspection, sessions for planner
    continuity, structured outputs for PASS/CONDITIONAL/FAIL
    verdicts. Highest-ROI SDK integration target.

E2. **`excludeDynamicSections=True`** on the Claude Code harness
    (outer process running the per-step protocol). Cwd/git-status
    currently reshape the system prompt per run; this flag moves
    them to user message, letting prompt cache hit across cycles.
    ~10-30% token cost reduction.

E3. **`enable_file_checkpointing=True`** for `optimizer_best.json`
    + masterplan backups. Replaces manual `*.bak-*` files with
    `rewind_files(uuid)`.

E4. **`ENABLE_TOOL_SEARCH=auto:10`.** 28 skills + 6 MAS agents + 4
    MCP servers approach the accuracy-degradation threshold;
    auto-activation costs nothing below 10%.

### Cluster F -- Claude Code hardening

F1. **`SubagentStop` hook on qa-evaluator + harness-verifier.**
    Refuse `ok:true` unless sibling evaluator also ran this cycle.
    Enforces dual-evaluator rule at permission layer (runbook-only
    today; violated on cycles 79-85).

F2. **`permissionMode: plan` on reviewers.** Belt-and-suspenders
    over existing read-only tool allowlist.

F3. **`SessionStart` + `UserPromptSubmit` + `InstructionsLoaded` +
    `PreCompact` hooks.** Inject current phase-id as
    `additionalContext`; debug lazy-loaded rule files; mark
    harness log before compaction.

F4. **Plugin-package the harness scaffolding.** Move agents +
    hooks + skills + `per-step-protocol.md` into a versioned
    `pyfinagent-harness` plugin. Versioning solves the current
    "no version" problem on per-step-protocol.md. Unlocks
    `strictKnownMarketplaces:[]` lockdown for go-live.

F5. **Hardening env vars:** `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`,
    `DISABLE_TELEMETRY=1` (if preferred), `BASH_MAX_TIMEOUT_MS=
    600000`, `CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY=5`.

### Cluster G -- Pattern migrations

G1. **XML tags + assistant prefill + multishot examples** on the 4
    Claude MAS skills (`bull`, `bear`, `devils_advocate`,
    `moderator`). Highest-ROI prompt-engineering wins.

G2. **Third-person YAML frontmatter** on `SKILL_TEMPLATE.md` and
    all 28 skills. Even without migration to native Skills,
    enforces discipline + portability.

G3. **Content-moderation `risk_level` 0-3 pattern** ported into
    `conflict_detector.py` + `bias_detector.py` (replace binary
    flags).

G4. **Legal-summarization `details_to_extract` + meta-summarization
    + summary-indexed RAG** patterns for `10-K` / earnings-call
    analysis. Cross-check against existing `compaction.py`.

G5. **Voyage-finance-2 embeddings.** Replace BQ BM25 in
    `backend/agents/memory.py` + Vertex `text-embedding-005` in
    `backend/tools/nlp_sentiment.py:15`. 1024-dim, finance-trained.

### Cluster H -- Managed Agents pilot

H1. **Port ONE harness cycle GENERATE phase to a Managed Agent
    session.** Use `agent_toolset_20260401`; attach a memory store
    in place of `harness_learning_log` BQ table; keep qa-evaluator
    + harness-verifier local until `callable_agents` leaves
    Research Preview. Wins: zombie-worker fix, free SSE stream for
    Harness tab, audited memory versioning. Request memory +
    multi-agent RP access via overview form. Phase-4.12 candidate.

H2. **Anthropic pre-built skills** (xlsx/pptx/docx/pdf) for Slack
    bot + investor-report flow. Zero migration cost.

### Cluster I -- CI / surface polish

I1. **Add `/ultrareview`** to pre-merge checklist for substantial
    PRs.

I2. **Devcontainer** with firewall rules for team onboarding +
    CI parity.

I3. **Channels (iMessage/Telegram)** plugin as cheaper phone bridge
    than custom Slack direct-responder path.

I4. **Managed Code Review** if we move to Team plan post-go-live
    (replaces `claude-code-review.yml`).

### Cluster J -- Opus 4.7 readiness (do BEFORE flipping mas_main)

J1. Strip `thinking.budget_tokens` on Opus-4.7 call sites.
J2. Strip `temperature`/`top_p`/`top_k` on Opus-4.7 call sites.
J3. Bump `max_tokens` +35% for new tokenizer.
J4. Set `thinking.display: "summarized"` where UI shows thinking.
J5. Adopt `xhigh` effort level on harness agents (researcher,
    qa-evaluator, harness-verifier, planner).
J6. Populate `model_tiers.py::_LIVE_TIER` before any
    `COST_TIER=live` flip: proposed `mas_main=opus-4-7`,
    `mas_qa=sonnet-4-6`, `mas_communication=haiku-4-5`,
    `autoresearch_{fast,smart,strategic}={haiku,sonnet,opus}-4-7`.

## Coverage audit

Full list of URLs actually read in full (phase-4.10 + 4.11 combined):

- **Platform.claude.com**: ~60 pages (prompt caching, batches,
  files, vision, PDF, citations, tool use + 14 subpages,
  compaction, context editing, context windows, token counting,
  embeddings, effort, adaptive + extended thinking, search
  results, Structured outputs, streaming, Service tiers, rate
  limits, errors, beta headers, versioning, supported regions,
  data residency, api-and-data-retention, models overview,
  choosing-a-model, migration guide, pricing, whats-new-claude-
  4-7, 4 use-case guides, admin API, workspaces, usage-cost
  API, Bedrock / Vertex / Foundry Anthropic integration, 9
  Managed Agents, 8 language SDK pages).

- **Code.claude.com**: ~70 pages (overview, quickstart, how-it-
  works, 17 Agent SDK, CLI reference, commands, memory, hooks,
  skills, 5 plugins, permissions, permission-modes, security,
  sandboxing, ZDR, data-usage, settings, server-managed-settings,
  claude-directory, model-config, tools-reference, env-vars,
  environment-variables, statusline, output-styles, keybindings,
  terminal-config, voice-dictation, fullscreen, fast-mode,
  remote-control, 2 channels, routines, scheduled-tasks, desktop-
  scheduled-tasks, github-actions, gitlab-ci-cd, code-review,
  slack, chrome, web-quickstart, 2 desktop, devcontainer,
  interactive-mode, headless, ultraplan, ultrareview,
  checkpointing, llm-gateway, network-config, vs-code,
  jetbrains, 4 third-party-integrations, monitoring-usage, costs,
  analytics, legal-and-compliance, changelog, whats-new index +
  W13/W14/W15, authentication, troubleshooting).

Total: ~130 platform + Claude Code pages verified read-in-full
across both audit phases. Zero URLs returned 404 on retry; a few
pages were persisted-to-disk during WebFetch due to >50KB size.

## Implementation sizing (for action-list approval)

- **MUST FIX items 1-20** cluster into ~10 phase-4.10.x
  cycles under the full harness loop. Items 1, 2, 3 are most
  urgent and span multiple files.
- **NICE TO HAVE clusters A, B, C (prompt caching + advisor + PTC
  + Files API + Priority tier)** are the highest cost/latency
  ROI. Dedicated phase-4.12 "Platform-feature adoption".
- **Cluster E (Agent SDK port of planner + evaluator)** is
  ~2-3 days, reversible, high-impact -- phase-4.13 candidate.
- **Cluster H (Managed Agents pilot)** requires Anthropic form
  submission for memory + multi-agent RP access; kick off
  proactively.
- Clusters D, F, G, I are tractable one-at-a-time in later
  phases.

## Decision required

Before any implementation begins:
1. Approve the 20 MUST-FIX items (or prune).
2. Pick 3-5 NICE-TO-HAVE clusters for phase-4.12 scope.
3. Decide routines-vs-cron for phase-10.7 Meta-Evolution (blocks
   `cron_budget.yaml` activation per MUST-FIX #19).
4. Decide Agent SDK adoption target: port planner + evaluator
   only (Cluster E1), or defer pending SDK maturation.
5. Decide Managed Agents pilot: submit access form now, or defer.

All implementation blocked pending approval per audit-only mandate.

## References

All 14 topic audit files under `handoff/audit/phase-4.10/` and
`handoff/audit/phase-4.11/`. Documentation accessed 2026-04-18
across platform.claude.com and code.claude.com.
