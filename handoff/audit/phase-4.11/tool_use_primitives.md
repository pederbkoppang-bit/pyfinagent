# Tool-use Primitives Deep Audit (phase-4.11.2)

Scope: full read of every Anthropic tool-use primitive page (14 URLs) and
mapping to pyfinAgent's current tool plumbing (`backend/agents/multi_agent_orchestrator.py`
`AGENT_TOOLS`, `backend/tools/*.py`, `.claude/agent-memory/`, harness scripts).

## URL coverage

All 14 pages fetched in full (auto-persisted where over 54 KB):

1. overview — pricing, tool_choice, system-prompt token overhead
2. how-tool-use-works — client vs server execution, `stop_reason=tool_use` loop
3. advisor-tool — `advisor_20260301`, Sonnet-executor + Opus-advisor
4. bash-tool — `bash_20250124`, client-executed persistent shell
5. code-execution-tool — `code_execution_20250825` / `_20260120`, sandboxed Python+Bash, PTC
6. computer-use-tool — `computer_20251124`, screenshot + mouse/keyboard
7. memory-tool — `memory_20250818`, `/memories` directory, client-executed
8. text-editor-tool — `text_editor_20250728`, view/str_replace/create/insert
9. web-search-tool — `web_search_20260209` (dynamic filtering) / `_20250305`
10. web-fetch-tool — `web_fetch_20260209` / `_20250910`, URL allowlist, 250-char URL cap
11. fine-grained-tool-streaming — `eager_input_streaming: true` per tool
12. programmatic-tool-calling — `allowed_callers: ["code_execution_20260120"]`
13. tool-search-tool — `tool_search_tool_regex_20251119` / `_bm25_20251119`, `defer_loading`
14. tool-reference — canonical type/version matrix

## Per-tool digest and pyfinAgent relevance

### 1. Overview + How it works
**Digest.** Client tools return `stop_reason: "tool_use"`, caller executes and
replies with `tool_result`. Server tools run on Anthropic infra and can trigger
`pause_turn`. Every non-empty `tools` array adds 313–346 system-prompt tokens
depending on `tool_choice`. Anthropic-schema tools (bash, text_editor, memory,
computer) are trained-in, so they outperform hand-rolled equivalents with the
same function.

**pyfinAgent.** Our MAS orchestrator (`AGENT_TOOLS`, 7 read-only tools) already
uses the client-side loop. The tool definitions are custom JSON schemas, not
Anthropic-schema — fine because they are domain-specific readers (contract,
harness_log, best_params), not shell/editor equivalents.

### 2. Advisor tool (`advisor_20260301`, beta)
**Digest.** Executor calls `advisor` with empty input; server injects the full
transcript and bills a sub-inference on the advisor model. Output 400–700 text
tokens. Caching pays off at ≥3 calls/conversation. Headline pairing: Sonnet 4.6
executor + Opus 4.7 advisor.

**pyfinAgent relevance — HIGH.** Our MAS orchestrator uses opus-4-6 for Ford
and sonnet-4-6 for Communication/Researcher, with explicit turn chaining for
Sonnet→Opus handoff. The advisor tool replaces this with mid-generation
consults — Sonnet drives the `MAX_TOOL_TURNS=5` subagent loop and only pulls
Opus when stuck. Rough savings: ~60% of current Opus output tokens move to
Sonnet, Opus billed only on advisor calls.

### 3. Bash tool (`bash_20250124`)
**Digest.** Schema-less, client-executed, persistent session, 245 input tokens.
**pyfinAgent — LOW.** Claude Code's built-in Bash subagent tool already covers
this. Our harness shells out from a Python driver, not an LLM loop. MAS
subagents read curated artifacts, not shell.

### 4. Code execution tool (`code_execution_20250825` / `_20260120`)
**Digest.** Server-side sandboxed Python+Bash. `_20260120` adds REPL state and
enables programmatic tool calling. Free when paired with web_search/fetch 2026
versions. Container 4.5min idle TTL, 30-day hard cap. Not ZDR-eligible.
**pyfinAgent — MEDIUM.** Backtests/migrations must stay local (BQ creds, 1167-line
`backtest_engine.py`). But small numerical checks inside the `qa-evaluator`
subagent could move to code_execution and remove inline Claude-math.

### 5. Computer use tool (`computer_20251124`, beta)
**Digest.** Screenshots + mouse/keyboard, Xvfb container, 735 tokens. Opus 4.7
supports 2576 px long edge at 1:1 coords.
**pyfinAgent — LOW.** No desktop-automation workload. Playwright owns frontend
e2e tests at lower cost than Opus clicks. Skip.

### 6. Memory tool (`memory_20250818`)
**Digest.** Client-executed tool with `/memories` directory and commands
view/create/str_replace/insert/delete/rename. Ships a built-in system prompt:
"ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE". Pairs with
context editing and compaction for long-running agents.

**pyfinAgent relevance — HIGH.** Two disjoint memory systems today:
`.claude/agent-memory/researcher/*.md` (5 files, in-repo) and `MEMORY.md` in
`~/.claude/projects/.../memory/` (user auto-memory, outside repo). Switching the
MAS Layer-2 agents onto `memory_20250818` with `/memories` mapped to
`.claude/agent-memory/` buys us: trained-in schema, native compaction
integration for 28-cycle harness runs, and the built-in "check memory first"
system prompt. Do *not* merge with `backend/memory.py` (BM25 financial memory
serves Layer-1 Gemini).

### 7. Text editor tool (`text_editor_20250728`)
**Digest.** `view`, `str_replace`, `create`, `insert`. `max_characters`
truncation. 700 tokens per definition.

**pyfinAgent relevance — LOW.** Claude Code subagents already use the
`Edit`/`Write`/`Read` tools from the harness. The MAS subagents themselves are
read-only over artifacts (they call `read_*` wrappers) and do not need to write
files — writes go through the harness Python driver. Adding `text_editor` would
just duplicate capability the harness already owns.

### 8. Web search tool (`web_search_20260209` / `_20250305`)
**Digest.** Server-executed, $10 / 1k searches, citations always enabled,
`allowed_domains` / `blocked_domains` / `user_location` filters. The 2026-02-09
version adds dynamic code-execution-backed filtering.

**pyfinAgent relevance — MEDIUM.** We have no generic web search in
`backend/tools/`. All external data is typed: `alphavantage`, `fred_data`,
`alt_data`, `sec_insider`, etc. For the `researcher` harness subagent (gated
per `research-gate.md`), native `web_search_20260209` replaces the current
pattern of manually pasting URLs into context. This is a clean win because
citations are emitted in the block format we already need for research gate
compliance.

### 9. Web fetch tool (`web_fetch_20260209` / `_20250910`)
**Digest.** Server-executed, no charge beyond tokens. URLs must already appear
in conversation (anti-exfiltration). 250-char URL limit. `max_content_tokens`
cap. JavaScript-rendered pages not supported.

**pyfinAgent relevance — MEDIUM.** Our custom `social_sentiment`, `patent_tracker`,
`earnings_tone` tools fetch from known APIs, not arbitrary URLs — so
`web_fetch` does *not* replace them. But for the `researcher` subagent, paired
with `web_search_20260209`, it eliminates our current hand-rolled fetcher stub.

### 10. Fine-grained tool streaming (`eager_input_streaming: true`)
**Digest.** Ships partial JSON deltas without buffering. GA on all models.
Caveat: inputs may be invalid JSON (esp. on `max_tokens` hits). Your code must
accumulate `input_json_delta` fragments manually.

**pyfinAgent relevance — LOW.** Our tool loop in `multi_agent_orchestrator.py`
uses `client.messages.create` synchronously (non-streaming, as is typical for
the Anthropic Python SDK default). We only surface final messages to the UI via
`mas_events.py`. Streaming would matter for the frontend "Harness tab" live
view, but we already poll. Nice-to-have, not a priority.

### 11. Programmatic tool calling
**Digest.** Requires `code_execution_20260120`. Tool gets `allowed_callers:
["code_execution_20260120"]` and becomes an async Python function inside the
sandbox. Intermediate tool results never hit the context window — massive token
savings for N>3 batch loops. Incompatible with `strict: true`, `tool_choice`
forcing, and MCP-connector tools.

**pyfinAgent relevance — HIGH for MAS, LOW for harness.** The MAS subagent
loop issues all 7 `read_*` tools sequentially per round. Wrapping them with
`allowed_callers: ["code_execution_20260120"]` lets Claude write one async-gather
block: 7 round-trips → 1, with intermediate results never hitting context.
Worth piloting on Ford (touches all 7 reads). Harness driver is not a
candidate — Python process, not Claude loop.

### 12. Tool search tool (`tool_search_tool_regex_20251119` / `_bm25_20251119`)
**Digest.** Mark tools `defer_loading: true`; tool-search surfaces 3–5 per
query; full definitions expanded inline via `tool_reference` blocks. Preserves
prompt cache. Sweet spot is ≥10 tools or ≥10k tokens of tool definitions. Max
10k tools.

**pyfinAgent relevance — LOW today, MEDIUM future.** Our 7-tool MAS registry
is below the 10-tool break. If we register the 28 Layer-1 skills as callable
tools (today they are prompt-templates), or expose `backend/tools/*.py` (17
files) via MCP, tool_search becomes relevant. Defer until MCP connector wiring.

### 13. Tool reference (`tool-reference` page)
**Digest.** Five cross-cutting properties compose on any tool: `cache_control`,
`strict`, `defer_loading`, `allowed_callers`, `input_examples`, `eager_input_streaming`.
Every server tool has a dated type except `mcp_toolset`, which versions via
`anthropic-beta` header.

**pyfinAgent relevance.** We are under-using `strict: true` (guarantees our
`read_*` inputs parse). Adding `strict` on the 7 AGENT_TOOLS is a one-line win.
We also have no `cache_control` breakpoint on the tool definitions — adding one
after `AGENT_TOOLS` would cache the 7-tool prefix across the 28 harness cycles.

## Findings table

| Primitive | Replaces existing pyfinAgent code? | Priority | Rough effort |
|---|---|---|---|
| Advisor tool | Replaces manual Sonnet→Opus handoff in MAS | **HIGH** | 1 day (feature-flag) |
| Memory tool | Replaces ad-hoc `.claude/agent-memory/researcher/*.md` loading | **HIGH** | 1–2 days (path mapping + tests) |
| Programmatic tool calling | Compresses 7-read subagent loop to 1 turn | **HIGH** | 2 days (needs `code_execution_20260120`) |
| Web search + fetch | Replaces `researcher` subagent's manual URL paste | **MEDIUM** | 0.5 day |
| `strict: true` on AGENT_TOOLS | Guarantee input schema conformance | **MEDIUM** | 15 min |
| `cache_control` on AGENT_TOOLS | Cache 7-tool prefix across cycles | **MEDIUM** | 15 min |
| Code execution (non-PTC) | Small numeric checks inside `qa-evaluator` | LOW–MED | 0.5 day |
| Tool search (`defer_loading`) | Only pays off if we register 28 skills as tools | LOW now | Future |
| Fine-grained streaming | UX-only (harness tab already polls) | LOW | 0.5 day |
| Bash tool | Duplicates Claude-Code's built-in Bash | SKIP | — |
| Text editor tool | Duplicates Claude-Code's Edit/Write/Read | SKIP | — |
| Computer use | No desktop automation workload | SKIP | — |

## MUST FIX

1. **Add `strict: true` and a `cache_control` breakpoint on `AGENT_TOOLS`.**
   `backend/agents/multi_agent_orchestrator.py:72-120`. Fifteen minutes of work,
   immediate token savings across every 28-cycle harness run, and closes the
   schema-drift footgun where a subagent could send `read_harness_log(last_n="5")`
   as a string instead of int.

2. **Stop rolling our own "memory" in two inconsistent places.** Pick one path:
   either port `.claude/agent-memory/researcher/*.md` onto `memory_20250818`
   with `/memories` mapped to that directory, or document explicitly why we do
   not use it (e.g., harness runs without network on that call site). Right now
   MEMORY.md lives in `~/.claude/projects/.../memory/` and the researcher
   memory lives in-repo — two systems, one discoverability problem.

3. **Remove dead scaffolding per Anthropic's stress-test doctrine.** We carry
   `mas_events.py` event emissions around our read tools that would be
   unnecessary if those tools ran under `allowed_callers: ["code_execution_20260120"]`
   (the intermediate reads never enter context). Stress-test this before next
   cycle: if Opus 4.7 can do the 7-read synthesis in one PTC block, prune
   the per-tool event emission.

## NICE TO HAVE (prioritized by impact on pyfinAgent)

1. **Pilot advisor tool on MAS subagent loop.** Feature-flag `ENABLE_ADVISOR=true`
   in `backend/agents/multi_agent_orchestrator.py`; set executor to
   `claude-sonnet-4-6` for subagent tool loops, advisor to `claude-opus-4-7`.
   Measure token cost vs current Opus-only path across 5 cycles. If neutral or
   better, default on. Risk: advisor does not stream, so the "harness tab"
   live-view will see pauses — acceptable because we already poll.

2. **Pilot programmatic tool calling for Ford's 7-read synthesis.** Flip
   `allowed_callers: ["code_execution_20260120"]` on all `AGENT_TOOLS`, require
   `code_execution_20260120`, and let Ford write one async-gather block. Expect
   ~80% context-window reduction on the planning turn.

3. **Switch the `researcher` harness subagent to native web_search + web_fetch
   (2026-02-09).** Replace the manual URL-paste pattern with `tools=[web_search_20260209,
   web_fetch_20260209]`. Citations fall out natively in the required
   `web_search_result_location` shape, which is more compliant with
   `.claude/context/research-gate.md`'s "cite per claim" rule than our current
   best-effort citation handling.

4. **Move `MEMORY.md` + `.claude/agent-memory/researcher/` behind `memory_20250818`.**
   One memory backend, one system-prompt, uniform view/create/str_replace semantics,
   and native compaction integration for long harness runs.

5. **Fine-grained tool streaming for the Harness tab.** Set
   `eager_input_streaming: true` on the `read_*` tools so the frontend can
   render "Ford is reading evaluator_critique..." with partial progress instead
   of a spinner. UX polish, not a cost win.

6. **Tool-search tool for the 28 Layer-1 skills — future work.** Only
   cost-effective if/when we expose those skills as first-class tools via an
   MCP connector. Tracked as deferred.

## References

1. https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview (pricing table, 313–346 system-prompt tokens)
2. https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works (agentic loop, pause_turn)
3. https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool (Sonnet+Opus pairing, `advisor_20260301`)
4. https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool (`bash_20250124`, 245 tokens)
5. https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool (`code_execution_20260120`, PTC prerequisite)
6. https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool (`computer_20251124`, 735 tokens)
7. https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool (`memory_20250818`, `/memories` directory)
8. https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool (`text_editor_20250728`, 700 tokens)
9. https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool (`web_search_20260209`, $10/1k, citations)
10. https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool (`web_fetch_20260209`, 250-char URL cap)
11. https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming (`eager_input_streaming`)
12. https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling (`allowed_callers`)
13. https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool (`tool_search_tool_regex_20251119`, `defer_loading`)
14. https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference (version matrix, 6 cross-cutting properties)

pyfinAgent anchors:
- /Users/ford/.openclaw/workspace/pyfinagent/backend/agents/multi_agent_orchestrator.py (AGENT_TOOLS, lines 72–120)
- /Users/ford/.openclaw/workspace/pyfinagent/backend/tools/ (17 custom data fetchers, 3043 lines)
- /Users/ford/.openclaw/workspace/pyfinagent/.claude/agent-memory/researcher/ (5 project memory files)
- /Users/ford/.openclaw/workspace/pyfinagent/backend/agents/skills/ (28 Layer-1 skill prompts)
- /Users/ford/.openclaw/workspace/pyfinagent/scripts/harness/ (Python-driven, subprocess-based, not LLM-callable)
