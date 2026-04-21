# Compliance Audit — Tool-Use Primitives (phase-4.15.8)

**Date:** 2026-04-18  
**Scope:** All 14 tool-primitive doc pages (overview + 13 specific tools)  
**Internal files inspected:** `multi_agent_orchestrator.py` L72-120 (AGENT_TOOLS) + L930-1030 (tool loop), `llm_client.py`, `backend/tools/*.py` (17 files, 3043 lines total), `backend/agents/mcp_servers/` (4 FastMCP servers)  
**Live grep baseline:** All 8 absence-checks returned 0 — confirmed no adoption of any versioned Anthropic tool primitives in `backend/`.

---

## Pattern Table (28 rows)

| # | Pattern / Doc page | Status | Evidence | Deviation | Risk | Recommended fix | MF-# |
|---|---|---|---|---|---|---|---|
| 1 | **Tool overview — client vs server execution model understood** | ✅ | Tool loop at L944-954 correctly handles `stop_reason == "tool_use"`, builds `tool_result` blocks, re-sends. Matches the canonical client-loop shape. | None | None | — | — |
| 2 | **Tool overview — `strict: true` on AGENT_TOOLS** | ❌ | L72-120: all 7 AGENT_TOOLS entries have no `strict` key. Doc tip: "Guarantee schema conformance with strict tool use." | Missing `strict: true` across entire tools array | Silent schema drift — model may call tools with wrong arg shapes; hard to catch without strict validation | Add `"strict": True` to each dict in `AGENT_TOOLS`. All 7 tools have simple schemas (mostly `{}`) so no migration cost. | MF-5 |
| 3 | **Tool overview — `cache_control` breakpoint on AGENT_TOOLS array** | ❌ | `grep cache_control multi_agent_orchestrator.py` → 0 hits. `llm_client.py` L602-608 applies `cache_control` on the system prompt block, but not on the tools array. Doc: cache breakpoint on last tool in array caches entire tools block. | Tools array uncached on every MAS call | Every tool-loop turn re-encodes the 7 AGENT_TOOLS definitions (~400+ tokens) as new input tokens | Add `"cache_control": {"type": "ephemeral"}` to the last entry in `AGENT_TOOLS` (or a separate breakpoint entry). Pairs naturally with the existing system-prompt caching in `llm_client.py`. | MF-7 |
| 4 | **How tool use works — agentic loop shape (while stop_reason == "tool_use")** | ✅ | L938: `for turn in range(max_turns)` → L962: `if response.stop_reason == "tool_use"` → execute and re-send → else break. Structurally equivalent to the canonical while loop. | Loop is `for` not `while` — bounded by `MAX_TOOL_TURNS` | Bounded loop is actually a safer pattern; doc recommends capping iterations. | — | — |
| 5 | **How tool use works — parallel tool execution** | ✅ | L968-983: multiple `tool_use` blocks detected → `ThreadPoolExecutor(max_workers=len(tool_blocks))` dispatches all in parallel. Matches doc guidance on parallel tool calls. | None | None | — | — |
| 6 | **How tool use works — `tool_runner()` helper not used** | ⚠️ | Hand-rolled dispatch at L962-993. Doc (overview) references `client.beta.messages.tool_runner()` as a convenience helper that manages the loop. | Not using the SDK helper | No runtime risk — hand-rolled loop is correct. Maintenance cost: more code to keep in sync with future loop semantics changes. | Consider migrating to `tool_runner()` in a future refactor cycle; low priority given loop is correct today. | MF-11 |
| 7 | **How tool use works — `tool_choice` not set** | ⚠️ | `grep tool_choice multi_agent_orchestrator.py` → 0 hits. `client.messages.create` at L944 passes no `tool_choice`. Default is `"auto"` which is correct for open-ended agents. | Implicit reliance on default `"auto"` | No deviation from recommended behavior, but implicit. If model ever gets a version that changes default, break is silent. | Add explicit `tool_choice={"type": "auto"}` for documentation clarity. | — |
| 8 | **Advisor tool (`advisor_20260301`) — not adopted** | ❌ | `grep advisor_20260301` → 0. `grep advisor-tool-2026-03-01` → 0. Not present anywhere in backend. | Zero adoption | Subagents (Sonnet 4.6 executor) call `client.messages.create` for every planning turn at full Opus-equivalent cost. Adding Opus 4.7 advisor to the subagent loop would provide a quality lift on complex planning turns at similar or lower marginal cost per token. | Evaluate adding `{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-7"}` to AGENT_TOOLS for the highest-value MAS agents (planner, evaluator). Requires `betas=["advisor-tool-2026-03-01"]` on the `client.messages.create` call. Gate on LLM API cost approval (per CLAUDE.md). | MF-23 |
| 9 | **Advisor tool — executor/advisor model pairing** | N/A | Not adopted. If adopted: current executor model `claude-opus-4-6` (set at L148) is valid with advisor `claude-opus-4-7` per the compatibility table. | — | — | Verify `agent_config.model` at call site when implementing. | — |
| 10 | **Advisor tool — `max_uses` cost control** | N/A | Not adopted. Doc requires client-side tracking for conversation-level caps. | — | — | If adopted: set `max_uses` on the advisor tool definition to prevent runaway advisor calls inside the tool loop. | — |
| 11 | **Memory tool (`memory_20250818`) — not adopted** | ❌ | `grep memory_20250818` → 0. `grep memory-tool` → 0. Harness uses file-based BM25 memory (`harness_memory.py`) and BQ persistence — entirely separate from the Messages-API `memory_20250818` tool. | Zero adoption of native memory tool | pyfinagent already has a custom memory system; native tool adds overhead for no gain here. Gap is in cross-session continuity for MAS subagents — currently state is re-loaded from BQ on each run. | No immediate action required. The custom file-based memory is more tightly integrated with BQ. Native `memory_20250818` is worth evaluating only if MAS subagents need cross-session persistence without BQ. | — |
| 12 | **Bash tool (`bash_20250124`) — not adopted** | ❌ | `grep bash_20250124` → 0. No bash tool defined anywhere in MAS loop. | Not adopted | harness runs shell commands via `subprocess` in Python directly, not via the Anthropic bash tool. No gap for current use case. | No action required for current architecture. Bash tool would only be relevant if subagents needed to invoke shell commands mid-conversation. | — |
| 13 | **Text editor tool (`text_editor_20250728`) — not adopted** | ❌ | `grep text_editor_20250728` → 0. File edits done via Python `open()`/`write()` in harness scripts. | Not adopted | No gap — harness writes files directly, not through Claude. | No action required. | — |
| 14 | **Computer use tool (`computer_20251124`) — not adopted** | ❌ | `grep computer_20251124` → 0. | Not adopted | No gap — pyfinagent has no desktop/browser automation use case. | No action required. | — |
| 15 | **Code execution tool (`code_execution_20250825` / `_20260120`) — not adopted** | ❌ | Both version strings: 0 hits. | Not adopted | Quant optimizer runs backtests via direct Python calls to `BacktestEngine`, not via a sandboxed code execution tool. | No immediate action. Sandboxed code execution would be relevant if Claude needed to dynamically generate and test backtest code mid-conversation — a potential phase-5 capability. | — |
| 16 | **Tool search tool (`tool_search_tool_regex` / `_bm25`) — not adopted** | ❌ | `grep defer_loading` → 0. `grep tool_search_tool` → 0. | Not adopted | AGENT_TOOLS has only 7 tools — well below the ~10-tool threshold where tool search provides value. | No action required at current scale. Revisit if AGENT_TOOLS grows beyond 15 tools. | — |
| 17 | **Programmatic tool calling (`allowed_callers`) — not adopted** | ❌ | `grep allowed_callers` → 0. Requires `code_execution_20260120` which is also not adopted. | Not adopted | No gap. Current parallel tool execution via ThreadPoolExecutor is the right pattern for synchronous Python context. | No action required. Programmatic tool calling would be relevant if Claude needed to batch many tool calls inside the sandbox. | — |
| 18 | **Fine-grained tool streaming (`eager_input_streaming`) — not adopted** | ❌ | `grep eager_input_streaming` → 0. Tool loop at L944 uses synchronous `messages.create` (not streaming). | Not adopted | No gap. MAS tool loop does not require streaming; latency is dominated by backtest execution, not API response buffering. | No action required for current use case. | — |
| 19 | **Web search tool (`web_search_20260209` / `_20250305`) — not adopted in Messages API** | ✅ (correct) | 0 hits in backend. The harness researcher agent uses Claude Code's built-in `WebSearch` tool (session-scoped), NOT the Messages-API server tool. Correct separation per contract anti-patterns. | No conflation | None | — | — |
| 20 | **Web fetch tool (`web_fetch_20260209` / `_20250910`) — not adopted in Messages API** | ✅ (correct) | 0 hits in backend. Same pattern as web search — harness uses Claude Code built-ins, not server-side Messages-API tools. | No conflation | None | — | — |
| 21 | **Tool loop — `betas` header for advisor tool** | ❌ | `grep betas multi_agent_orchestrator.py` → 0 hits. `client.messages.create` at L944 passes no `betas`. | No beta headers set | If advisor tool is adopted, the call will fail with a 400 unless `betas=["advisor-tool-2026-03-01"]` is added. Computer use similarly requires a beta header. | When adopting advisor tool: add `betas=["advisor-tool-2026-03-01"]` to the `client.messages.create` kwargs. | MF-23 |
| 22 | **`llm_client.py` — cache_control on system prompt** | ✅ | L602-608: system prompt sent as structured block with `{"type": "ephemeral"}` when `enable_prompt_caching=True`. Correct implementation per doc guidance. | None | None | — | — |
| 23 | **`llm_client.py` — no tool-use parameters (no AGENT_TOOLS)** | ✅ | `llm_client.py` handles Gemini calls (primary pipeline); it never passes `tools=`. Tool use is exclusively in `multi_agent_orchestrator.py` via direct `anthropic.Anthropic()` client. Correct separation. | None | None | — | — |
| 24 | **Custom domain tools (`backend/tools/*.py`) — 17 fetchers** | ✅ | 17 tool files (3043 total lines): alphavantage, alt_data, anomaly_detector, earnings_tone, fred_data, monte_carlo, nlp_sentiment, options_flow, patent_tracker, quant_model, screener, sec_insider, sector_analysis, slack, social_sentiment, yfinance_tool + __init__. These are called from orchestrator.py (Layer 1), not from AGENT_TOOLS. Not exposed as Messages-API tool schemas. | None — correct architecture | None | — | — |
| 25 | **FastMCP servers (`mcp_servers/*.py`) — 4 in-process servers** | ✅ | backtest_server, data_server, risk_server, signals_server: all use `@mcp.tool` decorator pattern (FastMCP). Distinct from Messages-API AGENT_TOOLS. Serve the harness via MCP protocol, not via Anthropic tool-use API. Correct. | None | None | — | — |
| 26 | **Tool definition `input_schema` correctness** | ✅ | All 7 AGENT_TOOLS have syntactically valid `input_schema` objects (type=object, properties dict, required list). No missing required fields. | None | None | — | — |
| 27 | **Tool naming conventions** | ✅ | All 7 AGENT_TOOLS use snake_case names matching the trained-in conventions (doc: tool names should be descriptive). No conflicts with Anthropic reserved names. | None | None | — | — |
| 28 | **`pause_turn` / server-side loop continuation** | N/A | No server-executed tools (web_search, code_execution etc.) are in use. `pause_turn` handling is therefore irrelevant for current architecture. | — | — | If code_execution or web_search is adopted: add `pause_turn` handling — re-send conversation to let model continue. | — |

---

## Summary by severity

### Critical gaps (production risk)

**P1 — MF-5: `strict: true` missing on AGENT_TOOLS**  
All 7 tools at `multi_agent_orchestrator.py` L72-120 lack `"strict": True`. The doc guarantees schema conformance only when strict is set. Without it, the model can emit malformed tool inputs that silently corrupt tool execution. Fix: add `"strict": True` to each tool dict. Cost: zero — all schemas are simple.

**P2 — MF-7: `cache_control` missing on AGENT_TOOLS array**  
The 7-tool array is re-sent as raw input tokens on every MAS call and every tool-loop turn. `llm_client.py` already correctly caches system prompts — the same pattern should be applied to AGENT_TOOLS by adding a `cache_control` breakpoint to the last tool entry. At ~400 tokens per AGENT_TOOLS array, and given multi-turn tool loops, this is a recurring cost leak.

### Adoption opportunities (quality/cost improvement)

**P3 — MF-23: Advisor tool (`advisor_20260301`) not evaluated**  
The MAS planner and evaluator subagents run on `claude-opus-4-6` for all turns uniformly. For the planning/strategy turns (where intelligence matters most), pairing a Sonnet executor + Opus 4.7 advisor would provide a quality lift at similar or lower cost per token. This requires Peder's explicit LLM API cost approval before adoption.

### Correct non-adoptions (not gaps)

The following tools are correctly absent from pyfinagent's backend: `memory_20250818` (custom BQ-backed memory is more appropriate), `bash_20250124` (subprocess used directly), `text_editor_20250728` (Python file I/O used directly), `computer_20251124` (no desktop automation use case), `code_execution_20250825/20260120` (backtests run via direct Python calls), `tool_search_tool_regex/bm25` (only 7 tools, below the threshold), `allowed_callers` / programmatic tool calling (no sandbox needed), `eager_input_streaming` (synchronous loop, not streaming), `web_search_20260209` / `web_fetch_20260209` in the Messages API (harness correctly uses Claude Code built-ins).

### Conflation anti-patterns — confirmed avoided

- Claude Code `WebSearch`/`WebFetch` built-ins vs Messages-API `web_search_20260209` / `web_fetch_20260209`: correctly separated.
- Claude Code session memory vs Messages-API `memory_20250818`: correctly separated.

---

## Sources

1. https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
2. https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works
3. https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool
4. https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
5. https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
6. https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
7. https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
8. https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
9. https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
10. https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
11. https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming
12. https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling
13. https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
14. https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
