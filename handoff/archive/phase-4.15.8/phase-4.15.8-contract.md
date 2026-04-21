# Contract — Cycle 4.15.8 — Tool-use primitives compliance

Step: phase-4.15.8 Tool-use primitives (advisor, bash, code-execution,
computer, memory, text-editor, web-search/fetch, programmatic tool
calling, tool-search, fine-grained tool streaming, tool-reference)

## Research gate
Spawn `researcher` (merged) to cover each of the 13 tool primitive
pages + internal AGENT_TOOLS inventory.

## Hypothesis
pyfinagent uses custom domain tools (AGENT_TOOLS in MAS
orchestrator) + 4 in-process FastMCP servers, but:
- Zero use of `advisor_20260301` (Sonnet-executor + Opus-advisor)
- Zero use of `memory_20250818` (native memory tool)
- Zero use of `web_search_20260209` / `web_fetch_20260209` (the
  `researcher` harness agent uses WebSearch/WebFetch built-ins,
  NOT the native server-side tools)
- Zero use of programmatic tool calling (`allowed_callers`)
- Zero use of tool-search (`defer_loading`)
- No `strict: true` on AGENT_TOOLS
- No `cache_control` breakpoint on AGENT_TOOLS tool-array

## Success criteria (immutable)
1. every_doc_pattern_status_evidenced
2. qa_runs_live_code_checks_not_review
3. deviations_cite_doc_page

## Scope
`docs/audits/compliance-tool-use.md` — pattern-per-row covering
each of 13+ tool primitive pages. Per row: Status / Evidence /
Deviation / Risk / Recommended fix / MF-# mapping.

## Anti-patterns guarded
- Don't conflate Claude Code's built-in `WebSearch`/`WebFetch`
  tools (available to the harness session) with the Anthropic
  Messages-API `web_search_20260209` server tool (billed server-
  side, citations native). Different products.
- Don't conflate `memory_20250818` (Messages-API tool, client-
  executed) with Claude Code's `memory` feature (CLAUDE.md + auto
  memory, session-scoped).

## Out of scope
Managed Agents tools (covered in 4.15.14).

## Risk
Advisor + PTC + memory-tool adoption could materially reshape MAS
tool loop cost/latency but must be gated on provider/model
compatibility (Opus 4.6+/4.7 only for several).

## References
- Phase-4.11 tool_use_primitives.md (prior deep read)
- Phase-4.13 messages_sidebar_sweep.md
