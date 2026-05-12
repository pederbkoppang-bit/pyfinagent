---
step: phase-24.9
topic: LLM provider conformance audit — Claude + Gemini
tier: complex
date: 2026-05-12
---

## Research: LLM Provider Conformance Audit (Claude + Gemini)

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching | 2026-05-12 | Official doc | WebFetch | Full cache mechanics: prefix model, cache_control blocks, 5-min/1-hr TTL, 0.1x read cost, 4096-token min on Opus/Haiku 4.5, workspace-level isolation since Feb 2026 |
| https://platform.claude.com/docs/en/docs/build-with-claude/extended-thinking | 2026-05-12 | Official doc | WebFetch | budget_tokens param; Opus 4.7 rejects manual thinking — adaptive only; thinking tokens billed at output rate; 5k budget for moderate tasks, 32k ceiling before diminishing returns |
| https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing | 2026-05-12 | Official doc | WebFetch | 50% flat discount, async up to 24 hrs, up to 10,000 requests per batch, 300k output tokens per response via beta header, no streaming |
| https://platform.claude.com/docs/en/docs/build-with-claude/tool-use | 2026-05-12 | Official doc | WebFetch | strict:true tool schemas, server-side tools (web_search, code_execution), tool_choice auto/any/tool, 346-token overhead per request regardless of tool count |
| https://jangwook.net/en/blog/en/anthropic-files-api-batch-document-processing-guide/ | 2026-05-12 | Tech blog | WebFetch | Files API: upload-once/reuse pattern with file_id, 500MB limit, requires files-api-2025-04-14 beta header, combinable with batch API for compounded savings |
| https://www.mager.co/blog/2026-04-29-claude-prompt-caching/ | 2026-05-12 | Tech blog | WebFetch | 7 cache-breaking patterns: timestamp injection, tool-definition churn, model switching, thinking-param changes, re-pasting context, session drift, attaching different images |
| https://ai.google.dev/gemini-api/docs/thinking | 2026-05-12 | Official doc | WebFetch | Gemini 2.5 Flash: 0-24,576 token budget; 2.5 Pro: 128-32,768; thinking tokens billed as output tokens; Gemini 3 uses thinkingLevel enum instead of budget_tokens |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://platform.claude.com/docs/en/release-notes/overview | Release notes | Feature set confirmed via other sources |
| https://www.anthropic.com/news/message-batches-api | Announcement | Core details duplicated in official batch doc |
| https://www.finout.io/blog/anthropic-api-pricing | Pricing guide | Pricing cross-checked against cost_tracker.py; no new details |
| https://dev.to/whoffagents/claude-prompt-caching-in-2026-the-5-minute-ttl-change-thats-costing-you-money-4363 | Blog | TTL change noted; llm_client.py:773 already handles 1h TTL |
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-with-google-search | Official doc | Grounding architecture confirmed; orchestrator.py already implements correctly |
| https://ai.google.dev/gemini-api/docs/google-search | Official doc | Grounding mechanics confirmed via snippet + WebFetch of thinking page |
| https://releasebot.io/updates/anthropic/claude | Release tracker | Confirmed adaptive thinking / managed-agents-2026-04-01 beta via snippet |
| https://beginnersinai.org/whats-new-claude-2026/ | Blog | Model catalog cross-check; no novel API surface |
| https://intuitionlabs.ai/articles/token-optimization-chatgpt-claude-costs | Blog | Thinking budget cost context confirmed |
| https://age-of-product.com/token-economics-2026/ | Blog | Diminishing-returns narrative confirmed; no new API surface |

### Recency scan (2024-2026)

Searched (three-variant discipline):
1. Current-year frontier: "Anthropic Claude API new features 2026 citations batch files adaptive thinking"
2. Last-2-year window: "Claude thinking budget optimization diminishing returns token cost 2025 2026"
3. Year-less canonical: "Anthropic prompt caching best practices cache hit rate optimization"

Findings:
- **Feb 2026**: Cache isolation changed from organization-level to per-workspace. The 5-min default TTL has been the default since at least Q1 2026. `llm_client.py:773-779` already hardcodes `"ttl": "1h"` to restore the longer window — correct and current.
- **Mar 2026**: Message Batches API gained 300k output tokens per response via `anthropic-beta: output-300k-2026-03-24`. Not implemented in pyfinagent; no harm from absence.
- **Apr 2026**: Files API beta launched (`files-api-2025-04-14` header). Not implemented; relevant for FACT_LEDGER and skill markdown files repeated across 28 pipeline calls.
- **Apr 2026**: Opus 4.7 launched with adaptive thinking only (no manual budget_tokens). `llm_client.py:800-807` already handles the branching correctly.
- **May 2026**: Claude Managed Agents public beta. Not relevant to pyfinagent's self-hosted harness.
- No superseding literature on prompt-caching prefix mechanics or Gemini grounding patterns.

---

### Key findings

1. **Prompt caching infrastructure is correct but hit rate is likely sub-optimal for short system prompts.** `ClaudeClient` (`llm_client.py:704-1083`) applies `cache_control: {type:ephemeral, ttl:1h}` to the system prompt on every call (`llm_client.py:773-779`). However, the assembled system prompt for calls without a schema is `"You are a financial analysis AI."` (~35 tokens), well below the 4096-token minimum required for Opus 4.7 and Haiku 4.5 (2048 for Sonnet 4.6). These calls never cache despite sending `cache_control`. The Anthropic doc confirms: "Shorter prompts are processed without caching (silently — no error returned)." Moving skill markdown content + schema into the system block would push most calls past the threshold. (Source: Anthropic prompt-caching doc, 2026-05-12)

2. **`cost_tracker.py` under-reports cache write cost.** `cost_tracker.py:147` applies a 1.25× premium (`cache_creation * pricing[0] * 1.25`), which is the 5-minute TTL rate. However `llm_client.py:773-779` already sends `"ttl": "1h"` on every caching call, which Anthropic bills at 2.0× base input. The comment at `cost_tracker.py:140-148` itself says "When we adopt 1h TTL via beta header, bump to 2.0" — but the beta header is already in use. Underreporting: for a 4096-token system prompt on Opus 4.7 ($5/MTok), actual write cost is $0.041 vs $0.026 reported (60% under). (Source: Anthropic prompt-caching pricing table, 2026-05-12; `cost_tracker.py:147`, `llm_client.py:773`)

3. **Message Batches API is entirely unused.** All 28 pipeline agents call `generate_content()` synchronously. The 50% batch discount is directly applicable to non-interactive Steps 1-3 (Enrichment), Steps 4-5 (Market/Competitor), and Steps 6-7 (Debate rounds) when run for multiple tickers in bulk backtesting mode. No streaming used by `multi_agent_orchestrator.py`, so there is no migration blocker. Max 10,000 requests/batch; most batches finish < 1 hour. (Source: Anthropic batch-processing doc, 2026-05-12)

4. **Files API is unused; skill markdowns are re-sent on every call.** Skill markdown files (loaded via `load_skill()` and injected into prompts each orchestrator run) are typically 500-3000 tokens each. These are static between requests. Uploading once and passing `file_id` would eliminate repeated file-content transfer. The `_add_citations()` call at `multi_agent_orchestrator.py:1317-1323` also re-sends the full `system_prompt` string on each invocation. Requires `files-api-2025-04-14` beta header. (Source: jangwook.net Files API guide, 2026-05-12)

5. **Thinking budgets set for Gemini 2.5 Flash; cost justified for Critic/Moderator, questionable for Synthesis.** `orchestrator.py:95-118` defines Critic/Moderator at `budget_tokens:8192` and RiskJudge/Synthesis at `budget_tokens:4096`. Thinking tokens bill at output-token rates (on Gemini 2.5 Flash at $0.60/MTok for outputs, 8192 thinking tokens = ~$0.005/call). This is low-cost per call but cumulative across all analysis cycles. Synthesis (a structured-output JSON call with `response_schema: SynthesisReport`) benefits less from extended thinking than open-ended critic roles; the Anthropic extended-thinking doc recommends starting at 5k for moderate tasks and notes diminishing returns. A reduction of Synthesis to 2048 tokens is worth profiling. (Source: Anthropic extended-thinking doc + Google AI thinking doc, 2026-05-12; `orchestrator.py:112-118`, `settings.py:35-36`)

6. **Bespoke CitationAgent replaces Anthropic's native Citations feature.** `multi_agent_orchestrator.py:1284-1333` implements `_add_citations()` — an extra Claude Sonnet call that post-processes responses with numbered footnotes. This costs ~$0.003/Q&A call. Anthropic's native Citations feature (citations.enabled=true on document content blocks) performs citation grounding server-side at no extra LLM call. The constraint: `llm_client.py:874-881` already enforces that citations and structured output cannot coexist (API 400). Native citations require context to be passed as named `document` blocks rather than embedded in the system prompt. Current architecture embeds context in the system prompt — a refactor to document blocks would enable native citations and eliminate the extra LLM call per Q&A. (Source: Anthropic tool-use/citations doc, 2026-05-12; `multi_agent_orchestrator.py:1284-1333`, `llm_client.py:874-881`)

7. **Google Search grounding is correctly Gemini-only.** `orchestrator.py:407-419` assigns `grounded_client` only when `client.supports_grounding=True`. `ClaudeClient.supports_grounding = False` (`llm_client.py:696`). Grounded steps (4, 5, 9, 10) always use `GeminiClient`. The Gemini grounding implementation at `orchestrator.py:603-615` correctly extracts `grounding_metadata` from responses. No action needed. (Source: Google Search Grounding doc, 2026-05-12; `orchestrator.py:407-419`)

8. **In-process cache hit counter may miss cross-instance hits.** `ClaudeClient._cache_hits/_cache_misses` are instance-level counters (`llm_client.py:709-713`). Each call to `make_client()` creates a new instance. Server-side cache hits DO occur across instances when the prefix is byte-identical within the 1h TTL window — but the in-process `cache_hit_rate` counter (`llm_client.py:725-728`) only counts hits observed in the current instance's lifetime. The `cache_stats` property (`llm_client.py:730-741`) and `cost_tracker.py:218-236` accumulate from `UsageMeta.cache_read_input_tokens`, which IS correct — it comes from the API response regardless of instance identity. The hit-rate counter is thus misleadingly low but the token counts are accurate. (Source: `llm_client.py:709-741`, Anthropic prompt-caching doc)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/llm_client.py` | 1155 | Provider router; ClaudeClient (690-1083), GeminiClient (305-580), OpenAIClient (587-683) | Active. Cache: 773-779 (1h TTL). Thinking dispatch: 797-865. Cache stats: 709-741. Citations guard: 874-881. |
| `backend/agents/cost_tracker.py` | 255 | Per-agent token/cost accounting; cache premium at line 147 | Active. Bug: line 147 uses 1.25x but 1h TTL is already in use (should be 2.0x). |
| `backend/agents/orchestrator.py` | ~1477 | 28-agent pipeline; Gemini thinking config defs 93-118; grounding routing 407-419 | Active. Thinking enabled opt-in via settings.enable_thinking. Grounding correctly Gemini-only. |
| `backend/agents/multi_agent_orchestrator.py` | 1512 | Layer-2 MAS; tool-use loop 1023-1200; bespoke CitationAgent 1284-1333 | Active. Citations are a synthetic post-processing LLM call, not native Anthropic citations. |
| `backend/config/settings.py` | N/A (partial read) | Thinking budget defaults and feature flags | Active. Defaults: Critic/Moderator 8192, RiskJudge/Synthesis 4096, enable_thinking=False. |

---

### Consensus vs debate (external)

**Consensus**: Prompt caching prefix placement (static content first, dynamic last) is universal. Batch API 50% savings are uncontested. Diminishing thinking returns above 32k tokens is noted in both official docs and community.

**Debate**: Files API vs prompt caching for large repeated documents. Files API offers permanent storage (survives restarts, no re-upload on each process start). Prompt caching with 1h TTL is simpler to implement but requires the prefix to be in-flight within the window. For skill markdowns that load at startup and are static, Files API is the cleaner fit for long-running processes.

### Pitfalls (from literature)

- Changing thinking parameters between calls invalidates the message cache. `llm_client.py:800-807` branches adaptive vs. manual per model — if calls switch models mid-pipeline, cache context is lost.
- Dynamic context injected into the system prompt (e.g., timestamps, per-request IDs) defeats caching. `ClaudeClient` uses only static text — safe. Callers that modify the system prompt before the call will break cache hits.
- Batch API is incompatible with streaming. `multi_agent_orchestrator.py` does not stream Claude responses; no migration blocker.
- Native Citations are incompatible with `response_schema`/`output_config.format` (`llm_client.py:874-881`). A refactor to native citations must drop structured output on citation-eligible Q&A calls.

### Application to pyfinagent (file:line anchors)

| Finding | File:line | Action |
|---------|-----------|--------|
| Cache write premium 1.25x but 1h TTL in use (should be 2.0x) | `cost_tracker.py:147` | Change `1.25` to `2.0` to match Anthropic 1h billing |
| System prompt often below 4096-token cache threshold | `llm_client.py:751-759` | Consolidate skill body + schema into system block; ensure composite >4096 tokens for Opus/Haiku |
| In-process hit counter misleadingly low; token counts accurate | `llm_client.py:709-728` | Document the known gap; rely on `cache_read_input_tokens` from `UsageMeta` not `_cache_hits` |
| Batch API unused for bulk non-interactive steps | `orchestrator.py` (all `_generate_with_retry` calls) | Wrap Steps 1-3 with batched dispatch for multi-ticker backtest runs; 50% cost saving |
| Files API unused for repeated skill markdown | Skill loader in orchestrator; `llm_client.py:750-759` | Pre-upload skill markdowns at startup; pass `file_id` references; requires beta header |
| Bespoke CitationAgent vs native Citations | `multi_agent_orchestrator.py:1284-1333` | Refactor Q&A to pass context as document blocks; enables native citations, eliminates extra LLM call |
| Synthesis thinking budget 4096 on structured-output call | `orchestrator.py:112-118`, `settings.py:35-36` | Profile Synthesis at 2048; structured-output JSON call benefits less from extended thinking |
| Gemini grounding correctly Gemini-only | `orchestrator.py:407-419`, `llm_client.py:696` | No action needed; conforms to spec |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched: 5 official docs + 2 authoritative tech blogs)
- [x] 10+ unique URLs total including snippet-only (17 total: 7 full + 10 snippet)
- [x] Recency scan (last 2 years) performed + reported (three-variant search discipline applied)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered all four target files (llm_client.py 1155 lines full read; cost_tracker.py 255 lines full read; orchestrator.py key sections; multi_agent_orchestrator.py key sections)
- [x] Contradictions / consensus noted (cache write premium mismatch; synthetic citation vs native)
- [x] All claims cited per-claim (URL + access date inline)

---

### Summary (<=200 words)

Infrastructure is largely correct: cache TTL is 1h, grounding is Gemini-only, thinking dispatch handles Opus 4.7's adaptive-only constraint, and stop-reason dispatch is complete. Three actionable bugs and three unused-feature candidates found.

**Bugs**: (1) `cost_tracker.py:147` applies 1.25x cache-write premium but the code already sends 1h-TTL requests that Anthropic bills at 2.0x — under-reporting cache write cost by 60%. (2) System prompts built by `ClaudeClient` are typically below the 4096-token minimum threshold, so `cache_control` is sent but the write never registers for most calls. (3) In-process `cache_hit_rate` counter resets per instance — the rate figure is misleadingly low, though token-level accounting in `UsageMeta` is accurate.

**Unused features**: (1) Message Batches API — 50% flat discount applicable to non-interactive pipeline steps (Steps 1-3, 6-7) in bulk backtest mode. (2) Files API — upload skill markdowns once at startup, pass `file_id` instead of full text across 28 pipeline calls. (3) Native Anthropic Citations — eliminates the extra `_add_citations()` LLM call per Q&A response if context is restructured as document blocks.

Thinking budgets for Critic/Moderator (8192) are defensible; Synthesis (4096 on a structured-output call) is a candidate for reduction to 2048.

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
