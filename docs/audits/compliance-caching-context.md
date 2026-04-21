# Compliance Audit: Prompt Caching, Context Management, Compaction
**Phase:** 4.15.5
**Date:** 2026-04-18
**Scope:** backend/ Python sources — ClaudeClient, MAS orchestrator, cost tracker, compaction, skill prompts

---

## Sources

- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- https://platform.claude.com/docs/en/build-with-claude/compaction
- https://platform.claude.com/docs/en/build-with-claude/context-editing
- https://platform.claude.com/docs/en/build-with-claude/context-windows
- https://platform.claude.com/docs/en/build-with-claude/token-counting

---

## Patterns Found: 16

---

### P-01 — Cache control wired for ClaudeClient.generate_content (PASS)

`backend/agents/llm_client.py:601-611`

`cache_control: {"type": "ephemeral"}` is placed on the system block when `enable_prompt_caching=True` (the default). The block-level breakpoint is correct per the docs — it marks the end of the static section so the API can cache everything up to that point across repeated calls. Cache hits and misses are tracked in `_cache_hits` / `_cache_misses` counters.

**Status:** Compliant.

---

### P-02 — MF-13: Cache threshold miss — MAS hot path sends plain string system prompt, no cache_control (FAIL)

`backend/agents/multi_agent_orchestrator.py:892-897` (`_call_agent`)
`backend/agents/multi_agent_orchestrator.py:944-953` (`_call_agent_with_tools`)

Both MAS call sites pass `system=agent_config.system_prompt` as a bare string — no `cache_control` block, no `betas` header. The system prompts for MAIN (Ford/Opus 4.6), QA (Analyst/Opus 4.6), and RESEARCH (Researcher/Sonnet 4.6) agents are defined as f-strings in `agent_definitions.py` and are fully static across turns within a session. They are never cached.

The Anthropic docs require the min-token threshold to be met AND `cache_control` to be present. Since neither condition is met in the MAS path, every MAS turn re-bills the full system prompt as regular input tokens.

**Estimated token loss per turn:**
- Ford system prompt: ~800 tokens (estimated from agent_definitions.py content)
- QA system prompt: ~700 tokens
- Research system prompt: ~700 tokens
- At $3/MTok for Sonnet 4.6/Opus 4.6, cache reads would cost $0.30/MTok (90% saving)

**Status:** Non-compliant. Highest-priority fix candidate because the MAS agents are called on every Slack/UI request.

---

### P-03 — Min-token threshold check for Sonnet 4.6 (PASS for ClaudeClient, GAP for MAS)

Per the docs (April 2026):
- claude-sonnet-4-6: **2,048 tokens** minimum
- claude-opus-4-7: **4,096 tokens** minimum
- claude-haiku-4-5: **4,096 tokens** minimum

`ClaudeClient.generate_content` is used for one-shot Claude calls (not the MAS path). The system prompt it builds is "You are a financial analysis AI." plus an optional JSON schema block. Without the schema, that is roughly 10 tokens — well below the 2,048 threshold for Sonnet 4.6 and 4,096 for Opus 4.7.

**This means `cache_control` is being placed on a system block that will never actually be cached.** The Anthropic API returns no error; both `cache_creation_input_tokens` and `cache_read_input_tokens` will be 0, but the code's miss counter increments and masks the problem.

**Mitigations:** Either (a) extend the system prompt to exceed the threshold (pad with domain context, trading rules, or system instructions), or (b) move caching to a route where the static block is genuinely large (e.g., skills prompts passed as system messages in a future refactor).

**Status:** Non-compliant for the ClaudeClient path. The `enable_prompt_caching=True` default is misleading — caching never actually activates.

---

### P-04 — Skill prompts are NOT sent as cached system messages in Layer-1 Gemini pipeline (INFORMATIONAL)

`backend/config/prompts.py`, `backend/agents/skills/*.md`

All 28 Layer-1 agents use `LLMClient` → `GeminiClient`. Skill prompts are injected into the user turn via `format_skill()`, not into a system block. Gemini does not support Anthropic-style `cache_control`. This is correct behavior for the Gemini path.

Estimated sizes of largest prompts (chars / 4 = rough tokens):
- synthesis_agent.md: ~3,007 tokens
- quant_strategy.md: ~4,569 tokens (optimizer, not a pipeline agent)
- moderator_agent.md: ~1,820 tokens
- critic_agent.md: ~1,830 tokens

If the Layer-1 pipeline ever migrates to Claude, synthesis_agent.md and critic_agent.md would need to route through a cached system block to break the 2,048 Sonnet threshold.

**Status:** Informational only.

---

### P-05 — Cost tracker correctly models cache pricing (PASS)

`backend/agents/cost_tracker.py:130-138`

Cache read discount is applied at 0.1x input price (90% saving) when `cache_read_input_tokens > 0`. The `prompt_caching` summary block is computed correctly. However, the tracker does not model the 1.25x write surcharge for 5-minute cache creation or the 2x write surcharge for 1-hour TTL — it accounts only for the read discount. This means cost projections will be slightly optimistic when caches are first being written.

**Status:** Partially compliant. Write surcharge is unmodeled.

---

### P-06 — 5-min vs 1-hour TTL: only 5-min TTL is used (INFORMATIONAL)

`grep: no "ttl.*1h" or "extended-cache-ttl" hits in backend/`

The 1-hour TTL (`"cache_control": {"type": "ephemeral", "ttl": "1h"}`) and the `anthropic-beta: extended-cache-ttl-2025-04-11` header are not used anywhere. For the MAS agents, where the same system prompt is reused across a session but sessions may be spaced hours apart (Slack queries arrive sporadically), the 1-hour TTL would be significantly more effective than 5 minutes.

The `extended-cache-ttl-2025-04-11` beta was released 2025-04-11 and is now production-stable. It requires adding the beta header to each request alongside the extended TTL shape.

**Status:** Gap. 1-hour TTL not evaluated or deployed.

---

### P-07 — Automatic cache-control (2026-02-19 release) not used (GAP)

Anthropic released top-level automatic `cache_control` on 2026-02-19. This moves the breakpoint forward automatically as a conversation grows, avoiding manual block-level marker placement entirely. The API signature is:

```python
client.messages.create(
    ...,
    cache_control={"type": "ephemeral"},   # top-level
)
```

No code in `backend/` uses this form. The MAS tool loop (`_call_agent_with_tools`) accumulates a growing `messages` list across turns; automatic cache-control would be a one-line addition that caches the growing conversation prefix on each turn.

**Status:** Gap.

---

### P-08 — Compaction: `compact_20260112` API beta not wired (FAIL)

`grep: no hits for "compact_20260112", "compact-2026-01-12", "context_management", "betas"` in backend/

The Anthropic server-side compaction feature (`betas=["compact-2026-01-12"]`, `context_management={"edits": [{"type": "compact_20260112"}]}`) is not wired anywhere in the codebase.

The MAS `_call_agent_with_tools` loop can run up to `MAX_TOOL_TURNS` turns with a growing message list. Each turn appends assistant content (including thinking blocks) plus tool results. There is no guard against the context window filling up. Without compaction, long tool loops will eventually hit the model's context limit and raise an exception.

Supported models include claude-sonnet-4-6 and claude-opus-4-6 — exactly the MAS agents in use.

**Status:** Non-compliant. Server-side compaction is the primary recommended strategy for long agentic loops (per the context-editing docs).

---

### P-09 — Local compaction helpers in `compaction.py` are a different thing (INFORMATIONAL)

`backend/agents/compaction.py`

The existing `compaction.py` module implements client-side string truncation helpers (`compact_text`, `compact_trace_summary`, `compact_quant_snapshot`, etc.) for the Layer-1 Gemini pipeline. These deterministically truncate payloads passed between steps. This is not the same as the Anthropic `compact_20260112` API feature, which generates a semantic summary via the model and drops earlier message blocks server-side.

Both mechanisms are valid and complementary. The local helpers address Gemini's context limits in the 15-step pipeline; the server-side API addresses the MAS Claude agents' growing conversation history.

**Status:** Informational — no conflict, but naming overlap (`compaction.py` vs `compact_20260112`) could cause confusion.

---

### P-10 — `tool_result_clear` context-editing not used (GAP)

`grep: no hits for "tool_result_clear", "thinking_block_clear", "context-editing"` in backend/

The Anthropic context-editing API (`betas=["context-editing-2025-10-22"]`) provides two targeted strategies:
- `tool_result_clear`: drops old tool results from the conversation history while preserving the tool call and a cleared marker
- `thinking_block_clear`: drops old thinking blocks (keeping only recent ones for context continuity)

The MAS `_call_agent_with_tools` accumulates both tool results and thinking blocks across turns. On a 10-turn research loop, thinking blocks alone could easily add 20,000 tokens (10 turns × 2,048 budget). Clearing thinking blocks from all but the last 1-2 turns would substantially reduce per-turn input cost without losing reasoning continuity.

The docs note that for most use cases server-side compaction (P-08) is preferred, but `thinking_block_clear` is specifically recommended for extended-thinking loops.

**Status:** Gap.

---

### P-11 — Token counting API (`count_tokens`) not used for pre-flight checks (GAP)

`grep: no hits for "count_tokens", "tiktoken"` in backend/

No code calls `client.messages.count_tokens()` before dispatching to the MAS or `ClaudeClient` paths. There is no guard that checks whether the system prompt + task will exceed the context window, nor any gate that could trigger early compaction. The Anthropic token counting endpoint is free (subject to RPM limits) and adds no billing cost.

Without pre-flight counting, the only signal that a request is too large is an API error, which is unhandled in the current MAS error path beyond a generic `logger.error` + `raise`.

**Status:** Gap.

---

### P-12 — Extended thinking invalidates message cache on mode change (RISK)

`backend/agents/multi_agent_orchestrator.py:950-953`

`_call_agent_with_tools` always enables extended thinking with `{"type": "enabled", "budget_tokens": 2048}`. Per the prompt-caching docs, changes to extended thinking settings invalidate both the tools cache and the system cache (but not the messages cache).

If the codebase ever introduces a code path where some turns use thinking and others do not (e.g., a simplified fast-path for simple tasks), the cache will be invalidated on the mode switch. Currently the setting is always-on in `_call_agent_with_tools`, so there is no intra-session flip — but `_call_agent` (no thinking) and `_call_agent_with_tools` (thinking enabled) are both used for the same agent configs. Mixing these within one agent's session would blow the cache.

**Status:** Latent risk. No active bug today because the two methods are not interleaved per agent within a session, but warrants a code comment.

---

### P-13 — Cache hit/miss metric uses boolean turn-level tracking, not token-weighted accuracy (MINOR)

`backend/agents/llm_client.py:650-662`

`_cache_hits` and `_cache_misses` count turns where `cache_read_input_tokens > 0` / == 0. A turn with 50 cache-read tokens counts the same as one with 100,000. The `estimated_savings_pct` formula in `cost_tracker.py:223` is token-weighted and more accurate, but the raw hit rate exposed via `cache_stats` is turn-level. This does not affect billing but may mislead dashboard interpretation.

**Status:** Minor. Low priority.

---

### P-14 — No `betas` header for extended-cache-ttl or context-editing in ClaudeClient (GAP)

`backend/agents/llm_client.py:599-630`

`ClaudeClient.generate_content` constructs the `client.messages.create()` call with no `betas` kwarg. Adding `extended-cache-ttl-2025-04-11` or `context-editing-2025-10-22` requires passing `betas=[...]` to the SDK call. The current code has no mechanism to thread beta headers through. A `betas: list[str] | None = None` parameter would need to be added to `generate_content` and propagated from callers.

**Status:** Gap — structural prerequisite for P-06 and P-10.

---

### P-15 — Cache creation cost tracking missing from `cost_tracker.record()` (MINOR)

`backend/agents/cost_tracker.py:129-138`

`cost_tracker.record()` applies the 90% read discount correctly but does not apply the 1.25x write surcharge for 5-min cache creation tokens or the 2x surcharge for 1-hour TTL. Cache creation tokens are tracked in `cache_creation_input_tokens` but priced at the base rate (effectively treating them as regular input). At scale, the first call in each session overpays relative to the model.

**Status:** Minor accounting error. Real-world impact is small (write surcharge is bounded per session; reads amortize it over N reuses).

---

### P-16 — No thinking_block_clear in multi-turn loop; thinking tokens accumulate unbounded (RISK)

`backend/agents/multi_agent_orchestrator.py:938-953`

Each turn appends the full assistant content block (including `thinking` blocks up to 2,048 tokens each) to the `messages` list. On a MAX_TOOL_TURNS run, thinking blocks alone could consume up to `MAX_TOOL_TURNS × 2,048` tokens of input on the next turn. The docs note that thinking blocks from previous assistant turns technically count as 0 input tokens via the token counting API, but they are still included in the transmitted payload and consume bandwidth/rate-limit headroom. More importantly, if compaction is ever enabled (P-08), thinking blocks from before the compaction point will be automatically dropped — which is the correct behavior.

**Status:** Latent risk. Tracked here for completeness; resolves automatically if P-08 is implemented.

---

## Summary Table

| ID | Area | Status | Priority |
|----|------|--------|----------|
| P-01 | ClaudeClient cache_control wiring | PASS | — |
| P-02 | MAS hot path: no cache_control (MF-13) | FAIL | HIGH |
| P-03 | Min-token threshold: ClaudeClient system too short | FAIL | HIGH |
| P-04 | Layer-1 Gemini skills: no Claude caching needed | INFO | — |
| P-05 | Cost tracker: read discount correct, write surcharge missing | PARTIAL | LOW |
| P-06 | 1-hour TTL not evaluated | GAP | MEDIUM |
| P-07 | Automatic cache-control (2026-02-19) not used | GAP | MEDIUM |
| P-08 | compact_20260112 server-side compaction not wired | FAIL | HIGH |
| P-09 | Local compaction.py vs API compaction naming overlap | INFO | — |
| P-10 | tool_result_clear / thinking_block_clear not used | GAP | MEDIUM |
| P-11 | count_tokens pre-flight checks absent | GAP | MEDIUM |
| P-12 | Thinking-mode flip invalidates message cache | RISK | LOW |
| P-13 | Hit/miss counter is turn-level not token-weighted | MINOR | LOW |
| P-14 | No betas parameter in ClaudeClient.generate_content | GAP | MEDIUM |
| P-15 | Cache write surcharge unmodeled in cost_tracker | MINOR | LOW |
| P-16 | Thinking blocks accumulate unbounded in tool loop | RISK | LOW |

**Critical (FAIL) count: 3** — P-02, P-03, P-08
**High-priority gaps: 2** — P-06 (1h TTL), P-11 (count_tokens)
**Structural prerequisite for multiple gaps: 1** — P-14 (betas parameter)
