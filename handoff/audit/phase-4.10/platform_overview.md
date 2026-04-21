# Platform Overview Audit -- Claude Doc Alignment (phase-4.10.5)

Scope: Read-only audit of pyfinAgent against the public Claude Platform docs at
`platform.claude.com/docs/en/home` and its linked sub-pages. No source code
modified. Focus: cost and latency wins we are leaving on the table.

---

## Documentation summary

| Feature | URL | Key facts (verbatim where possible) |
|---|---|---|
| Models overview | `/docs/en/about-claude/models/overview` | Current GA: `claude-opus-4-7` ($5/$25 per MTok, 1M ctx, 128k out), `claude-sonnet-4-6` ($3/$15, 1M ctx, 64k out), `claude-haiku-4-5` ($1/$5, 200k ctx, 64k out). Legacy: `claude-opus-4-6` ($5/$25), `claude-sonnet-4-5-20250929`, `claude-opus-4-1-20250805` ($15/$75). Sonnet 4 / Opus 4 deprecated, retire 2026-06-15. Haiku 3 retires 2026-04-19. |
| Prompt caching | `/docs/en/build-with-claude/prompt-caching` | `cache_control: {type: "ephemeral"}` (5-min default) or `{type: "ephemeral", ttl: "1h"}` (2x write cost). Writes 1.25x base, reads 0.1x base (90% off). Min cacheable tokens: Opus 4.7/4.6/4.5 = **4096**, Sonnet 4.6 = 2048, Sonnet 4.5/4/3.7 = 1024, Opus 4.1/4 = 1024, Haiku 4.5 = 4096. Up to 4 breakpoints; 20-block lookback. Cached tokens do **not** count towards ITPM on modern models. |
| Batches API | `/docs/en/build-with-claude/batch-processing` | `POST /v1/messages/batches`, up to 100,000 requests per batch, 256 MB payload, **50% discount**, most finish in <1h. Per-tier queue caps: T1 100k, T2 200k, T3 300k, T4 500k. Not ZDR-eligible. |
| Files API | `/docs/en/build-with-claude/files` | Beta header `anthropic-beta: files-api-2025-04-14`. 500 MB/file, 500 GB/org. PDFs, images, plain text; referenced as `{source: {type: "file", file_id: ...}}`. Free to upload/list/delete. Not on Bedrock / Vertex. |
| Vision | `/docs/en/build-with-claude/vision` | base64/URL/file_id; up to 100 images per 200k-ctx request, 600 per request otherwise; max 8000x8000 px. Standard 32 MB request cap. |
| PDF support | `/docs/en/build-with-claude/pdf-support` | 32 MB request / 600 pages (100 for 200k-ctx models). URL, base64, or file_id. Each page is rasterised AND OCR'd. Combines well with prompt caching + batches. |
| Citations | `/docs/en/build-with-claude/citations` | `citations: {enabled: true}` on document blocks. `cited_text` is free (does not count as output tokens). Incompatible with Structured Outputs. Works with caching + batches. |
| Service tiers / Priority Tier | `/docs/en/api/service-tiers` | `service_tier: "auto"` or `"standard_only"`. Priority reduces 529 overloaded errors, targets 99.5% uptime. Requires sales contact (commit to ITPM/OTPM, 1/3/6/12 months). Response `usage.service_tier` echoes assignment. |
| Rate limits | `/docs/en/api/rate-limits` | Token-bucket. Tier-based: T1 50 RPM / 30k ITPM (Opus, Sonnet), T2 1,000 RPM / 450k ITPM, T3 2,000 / 800k, T4 4,000 / 2M. Response headers `anthropic-ratelimit-{requests,input-tokens,output-tokens}-{limit,remaining,reset}`, plus `retry-after`. Cached reads do NOT count toward ITPM on modern models. |
| Errors | `/docs/en/api/errors` | 400 `invalid_request_error`, 401, 402 `billing_error`, 403, 404, 413 `request_too_large`, 429 `rate_limit_error`, 500 `api_error`, 504 `timeout_error`, 529 `overloaded_error`. Every response carries `request-id` header. 10-min non-streaming timeout. |
| Streaming | `/docs/en/build-with-claude/streaming` | SSE with `stream: true`. Events: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`, `ping`, `error`. SDK `.stream()` + `.get_final_message()` recommended for long requests. |
| Tool use / structured output | `/docs/en/build-with-claude/tool-use` | `tools=[...]` with `input_schema`, `tool_choice` ∈ {auto, any, tool, none}. `strict: true` for schema conformance. Separate Structured Outputs (`output_config.format`) also available (incompatible with Citations). |
| Admin API | `/docs/en/administration/administration-api` | `x-api-key: sk-ant-admin...`. Endpoints: `/v1/organizations/{users,invites,workspaces,api_keys,me}`, Usage & Cost API, Claude Code Analytics API. Roles: user, claude_code_user, developer, billing, admin. |
| Extended / Adaptive thinking | (linked from models page) | Opus 4.7 = adaptive only (no manual budget), Sonnet 4.6 / Haiku 4.5 = both. `thinking: {type: "enabled", budget_tokens: N}` forces temperature=1. |

---

## Codebase audit

### Feature x pyfinAgent matrix

| Platform feature | In use? | Where | Notes |
|------------------|---------|-------|-------|
| Prompt caching | Partial | `backend/agents/llm_client.py:601-608` (`ClaudeClient`); metrics at `cost_tracker.py:125-151`, `:201-224` | Enabled in the factory `ClaudeClient` path (system prompt wrapped in ephemeral block). **NOT** applied in `multi_agent_orchestrator.py:944-954`, which is the 6-agent MAS hot path and builds raw `client.messages.create` calls without `cache_control`. No caching on `tools=AGENT_TOOLS` array (stable, cacheable). No 1h TTL used anywhere. No cache breakpoints on skill prompts (28 skills in `backend/agents/skills/*.md` -- prime candidates). |
| Batches API | No | -- | Zero references to `messages.batches` in repo. Overnight research, 28-agent analysis pipeline, gauntlet regimes in `backend/backtest/gauntlet/regimes.py`, and backtest re-runs are all sync. Leaves the 50% discount on the table. |
| Files API | No | -- | No `beta.files`, no `files-api-2025-04-14` header anywhere. SEC/EDGAR ingestion (`backend/tools/sec_insider.py`, `earnings_tone.py`) predates the Files API and ingests text directly. |
| Vision / PDF input | No | -- | `backend/backtest/gauntlet/regimes.py:108,191` only references PDFs as *research citation URLs*, never passes them to Claude. No `type: document`, no `media_type: application/pdf` call sites. 10-K / 10-Q / earnings-deck ingestion goes through custom text extraction, not Claude's native PDF parser. |
| Citations | No | -- | No `citations: {enabled: true}` call sites. Custom evidence/grounding metadata is accumulated by Gemini grounding only (`llm_client.py:397-411`). No cost-free `cited_text` on Claude-side. |
| Priority tier | No | -- | No `service_tier` parameter in `multi_agent_orchestrator.py` or `ClaudeClient`. For a long-running autonomous trader this is the biggest availability gap: 529s during market hours will stall the MAS. |
| Admin API | No | -- | No `sk-ant-admin` keys, no programmatic workspace/usage polling. Budget dashboard reads from local `cost_tracker.summarize()` only. |
| Rate-limit handling | Partial | `debate.py:50-82`, `risk_debate.py:47`, `services/ticket_queue_processor.py:348-354`, `tools/alt_data.py:88-90` | Generic exponential backoff loop (5s -> 10s -> 20s, 3 retries), triggered on string match `"ratelimit"`/`"overload"`/`"unavailable"` in exception class name. **Does NOT read `retry-after` header**, does NOT inspect `anthropic-ratelimit-*-remaining` for proactive throttling. 2.12 `ticket_queue_processor.py` has retry-count-based backoff but same gap. |
| Streaming | No (for Claude) | `backend/agents/orchestrator.py:456,473`, `openclaw_client.py:183` | HTTP streaming used for *internal FastAPI endpoints* (ingestion/quant agent), not for Anthropic SSE. `ClaudeClient.generate_content` uses non-streaming `client.messages.create`. Risk: on the 10-minute non-streaming timeout and when MAS subagents with `max_tokens + 2048` thinking budget run long. |
| Tool use / structured output | Yes | `multi_agent_orchestrator.py:76-114,948,962-993` | Proper `input_schema`, `stop_reason == "tool_use"` loop, parallel tool execution via `ThreadPoolExecutor`. Interleaved thinking with `budget_tokens=2048`. Clean implementation. Not using `strict: true` though. |
| Error handling / backoff | Partial | see Rate-limit row | Transient classification by substring (`"ratelimit"`/`"overload"`/`"unavailable"` in type name). No 529 overloaded vs 429 rate-limit differentiation. No circuit breaker. No `request_id` logging on failures (harder to file support tickets). |
| Extended thinking | Yes | `llm_client.py:621-628`, `multi_agent_orchestrator.py:950-953` | Correct shape `{type: "enabled", budget_tokens: N}` and forced `temperature=1`. Tiered budgets (Critic 8192 / Synthesis 4096) per `backend-agents.md`. |

---

## Findings

| Aspect | Status | Evidence | Notes |
|--------|--------|----------|-------|
| `ClaudeClient` system-prompt caching | Correct | `llm_client.py:601-611` | Ephemeral block shape matches docs. BUT system prompt is a short "You are a financial analysis AI." + optional schema hint -- likely **below 4096-token minimum for Opus 4.7**, so cache almost never triggers. Observed cache metrics will stay at 0/0 until the cached content crosses threshold. |
| MAS orchestrator bypasses caching | Missing | `multi_agent_orchestrator.py:944-954` | Calls `client.messages.create` directly with raw `system=agent_config.system_prompt` (no wrapping). Agent system prompts in `agent_definitions.py` + `AGENT_TOOLS` are stable, ideal cache candidates for the hot path. |
| Cost-tracker caching math | Correct | `cost_tracker.py:131-138` | Applies 10% of base input for `cache_read_input_tokens`. Matches docs. |
| Model catalog out of date | Incorrect | `cost_tracker.py:17-28`, `llm_client.py:63-67` | `GITHUB_MODELS_CATALOG` still advertises `claude-3-5-sonnet-20241022` and `claude-3-7-sonnet-20250219` as first-class. Docs list Sonnet 3.7 as deprecated; GA is now `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`. `MODEL_PRICING` has no entries for opus-4-6/4-7 or haiku-4-5. Costs for Claude calls going through `make_client` -> ClaudeClient fall back to `_DEFAULT_PRICING (0.10, 0.40)` -- under-reporting. |
| No `retry-after` header respect | Missing | `debate.py:67-69`, `risk_debate.py:51` | Hard-coded exponential. On 429, we should sleep for exactly `retry-after` seconds and log `request_id`. Current approach risks further throttling under sustained load. |
| No `service_tier` parameter | Recommended | `multi_agent_orchestrator.py:944` | For the live/paper-trading MAS add `service_tier="auto"` once on Priority; currently all calls run in Standard. |
| No Batches for gauntlet / research | Recommended | `backend/backtest/gauntlet/regimes.py`, `scripts/harness/run_harness.py` | Nightly backtest sweeps and multi-regime replays fire a lot of Claude requests where latency is irrelevant. 50% cost cut is available. |
| No Files API for 10-K/10-Q | Recommended | `backend/tools/sec_insider.py`, `earnings_tone.py` | Financial filings are re-ingested per call. Upload once, reuse `file_id` + `citations: {enabled: true}` for free `cited_text` in decisions. |
| No streaming on long MAS calls | Recommended | `llm_client.py:630`, `multi_agent_orchestrator.py:944` | With `thinking.budget_tokens=2048` + tool loop of up to MAX_TOOL_TURNS, calls can approach the 10-min non-streaming limit. Switch to `client.messages.stream(...).get_final_message()`. |
| No Admin API integration | Recommended | -- | Budget dashboard relies on in-process `cost_tracker`. Admin API's Usage & Cost endpoint would give server-side truth + per-workspace attribution (separate workspace for MAS vs harness vs paper-trader). |
| `strict: true` on tools | Recommended | `multi_agent_orchestrator.py:76-114` | Add `strict: true` to the 7 agent tool definitions to guarantee schema conformance (docs recommend it). |
| Request-id logging | Recommended | error paths in `llm_client.py`, `debate.py`, `multi_agent_orchestrator.py:955-957` | Log `response._request_id` on failure/ratelimit to speed support escalations. |

---

## Gaps & Opportunities

### MUST FIX (correctness)
- **Claude model pricing table is stale.** `cost_tracker.MODEL_PRICING` has no `claude-opus-4-7`, `claude-opus-4-6`, `claude-haiku-4-5` entries. Any Claude call made through the MAS hot path falls back to `_DEFAULT_PRICING = (0.10, 0.40)` -- a 50x underestimate for Opus 4.7 input and 187x for output. Every autonomous budget check is wrong.
- **Deprecated models still advertised.** `GITHUB_MODELS_CATALOG` lists `claude-3-5-sonnet-20241022`, `claude-3-7-sonnet-20250219` as options without a deprecation note; Anthropic retires Sonnet 4 / Opus 4 on 2026-06-15 and Haiku 3 on 2026-04-19. Need a migration path to Sonnet 4.6 / Opus 4.7.
- **Prompt caching min-token threshold probably not met.** System prompt in `ClaudeClient.generate_content` is ~20 chars of boilerplate. Under the 4096 min for Opus 4.7 / Haiku 4.5, so cache entries never get created. `enable_prompt_caching=True` is silently a no-op on those models. Either move skill prompts into the cached system block or switch default to Sonnet 4.5 where the min is 1024.
- **Rate-limit loop ignores `retry-after`.** Exponential 5/10/20s backoff is blind to the server's actual window. On a real 429 we may retry too early three times and get banned.

### NICE TO HAVE (cost / latency / observability wins)
- **Prompt caching on MAS tool definitions and skill prompts.** `AGENT_TOOLS` (7 definitions, ~300 tokens of tool schema JSON) and each agent's system prompt are static across turns of the tool loop. Wrap the last tool definition and the `system=...` string in `cache_control: {type: "ephemeral"}` per the docs' best-practice "cache at prompt beginning" rule. Expected: ~90% reduction on input costs for turns 2..N of every MAS query, 85% latency reduction on cache hits.
- **Batches API for overnight work.** `scripts/harness/run_harness.py` cycles and gauntlet regime re-runs are async by nature. Submitting the combined request set to `/v1/messages/batches` halves the Anthropic bill for that workload. Max 100k requests/batch, 256 MB payload -- well under our per-cycle load.
- **Files API + Citations for SEC filings.** Upload each 10-K/10-Q once (`anthropic-beta: files-api-2025-04-14`), reuse `file_id` across every agent that references it, enable citations so `cited_text` is free on output. Combine with caching on the `document` block for repeat reads. Huge win on `deep_dive_agent.md` and `insider_agent.md` workloads.
- **PDF ingestion via native Claude document blocks.** Replace custom text extraction in `earnings_tone.py` / `sec_insider.py` with `{"type": "document", "source": {...}}` so Claude sees charts, tables, and page layout -- materially better on earnings decks.
- **Priority Tier for the live paper-trader / slack-bot SLA loop.** `service_tier="auto"` with a small Opus 4.7 commit protects us from 529s during market hours. Currently any Anthropic capacity pinch stalls the ticket queue.
- **Streaming for long MAS turns.** `client.messages.stream(...).get_final_message()` removes the 10-min timeout risk on subagent calls with thinking + multi-tool loops.
- **Admin API + Usage & Cost API for budget dashboard.** Replace the in-process `cost_tracker` as the source of truth with server-side `/v1/organizations/usage` data. Split workloads into separate workspaces (harness vs MAS vs paper-trader) for attribution.
- **`strict: true` on tool schemas.** Cheap insurance against malformed tool calls (documented best practice).
- **Log `response._request_id` on every Claude failure.** Free, makes support tickets actionable.

---

## References

1. Claude Platform home -- https://platform.claude.com/docs/en/home
2. Models overview -- https://platform.claude.com/docs/en/about-claude/models/overview
3. Prompt caching -- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
4. Batch processing -- https://platform.claude.com/docs/en/build-with-claude/batch-processing
5. Files API -- https://platform.claude.com/docs/en/build-with-claude/files
6. Vision -- https://platform.claude.com/docs/en/build-with-claude/vision
7. PDF support -- https://platform.claude.com/docs/en/build-with-claude/pdf-support
8. Citations -- https://platform.claude.com/docs/en/build-with-claude/citations
9. Service tiers / Priority Tier -- https://platform.claude.com/docs/en/api/service-tiers
10. Rate limits -- https://platform.claude.com/docs/en/api/rate-limits
11. Errors -- https://platform.claude.com/docs/en/api/errors
12. Streaming -- https://platform.claude.com/docs/en/build-with-claude/streaming
13. Tool use -- https://platform.claude.com/docs/en/build-with-claude/tool-use
14. Admin API overview -- https://platform.claude.com/docs/en/administration/administration-api

Code references (pyfinAgent):
- `backend/agents/llm_client.py` (ClaudeClient + factory)
- `backend/agents/multi_agent_orchestrator.py` (hot path, tool loop)
- `backend/agents/cost_tracker.py` (pricing + cache metrics)
- `backend/agents/debate.py`, `risk_debate.py` (retry loops)
- `backend/services/ticket_queue_processor.py` (backoff)
- `backend/tools/sec_insider.py`, `earnings_tone.py` (SEC ingestion)
- `scripts/harness/run_harness.py` (batch candidate)
