# API reference + SDKs Deep Audit (phase-4.11.6)

Audit of pyfinAgent's direct Anthropic API + SDK usage against the full Claude
API reference and every language-SDK page. Mode: AUDIT ONLY (no code changes).

## URL coverage

Read in full (23/23):

1. overview, 2. messages, 3. messages/create, 4. messages/batches,
5. creating-message-batches, 6. messages-count-tokens, 7. models-list,
8. files-create, 9. rate-limits, 10. service-tiers, 11. beta-headers,
12. versioning, 13. errors, 14. supported-regions, 15. client-sdks,
16-22. sdks/{python,typescript,go,java,csharp,ruby,php}, 23. sdks/cli.

(TypeScript/Go/Java/C#/Ruby/PHP pages skimmed for version currency + feature
parity only; pyfinAgent is Python-only.)

## Messages API usage in pyfinAgent vs doc

Direct `anthropic.Anthropic()` / `client.messages.create()` call sites:

| File | Model | Role |
|------|-------|------|
| `backend/agents/llm_client.py:560` | routed | `ClaudeClient` wrapper (prompt caching + thinking) |
| `backend/agents/multi_agent_orchestrator.py:165` | opus-4-6 / sonnet-4-6 | tool-loop for MAS agents |
| `backend/agents/planner_agent.py:37` | `claude-opus-4-6` | LLM-as-Planner |
| `backend/agents/planner_enhanced.py:19` | — | Planner v2 |
| `backend/services/ticket_queue_processor.py:178` | opus-4-6 / sonnet-4-6 | Slack ticket agent |
| `backend/services/autonomous_loop.py:419` | `claude-sonnet-4-20250514` | paper-trading analysis |

Observations vs `POST /v1/messages` spec:
- All sites correctly set the 3 required params (`model`, `max_tokens`,
  `messages`). `system` is used as top-level (not a message role) per spec.
- `temperature=0.0` passed for deterministic JSON. Compatible with spec
  (range 0.0-1.0).
- No call site uses `stream=True`. Doc warns: "SDK throws ValueError if a
  non-streaming request is expected to exceed ~10 min." pyfinAgent's
  `ticket_queue_processor` wraps each call in a 60s `ThreadPoolExecutor` so
  this is fine in practice, but the SDK's own TCP keep-alive + 10-min default
  timeout are being bypassed.
- `autonomous_loop.py:438` still pins `claude-sonnet-4-20250514` — a
  2025-05 snapshot. Current convention is Sonnet 4.6 (`claude-sonnet-4-6`).
  All other call sites have moved to the 4-6 aliases. This is stale and will
  be deprecated per the model-deprecation policy referenced in errors doc.
- `ticket_queue_processor.py:206` passes exactly the 4 required kwargs —
  no `service_tier`, no `metadata`, no `user_id`. `metadata.user_id` is
  strongly recommended by the doc for abuse-detection attribution.
- No call site uses `output_config` for structured JSON. `llm_client.py`
  injects a JSON schema hint via `system` prompt. This works but now that
  Sonnet/Opus 4.6+ support **structured outputs** via `output_config.format`
  (see errors doc: "Use structured outputs... or `output_config.format`
  instead"), prompt-hint mode is strictly inferior (no retry on malformed,
  no schema enforcement).
- `planner_agent.py:136` uses `response_text.find('{')` / `rfind('}')`
  regex-free JSON extraction — a classic parse-by-luck anti-pattern that
  structured outputs would eliminate.

## Rate-limit + retry-after + request-id compliance

Reference: `rate-limits.md` response-headers table (18 headers), `errors.md`
`request_id` section.

**Finding: pyfinAgent reads ZERO rate-limit headers and does not respect
`retry-after`.** Grep for `anthropic-ratelimit`, `retry-after`, `retry_after`
across the whole tree returns nothing.

Per-file evidence:
- `agents/debate.py:50-82` — `_generate_with_retry` catches generic
  `Exception`, matches `"ratelimit"|"overload"|"unavailable"` in the error
  class name, then sleeps a hard-coded `delay=5` doubled. The 429 response's
  `retry-after` header (which the API returns *exactly for this purpose*) is
  never read. If the server says "retry in 30s" we retry in 5s and burn the
  next retry.
- `agents/risk_debate.py:47-76` — identical pattern.
- `agents/orchestrator.py:387-440` — identical pattern (delay=5, ×2).
- `services/ticket_queue_processor.py:348-354` — `wait_time = min(60 * (2**
  (retry_count-1)), 240)`. String-matches "429" in error message (`:244`)
  but again ignores `retry-after`.
- `tools/alt_data.py:90` — pytrends 429 handler, not Anthropic. OK.

**`request_id` logging is also missing.** The Python SDK exposes
`message._request_id` on every response; errors include `request_id` in the
error body. pyfinAgent never logs it. This makes Anthropic support tickets
much harder. Flagged in phase-4.10 and still unresolved.

**Structured logging of the 18 `anthropic-ratelimit-*` response headers is
flat-out absent.** To access these we'd need
`client.messages.with_raw_response.create(...)` which is documented in
`sdks/python.md` but unused in the repo. Without this we have no visibility
into remaining tokens / remaining requests, so we can't do proactive back-off
before hitting 429.

## Beta-header + versioning compliance

Reference: `versioning.md` (only `2023-06-01` is current) and `beta-headers.md`
(23 enumerated beta headers, including `interleaved-thinking-2025-05-14`,
`output-300k-2026-03-24`, `context-1m-2025-08-07`, `extended-cache-ttl-
2025-04-11`, `skills-2025-10-02`, `fast-mode-2026-02-01`, `context-management-
2025-06-27`, `prompt-caching-2024-07-31`, `files-api-2025-04-14`, `token-
efficient-tools-2025-02-19`, `mcp-client-2025-11-20`).

Grep for `anthropic-beta`, `betas=`, `anthropic-version` in backend returns
**zero hits**. pyfinAgent never sets any beta header. Consequences:

- **Prompt caching**: `llm_client.ClaudeClient` passes `cache_control:
  {type: ephemeral}` on system blocks. Per doc this is the "prompt-caching-
  2024-07-31" feature. It's been GA since Nov 2024 so the beta header is no
  longer required, but we have no `extended-cache-ttl-2025-04-11` header, so
  we're locked into the default 5-minute TTL instead of opting into 1h TTL.
  Given the harness runs for >5 min the 1h TTL could materially lift our
  cache-hit rate (currently a `ClaudeClient._cache_hits` metric is kept but
  no header is sent to extend it).
- **Interleaved thinking**: we support `thinking: {type: enabled, budget_
  tokens: N}` in `llm_client.py:622-627`, but interleaved thinking across
  tool calls requires `anthropic-beta: interleaved-thinking-2025-05-14`.
  Not set, so MAS tool loops do not get interleaved reasoning.
- **Files API**: we never call it; no action needed now, but when
  uploading large analyst PDFs we'd need `files-api-2025-04-14`.
- **Output token budget**: for long synthesis outputs, `output-300k-
  2026-03-24` enables up to 300k output tokens on Opus 4.x. Synthesis max
  is 4096 today so we're far from the cap, no impact.
- **`anthropic-version`**: SDK auto-sends `2023-06-01`. pyfinAgent doesn't
  override — OK, but there is no outer pin either, so if Anthropic ships a
  breaking version bump we wouldn't notice.

## Python SDK version currency + unused helpers

Pinned: `anthropic==0.87.0` (`backend/requirements.txt:37`). Latest on PyPI:
**0.96.0** (9 point releases behind — 0.88 through 0.96). Pin is explicit
per supply-chain-hardening comment citing CVE-2026-34450/34452 (0.87 is the
fixed version; the CVE does not prevent moving forward).

Unused SDK features documented in `sdks/python.md` that would materially
help pyfinAgent:

1. **`max_retries`** — the SDK has built-in exponential backoff and retries
   408/409/429/>=500 twice by default. We have 3 separate hand-rolled retry
   loops (`debate`, `risk_debate`, `orchestrator`) plus one in
   `ticket_queue_processor`. None of them read `retry-after`. The SDK's
   retry respects the header. Migrating to `Anthropic(max_retries=3)` +
   `client.with_options(max_retries=N)` would delete ~120 lines and fix the
   retry-after bug simultaneously.
2. **`with_raw_response` / `with_streaming_response`** — gives direct access
   to `response.headers` for rate-limit observability. Zero uses.
3. **`client.messages.count_tokens()`** — would replace the hand-written
   `get_model_max_input_chars()` char-count safety-net in `llm_client.py:
   100-146`. We're approximating "1 token ~ 3.5-4 chars" when the real
   counter is one API call away.
4. **`client.messages.stream()` / `get_final_message()`** — we never use
   streaming, but for the synthesis step (4096 tokens, 15-60s) streaming
   would eliminate the TCP-drop risk called out in the errors doc under
   "Long requests".
5. **`client.messages.batches`** — Message Batches API is 50% cheaper. Our
   nightly `autonomous_loop._run_claude_analysis()` runs Claude on ~100
   tickers; this is the archetypal batch workload and is currently run
   synchronously one ticker at a time.
6. **`AsyncAnthropic`** — we wrap sync calls in `asyncio.to_thread` and
   `ThreadPoolExecutor(max_workers=1)`. The async client is native to the
   SDK and would clean up both.
7. **`client.beta.messages.tool_runner()` with `@beta_tool`** — the MAS tool
   loop in `multi_agent_orchestrator.py` hand-rolls the tool dispatch; the
   SDK now provides a tool-runner helper.
8. **Error class hierarchy** — grepping for `APIError|APIStatusError|
   APIConnectionError|APITimeoutError|BadRequestError|AuthenticationError`
   returns **zero matches** in backend. All retry loops catch bare
   `Exception` and string-match the class name (`"ratelimit" in
   err_name.lower()`). This is fragile and skips `APIConnectionError` (no
   "rate" substring). Correct pattern (per `sdks/python.md`):
   ```python
   except anthropic.RateLimitError as e:
       retry_after = int(e.response.headers.get("retry-after", 5))
   except anthropic.APIConnectionError: ...
   except anthropic.APIStatusError as e: ...
   ```

## Other-language SDK notes (informational — we're Python-only)

| SDK | Latest | Notable capability we don't have |
|-----|--------|----------------------------------|
| TypeScript | `@anthropic-ai/sdk` (Node 20+) | identical feature parity; used if we ever move Slack bot to Node |
| Go | `anthropic-sdk-go` (Go 1.23+) | Context-based cancellation, functional options |
| Java | `com.anthropic:anthropic-java:2.20.0` | Builder pattern, `CompletableFuture` |
| C# | `Anthropic` NuGet, .NET Standard 2.0+ | `IChatClient` integration |
| Ruby | 3.2+, Sorbet types | — |
| PHP | 8.1+, Composer | — |
| CLI (`ant`) | `brew install anthropics/tap/ant` | Shell scripting, typed flags, response transforms — could replace the harness's Python-wrapped curl-equivalents |

All SDKs share the same retry/streaming/beta-namespace design. If we ever
split autonomous_loop into a Go service (for latency), the SDK parity is
there.

## Findings

F1. **No `retry-after` honoring across 4 retry loops.** Hard-coded 5s×2.
    Flagged in phase-4.10 and still unresolved. Impacts throughput under
    burst load.
F2. **No rate-limit header observability.** `anthropic-ratelimit-*` never
    read. No ability to back-off proactively.
F3. **No `request_id` logging.** Errors land in Sentry without an Anthropic
    correlation ID — support triage is harder than it needs to be.
F4. **Hand-rolled retry instead of SDK built-in.** 4 retry loops duplicate
    SDK functionality and get it wrong (no retry-after).
F5. **Exception catching via `isinstance(...) == "ratelimit"` string match.**
    `anthropic.RateLimitError` / `APIStatusError` / `APIConnectionError`
    classes never imported. Fragile and miss some error classes.
F6. **SDK pin 9 versions behind** (0.87.0 vs 0.96.0). No review since
    phase-3.7.6. The CVE cited in the pin comment is resolved in >=0.87, so
    nothing blocks moving to 0.96.
F7. **Stale model ID in `autonomous_loop.py:438`** — `claude-sonnet-4-20250514`
    when the rest of the codebase uses `claude-sonnet-4-6`.
8. **Never set `anthropic-beta`.** Missing out on:
    - `extended-cache-ttl-2025-04-11` (1h cache; could ~2x hit rate on
      >5-min harness cycles)
    - `interleaved-thinking-2025-05-14` (better MAS tool-loop reasoning)
    - optional `context-1m-2025-08-07` for Sonnet 4.6 if we ever need >200k.
F9. **Never set `service_tier`.** Default is "auto" per spec, which is fine,
    but we can't force `standard_only` for batch-like non-critical paths.
F10. **No Batch API usage.** The nightly paper-trading Claude sweep is the
    textbook batch workload; we pay 2x cost today.
F11. **No structured outputs (`output_config.format`).** Planner, paper-
    trading analysis, and MAS responders all regex-extract JSON from prose.
    Opus 4.6+ and Sonnet 4.6+ support structured outputs natively.
F12. **No `count_tokens`.** `get_model_max_input_chars()` estimates chars
    instead of calling the free token-counting endpoint.
F13. **No streaming.** Synthesis step is a 4096-token single-shot blocking
    call; the errors doc explicitly recommends streaming for long calls.

## MUST FIX / NICE TO HAVE

### MUST FIX (before phase-4.12 go-live)
- **M1 (F1, F2, F4, F5)**: Refactor the 4 retry loops to either (a) rely on
  SDK's `max_retries` (which respects `retry-after`), or (b) catch
  `anthropic.RateLimitError` explicitly and read `e.response.headers.get
  ("retry-after")`. Delete the string-matching error classifier.
- **M2 (F3)**: Log `message._request_id` on every success AND failure. Add
  to `cost_tracker.record()` + every `logger.error("Agent ... failed ...")`
  path in `ticket_queue_processor.py:225`, `orchestrator.py:420`,
  `debate.py:68`, `risk_debate.py:65`.
- **M3 (F7)**: Bump `autonomous_loop.py:438` to `claude-sonnet-4-6` (or
  the current CLAUDE.md default) and add a pre-commit grep to prevent
  re-introducing dated snapshot IDs.

### NICE TO HAVE (phase-4.12+)
- **N1 (F6)**: Bump `anthropic==0.87.0` -> `0.96.0`, run backend tests.
- **N2 (F8)**: Add `anthropic-beta: extended-cache-ttl-2025-04-11` to
  `ClaudeClient`; set `cache_control: {type: ephemeral, ttl: "1h"}`.
  Expected: cache-hit rate up from ~30% (5m window) to 60-80%.
- **N3 (F10)**: Move `autonomous_loop._run_claude_analysis` to
  `client.messages.batches.create(...)` for the nightly ticker sweep -> 50%
  cost reduction on that path.
- **N4 (F11)**: Switch `planner_agent.py` + `paper_trader` LLM parsing to
  `output_config.format={...json-schema...}`. Drop regex JSON extraction.
- **N5 (F13)**: Stream synthesis step via `client.messages.stream(...)`.
- **N6 (F12)**: Call `client.messages.count_tokens` in the `_
  MODEL_MAX_INPUT_CHARS` safety net for exact counts on Anthropic models.
- **N7**: Add `metadata={"user_id": <ticket.submitter>}` to Slack-ticket
  calls for abuse-detection attribution.
- **N8**: Use `AsyncAnthropic` in `autonomous_loop.py` instead of
  `asyncio.to_thread`.

## References

1. https://platform.claude.com/docs/en/api/overview
2. https://platform.claude.com/docs/en/api/messages
3. https://platform.claude.com/docs/en/api/messages/create
4. https://platform.claude.com/docs/en/api/messages/batches
5. https://platform.claude.com/docs/en/api/creating-message-batches
6. https://platform.claude.com/docs/en/api/messages-count-tokens
7. https://platform.claude.com/docs/en/api/models-list
8. https://platform.claude.com/docs/en/api/files-create
9. https://platform.claude.com/docs/en/api/rate-limits
10. https://platform.claude.com/docs/en/api/service-tiers
11. https://platform.claude.com/docs/en/api/beta-headers
12. https://platform.claude.com/docs/en/api/versioning
13. https://platform.claude.com/docs/en/api/errors
14. https://platform.claude.com/docs/en/api/supported-regions
15. https://platform.claude.com/docs/en/api/client-sdks
16. https://platform.claude.com/docs/en/api/sdks/python
17. https://platform.claude.com/docs/en/api/sdks/typescript
18. https://platform.claude.com/docs/en/api/sdks/go
19. https://platform.claude.com/docs/en/api/sdks/java
20. https://platform.claude.com/docs/en/api/sdks/csharp
21. https://platform.claude.com/docs/en/api/sdks/ruby
22. https://platform.claude.com/docs/en/api/sdks/php
23. https://platform.claude.com/docs/en/api/sdks/cli
24. pyfinagent `backend/agents/llm_client.py` lines 536-672
25. pyfinagent `backend/agents/multi_agent_orchestrator.py` lines 155-168
26. pyfinagent `backend/agents/planner_agent.py` lines 17, 37, 115-122
27. pyfinagent `backend/services/ticket_queue_processor.py` lines 154-248
28. pyfinagent `backend/services/autonomous_loop.py` lines 384-441
29. pyfinagent `backend/agents/debate.py` lines 50-82
30. pyfinagent `backend/agents/risk_debate.py` lines 47-76
31. pyfinagent `backend/agents/orchestrator.py` lines 387-440
32. pyfinagent `backend/requirements.txt` line 37 (anthropic==0.87.0)
33. https://pypi.org/project/anthropic/ (latest 0.96.0)
