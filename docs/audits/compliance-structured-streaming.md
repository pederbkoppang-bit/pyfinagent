# Compliance Audit: Structured Outputs, Stop Reasons, Streaming, Refusals
Phase 4.15.7 — 2026-04-18

---

## Summary Counts

| Check | Count | Status |
|---|---|---|
| `json.loads` calls (agents/ + services/) | 34 | -- |
| `JSONDecodeError` handlers (same scope) | 10 | -- |
| Unguarded `json.loads` calls (no handler in same try-block) | ~8 | FAIL |
| `output_config` / `structured_output` usage | 0 | FAIL |
| stop_reason values dispatched (of 7 required) | 2 of 7 | FAIL |
| `max_tokens` + `tool_use` tail retry | 0 | FAIL |
| Anthropic SDK streaming (`messages.stream` / `stream=True`) | 0 | NOTE |
| Prefill usage | 0 | PASS |
| Citations + structured output guard | N/A | PASS |

---

## Pattern 1 — FAIL: `output_config.format` not used anywhere (MF-5)

`grep -rn 'output_config\|structured_output' backend/ --include='*.py'` returned no matches.

The GA replacement for prompt-level JSON coercion is `output_config={"format": {"type": "json_schema", "schema": {...}}}` on `client.messages.create()`. The Anthropic docs mark this as the correct path for all claude-opus-4-6, claude-opus-4-7, and claude-sonnet-4-6 calls that need guaranteed JSON. Zero callsites use it.

**Affected files:** `llm_client.py` (ClaudeClient.generate_content), `multi_agent_orchestrator.py` (all `client.messages.create` calls), `planner_agent.py`, `planner_enhanced.py`, `autonomous_loop.py`.

---

## Pattern 2 — FAIL: JSON schema injected via system prompt instead of output_config (MF-5)

`backend/agents/llm_client.py:586-597` — when `response_mime_type == "application/json"` or a schema is present, the code appends schema instructions to the system prompt string:

```python
system_prompt += f"\n\nYou MUST respond with valid JSON matching this exact schema:\n{json.dumps(...)}"
```

This is the deprecated pattern. `output_config.format` guarantees schema conformance at the API level; prompt injection only nudges the model and can be ignored, especially on long contexts or high-temperature calls.

---

## Pattern 3 — FAIL: `find('{')/rfind('}')` JSON extraction is fragile (MF-21)

`planner_agent.py:129-132`, `planner_agent.py:234-236`, `evaluator_agent.py:404-408`, `planner_enhanced.py:119`, `skill_optimizer.py:533`, `compaction.py:146-148`.

All six sites use the same pattern:
```python
json_start = response_text.find('{')
json_end   = response_text.rfind('}') + 1
data = json.loads(response_text[json_start:json_end])
```

This silently corrupts output when the model returns nested JSON with prose before or after (e.g., a markdown code fence containing `{...}` followed by an explanation). The outermost `{}` delimiters are not necessarily the right boundaries. `output_config.format` eliminates this entirely.

---

## Pattern 4 — FAIL: ~8 unguarded `json.loads` calls (MF-21)

34 `json.loads` calls in `backend/agents/` and `backend/services/`. Only 10 `JSONDecodeError` handlers exist in the same scope. The gap is approximately 8 unguarded sites.

Confirmed unguarded (no `JSONDecodeError` or broad `except` covering the call):

| File | Line | Context |
|---|---|---|
| `orchestrator.py` | 478 | Parses streaming SSE line `FINAL_JSON:...` |
| `autonomous_loop.py` | 449 | Parses Claude text response via regex |
| `skill_optimizer.py` | 368 | `json.loads(json_match)` where match is a string |
| `compaction.py` | 148 | Double-decode: `json.loads(json.loads(...))` |

`orchestrator.py:478` is particularly exposed — it parses a line from an httpx streaming response and throws an unhandled exception that would abort the entire analysis pipeline for a ticker.

`autonomous_loop.py:449` uses `re.search(r'\{[^}]+\}', text)` which only matches single-depth JSON objects (no nested keys). If the model response contains nested braces the regex fails silently and `json_match` is `None`, but `json.loads` is called on `.group()` of a confirmed match — the `None` branch is handled. However the single-depth regex is wrong for the schema used.

---

## Pattern 5 — FAIL: stop_reason dispatch covers only 2 of 7 values (MF-26, MF-27)

`grep -rn 'stop_reason' backend/` found exactly two references:

- `multi_agent_orchestrator.py:962` — `if response.stop_reason == "tool_use":`
- `multi_agent_orchestrator.py:1024` — comment: `"stop_reason is 'end_turn' or 'max_tokens'"`

Documented stop_reason values and their required handling:

| Value | Required action | Handled? |
|---|---|---|
| `end_turn` | Read content normally | Yes (implicit else branch) |
| `tool_use` | Execute tool, append result, loop | Yes — line 962 |
| `max_tokens` | Check if last block is `tool_use`, retry with higher limit | No |
| `stop_sequence` | Read `response.stop_sequence`, consume partial | No |
| `pause_turn` | Append assistant content as-is, re-call API | No |
| `refusal` | Surface user-facing message, do not re-try raw | No |
| `model_context_window_exceeded` | Summarize/compress history, retry | No |

The `else` branch at line 1023 handles both `end_turn` and `max_tokens` identically — it extracts text and exits the loop. A truncated response on `max_tokens` is silently returned as complete output to the caller. There is no retry and no signal to the caller that output is incomplete.

---

## Pattern 6 — FAIL: no `max_tokens` + incomplete `tool_use` tail retry (MF-26)

Per the Anthropic stop-reasons doc:
> If the truncated response contains an incomplete tool use block, retry with a higher `max_tokens` value.

The tool loop in `multi_agent_orchestrator.py:944-1030` sets `max_tokens=agent_config.max_tokens + 2048` (thinking budget added). If that budget is still insufficient and `stop_reason == "max_tokens"` with the last block being a `tool_use`, the code falls through to the `else` branch, assembles partial text, and exits. The incomplete tool call is silently dropped.

---

## Pattern 7 — FAIL: `refusal` stop_reason not handled (MF-30)

No callsite checks `response.stop_reason == "refusal"`. The Anthropic doc states:

> Refusals bypass schema — safety refusals may return non-schema-compliant output with `stop_reason: "refusal"`.

In the Slack assistant path (`assistant_handler.py`, `streaming_integration.py`) the response text is streamed directly to the user regardless of stop_reason. A refusal would surface raw refusal prose in the Slack thread without any classification or fallback. The governance layer in `assistant_handler.py` handles rate-limit errors and network errors via `classify_error()`, but `refusal` is not a Python exception — it arrives as a successful API response and bypasses that handler entirely.

---

## Pattern 8 — FAIL: `pause_turn` not handled in tool loop (MF-27)

`pause_turn` is returned when a server-side sampling loop (web_search, web_fetch server tools) hits the iteration limit. The tool loop at `multi_agent_orchestrator.py:1023` treats any non-`tool_use` stop_reason as terminal. If `AGENT_TOOLS` ever includes server tools and `pause_turn` is returned, the agent exits mid-task instead of continuing. Currently `AGENT_TOOLS` appears to be client-side tools only, but this is a latent bug for any future server-tool addition.

---

## Pattern 9 — FAIL: `model_context_window_exceeded` not handled (MF-26)

No handler exists. The `model_context_window_exceeded` stop reason (enabled by default on Sonnet 4.5+ and newer) would fall through to the `else` branch alongside `end_turn`. With long conversation histories in the MAS tool loop (observation masking triggers at 60% window), context can fill, and the partial output would be silently treated as complete.

---

## Pattern 10 — PASS: No prefill usage detected

`grep -rn 'prefill\|assistant.*pre' backend/ --include='*.py'` returned no matches. No callsite places a synthetic assistant turn as the final message before calling `messages.create`. This is correct — the Anthropic docs explicitly state:

> Prefilling is not supported on Claude Opus 4.7, Claude Opus 4.6, and Claude Sonnet 4.6. Requests using prefill with these models return a 400 error.

All three models are in active use here (opus-4-6 for Ford/QA/Planner; sonnet-4-6 for Communication Agent/Researcher). No 400 errors from this source.

---

## Pattern 11 — PASS: No citations + structured output conflict

`grep -rn 'citations' backend/ --include='*.py'` found only patent citation counts (`patent_tracker.py`) and prompt reference arrays (`prompts.py`). No callsite uses the Anthropic citations API feature alongside `output_config.format`. Since `output_config.format` is also not used (Pattern 1), the mutex conflict is not triggerable today. If `output_config.format` is added, verify citations are not simultaneously enabled.

---

## Pattern 12 — NOTE: No Anthropic SDK streaming on the Claude path

`grep -rn 'stream=True\|messages\.stream\|text_stream' backend/ --include='*.py'` returned no matches. All Claude calls use `client.messages.create()` (blocking). The Slack "streaming" in `assistant_handler.py` and `streaming_integration.py` is Slack SDK streaming (word-by-word chunks via `client.chat_stream()`), not Anthropic SSE streaming. The blocking Claude call completes first; text is then chunked and sent to Slack.

This is architecturally sound for the current setup, but means the Anthropic streaming note about `stop_reason` placement (null in `message_start`, present in `message_delta`) is irrelevant today. If Anthropic streaming is added later, the `message_delta` event is where `stop_reason` must be read.

---

## Pattern 13 — FAIL: `evaluator_agent.py` uses Gemini, not Claude, for structured JSON

`evaluator_agent.py:84` initializes with `model_name="gemini-2.0-flash"` via Vertex AI. The evaluation prompt asks for JSON output enforced by prose instruction only (no `response_schema` or `output_config`). When `VERTEX_AVAILABLE` is false the file falls back to a mock evaluator that returns hardcoded JSON strings. Neither path uses Claude's `output_config.format`.

This is consistent (Gemini has its own structured output mechanism), but the mock path means evaluations during CI/test runs with no Vertex credentials return the same hardcoded PASS/CONDITIONAL/FAIL regardless of input — a silent fidelity risk for harness dry-runs.

---

## Pattern 14 — WARN: `stop_sequence` never configured or checked

No callsite sets `stop_sequences` on any Claude call. The stop_reason value `stop_sequence` is therefore unreachable today. If prompt templates are ever updated to include a sentinel (e.g., `END_JSON`) to terminate generation early, the response handler must be updated to check for `stop_sequence` and parse `response.stop_sequence` to know which sentinel fired.

---

## Pattern 15 — WARN: `orchestrator.py:478` parses SSE line without error handler

```python
final_json = json.loads(json_str)  # line 478, no except
```

This is inside an `async for line in r.aiter_lines()` block that itself has no surrounding try/except. A malformed JSON line from the quant agent SSE stream will raise `json.JSONDecodeError` and propagate up through the analysis pipeline, aborting the entire ticker analysis without logging the bad line. A `try/except json.JSONDecodeError` wrapping this single line would allow the loop to continue and log the bad line for debugging.

---

## Priority Fix List

| Priority | Pattern | Fix |
|---|---|---|
| P0 | 5, 6 | Add `stop_reason` dispatch to tool loop: handle `max_tokens` + incomplete `tool_use` tail (retry with 2x tokens), `refusal` (surface message, abort), `model_context_window_exceeded` (compress and retry), `pause_turn` (append and continue) |
| P0 | 7 | Add `refusal` check in `assistant_handler.py` before streaming to Slack |
| P1 | 1, 2 | Replace system-prompt JSON coercion with `output_config={"format": {"type": "json_schema", ...}}` on all Claude `messages.create` calls that expect JSON |
| P1 | 3 | Remove `find('{')/rfind('}')` extraction; rely on `output_config.format` guarantee |
| P2 | 4 | Wrap `orchestrator.py:478` and `autonomous_loop.py:449` `json.loads` calls in `try/except json.JSONDecodeError` |
| P3 | 13 | Add note to evaluator: mock path must be gated by an env flag so it cannot fire silently in staging |

---

## Reference Map

- MF-5: Structured outputs — `output_config.format` is the GA JSON enforcement mechanism
- MF-21: JSON parsing — `find/rfind` fragility; unguarded `json.loads`
- MF-26: Stop reasons — `max_tokens`, `model_context_window_exceeded` truncation handling
- MF-27: Stop reasons — `pause_turn` continuation pattern for server tools
- MF-30: Stop reasons — `refusal` is a successful response, not a Python exception; must be checked explicitly
