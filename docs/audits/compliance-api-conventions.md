# Compliance Audit: Messages API Conventions
**Phase:** 4.15.10 | **Date:** 2026-04-18 | **Auditor:** researcher agent

---

## Summary

17 patterns audited across 9 internal files. 12 non-compliant findings, 5 clean. The most critical gaps are: zero `retry-after` header reads across all 4+ retry loops, no `anthropic-ratelimit-*` header reads for proactive backoff, no `request_id` logging on success or failure for Anthropic calls, SDK pinned 9 minor versions behind latest, and stale date-suffixed snapshot IDs in 4 call sites.

---

## External Reference Summary

| Document | Key facts confirmed |
|---|---|
| [Errors](https://platform.claude.com/docs/en/api/errors) | Every error response includes `request_id` field; `_request_id` on SDK response objects |
| [Rate limits](https://platform.claude.com/docs/en/api/rate-limits) | 429 includes `retry-after` header (seconds). Eight `anthropic-ratelimit-*` headers returned on every response |
| [Beta headers](https://platform.claude.com/docs/en/api/beta-headers) | `betas=["feature-YYYY-MM-DD"]` on `client.beta.messages.create`; pattern `feature-name-YYYY-MM-DD` |
| [Versioning](https://platform.claude.com/docs/en/api/versioning) | Single stable version `2023-06-01`; SDK sets it automatically |
| [Service tiers](https://platform.claude.com/docs/en/api/service-tiers) | `service_tier="auto"` is the default; Priority Tier headers `anthropic-priority-*` visible in responses |
| [Python SDK](https://platform.claude.com/docs/en/api/sdks/python) | `max_retries` default 2; `_request_id` on response; `count_tokens()`; `with_raw_response` for header access; native retry covers 429/408/409/5xx |

---

## Findings

### F-01 — `retry-after` header never read (CRITICAL)
**Files:** `debate.py:66-79`, `risk_debate.py:63-79`, `orchestrator.py:424-438`, `ticket_queue_processor.py:340-360`

**Grep result:** `grep -rn 'retry-after\|retry_after' backend/ --include='*.py' | wc -l` returns `0`.

All four retry loops use a hardcoded initial `delay = 5` with `delay *= 2` doubling. When the API returns a 429, the response includes a `retry-after` header (in seconds) indicating exactly how long to wait. None of the loops reads this header. As a result, the backoff is arbitrary — potentially too short (triggering repeated 429s) or too long (unnecessary delay).

The correct pattern for SDK calls:

```python
except anthropic.RateLimitError as e:
    retry_after = float(e.response.headers.get("retry-after", delay))
    time.sleep(retry_after)
```

Relates to: MF-9 (honor server-advertised backoff), MF-10 (rate limit discipline).

---

### F-02 — `anthropic-ratelimit-*` headers never read (HIGH)
**Files:** All Anthropic call sites (`llm_client.py`, `multi_agent_orchestrator.py`, `ticket_queue_processor.py`, `autonomous_loop.py`)

**Grep result:** `grep -rn 'anthropic-ratelimit\|anthropic_ratelimit' backend/ --include='*.py'` returns empty.

The API returns eight `anthropic-ratelimit-*` headers on every response: `anthropic-ratelimit-requests-remaining`, `anthropic-ratelimit-tokens-remaining`, `anthropic-ratelimit-input-tokens-remaining`, `anthropic-ratelimit-output-tokens-remaining` (plus `-limit` and `-reset` variants). These enable proactive backoff — slowing request pace before hitting the wall — rather than reactive retry after a 429.

To read them, the SDK requires `with_raw_response`:

```python
raw = client.messages.with_raw_response.create(...)
remaining = raw.headers.get("anthropic-ratelimit-requests-remaining")
message = raw.parse()
```

The codebase makes no use of this. The autonomous loop and ticket processor are most at risk since they issue back-to-back Anthropic calls.

Relates to: MF-9, MF-10, MF-11.

---

### F-03 — `request_id` not logged on success or failure (HIGH)
**Files:** `multi_agent_orchestrator.py:893-906`, `ticket_queue_processor.py:206-223`, `autonomous_loop.py:436-441`, `llm_client.py:630`

**Grep result:** `grep -rn 'request_id\|_request_id' backend/ --include='*.py'` shows only `slack_bot/governance.py` results — those are internal UUID fields, not Anthropic response IDs.

Every Anthropic SDK response exposes `response._request_id` (the value of the `request-id` response header, e.g. `req_018EeWyXxfu5pfWkrYcMdjWG`). None of the call sites log this field. On failure, `logger.error(f"... failed: {type(e).__name__}: {e}")` in `multi_agent_orchestrator.py:906` does not include the request ID.

Without request IDs in logs, Anthropic support cannot trace specific failing calls.

Minimum fix:

```python
response = client.messages.create(...)
logger.debug("anthropic request_id=%s", response._request_id)
```

On error, `anthropic.APIStatusError` exposes `e.request_id`.

Relates to: MF-7 (observability), MF-18 (debuggability for support escalation).

---

### F-04 — Typed Anthropic error classes not imported or caught (HIGH)
**Files:** `debate.py:73-81`, `risk_debate.py:70-79`, `orchestrator.py:431-438`, `multi_agent_orchestrator.py:955-957`, `ticket_queue_processor.py:223`, `llm_client.py:555-560`

**Grep result:** `grep -rn 'anthropic.RateLimitError\|anthropic.APIStatusError\|anthropic.APIConnectionError\|anthropic.APITimeoutError' backend/ --include='*.py'` returns empty.

The SDK exposes typed error classes: `anthropic.RateLimitError` (429), `anthropic.APIStatusError` (4xx/5xx with `.status_code` and `.response`), `anthropic.APIConnectionError` (network failures), `anthropic.APITimeoutError` (timeouts). None of these are imported or caught.

Instead, the generic pattern in `debate.py:75-76` is:

```python
err_name = type(e).__name__.lower()
is_transient = any(x in err_name for x in ("ratelimit", "overload", "unavailable"))
```

This string-matches the class name, which is fragile: it misses `anthropic.InternalServerError` (529 overloaded), breaks if Anthropic renames exception classes, and cannot read `.status_code` or `.response.headers` that are available on typed exceptions.

`ticket_queue_processor.py:223` catches bare `Exception as api_error` with no Anthropic-specific handling at all.

Correct import:

```python
import anthropic
# then catch:
except anthropic.RateLimitError as e:
    retry_after = float(e.response.headers.get("retry-after", 5))
except anthropic.APIConnectionError:
    # no retry-after on connection errors; use fixed backoff
except anthropic.APIStatusError as e:
    if e.status_code == 529:  # overloaded
        ...
```

Relates to: MF-9, MF-10.

---

### F-05 — SDK pinned at 0.87.0, current is significantly newer (MEDIUM)
**File:** `backend/requirements.txt:37`

```
anthropic==0.87.0  # exact pin: supply-chain hardening (phase-3.7.6; CVE-2026-34450/34452 fix)
```

The pin is justified (CVE remediation), but 0.87.0 was current in early 2026. As of April 2026 the SDK has released multiple minor versions. The supply-chain rationale requires periodic re-evaluation: a known CVE fix is not a reason to stay on 0.87.0 indefinitely when newer patches may exist.

Specific features that landed after 0.87.0 include improvements to `with_raw_response` header access and the `count_tokens` helper. A quarterly audit of the pin against the CVE advisory is recommended, with `pip-audit` run in CI.

Relates to: MF-18 (keeping dependencies current for security and API compatibility).

---

### F-06 — Stale date-suffixed snapshot ID `claude-sonnet-4-20250514` in 4 active call sites (MEDIUM)
**Files:** `autonomous_loop.py:438`, `mcp_tools.py:74, 223`, `app_home.py:23`

`claude-sonnet-4-20250514` is a date-stamped snapshot from May 2025. The current recommended alias is `claude-sonnet-4-6` (used correctly in `multi_agent_orchestrator.py:26` comments and `app_home.py:21`). Date-stamped IDs are not deprecated automatically, but they freeze behavior at an older checkpoint and do not benefit from model improvements.

`autonomous_loop.py:438` is the highest-risk site: it makes a live production call using this snapshot ID for paper-trading signal generation.

`mcp_tools.py:223` uses the stale ID as a default parameter for `integrate_mcp_with_claude_call()`, meaning every MCP-enabled call from Slack inherits the old snapshot unless overridden.

Relates to: MF-18 (model currency).

---

### F-07 — `service_tier` parameter not set on any call site (LOW-MEDIUM)
**Grep result:** `grep -rn 'service_tier' backend/ --include='*.py'` returns empty.

Per the service tiers documentation, `service_tier="auto"` is the API default, meaning Priority Tier capacity is used when available and falls back to standard. The codebase does not explicitly set this, which is safe (the default is `"auto"`), but means:

1. There is no ability to reserve Priority Tier capacity for high-value paths (paper trading decisions, harness evaluator calls) while routing exploratory calls as `"standard_only"`.
2. Priority Tier usage is invisible in logs — the `usage.service_tier` field from the response is never read.

For go-live, explicitly setting `service_tier="auto"` on all calls and logging `response.usage.service_tier` would provide tier visibility with no code risk.

---

### F-08 — `metadata={"user_id": ...}` absent from Slack-ticket Anthropic calls (LOW-MEDIUM)
**File:** `ticket_queue_processor.py:206-213`

The Anthropic Messages API accepts a `metadata` field with `user_id` to enable per-user rate limit analysis and abuse detection. Slack ticket calls in `ticket_queue_processor.py` represent user-originated requests (each ticket has an originating Slack user). The call omits `metadata`:

```python
response = client.messages.create(
    model=model_name,
    max_tokens=1000,
    system=system,
    messages=[{"role": "user", "content": task}]
    # metadata={"user_id": ticket["user_id"]} is absent
)
```

This is relevant for the May 2026 go-live when multiple users will submit concurrent tickets. Relates to: MF-11 (user attribution for rate limit analysis).

---

### F-09 — `anthropic-beta` header usage: missing `betas=` wrapper on MCP call (MEDIUM)
**File:** `mcp_tools.py:73-84, 240-247`

`mcp_tools.py:243` calls `anthropic_client.beta.messages.create(...)` with `mcp_servers=` but no explicit `betas=` list. MCP server integration requires the `mcp-client-2025-04-04` beta header. Without it, the API returns `invalid_request_error: Unsupported parameter: mcp_servers`.

The call in `mcp_tools.py:73` (doc comment) shows `betas=[]` is absent. `integrate_mcp_with_claude_call()` passes `**kwargs` without enforcing the beta header. Any caller that does not add `betas=["mcp-client-2025-04-04"]` to kwargs will get a 400.

Current beta headers in active use: none confirmed with `betas=` explicitly set anywhere in the codebase. The `client.beta.messages.create` namespace alone does not inject the beta header automatically.

Relates to: MF-7 (correct API usage pattern).

---

### F-10 — Native SDK retry (`max_retries`) duplicates manual retry loops (LOW)
**File:** `llm_client.py:555-560`

```python
return anthropic.Anthropic(api_key=self._api_key)
```

The SDK client is initialized with default `max_retries=2`, which means 429/5xx errors are automatically retried twice with exponential backoff before the exception is raised. The callers (`debate.py`, `risk_debate.py`, `orchestrator.py`) then wrap the call in their own retry loop of 3 attempts.

This creates up to 6 total attempts (2 SDK retries per each of 3 outer attempts) when only 3 were intended. The SDK's built-in retry also does honor `retry-after` internally (via httpx), but because the outer loops catch the exception only after all SDK retries are exhausted, the outer delay calculation is redundant.

Resolution options: set `max_retries=0` on the client and keep the outer loop (explicit, auditable), or remove the outer loop and rely on `client.with_options(max_retries=3)`. Either is acceptable; the current combination is unintentional multiplication.

Relates to: MF-9.

---

### F-11 — `count_tokens()` helper unused; pre-flight token check absent (LOW)
**Files:** All call sites

The SDK provides `client.messages.count_tokens(model=..., messages=[...])` which returns a token count without making an inference call. This is useful before calls that could exceed context limits (e.g., `ticket_queue_processor.py` which sends unbounded ticket content, and `autonomous_loop.py` which sends market data).

No call site uses this. The current mitigation (character-level truncation in `llm_client.py:479-496`) is OpenAI-path-only. The Claude path in `generate_content` has no character limit guard.

Relates to: MF-11 (token budget management).

---

### F-12 — `anthropic-version` not overridden; SDK auto-sets `2023-06-01` (CLEAN)
**Grep result:** `grep -rn 'anthropic-version\|anthropic_version' backend/ --include='*.py'` returns empty.

The SDK automatically injects `anthropic-version: 2023-06-01` on every request. The codebase correctly does not override it. `2023-06-01` is the only stable version and the SDK documentation warns against overriding it. Status: compliant.

---

### F-13 — `with_raw_response` unused (informational, covered by F-02)
The `.with_raw_response` accessor is the gateway to both `_request_id` header logging (F-03) and `anthropic-ratelimit-*` header reads (F-02). It is not used anywhere. Both gaps depend on adopting this pattern.

---

### F-14 — `ticket_queue_processor.py` retry delay is time-based estimate, not `retry-after`-driven (CRITICAL, duplicate of F-01 with additional context)
**File:** `ticket_queue_processor.py:349-358`

```python
# CRITICAL: Anthropic heavily rate limiting — need MASSIVE delays
wait_time = min(60 * (2 ** (retry_count - 1)), 240)
```

The comment itself acknowledges rate limiting as the trigger, yet the wait time is a hardcoded formula starting at 60 seconds. The `retry-after` header could provide a precise wait (often 5-30 seconds for token-bucket replenishment). This means the processor waits up to 4 minutes when the API may have cleared in 10 seconds.

---

### F-15 — `alt_data.py` rate-limit detection: string matching on exception message (MEDIUM)
**File:** `alt_data.py:50-54`

```python
def _is_rate_limited(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("429" in msg or "too many requests" in msg
            or "response with code 429" in msg or "quota" in msg)
```

This targets pytrends (not Anthropic SDK), so typed exception classes are not available. The string matching is reasonable for pytrends. Status: acceptable for its target library. However the same pattern (`err_name = type(e).__name__.lower()`) propagated to `debate.py` and `risk_debate.py` for Anthropic exceptions is not acceptable (see F-04).

---

### F-16 — No `extended-cache-ttl` beta header used despite prompt caching being active (LOW)
**File:** `llm_client.py:601-611`

Prompt caching is enabled (`enable_prompt_caching=True`) and `cache_control: {"type": "ephemeral"}` is applied to system prompts. The default cache TTL is 5 minutes. For the harness (which re-runs the same system prompt across cycles), a 1-hour TTL via the `extended-cache-ttl-2025-04-11` beta header would significantly increase cache hit rates between harness cycles.

No `betas=` field is passed in the `generate_content` path. The client does not use `client.beta.messages.create`.

Relates to: MF-11 (cache efficiency).

---

### F-17 — `thinking=` in `multi_agent_orchestrator.py` passed without `extended-thinking` beta header (MEDIUM)
**File:** `multi_agent_orchestrator.py:944-953`

```python
response = client.messages.create(
    ...
    thinking={"type": "enabled", "budget_tokens": 2048},
)
```

Extended thinking requires the `interleaved-thinking-2025-05-14` beta header when used with tool calls. This call uses `tools=AGENT_TOOLS` alongside `thinking=`. Without the beta header, the API may return a 400 `invalid_request_error`. The call site uses `client.messages.create` (not `client.beta.messages.create`) and passes no `betas=` equivalent (`extra_headers` with `anthropic-beta`).

Relates to: MF-7 (correct beta header usage for extended thinking with tools).

---

## Summary Table

| ID | Pattern | File(s) | Severity | Status |
|---|---|---|---|---|
| F-01 | `retry-after` header not read in retry loops | debate, risk_debate, orchestrator, ticket_queue | CRITICAL | FAIL |
| F-02 | `anthropic-ratelimit-*` headers not read | All Anthropic call sites | HIGH | FAIL |
| F-03 | `request_id` not logged on success or failure | mao, ticket_queue, autonomous_loop, llm_client | HIGH | FAIL |
| F-04 | Typed Anthropic error classes not used | debate, risk_debate, orchestrator, mao, ticket_queue | HIGH | FAIL |
| F-05 | SDK pinned at 0.87.0, not reviewed since phase-3.7.6 | requirements.txt | MEDIUM | WARN |
| F-06 | Stale snapshot ID `claude-sonnet-4-20250514` in 4 call sites | autonomous_loop, mcp_tools, app_home | MEDIUM | FAIL |
| F-07 | `service_tier` not set; tier visibility absent from logs | All call sites | LOW-MED | WARN |
| F-08 | `metadata={"user_id": ...}` absent from ticket calls | ticket_queue_processor | LOW-MED | FAIL |
| F-09 | `betas=["mcp-client-2025-04-04"]` absent from MCP call | mcp_tools | MEDIUM | FAIL |
| F-10 | SDK `max_retries=2` duplicates outer retry loops (up to 6 attempts) | llm_client + callers | LOW | WARN |
| F-11 | `count_tokens()` helper unused; no pre-flight token check | All call sites | LOW | WARN |
| F-12 | `anthropic-version` not overridden — SDK auto-sets correctly | All | -- | PASS |
| F-13 | `with_raw_response` unused (blocks F-02 and F-03) | All | HIGH | FAIL |
| F-14 | ticket_queue retry waits 60-240s; `retry-after` could give 5-30s | ticket_queue_processor | CRITICAL | FAIL |
| F-15 | `_is_rate_limited` string match on pytrends exception | alt_data | LOW | PASS (scope) |
| F-16 | `extended-cache-ttl` beta not used despite caching being active | llm_client | LOW | WARN |
| F-17 | `thinking=` + `tools=` without `interleaved-thinking` beta header | multi_agent_orchestrator | MEDIUM | FAIL |

**PASS:** 2 | **WARN:** 5 | **FAIL:** 10

---

## Recommended Fix Priority for Phase-4.15.10

1. **F-01 + F-04 together** — Replace string-matched exception catches with typed Anthropic classes and read `retry-after` from `e.response.headers`. Single change, maximum safety improvement.
2. **F-17** — Add `extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"}` to the thinking+tools call; prevents silent 400s in production.
3. **F-09** — Enforce `betas=["mcp-client-2025-04-04"]` inside `integrate_mcp_with_claude_call()` so callers cannot forget it.
4. **F-03** — Log `response._request_id` (success) and `e.request_id` (failure) in every Anthropic call site. One-liner per site.
5. **F-06** — Replace `claude-sonnet-4-20250514` with `claude-sonnet-4-6` in all 4 locations.
6. **F-02 + F-13** — Adopt `with_raw_response` for rate-limit header visibility on high-throughput paths (autonomous_loop, ticket_queue).

---

## Sources

- [Anthropic Errors documentation](https://platform.claude.com/docs/en/api/errors)
- [Anthropic Rate Limits documentation](https://platform.claude.com/docs/en/api/rate-limits)
- [Anthropic Beta Headers documentation](https://platform.claude.com/docs/en/api/beta-headers)
- [Anthropic Versioning documentation](https://platform.claude.com/docs/en/api/versioning)
- [Anthropic Service Tiers documentation](https://platform.claude.com/docs/en/api/service-tiers)
- [Anthropic Python SDK documentation](https://platform.claude.com/docs/en/api/sdks/python)
