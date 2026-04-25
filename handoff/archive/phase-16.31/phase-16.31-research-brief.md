# Research Brief: phase-16.31 — MAS Gemini Fallback in `_get_client()`

**Tier:** moderate  
**Date:** 2026-04-24  
**Researcher:** researcher agent (merged researcher + Explore)

---

## Read in Full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://platform.claude.com/docs/en/api/sdks/python | 2026-04-24 | Official doc | WebFetch | Full error table: 401 = AuthenticationError; retries only fire on 408, 409, 429, >=500 — NOT 401 |
| https://platform.claude.com/docs/en/api/errors | 2026-04-24 | Official doc | WebFetch | HTTP 401 = `authentication_error` "There's an issue with your API key." 529 = `overloaded_error`. |
| https://www.getmaxim.ai/articles/retries-fallbacks-and-circuit-breakers-in-llm-apps-a-production-guide/ | 2026-04-24 | Authoritative blog | WebFetch | 401/403 are non-retryable, trigger immediate fallback; 429/5xx trigger retry first then fallback. |
| https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/ | 2026-04-24 | Authoritative blog | WebFetch | Multi-hop agent chains: circuit breakers prevent per-turn re-hitting a dead provider; pairing fallbacks with circuit breakers reduces latency. |
| https://pydantic.dev/docs/ai/api/models/fallback/ | 2026-04-24 | Official doc | WebFetch | FallbackModel triggers on ModelAPIError by default; 401 (AuthenticationError) warrants immediate fallback, not retry. |
| https://arxiv.org/html/2603.03111 | 2026-04-24 | Peer-reviewed preprint | WebFetch | Provider switch mid-conversation: single-turn handoff causes F1 swing of +-4 on CoQA; effect is asymmetric between models. For multi-turn tool loops, restart from turn 1 with fallback provider minimizes drift vs mid-turn switch. |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://deepwiki.com/anthropics/anthropic-sdk-python | Community wiki | Fetched in full — see above |
| https://github.com/anthropics/anthropic-sdk-python/issues/1258 | GitHub issue | Snippet: SSE mid-stream errors get status_code=200 bug; relevant for streaming (MAS does not stream) |
| https://github.com/anthropics/claude-agent-sdk-python/issues/472 | GitHub issue | Snippet: API errors returned as text instead of exceptions in agent SDK |
| https://www.ravchat.com/resilient-llm-build-fault-integration | Blog | Snippet: multi-provider auto-switch on rate-limit errors |
| https://github.com/gitcommitshow/resilient-llm | Code | Snippet: circuit breaker pattern library |
| https://dev.to/sandhu93/circuit-breaker-for-llm-provider-failure-53f6 | Blog | Snippet: general circuit breaker pattern |
| https://arxiv.org/abs/2505.06120 | Preprint | Snippet: LLMs get lost in multi-turn conversation (39% avg drop); context only, not directly about provider fallback |
| https://github.com/pydantic/pydantic-ai/issues/3267 | GitHub issue | Snippet: FallbackModel + SDK retry conflict; confirms max_retries=0 needed when using external fallback |
| https://medium.com/@kamyashah2018/the-complete-guide-to-llm-routing-5-ai-gateways-transforming-production-ai-infrastructure-b5c68ee6d641 | Blog | Snippet: LLM routing gateways overview |
| https://www.zenml.io/llmops-database/implementing-llm-fallback-mechanisms-for-production-incident-response-system | Blog | Snippet: fallback patterns in production incident response |

---

## Recency Scan (2024-2026)

Searched for: "Anthropic SDK exception fallback 2026", "LLM provider fallback circuit breaker 2025", "multi-turn conversation provider switch 2025".

**Findings:**
- 2025 (arxiv 2505.06120, 2603.03111): Two papers directly address multi-turn LLM provider switching. Key finding: switch from turn N to N+1 via a different provider causes measurable performance drift (+-4 F1, -8 to +13 pp on instruction-following). Recommendation: restart the full conversation with the fallback provider rather than switching mid-turn. This directly shapes the MAS fallback design (restart full agent interaction with Gemini client, not inject mid-tool-loop).
- 2025 (pydantic-ai FallbackModel): The canonical open-source implementation triggers fallback on ModelAPIError including 401; explicitly recommends setting max_retries=0 on the primary provider client to prevent the SDK's own retry from delaying fallback.
- 2025-2026 (Portkey, Maxim): Industry consensus solidified around: 401/403 = immediate fallback (no retry); 429/5xx = retry-first then fallback. No new papers supersede this.
- No new finding contradicts the design of wiring `AuthenticationError` as the fallback trigger class.

---

## Key Findings

1. **401 is NOT retried by the SDK.** The Anthropic SDK retries on 408, 409, 429, >=500 only. `AuthenticationError` (HTTP 401) fires immediately and propagates. The MAS's `_call_agent` and `_call_agent_with_tools` catch bare `Exception` and re-raise — the 401 propagates up to `execute_classified_sync` (line 209) which catches it and stuffs it into `response: "Error: ..."`. (Source: platform.claude.com/docs/en/api/sdks/python, 2026-04-24)

2. **AuthenticationError is a permanent signal for that key.** `sk-ant-oat-*` is an OAuth bearer token for Claude Code's own session — it is categorically not a valid Messages API key. Every call will 401. Retrying it is wasteful; falling back to Gemini is correct and permanent for the lifetime of this deployment. (Source: Anthropic errors doc, 2026-04-24; Q/A-16.20 context)

3. **Production consensus: 401/403 = immediate fallback, skip retry.** Industry pattern (Maxim, Portkey, pydantic-ai) is: non-retryable errors (401, 403, 404, 400) trigger fallback on the FIRST failure without burning retry budget. (Source: getmaxim.ai, portkey.ai, 2026-04-24)

4. **Multi-turn fallback: restart the turn, not mid-loop switch.** Research (arxiv 2603.03111) shows switching provider mid-conversation thread causes F1 drift; the 2025 recommendation is to restart from turn 1 with the fallback provider. For MAS's `_call_agent_with_tools`, if the 401 fires on turn 0 (before any tool results), the restart is clean. If it fires on turn N>0, the correct behavior is still to retry the whole agent call from scratch with Gemini rather than stitching Gemini responses onto an Anthropic tool-loop conversation. (Source: arxiv 2603.03111, 2026-04-24)

5. **`make_client()` IS the right abstraction; `_get_client()` bypasses it.** `llm_client.py::make_client()` already routes `claude-*` models to `ClaudeClient` (Anthropic direct) and defaults to `GeminiClient` for anything else. The MAS orchestrator bypasses it entirely by calling `anthropic.Anthropic(api_key=...)` directly in `_get_client()`. The correct fix is NOT to build a parallel fallback from scratch, but to route through `make_client()` which already knows about Gemini. (Source: llm_client.py:1090-1154 internal, 2026-04-24)

6. **The `autonomous_loop.py:370-373` Gemini fallback pattern differs meaningfully.** That pattern is a service-level fallback: it constructs an entirely new `AnalysisOrchestrator(fallback_settings)` and calls `run_full_analysis()`. The MAS is a different layer — it is the MAS orchestrator itself that needs to fall back, not an outer service calling into it. The pattern to borrow is the `fallback_settings.model_copy()` + model override, not the orchestrator construction. (Source: autonomous_loop.py:370-396 internal, 2026-04-24)

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/multi_agent_orchestrator.py` | 155-168 | `_get_client()` — lazy-init Anthropic client, cached in `self._client` | PATCH TARGET: bypasses make_client, no Gemini fallback |
| `backend/agents/multi_agent_orchestrator.py` | 889-907 | `_call_agent()` — one-shot API call, bare `except Exception` re-raises | PATCH TARGET: needs try/except around messages.create specifically |
| `backend/agents/multi_agent_orchestrator.py` | 920-974 | `_call_agent_with_tools()` — tool-loop, bare `except Exception` re-raises on turn N | PATCH TARGET: needs same fallback path |
| `backend/agents/multi_agent_orchestrator.py` | 201-218 | `execute_classified_sync()` — outer catch, stuffs 401 into response string | Downstream from the fix; will work correctly once _get_client handles fallback |
| `backend/agents/multi_agent_orchestrator.py` | 1317-1363 | `run_orchestrated_round()` — harness entry point | Not patched; already handles exception at dict level |
| `backend/agents/llm_client.py` | 1090-1154 | `make_client()` — multi-provider router, Gemini is default fallback | RE-USE: this is the right abstraction; _get_client should delegate here |
| `backend/agents/llm_client.py` | 690-1083 | `ClaudeClient` — Anthropic client with typed exception handling | EXISTS: already has RateLimitError + APIStatusError catches; no AuthenticationError -> fallback |
| `backend/config/model_tiers.py` | 42-62 | `_BUILD_TIER` — role -> model ID map | Defines: mas_main=claude-opus-4-6, mas_research=claude-sonnet-4-6, gemini_enrichment=gemini-2.0-flash |
| `backend/config/model_tiers.py` | 59-61 | Gemini model IDs | gemini-2.0-flash (enrichment), gemini-2.5-flash (deep think) — these are the fallback targets |
| `backend/services/autonomous_loop.py` | 359-396 | Service-level Claude->Gemini fallback | REFERENCE ONLY: different layer, different shape; borrow model_copy() pattern |

---

## Exact Construction Path: `_get_client()` (lines 155-168)

```python
def _get_client(self):
    if self._client is None:
        try:
            import anthropic
            from backend.config.settings import get_settings
            settings = get_settings()
            api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured.")
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("pip install anthropic")
    return self._client
```

**Critical observation:** `self._client` is cached at the instance level. Once set to an Anthropic client, all subsequent calls reuse it. The 401 fires at `client.messages.create()` time (in `_call_agent` or `_call_agent_with_tools`), not at `anthropic.Anthropic(api_key=...)` construction time. The constructor accepts any non-empty string as `api_key` — a `sk-ant-oat-*` token passes truthiness check. The 401 surfaces only on the first actual API call.

---

## `_call_agent` Exception Path (lines 889-907)

```python
def _call_agent(self, agent_config, task):
    client = self._get_client()
    try:
        response = client.messages.create(...)
        ...
    except Exception as e:
        logger.error(f"API call to {agent_config.name} failed: {type(e).__name__}: {e}")
        raise
```

The bare `except Exception` re-raises. The caller is `_execute_full_flow` via `loop.run_in_executor`. That propagates up to `execute_classified_sync` line 209, which catches and returns a dict with `response: "Error: ..."`. The `run_orchestrated_round` function then adds `iterations=1` and returns. So `iterations >= 1` PASSES today, but `response` contains the error string, not real analysis.

---

## `_call_agent_with_tools` Exception Path (lines 920-974)

Same shape: `except Exception` on line 972 logs and re-raises. The 401 from turn 0 propagates identically.

---

## `_execute_full_flow` Failure Propagation (lines 268-350)

Uses `loop.run_in_executor(None, self._call_agent, ...)` for most agent calls. The `run_in_executor` wraps the exception in a Future; the `await` re-raises it. `_execute_full_flow` does NOT have a general try/except — it lets exceptions propagate to `execute_classified_sync`'s outer catch. This means the fallback must be placed INSIDE `_call_agent` / `_call_agent_with_tools` (before re-raising), or in `_get_client()`.

---

## Gemini Fallback Model IDs (from `model_tiers.py`)

- `gemini-2.0-flash` — used for enrichment (fast, cheap)
- `gemini-2.5-flash` — used for deep think (slower, smarter)

For MAS agent roles (mas_main = claude-opus-4-6, mas_research = claude-sonnet-4-6), the appropriate Gemini fallback is `gemini-2.0-flash` (matches latency profile of Sonnet) or `gemini-2.5-flash` for mas_main. There is no role-to-Gemini-fallback mapping defined yet.

---

## Consensus vs Debate (External)

**Consensus:** 401 AuthenticationError = permanent key problem, immediate fallback appropriate, no retry. All sources agree.

**Debate:** The right fallback TARGET for MAS agent roles. Options:
1. Use `gemini-2.0-flash` for all MAS roles (simple, cheap, fast).
2. Map each Claude role to a Gemini tier (mas_main -> gemini-2.5-flash, others -> gemini-2.0-flash).
3. Use the `make_client()` abstraction and pass a Gemini model name.

Option 3 is the cleanest because it reuses existing tested routing; the Gemini model name becomes a config parameter.

**Debate on scope:** Whether to patch `_get_client()` directly (small, targeted) or refactor `_get_client()` to delegate to `make_client()` (larger, cleaner long-term). See "Patch Shape" section below.

---

## Pitfalls (from Literature)

1. **SDK retry delay before fallback.** If `max_retries` is left at default (2 for the `anthropic.Anthropic` client in `multi_agent_orchestrator._get_client`, but 3 in `ClaudeClient._get_client`), the 401 might still fire immediately since 401 is NOT in the SDK retry set. Confirmed: no delay for 401. No change needed on retry count.

2. **Mid-turn fallback causes conversation drift** (arxiv 2603.03111). If the fallback is placed in the tool-loop at turn N>0, the Gemini client inherits an Anthropic-formatted conversation thread. Safer to fail the whole `_call_agent_with_tools` call and retry from scratch with Gemini. The current re-raise pattern already does this correctly.

3. **Cached `self._client`** — if the first call 401s, `self._client` is already set to the Anthropic client. The fallback must either (a) not use `self._client` caching (reconstruct on each call), or (b) set `self._client` to the fallback Gemini client on 401 detection so subsequent calls route to Gemini automatically. Option (b) is the right design: once we know the Anthropic key is bad, cache the Gemini client.

4. **Tool-loop API surface mismatch.** `_call_agent_with_tools` calls `client.messages.create(tools=AGENT_TOOLS, thinking=...)`. These are Anthropic-native kwargs. A Gemini fallback via `make_client()` returns a `GeminiClient` whose `.generate_content()` API is completely different. The fallback cannot transparently replace the native Anthropic client mid-loop. This is the core architecture challenge.

5. **`run_orchestrated_round` already passes** with `iterations >= 1` (per phase-16.25). The QUALITY of the `response` field is the real delta — it currently contains an error string, not analysis.

---

## Application to pyfinagent — Concrete Patch Shape

### Option A: Minimal patch at `_call_agent` / `_call_agent_with_tools` — fail to Gemini via `make_client()`

The fundamental incompatibility is the API surface: `client.messages.create(tools=..., thinking=...)` is Anthropic-native. `GeminiClient.generate_content()` is different. So a transparent drop-in is not possible.

The **right shape** for Option A is:
1. In `_call_agent()`: catch `anthropic.AuthenticationError` (subclass of `APIStatusError`, status 401) specifically.
2. On catch: set `self._client = None` (invalidate cache) + set a flag `self._anthropic_unavailable = True`.
3. On fallback: call `make_client("gemini-2.0-flash", vertex_model, settings)` and call `gemini_client.generate_content(prompt)` — but this requires reformatting the tool-loop conversation into a text prompt. Loss of tool-calling capability on fallback.
4. Return the Gemini text as the agent response.

**Scope assessment:** Moderate-to-large. Converting the tool-loop messages format to a Gemini prompt string requires a helper; the agent_config.system_prompt + task formatting must be preserved; AGENT_TOOLS are Anthropic-specific and have no Gemini equivalent in this codebase.

### Option B: Replace `_get_client()` with a stateful primary/fallback client wrapper

Introduce a `_MASClientWrapper` class that:
1. Tries the Anthropic client on first call.
2. On `AuthenticationError` (or `PermissionDeniedError`): marks Anthropic as permanently unavailable, switches to a fallback.
3. The fallback is NOT a `GeminiClient` (different API surface) but rather a `ClaudeClient` via GitHub Models (which is already in `make_client()` routing for claude-* models if `github_token` is set).

**Scope assessment:** This requires knowing whether `GITHUB_TOKEN` is set in the environment. If it is, GitHub Models serves claude-* models identically to the Anthropic API — the `_get_client()` replacement just returns an OpenAI-compatible client pointing at GitHub Models, with the same `messages.create()` API surface. The tool loop runs unchanged. This is the cleanest option IF `GITHUB_TOKEN` is available.

### Option C: Targeted minimal patch — detect 401, reroute via GitHub Models, fall through to Gemini text-only

The `autonomous_loop.py` Gemini pattern (lines 370-373) constructs `AnalysisOrchestrator(fallback_settings)` — a completely different object. The MAS equivalent would be: on 401 detection, delegate the analysis to the Layer-1 `AnalysisOrchestrator` (Gemini pipeline), not continue inside the MAS tool loop. This mirrors `autonomous_loop.py:373` exactly.

Concretely, `_call_agent_with_tools` on 401:
1. Catches `anthropic.AuthenticationError`.
2. Calls `self._gemini_fallback(agent_config, task)` which instantiates `GeminiClient` with `gemini-2.0-flash` via `make_client()` and calls `generate_content(prompt)` where prompt = system_prompt + task formatted as one string.
3. Returns the text + empty usage.

This loses tool-calling on fallback but is safe, simple (~30 lines), and produces real analysis text instead of an error string. The verification command's `iterations >= 1` assertion already passes; the new benefit is `response` contains real Gemini analysis.

### Recommended shape for this cycle

**Option C** (targeted patch, ~35-45 lines total across `_call_agent` and `_call_agent_with_tools`):

```python
# In __init__: add
self._anthropic_unavailable = False
self._gemini_fallback_client = None  # lazy-init

def _get_gemini_fallback_client(self):
    if self._gemini_fallback_client is None:
        from backend.agents.llm_client import make_client, GeminiModelBundle
        from backend.config.settings import get_settings
        # build a minimal GeminiModelBundle — vertex_model is None, 
        # GeminiClient handles None bundle gracefully (returns empty LLMResponse)
        # so we need the google-genai client. Reuse the existing orchestrator's
        # GeminiClient construction path from agent_orchestrator.py if available,
        # or construct directly.
        settings = get_settings()
        # make_client with a non-claude, non-gpt, non-catalog name -> GeminiClient
        # But vertex_model is required... grab from the existing AnalysisOrchestrator
        # OR construct a GeminiModelBundle manually.
        pass
    return self._gemini_fallback_client
```

**Key complication:** `make_client()` for Gemini requires a pre-built `vertex_model` (a `GeminiModelBundle`). The MAS orchestrator does not have one — it was designed Anthropic-only. The `GeminiModelBundle` construction requires a `google.genai.Client` which requires Vertex AI credentials (ADC, project ID). This is already wired in the Layer-1 `AnalysisOrchestrator` but not in the MAS.

**Revised recommended shape:** Use the existing `AnalysisOrchestrator` as the Gemini proxy. On `AuthenticationError` in `_call_agent_with_tools`, construct a minimal single-question Gemini analysis via the Layer-1 orchestrator's `run_full_analysis()` call — exactly as `autonomous_loop.py:373` does. This is slightly heavier but requires no new Vertex AI client wiring.

**Alternatively, simplest viable fix:** Wrap the `_get_client()` call with GitHub Models routing. If `GITHUB_TOKEN` is set, `make_client("claude-sonnet-4-6", None, settings)` returns an `OpenAIClient` (GitHub Models) that has a `.generate_content()` API surface, NOT `.messages.create()`. Still API-surface incompatible with the tool loop.

**Honest scope call:** This is LARGER than a single-cycle 30-50 line patch if we want real tool-calling capability on the Gemini fallback path. The minimal viable cycle-1 fix is:

1. Add `anthropic.AuthenticationError` catch in `_call_agent` and `_call_agent_with_tools`.
2. On catch: call a `_gemini_text_fallback(agent_config, task)` helper that constructs a `GeminiModelBundle` (requiring ADC — already available via Vertex AI credentials on Peder's Mac) and calls `generate_content(system_prompt + "\n\n" + task)`.
3. Return the Gemini text response (no tools — Gemini tool loop not implemented in MAS).
4. Set `self._anthropic_unavailable = True` so subsequent calls skip the Anthropic client entirely.

This is ~40-50 lines and changes the `response` field from "AuthenticationError: ..." to real Gemini analysis. The tool-calling capability is degraded (no tool calls on fallback), but the analysis quality is restored.

**Does this change `run_orchestrated_round('AAPL', max_iterations=2)` outcome?**
- Today: `iterations=1`, `response="Error: AuthenticationError: ..."` — assertion PASSES but response is useless.
- After fix: `iterations=1`, `response=<Gemini analysis of AAPL>` — assertion PASSES, response is useful.
- The `iterations >= 1` assertion does NOT change. The quality delta is in `response`.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total including snippet-only (16 URLs)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (multi_agent_orchestrator.py, llm_client.py, model_tiers.py, autonomous_loop.py)
- [x] Contradictions / consensus noted (api surface mismatch is the key tension)
- [x] All claims cited per-claim

---

### Queries Run (3-variant discipline)

1. **Current-year frontier (2026):** "Anthropic Python SDK AuthenticationError APIStatusError exception types fallback 2026"
2. **Last-2-year window (2025):** "multi-turn LLM conversation fallback provider switch mid-conversation state preservation 2025"; "LLM provider fallback circuit breaker pattern multi-provider production 2025"
3. **Year-less canonical:** "Anthropic SDK exception hierarchy RateLimitError AuthenticationError APIStatusError"; "pydantic-ai FallbackModel LiteLLM provider fallback implementation pattern"
