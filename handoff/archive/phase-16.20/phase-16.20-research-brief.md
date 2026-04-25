# Research Brief: phase-16.20 — MAS Orchestrator Live Round-Trip

**Tier:** simple | **Accessed:** 2026-04-24 | **Researcher:** researcher subagent

---

## Read in Full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://platform.claude.com/docs/en/api/getting-started | 2026-04-24 | Official doc | WebFetch | API auth requires `x-api-key` header; API keys from console only; OAuth tokens are NOT listed as a supported auth mechanism |
| https://www.anthropic.com/engineering/multi-agent-research-system | 2026-04-24 | Engineering blog | WebFetch | Lead agent decomposes, subagents run with independent context windows; graceful tool failure handling; file-based state handoffs |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-04-24 | Engineering blog | WebFetch | Single-provider (Claude) harness; no multi-provider fallback documented; context reset vs compaction patterns |
| https://docs.openclaw.ai/providers/anthropic | 2026-04-24 | Vendor doc | WebFetch | `sk-ant-oat-*` OAuth tokens are rejected for 1M context requests; `openclaw models status --json` shows `auth.unusableProfiles` for rate-limited/invalid creds |
| https://medium.com/@FrankGoortani/designing-resilient-llm-architectures-disaster-recovery-strategies-6ad2e2f65942 | 2026-04-24 | Practitioner blog | WebFetch | Fail-fast vs graceful fallback; circuit breaker pattern; logging/monitoring required to detect which provider handled a request |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://github.com/anthropics/claude-code/issues/28091 | Bug report | Snippet sufficient: OAuth tokens disabled for third-party apps — confirmed `sk-ant-oat-*` does not work via Messages API |
| https://lalatenduswain.medium.com/claude-api-authentication-in-2026-oauth-tokens-vs-api-keys-explained-12e8298bed3d | Blog | Fetched in full but content was conceptual only — no HTTP status codes documented |
| https://github.com/badlogic/pi-mono/issues/2751 | Bug report | Snippet: "Anthropic API rejects OAuth tokens sent via Bearer auth" — confirms 401 on OAuth bearer; `x-api-key` header works instead |
| https://www.merge.dev/blog/llm-routing | Guide | Snippet sufficient for fallback pattern overview |
| https://openrouter.ai/docs/guides/routing/provider-selection | Vendor doc | Snippet: priority-ordered provider routing pattern |
| https://tianpan.co/blog/2026-03-11-llm-api-resilience-production | Blog | Snippet: transient vs persistent failure taxonomy; retry vs fallback distinction |
| https://blog.laozhang.ai/en/posts/openclaw-anthropic-api-key-error | Blog | Snippet: key format guide `sk-ant-api03-*` vs `sk-ant-oat-*` |
| https://github.com/openclaw/openclaw/issues/9938 | Bug report | Snippet: "OAuth authentication is currently not supported" error message literal |
| https://dev.to/kuldeep_paul/how-to-build-multi-provider-failover-strategies-with-bifrost-for-ultra-reliable-ai-applications-1keo | Dev blog | Snippet: Bifrost failover pattern overview |
| https://blog.bytebytego.com/p/how-anthropic-built-a-multi-agent | Blog | Snippet: confirms orchestrator-worker diagram; already read primary Anthropic source |

---

## Recency Scan (2024–2026)

**Searched:** "Anthropic API key format sk-ant-api03 vs OAuth token 2026", "multi-provider LLM fallback 2025 2026", "Anthropic Messages API authentication 401 sk-ant-oat 2026", "Anthropic multi-agent MAS 2026".

**Findings:** Three directly relevant 2026 sources found:
1. GitHub issue #28091 (anthropics/claude-code): Anthropic **disabled** OAuth tokens for third-party apps in 2026 — this is an active, confirmed break, not a historical bug.
2. OpenClaw issue #9938 (2026): literal error "OAuth authentication is currently not supported" returned by Anthropic Messages API.
3. Anthropic official API docs (2026): authentication section lists only `x-api-key` header with console-issued keys; no OAuth token path documented.

No newer canonical references on MAS architecture supersede the 2025 multi-agent blog post.

---

## Key Findings

1. **`sk-ant-oat-*` tokens are rejected by the Anthropic Messages API.** The API explicitly returns "OAuth authentication is currently not supported" when such a token is used via the `Authorization: Bearer` header. The same token sent via `x-api-key` header may authenticate but is architecturally unsupported for production use. (Source: GitHub anthropics/claude-code #28091, OpenClaw #9938, Anthropic official docs, 2026.)

2. **`sk-ant-api03-*` is the only supported API key prefix for programmatic access.** This key is obtained from console.anthropic.com and is billed per-token. The `sk-ant-oat01-*` prefix is an OAuth token tied to a Claude.ai subscription, not an API console credential. (Source: LaoZhang blog, Anthropic official docs.)

3. **`multi_agent_orchestrator.py` uses ONLY the Anthropic SDK directly — there is no Gemini fallback path inside the orchestrator.** `_get_client()` (line 156) calls `anthropic.Anthropic(api_key=...)` directly. If this call raises (e.g., due to a 401), the exception propagates up to `execute_classified_sync` (line 201), which returns an error dict — it does NOT fall back to Gemini. (Source: internal audit, `multi_agent_orchestrator.py` lines 155–168, 201–218.)

4. **`make_client()` in `llm_client.py` does NOT apply to the orchestrator.** `make_client()` (line 1090) handles the pipeline agents (Gemini, GitHub Models, direct Claude). The MAS orchestrator has its own private `_get_client()` that bypasses `make_client()` entirely — it reads `settings.anthropic_api_key` or `os.getenv("ANTHROPIC_API_KEY")` and constructs `anthropic.Anthropic(api_key=...)` directly. There is no fallback-to-Gemini logic in this path. (Source: `multi_agent_orchestrator.py` lines 155–168; `llm_client.py` lines 1090–1154.)

5. **`run_orchestrated_round` does not exist in the codebase.** The verification command imports `run_orchestrated_round` from `backend.agents.multi_agent_orchestrator`, but a grep of the entire `backend/` tree finds no such function. The module exposes: `MultiAgentOrchestrator` (class), `get_orchestrator()` (singleton factory), `handle_message()` (async), `call_single_agent_sync()`, `execute_classified_sync()`, `classify_message_sync()`. `run_orchestrated_round` is absent. The GENERATE phase must add this function. (Source: `multi_agent_orchestrator.py` full read; `grep -rn "run_orchestrated_round" backend/` returned no output.)

6. **`settings.anthropic_api_key` is declared as an empty-string default.** `settings.py` line 86: `anthropic_api_key: str = Field("", ...)`. It is populated from `ANTHROPIC_API_KEY` in `backend/.env`. An `sk-ant-oat-*` value will pass the truthiness check (non-empty string), so `_get_client()` will construct `anthropic.Anthropic(api_key="sk-ant-oat-...")` without raising at construction time. The 401 only surfaces on the first API call. (Source: `backend/config/settings.py` line 86; `multi_agent_orchestrator.py` lines 160–165.)

7. **The Anthropic SDK logs `anthropic.APIStatusError` with request_id and status code.** `llm_client.py` ClaudeClient lines 938–943 catch `_anthropic_sdk.APIStatusError` and log `request_id + status`. The orchestrator's `_call_agent` (line 889) does NOT have this typed catch — it only has a bare `except Exception` that re-raises. A 401 will surface as an unhandled exception logged at ERROR level. (Source: `llm_client.py` lines 929–944; `multi_agent_orchestrator.py` lines 905–907.)

8. **Multi-provider fallback in production LLM systems: the consensus pattern is "distinguish transient vs persistent failure before routing."** 401 (auth) is persistent — never retry to the same provider; fail immediately. 429 (rate limit) and 503 (availability) are transient — use exponential backoff first, then route to backup. Circuit breaker pattern: 3 consecutive errors open the breaker, route to secondary, probe primary after 60s. (Source: Frank Goortani Medium blog, tianpan.co blog 2026.)

9. **All four MAS agents (Communication, Ford/Main, Q&A, Researcher) resolve to Claude models.** `agent_definitions.py` lines 129, 179, 228, 272 call `resolve_model("mas_*")`. `model_tiers.py` build tier maps all four to `claude-sonnet-4-6` or `claude-opus-4-6`. No MAS agent resolves to a Gemini model. The Gemini models (`gemini-2.0-flash`, `gemini-2.5-flash`) are only in the pipeline agents (Layer 1 orchestrator). (Source: `agent_definitions.py` full read; `model_tiers.py` lines 42–62.)

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/multi_agent_orchestrator.py` | 1315 | MAS orchestrator — `run_orchestrated_round` target | **`run_orchestrated_round` absent — must be added in GENERATE** |
| `backend/agents/llm_client.py` | 1154 | LLM provider abstraction + `make_client()` factory | Active; does NOT apply to orchestrator's `_get_client()` |
| `backend/agents/agent_definitions.py` | 426 | Agent configs — model, system prompt, delegation rules | Active; all 4 MAS agents resolve to Claude models |
| `backend/config/model_tiers.py` | 214 | Role-to-model-ID registry; `resolve_model()` | Active; `cost_tier=build` maps all MAS roles to claude-* |
| `backend/config/settings.py` | ~120 | App settings; `anthropic_api_key` field at line 86 | Active; empty-string default, populated from `.env` |
| `handoff/logs/mas-harness.log` | large | Runtime MAS harness logs | No "anthropic" provider invocation found in last 24h grep |

---

## Consensus vs Debate

**Consensus:** OAuth tokens (`sk-ant-oat-*`) do not work with the Anthropic Messages API. This is confirmed by Anthropic's official docs, multiple GitHub issues (2025–2026), and OpenClaw's own issue tracker. The only question is whether the error is a silent degradation or a hard 401 — evidence says **hard 401** (HTTPStatusError from SDK), not silent.

**Debate:** Whether to implement Gemini fallback inside `run_orchestrated_round` or to require a valid `sk-ant-api03-*` key. The current architecture makes a hard choice: all MAS agents are Claude-only, and there is no fallback path. Adding a Gemini fallback would require either: (a) routing through `make_client()` which Gemini-routes non-claude models, or (b) adding explicit exception handling with a separate Gemini client. This is a design decision for GENERATE.

---

## Pitfalls

1. **`sk-ant-oat-*` key silently accepted at construction, fails on first API call.** The truthiness check passes, SDK constructs `anthropic.Anthropic(...)`, but a 401 `APIAuthenticationError` fires on the first `client.messages.create()`. The orchestrator's bare `except Exception` in `_call_agent` will log "API call to [name] failed" and re-raise — the round-trip returns an error dict, not a fallback response.

2. **`run_orchestrated_round` must be a synchronous wrapper.** The verification command calls it with a plain `python3 -c "..."` invocation. The existing `execute_classified_sync` pattern (lines 201–218) is the right template: it creates a new event loop, runs the async flow, closes the loop. `run_orchestrated_round(ticker, max_iterations)` should follow this pattern.

3. **`max_iterations=2` in the verification command maps to `MAX_RESEARCH_ITERATIONS`.** The orchestrator caps at 3 iterations (line 59, redeclared at 121). The output dict must contain an `iterations` key — the existing `_build_result` (line 1296) does not include this. The GENERATE phase must add `iterations` to the return dict.

4. **No Gemini fallback path exists.** If `ANTHROPIC_API_KEY=sk-ant-oat-*` and no key swap, the orchestrator 401s and the `assert out.get('iterations', 0) >= 1` fails. The documented fallback option is: catch `APIStatusError` with status 401, log the caveat, run a minimal iteration using a Gemini client directly (bypassing the orchestrator's `_get_client()`). This is an explicit design choice Main must make.

5. **Harness logs show no recent Anthropic provider invocations.** The `mas-harness.log` grep found no "anthropic" entries in the last 24 hours. This corroborates that the orchestrator has not been exercised recently with a valid API key.

---

## Application to pyfinagent

| Finding | File:Line anchor | Implication for GENERATE |
|---------|-----------------|--------------------------|
| `run_orchestrated_round` absent | `multi_agent_orchestrator.py` (no line) | Must add this function; sync wrapper over `execute_classified_sync` |
| `_get_client()` raises on bad key | `multi_agent_orchestrator.py:155–168` | Wrap in try/except `APIStatusError(status=401)` for Gemini fallback or fail-fast |
| `_build_result` lacks `iterations` key | `multi_agent_orchestrator.py:1296–1304` | Add `iterations` field or return it from `run_orchestrated_round` wrapper |
| All MAS agents = Claude | `model_tiers.py:42–62`, `agent_definitions.py:129,179,228,272` | No automatic Gemini fallback; must be explicit |
| `APIStatusError` catch in ClaudeClient | `llm_client.py:938–943` | Orchestrator's `_call_agent` lacks this — add typed catch for observability |
| `settings.anthropic_api_key` empty default | `settings.py:86` | Key presence check before constructing client gives better error message |

---

## Search Queries Run

1. **2026-scoped (frontier):** "Anthropic API key format sk-ant-api03 vs OAuth token 2026", "Anthropic Messages API authentication 401 sk-ant-oat 2026"
2. **2025-scoped (last 2 years):** "multi-provider LLM routing graceful degradation fallback pattern 2025", "Anthropic multi-agent orchestrator MAS architecture 2025 2026"
3. **Year-less canonical:** "Anthropic API key validation smoke test production round-trip verification", "multi-agent orchestrator architecture"

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched: Anthropic official docs, Anthropic multi-agent blog, Anthropic harness blog, OpenClaw docs, Frank Goortani resilience blog)
- [x] 10+ unique URLs total (10 snippet-only + 5 full = 15 total)
- [x] Recency scan (last 2 years) performed + reported (2026 sources found and reported)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (orchestrator, llm_client, agent_definitions, model_tiers, settings, logs)
- [x] Contradictions / consensus noted (OAuth token rejection consensus; Gemini fallback design debate documented)
- [x] All claims cited per-claim (not just listed in footer)

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "phase-16.20-research-brief.md",
  "gate_passed": true
}
```
