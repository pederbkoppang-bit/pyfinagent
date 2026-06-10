# Research Brief: Claude as Default LLM Provider (task #49)

**Tier:** moderate | **Date:** 2026-04-24

---

## Read in Full (>=5 required)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://docs.litellm.ai/docs/routing | 2026-04-24 | Official docs | WebFetch | Model-alias routing; `order` param for prioritized fallback chains; simple-shuffle recommended for prod |
| https://portkey.ai/blog/failover-routing-strategies-for-llms-in-production/ | 2026-04-24 | Authoritative blog | WebFetch | 4 failover strategies; status-code-based (401 vs 429); sequential vs parallel; "avoid DIY" warning |
| https://dev.to/ash_dubai/multi-provider-llm-orchestration-in-production-a-2026-guide-1g10 | 2026-04-24 | Blog (2026) | WebFetch | Provider-agnostic routing layer via model-name map; wrap calls in try/catch for silent fallback; avoid complex ML-based routing |
| https://www.edenai.co/post/anthropics-claude-opus-4-vs-google-deepminds-gemini-2-5-pro | 2026-04-24 | Authoritative blog | WebFetch | Claude Opus 4: 72.5% SWE-bench vs Gemini 63.2%; Gemini wins visual reasoning; hybrid approach recommended for financial AI |
| https://docs.litellm.ai/docs/proxy/reliability | 2026-04-24 | Official docs | WebFetch | Ordered fallback chains; 401 and 429 both trigger `allowed_fails`/`cooldown_time` uniformly; content_policy_fallbacks distinct from general fallbacks |

---

## Identified but Snippet-only

| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://www.getmaxim.ai/articles/top-5-llm-failover-routing-gateways-in-2026/ | Blog | Covered by other routing sources |
| https://artificialanalysis.ai/models/comparisons/claude-4-sonnet-thinking-vs-gemini-2-5-flash-reasoning | Benchmark | Page returned framework, not data |
| https://skywork.ai/blog/ai-agent/gemini-vs-claude/ | Blog | Covered by edenai.co |
| https://blog.getbind.co/2025/08/02/gemini-2-5-deep-think-vs-claude-4-opus-vs-openai-o3-pro-coding-comparison/ | Blog | Covered by edenai.co |
| https://www.statsig.com/perspectives/providerfallbacksllmavailability | Blog | Covered by portkey.ai |
| https://pinggy.io/blog/best_ai_llm_routers_openrouter_alternatives/ | Blog | Survey article, no new patterns |
| https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system | Official docs | Not relevant to provider switching |
| https://github.com/openai/codex/issues/17312 | Community | Confirmed UI demand for provider visibility |

---

## Recency Scan (2024-2026)

Searched "multi-provider LLM router production design fallback chain 2026", "Anthropic Claude 4 vs Gemini 2.5 deep think 2025 2026", "agent system provider switching settings UI 2025".

**Findings:** Claude Opus 4 (May 2025) significantly outperforms Gemini 2.5 Pro on SWE-bench (72.5% vs 63.2%) and math (90% vs 83%). LiteLLM and Portkey routing docs are actively updated for 2025-2026. The "provider-agnostic model-name map" pattern (dev.to, 2026) validates the existing `make_client()` approach in pyfinagent. No finding supersedes canonical routing patterns; newer material confirms them.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/config/settings.py` | 177 | Pydantic settings; `gemini_model` + `deep_think_model` fields | Active — needs rename + Claude defaults |
| `backend/config/model_tiers.py` | 214 | Role-based model registry; `_BUILD_TIER` has Claude models already | Active — already has `claude-sonnet-4-6` + `claude-opus-4-6` |
| `backend/api/settings_api.py` | 345 | REST API; `_VALID_MODELS` whitelist; field names use `gemini_model` | Active — field names mislead; Claude models already in whitelist |
| `backend/agents/llm_client.py` | 1155 | `make_client()` factory; priority: GitHub > Claude > OpenAI > Gemini | Active — routing works; Claude path gated on `anthropic_api_key` being set |
| `backend/agents/orchestrator.py` | ~340+ | `_resolve_gemini()` at line 313; `_GEMINI_FALLBACK = "gemini-2.0-flash"` | Active — Gemini-only features already guarded |
| `backend/services/autonomous_loop.py` | ~392-470 | `_run_claude_analysis()` hard-codes `claude-sonnet-4-6`; falls back to Gemini | Active — hard-codes model; ignores `settings.*_model` |
| `frontend/src/app/settings/page.tsx` | ~736-760 | ModelPicker labels "Standard Model (all agents)" + "Deep Think Model"; `form.gemini_model` | Active — label is already generic; underlying field name is misleading |
| `frontend/src/lib/types.ts` | ~514-523 | `FullSettings.gemini_model` + `deep_think_model` | Active — field name needs update or aliasing |

---

## Key Findings

1. **The routing infrastructure already works for Claude** — `make_client()` routes `claude-*` models to `ClaudeClient` (line 1131 in llm_client.py). The only blocker is that `ANTHROPIC_API_KEY` is currently an OAuth token (`sk-ant-oat*`), not a usable API key. (Source: task context + llm_client.py:1130-1139)

2. **Model-name-driven routing is the right approach** — The existing `make_client()` pattern is validated by LiteLLM's model-alias system and the 2026 production guide's "provider map" pattern. No need for a separate `default_provider: Literal["claude","gemini"]` field. (Source: litellm docs; dev.to 2026 guide)

3. **Three Gemini-only features are already guarded** — `_resolve_gemini()` at orchestrator.py:313-315 forces `gemini-2.0-flash` fallback for RAG (Vertex AI Search), Google Search Grounding, and Vertex AI structured output schemas. These cannot be migrated to Claude. (Source: orchestrator.py:313-315; llm_client.py header comment)

4. **The fallback in `_run_single_analysis` is robust** — autonomous_loop.py:360-389 wraps `_run_claude_analysis` in a broad `except Exception` block that falls back to `AnalysisOrchestrator.run_full_analysis` (Gemini). Monday's cycle will use Gemini if the Anthropic key is still invalid. (Source: autonomous_loop.py:353-389)

5. **`_run_claude_analysis` ignores settings** — It hard-codes `claude-sonnet-4-6` at line 454 and constructs its own `anthropic.Anthropic` client instead of using `make_client()`. This is a secondary concern but means changing the settings model won't affect the paper-trading analysis path. (Source: autonomous_loop.py:425-457)

6. **401 auth errors trigger the same fallback as 429** — `ClaudeClient.generate_content` raises `APIStatusError` on 401, which propagates out of `_run_claude_analysis`, triggering the `except Exception` Gemini fallback in `_run_single_analysis`. (Source: llm_client.py:938-944; autonomous_loop.py:361-363)

---

## Implementation Plan: Exact Edit List

### Defaults to set
- **Standard model:** `claude-sonnet-4-6` (effort="medium" per model_tiers.py, 200K context, best for enrichment + debate chains)
- **Deep-think model:** `claude-opus-4-6` (effort="high", 72.5% SWE-bench vs Gemini 63.2%; superior for Moderator, Synthesis, Critic, Risk Judge)

### Touchpoint 1 — `backend/config/settings.py` (lines 29-30)

```
BEFORE:
    gemini_model: str = Field("gemini-2.0-flash", description="Gemini model name for standard agents")
    deep_think_model: str = Field("gemini-2.5-flash", description="Model for deep-think agents ...")

AFTER:
    gemini_model: str = Field("claude-sonnet-4-6", description="Standard model for all pipeline agents. Any model in the Settings UI whitelist is valid (Claude, Gemini, GitHub Models, OpenAI). Gemini-only features (RAG, Search Grounding) always fall back to gemini-2.0-flash regardless of this setting.")
    deep_think_model: str = Field("claude-opus-4-6", description="Deep-think model for Moderator, Synthesis, Critic, Risk Judge. Same provider flexibility as gemini_model.")
```

No rename needed for now — `gemini_model` is a misleading but functional field name; renaming it requires touching types.ts, api.ts, settings_api.py, and the frontend in one atomic commit. Deferring the rename to a follow-up avoids a risky multi-file refactor. The label in the UI already says "Standard Model (all agents)" which is correct.

### Touchpoint 2 — `backend/services/autonomous_loop.py` (lines 425-429)

```
BEFORE:
    api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("No ANTHROPIC_API_KEY available")
    client = anthropic.Anthropic(api_key=api_key)

AFTER:
    # Use the centralized make_client() so the model routes correctly per settings
    # (falls back to Gemini in the outer _run_single_analysis wrapper if key is invalid)
    from backend.agents.llm_client import make_client
    # For the paper-trading lightweight path, build a minimal vertex_model=None bundle
    # and honor settings.gemini_model (now Claude by default)
    _model_name = settings.gemini_model  # respects user selection
    api_key = settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key and _model_name.startswith("claude-"):
        raise ValueError(f"No ANTHROPIC_API_KEY available for model {_model_name}")
```

AND line 454 changes from hard-coded `model="claude-sonnet-4-6"` to use `_model_name`.

### Touchpoint 3 — `backend/api/settings_api.py` (line 218)

```
BEFORE:
    deep_think_model=s.deep_think_model or s.gemini_model,

AFTER:
    deep_think_model=s.deep_think_model or s.gemini_model,  # no change needed
```

No change needed — the fallback logic is already correct. The `_VALID_MODELS` whitelist already includes all Claude models.

### Touchpoint 4 — `frontend/src/app/settings/page.tsx`

Add Claude models to `PRIMARY_MODEL_NAMES` set (line 126) so they appear in the top section of the picker rather than under "Other models":

```
BEFORE:
    "claude-sonnet-4",
    "claude-sonnet-4-6",
AFTER (add):
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
```

Also add display name:
```
BEFORE (MODEL_DISPLAY_NAMES, missing):
    // No claude-opus-4-6 entry
AFTER:
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-haiku-4-5": "Claude Haiku 4.5",
```

Add a note below the ModelPicker grid (near line 750) explaining Gemini-only features:

```tsx
<div className="mt-3 rounded-lg border border-sky-800/50 bg-sky-950/30 px-3.5 py-2.5 text-sm text-sky-300">
  <span className="font-medium">Note:</span> Three features always use Gemini regardless of model selection:
  RAG (Vertex AI Search), Google Search Grounding (Steps 4/5/9/10), and Vertex AI structured output schemas.
  These are Google-only APIs. All other pipeline steps use your selected model.
</div>
```

### Touchpoint 5 — `backend/.env`

Replace the OAuth token with a real API key:
```
BEFORE:
    ANTHROPIC_API_KEY=sk-ant-oat...

AFTER:
    ANTHROPIC_API_KEY=sk-ant-api...  (real key from console.anthropic.com)
```

This is the PRIMARY unblock. Without it, all Claude routing silently falls to Gemini.

---

## Provider Approach Recommendation

**Stay model-name-driven. Do NOT add `default_provider: Literal["claude","gemini"]`.**

Rationale:
- `make_client()` already routes by model prefix. A separate `default_provider` field creates a second source of truth and requires the orchestrator to resolve conflicts when they disagree.
- LiteLLM, Portkey, and the 2026 production guide all validate the "model-name alias" pattern as the production standard.
- The only gap is that `gemini_model` and `deep_think_model` field names are misleading. Fix this with a rename in a follow-up (task #50 candidate), not by adding a parallel provider field.

---

## Three Gemini-Only Features (UI copy)

1. **RAG / Vertex AI Search** — Financial document retrieval uses `VertexAISearch` datastore. Google-only API; no Claude equivalent.
2. **Google Search Grounding** — Real-time web grounding in analyst agents (Steps 4, 5, 9, 10). `google.genai.types.Tool(google_search=...)` is Gemini-only.
3. **Vertex AI Structured Output Schemas** — `response_schema` enforcement at the SDK layer. Falls back to system-prompt injection for Claude (already handled in `ClaudeClient.generate_content`).

These are already handled by `_resolve_gemini()` (orchestrator.py:313-315) and the `_GEMINI_FALLBACK = "gemini-2.0-flash"` constant (orchestrator.py:310). No code change needed.

---

## Risk Analysis: Invalid Key Monday Morning

**Scenario:** `ANTHROPIC_API_KEY` is still `sk-ant-oat*` Monday morning.

**What happens:**
1. `_run_single_analysis` calls `_run_claude_analysis`.
2. `_run_claude_analysis` constructs `anthropic.Anthropic(api_key="sk-ant-oat*")` and calls `client.messages.create(model="claude-sonnet-4-6", ...)`.
3. Anthropic returns HTTP 401. The SDK raises `anthropic.APIStatusError`.
4. This propagates out of `_run_claude_analysis` as an unhandled exception.
5. `_run_single_analysis` catches it at line 361 (`except Exception as e`) and logs a warning.
6. The Gemini fallback path (`AnalysisOrchestrator.run_full_analysis`) is invoked.
7. **Result: cycle runs normally on Gemini.** No trade decisions are skipped.

**Residual risk:** After touchpoint 2 (wiring `autonomous_loop` to use `make_client()`), the behavior changes slightly. `make_client()` raises `ValueError` if `model.startswith("claude-")` and `anthropic_api_key` is empty/invalid (llm_client.py:1124-1139). This ValueError also propagates up to the `except Exception` block in `_run_single_analysis`, so Gemini fallback still fires. The only failure mode is if `_run_single_analysis` itself lacks the try/except wrapper — it does (line 359-363).

**Conclusion:** Gemini fallback is robust for both the current code and after all edits. Monday's cycle is safe.

---

## Consensus vs Debate

**Consensus:** Model-name-driven routing (not provider-field-driven) is the production standard. Fallback chains should be ordered and silent to users.

**Debate:** Whether to rename `gemini_model` -> `standard_model` now or defer. Renaming is cleaner semantically but touches 8+ files atomically. Given the Monday risk window, defer the rename.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched)
- [x] 10+ unique URLs total (13 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (8 files)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```
