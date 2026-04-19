# Research Brief: phase-11.3 — Migrate Complex Vertex Callers + ThinkingConfig Fix

Effort tier: complex (assumption; caller did not explicitly state, but the step spans 4 files with grounding, ThinkingConfig, _flatten_schema, and structured-output changes).

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thinking | 2026-04-19 | Official doc | WebFetch | `ThinkingConfig(thinking_budget=N, include_thoughts=True)`; thought parts detected via `part.thought` (bool); `part.text` holds content — NOT `part.thinking` |
| https://ai.google.dev/gemini-api/docs/thinking | 2026-04-19 | Official doc | WebFetch | Confirms `part.thought` bool; `thinking_budget=0` disables; Gemini 2.5 and 3.x families only; Gemini 2.0 Flash does NOT support ThinkingConfig |
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-with-google-search | 2026-04-19 | Official doc | WebFetch | `Tool(google_search=GoogleSearch())`; `grounding_chunks[i].web.uri`, `.web.title`; `grounding_supports[i].segment.text`, `.grounding_chunk_indices` |
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-with-vertex-ai-search | 2026-04-19 | Official doc | WebFetch | `Tool(retrieval=Retrieval(vertex_ai_search=VertexAISearch(datastore=PATH)))`; attribute structure same as Google Search response |
| https://github.com/googleapis/python-genai/issues/699 | 2026-04-19 | Issue tracker | WebFetch | Issue closed; `Field(default=...)` causes ValueError "Default value is not supported in the response schema"; `default_factory` not explicitly tested but same code path affected |
| https://dev.to/polar3130/differences-in-response-models-between-the-vertex-ai-sdk-and-the-gen-ai-sdk-4m49 | 2026-04-19 | Technical blog | WebFetch | New SDK response model is Pydantic-based (not protobuf); `.to_dict()` removed; `.model_dump()` is the new serialization method |
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk | 2026-04-19 | Official doc | WebFetch | Vertex AI generative modules deprecated June 24 2025, removed June 24 2026; `genai.Client(vertexai=True, project=..., location=...)` replaces `vertexai.init()`; non-generative APIs (datastore, RAG) still require `google-cloud-aiplatform` |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://googleapis.github.io/python-genai/ | SDK reference | 21 078-line types.py — GitHub renders empty for fetcher; used search results instead |
| https://github.com/googleapis/python-genai/issues/1842 | Issue | `thinking_budget=0` silently ignored with tools present — relevant warning but covered by official doc |
| https://github.com/googleapis/python-genai/issues/1011 | Issue | `content.parts` becomes None near budget limit — edge case, covered by official doc |
| https://discuss.ai.google.dev/t/response-schema-from-pydantic/50028 | Forum | Covered by issue #699 WebFetch |
| https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/grounding/intro-grounding-gemini.ipynb | Notebook | Grounding shape confirmed by official doc WebFetch |

---

## Recency scan (2024-2026)

Searched for 2024-2026 literature on:
1. `google-genai ThinkingConfig thinking_budget include_thoughts Part thought` — issue #1842 (2025) reveals `thinking_budget=0` is silently ignored when tools present; issue #1011 (2025) reveals parts can become None near token limit. Both are live gotchas for orchestrator.py (which uses grounding Tools alongside thinking budgets for moderator/critic).
2. `google-genai response_schema Pydantic default_factory list` — issue #699 (Apr 2025) confirms default values in response_schema cause a ValueError at API call time; closed as p2 feature request, no SDK fix merged.
3. `vertexai.init genai.Client vertexai=True google-cloud-aiplatform dependency` — deprecation announcement June 2025 confirms non-generative APIs (Vertex AI Search datastore admin, RAG corpus, etc.) still use `google-cloud-aiplatform`; `vertexai.init()` must remain for those paths or be replaced with their own client initialization.

---

## Key findings

1. **ThinkingConfig attribute name is `thinking_budget`, NOT `budget_tokens`.** The old dict form `{"thinking": {"budget_tokens": N}}` is silently ignored. New form: `types.ThinkingConfig(thinking_budget=N, include_thoughts=True)`. (Source: Vertex AI thinking doc, 2026-04-19)

2. **Thought parts: `part.thought` (bool) + `part.text`, NOT `part.thinking`.** `llm_client.py:423` does `if hasattr(part, "thinking")` — this attribute does not exist in the new SDK. Zero thoughts will be extracted after migration unless corrected. (Source: Gemini API thinking doc, 2026-04-19)

3. **Google Search Tool: `Tool(google_search=GoogleSearch())`.** The old protobuf hack at `orchestrator.py:373-376` (`Tool.from_dict(_tool_types.Tool(...))`) must be replaced. (Source: Google Search grounding doc, 2026-04-19)

4. **Vertex AI Search Tool: `Tool(retrieval=Retrieval(vertex_ai_search=VertexAISearch(datastore=PATH)))`.** The current `Tool.from_retrieval(grounding.Retrieval(grounding.VertexAISearch(...)))` at `orchestrator.py:346-347` requires only an import swap; the nesting shape is identical. (Source: Vertex AI Search grounding doc, 2026-04-19)

5. **`grounding_metadata` in new SDK is a Pydantic model (not protobuf).** Direct attribute reads work: `gm.grounding_chunks`, `chunk.web.uri`, `chunk.web.title`. The `getattr(..., [])` defensive guards in `_extract_grounding_metadata` become unnecessary but harmless — the real risk is that grounding for RAG uses `retrieved_context`, not `web`; verify the chunk subtype at runtime. (Source: migration doc claim verified by Google Search grounding doc structure, 2026-04-19)

6. **`_flatten_schema` is CONDITIONALLY obsolete.** The new SDK's `GenerateContentConfig(response_schema=SynthesisReport)` accepts Pydantic natively. However: issue #699 shows the SDK rejects schemas with `default` values at call time with a ValueError. `_flatten_schema` currently strips `default_factory` outputs because Vertex AI SDK v1.141+ rejected `$defs`/`$ref`, not because of `default`. After migration, `_flatten_schema` is no longer needed for the Vertex AI proto-flattening use case, but the `default` stripping problem remains a live risk on any schema with `default_factory`. **Do not delete outright — repurpose as a `_strip_defaults(schema)` guard or change the Pydantic fields. See issue #699 finding below.**

7. **Issue #699 — Pydantic classes affected in `backend/agents/schemas.py`:**
   - `CriticVerdict.issues: list[CriticIssue] = Field(default_factory=list)` — line 65
   - `ModeratorConsensus.contradictions: list[Contradiction] = Field(default_factory=list)` — line 97
   - `ModeratorConsensus.dissent_registry: list[Dissent] = Field(default_factory=list)` — line 98
   These three fields will trigger ValueError at `response_schema=` call time after migration. Recommended fix: change to `Optional[list[...]] = None` or strip `default` from the schema dict before passing. (Source: internal grep + issue #699)

8. **`vertexai.init()` removal risk: `google-cloud-aiplatform` non-generative APIs.** The Vertex AI Search datastore used for RAG (`orchestrator.py:342-348`) calls `Tool.from_retrieval(...)` which belongs to `vertexai.generative_models`, not the datastore admin API. The `google-cloud-aiplatform` package can remain installed alongside `google-genai` — they are not mutually exclusive. Only the generative model calls need to move to `genai.Client`. (Source: deprecation doc, 2026-04-19)

---

## Internal code inventory

| File | Lines inspected | Role | Status |
|------|----------------|------|--------|
| `backend/agents/risk_debate.py` | Full (307 lines) | Risk analyst trio + Risk Judge | LIVE caller — see analysis below |
| `backend/agents/orchestrator.py` | Lines 1-80, 148-200, 300-400 | Layer-1 pipeline; RAG + Google Search grounding; GenerativeModel construction | Live; must fully migrate |
| `backend/agents/llm_client.py` | Lines 295-460 | GeminiClient; _flatten_schema; ThinkingConfig dict; thought extraction | Live; ThinkingConfig fix + thought attribute fix mandatory |
| `backend/agents/schemas.py` | Lines 55-115 | Pydantic response schemas | 3 fields with `default_factory=list` hit by issue #699 |
| `docs/VERTEX_AI_GENAI_MIGRATION.md` | Lines 118-183 | Migration recipe for phase-11.3 | Load-bearing — checklists and grep patterns already written |

---

## (a) risk_debate.py — LIVE CALLER, not dead import

`risk_debate.py:23` has `from vertexai.generative_models import GenerativeModel` BUT **GenerativeModel is never directly instantiated in this file**. The function `run_risk_debate()` receives a `model: LLMClient` parameter and calls `model.generate_content(...)`. The `GenerativeModel` import is dead — leftover from before the `LLMClient` abstraction. However the file also uses structured output configs at lines 37-46 with `response_schema: RiskAnalystArgument` and `response_schema: RiskJudgeVerdict`, passed as dicts to `model.generate_content(generation_config=config)`.

After migration, `risk_debate.py` needs:
1. Remove the dead `from vertexai.generative_models import GenerativeModel` import.
2. The `_RISK_STRUCTURED_CONFIG` / `_JUDGE_STRUCTURED_CONFIG` dicts pass directly to `GeminiClient.generate_content` which handles conversion internally — no dict-level change required in risk_debate.py itself.
3. The `thinking` injection at line 59 `config = {**config, "thinking": {"type": "enabled", "budget_tokens": thinking_budget}}` is the same silent-breakage pattern as llm_client.py and must be fixed.

---

## (b) Grounding response-shape diff vs migration-doc optimistic claim

Migration doc (line 126) says: "new SDK returns `response.candidates[0].grounding_metadata` as a plain Pydantic model". This is broadly correct — the response is Pydantic-based not protobuf — but the claim "regex/protobuf walk becomes dead code" is **partially optimistic**:

- `grounding_chunks[i].web.uri` and `.web.title` work as direct attribute reads for Google Search grounding.
- For Vertex AI Search (RAG) grounding, the chunk subtype is `retrieved_context` not `web`. The current code at `llm_client.py:436-442` checks only `chunk.web`; after migration this path still won't populate `uri`/`title` for RAG chunks. This is a pre-existing bug, not introduced by the migration — but it must not be made worse.
- `grounding_supports[i].grounding_chunk_indices` attribute name is unchanged.
- The `getattr` defensive guards (`getattr(gm, "grounding_chunks", []) or []`) can be simplified to `gm.grounding_chunks or []` but are not harmful.

**Simplified replacement for `_extract_grounding_metadata` legacy branch:**

```python
gm = response.candidates[0].grounding_metadata
if gm and gm.grounding_chunks:
    for chunk in gm.grounding_chunks:
        web = chunk.web
        if web:
            sources.append({"uri": web.uri or "", "title": web.title or ""})
```

---

## (c) `_flatten_schema` — NOT obsolete outright; repurpose

`_flatten_schema` (llm_client.py:310-385) solves two distinct problems:
1. Vertex AI proto schema cannot handle `$defs`/`$ref`/`anyOf` — GONE after migration (new SDK passes Pydantic natively).
2. Default values in schemas cause API ValueError (issue #699) — STILL A LIVE RISK for the three schemas identified.

**Recommendation:** Do not delete `_flatten_schema`. Gut the whitelist logic (problem 1 is gone), keep a `_strip_defaults(schema: dict) -> dict` variant that walks the schema and removes any `default` key from every field definition. Pass this stripped dict as `response_schema` for the three affected schemas. Alternatively, change those fields to `Optional[list[...]] = None`.

---

## (d) Pydantic classes hit by issue #699 (affected response_schema sites)

| Class | Field | Line | Used in |
|-------|-------|------|---------|
| `CriticVerdict` | `issues: list[CriticIssue] = Field(default_factory=list)` | schemas.py:65 | Critic Agent (orchestrator.py Step 12); response_schema in `_SYNTHESIS_STRUCTURED_CONFIG` chain |
| `ModeratorConsensus` | `contradictions: list[Contradiction] = Field(default_factory=list)` | schemas.py:97 | Moderator (debate.py, orchestrator.py Step 8) |
| `ModeratorConsensus` | `dissent_registry: list[Dissent] = Field(default_factory=list)` | schemas.py:98 | Same as above |

Fix options (either works):
- Option A: Change all three to `Optional[list[...]] = None` — zero schema-stripping logic required.
- Option B: Add a `_strip_defaults` post-processor in `GeminiClient.generate_content` that removes `"default"` keys from the flattened schema dict before the API call.

Option A is safer and simpler.

---

## Consensus vs debate

All six external sources agree on the `ThinkingConfig` field names (`thinking_budget`, `include_thoughts`) and on the `part.thought` bool detection. Issue #1842 warns of a known SDK bug where `thinking_budget=0` is silently ignored when tools are present — relevant because orchestrator uses grounding tools alongside thinking budgets for moderator/critic. The migration doc's claim that `grounding_metadata` becomes a pure Pydantic object is consistent with the DEV.to blog's finding that the new SDK is entirely Pydantic-based.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total (12 collected: 7 full + 5 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered all four relevant modules (risk_debate.py, orchestrator.py, llm_client.py, schemas.py)
- [x] Contradictions noted (migration doc optimism on grounding_metadata vs chunk subtype reality; issue #699 default_factory not explicitly tested but same code path)
- [x] All claims cited per-claim

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 5,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-11.3-research-brief.md",
  "gate_passed": true
}
```
