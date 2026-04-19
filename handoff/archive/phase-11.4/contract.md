# Sprint Contract -- phase-11.3 Migrate Complex Vertex Callers + ThinkingConfig

**Written:** 2026-04-19 PRE-commit.
**Step id:** `11.3` in phase-11.
**Immutable verification:** `source .venv/bin/activate && pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py && grep -rn "from vertexai.generative_models\|from vertexai import generative_models\|vertexai\.init" backend/ --include="*.py" | grep -v __pycache__ | grep -v "^backend/agents/debate.py:18:#" | wc -l | awk '{exit ($1>0)}'` — all tests pass AND zero live Vertex imports/init remaining.

## Research-gate summary

Researcher envelope `{tier: complex, external_sources_read_in_full: 7, snippet_only_sources: 5, urls_collected: 12, recency_scan_performed: true, internal_files_inspected: 5, gate_passed: true}`. Three-query discipline confirmed.

Key findings:
- **risk_debate.py**: LIVE caller despite phase-11.2 research thinking it was dead-like-debate. The `GenerativeModel` import at `:23` IS dead (never instantiated), but the file has the **same ThinkingConfig dict-form silent breakage at `:59`** that `llm_client.py:671-679` has.
- **Grounding response shape**: migration doc was optimistic. Direct attribute reads work for Google Search grounding (`gm.grounding_chunks[i].web.uri`/`title`), but Vertex AI Search (RAG) uses `retrieved_context` subtype, not `web`. Current code at `llm_client.py:436-442` only checks `chunk.web` — so RAG citation extraction has **always been a no-op** (pre-existing bug). Not worse after migration; flagged for a follow-up fix.
- **`_flatten_schema` NOT fully obsolete**: Pydantic issue #699 still open — the new SDK rejects `response_schema` dicts that contain `default` keys. Solution: keep `_flatten_schema` OR add a simpler `_strip_defaults` pass.
- **3 Pydantic fields hit issue #699**: `schemas.py::CriticVerdict.issues` (line 65), `schemas.py::ModeratorConsensus.contradictions` (line 97), `schemas.py::ModeratorConsensus.dissent_registry` (line 98). All use `Field(default_factory=list)`.
- **ThinkingConfig attribute name**: `part.thought` (bool), NOT `part.thinking`. `llm_client.py:423` uses the wrong attribute — pre-existing bug compounded by the dict-form ignore.
- **Google Search tool kwarg**: `types.Tool(google_search=types.GoogleSearch())` in 1.73.1 (new) vs old `Tool(google_search={})` dict form.

## Hypothesis

A `GeminiModelBundle` lightweight dataclass replacing `GenerativeModel` in `GeminiClient.__init__`, plus a schema post-pass that strips `default` keys, plus correct `types.ThinkingConfig` + `part.thought` + `types.Tool(google_search=...)` wiring, migrates orchestrator + llm_client + risk_debate without changing any public API. `schemas.py` is left unchanged (issue #699 worked around at the SDK boundary via the schema post-pass).

## Success criteria

**Functional:**
1. `backend/agents/llm_client.py`:
   - New `GeminiModelBundle` dataclass with 4 fields: `client` (genai.Client or None), `model_name`, `tools` (list, default []), `base_config` (dict, default {}).
   - `GeminiClient.__init__(model: GeminiModelBundle, model_name)` accepts a bundle instead of a `GenerativeModel` (bundle already carries model_name; keep explicit param for backward-compat with existing callsites that pass both).
   - `GeminiClient.generate_content` rewritten to use `self._model.client.models.generate_content(model=self._model.model_name, contents=prompt, config=types.GenerateContentConfig(..., tools=self._model.tools, thinking_config=..., response_schema=...))`.
   - `_flatten_schema` kept but augmented with a final `_strip_defaults` pass that removes `default` keys recursively (fixes issue #699 at the SDK boundary).
   - ThinkingConfig at `:671-679`: swap the dict form for `types.ThinkingConfig(thinking_budget=N, include_thoughts=True)`.
   - `part.thought` replaces `part.thinking` attribute check at `:423`.
   - Grounding extraction at `:436-442` left as-is (pre-existing RAG-no-op bug; adding `retrieved_context` support is a follow-up, documented in Known Caveats).
2. `backend/agents/orchestrator.py`:
   - Remove `import vertexai` at `:26` + `from vertexai.generative_models import ...` at `:29`.
   - Replace `vertexai.init(project=..., location=..., credentials=...)` at `:321-326` with a `genai.Client(vertexai=True, project=..., location=..., credentials=...)` built via `get_genai_client()` shim (which already does this). Assign to `_genai_client` local.
   - Replace each `GenerativeModel(name, tools=..., generation_config=...)` with `GeminiModelBundle(client=_genai_client, model_name=name, tools=tools_or_[], base_config=config)`.
   - Replace `Tool.from_dict(...)` google_search construction with `types.Tool(google_search=types.GoogleSearch())`.
   - Replace `grounding.Retrieval(grounding.VertexAISearch(datastore=datastore_path))` with `types.Tool(retrieval=types.Retrieval(vertex_ai_search=types.VertexAISearch(datastore=datastore_path)))` (new SDK shape).
3. `backend/agents/risk_debate.py`:
   - Remove dead `from vertexai.generative_models import GenerativeModel` at `:23`.
   - Fix ThinkingConfig dict at `:59`: `{"thinking": {"type": "enabled", "budget_tokens": N}}` dict → use `types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=N, include_thoughts=True))` when calling `generate_content`.
4. `backend/agents/schemas.py`: NOT modified — the `default_factory=list` fields stay; the `_strip_defaults` helper in `llm_client.py` strips emitted `default` keys at the SDK boundary.
5. Q/A grep patterns lock-in (from migration doc):
   - `grep -rn 'generation_config.*thinking.*budget_tokens\|thinking.*{.*budget_tokens' backend/ --include="*.py" | grep -v __pycache__` must return 0 hits after phase-11.3.
   - `grep -rn 'ThinkingConfig\|thinking_config=' backend/ --include="*.py" | grep -v __pycache__ | grep -v '#'` must return ≥2 hits (one per live judge agent path: llm_client + risk_debate).
6. Zero live Vertex imports remain: `grep -rn "from vertexai.generative_models\|from vertexai import generative_models\|vertexai\.init" backend/ --include="*.py" | grep -v __pycache__ | grep -v "^backend/agents/debate.py:18:#"` returns 0.
7. Full test regression zero failures: `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` same or better than the 79p/1s from 11.2.

**Non-goals:**
- NOT removing the `vertexai` top-level dep from requirements.txt (phase-11.4).
- NOT adding `retrieved_context` RAG citation extraction (pre-existing bug, future phase).
- NOT touching `schemas.py`.
- NOT adding `ModeratorConsensus` / `CriticVerdict` tests.

## Plan

1. Fix `schemas.py`? → No (using _strip_defaults at SDK boundary per above).
2. Add `GeminiModelBundle` dataclass to `llm_client.py`.
3. Rewrite `GeminiClient.generate_content` for new SDK.
4. Add `_strip_defaults` helper + chain after `_flatten_schema`.
5. Fix ThinkingConfig at `:671-679` (llm_client.py).
6. Fix `part.thought` attribute check.
7. Rewrite orchestrator.py: remove vertexai.init, build `GeminiModelBundle`s, new Tool shapes.
8. Fix risk_debate.py: remove dead import + ThinkingConfig dict.
9. Run immutable verify + regression + Q/A grep patterns.

## Researcher agent id

`a42ba45bfe138fffb`
