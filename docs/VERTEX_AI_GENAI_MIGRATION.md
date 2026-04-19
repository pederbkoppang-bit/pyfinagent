# Vertex AI `generative_models` → `google-genai` SDK Migration

**Status:** planning doc (phase-11.0 deliverable, 2026-04-19).
**Deadline:** 2026-06-24 (Google's hard removal date; no extensions).
**Phases covered:** masterplan `phase-11` steps 11.0 through 11.4.
**Rollout pattern:** final cutover uses Rainbow Deploys (phase-12 step 12.4).

## Why

Google deprecated `vertexai.generative_models` on **2025-06-24** and will remove it on **2026-06-24**. Every `pytest` run of `test_evaluator_agent.py` currently surfaces:

```
UserWarning: This feature is deprecated as of June 24, 2025 and will be
removed on June 24, 2026. For details, see
https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk.
```

Replacement SDK: **`google-genai`** (`from google import genai`), version `1.73.1` (2026-04-14 stable). Pinning policy matches pyfinagent's exact-equals supply-chain rule.

## Deadline + rollout plan

| Date | Milestone |
|------|-----------|
| 2026-04-19 | phase-11.0 (this doc) — audit + plan |
| 2026-04-19 → 2026-05 | phase-11.1 — add `google-genai==1.73.1` + `backend/agents/_genai_client.py` shim |
| 2026-05 | phase-11.2 — migrate trivial + one moderate caller (`evaluator_agent`, `skill_optimizer`, `debate`) |
| 2026-05 | phase-11.3 — migrate complex callers (`orchestrator`, `llm_client`), including ThinkingConfig fix |
| 2026-06 (w/ phase-12.4) | phase-11.4 — remove `vertexai` from `requirements.txt` via Rainbow cutover |

Buffer of ~3 weeks before the 2026-06-24 hard deadline. If anything slips, keep `google-cloud-aiplatform` pinned (still required for BigQuery + Vertex AI Search non-generative APIs + `google.oauth2.service_account.Credentials`); only the `vertexai` top-level package is deprecated.

## Call-site inventory

Produced by:

```
grep -rn "vertexai\.generative_models\|from vertexai import generative_models\|\.GenerativeModel(\|vertexai\.init" backend/ scripts/ --include="*.py" | grep -v __pycache__
```

8 matches across 6 files in `backend/agents/`. Zero matches in `scripts/`. Zero matches outside `backend/agents/`.

| # | File:line | Match | Bucket |
|---|-----------|-------|--------|
| 1 | `backend/agents/evaluator_agent.py:40` | `from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration` | trivial |
| 2 | `backend/agents/debate.py:18` | `from vertexai.generative_models import GenerativeModel` | moderate |
| 3 | `backend/agents/risk_debate.py:23` | `from vertexai.generative_models import GenerativeModel` | moderate |
| 4 | `backend/agents/llm_client.py:303` | docstring mentioning `vertexai.generative_models.GenerativeModel` (no runtime import) | trivial (comment-only) |
| 5 | `backend/agents/orchestrator.py:29` | `from vertexai.generative_models import GenerativeModel, Tool, grounding, GenerationConfig` | **complex** |
| 6 | `backend/agents/orchestrator.py:321` | `vertexai.init(project=..., location=..., credentials=...)` | **complex** |
| 7 | `backend/agents/skill_optimizer.py:102` | `from vertexai.generative_models import GenerativeModel` (lazy, inside `_get_model`) | trivial |
| 8 | `backend/agents/skill_optimizer.py:103` | `vertexai.init(project=..., location=...)` (lazy) | trivial |

**Why `llm_client.py` is complex despite the comment-only match at line 303:** line 303 is the docstring reference, but the file holds the provider-switch logic (`GeminiClient` class at `:288+`) that builds a `GenerativeModel` object indirectly via the orchestrator-injected factory, *and* owns the ThinkingConfig injection at `:671-679`. The actual code that instantiates `GenerativeModel(...)` for Gemini lives in `orchestrator.py`; `llm_client.py` is the consumer — both files have to move together.

**Why `nlp_sentiment.py` is not in the grep output** — it used to import `vertexai.generative_models` in an earlier iteration but now routes through `llm_client.make_client()`. The research brief's mention of it is stale; audit-verified.

## Bucket recipes

### Trivial (evaluator_agent.py, skill_optimizer.py, llm_client.py docstring)

Behavior: single `GenerativeModel(model_name)` instantiation, single `.generate_content(prompt)` call, no grounding, no tool-use, no structured output, no ThinkingConfig.

**Migration recipe:**

```python
# BEFORE
from vertexai.generative_models import GenerativeModel
import vertexai

vertexai.init(project=..., location=...)
model = GenerativeModel(model_name)
response = model.generate_content(prompt)

# AFTER (via shim `_genai_client.py`)
from backend.agents._genai_client import get_genai_client

client = get_genai_client()
response = client.models.generate_content(
    model=model_name,
    contents=prompt,
)
```

### Moderate (debate.py, risk_debate.py)

Behavior: `GenerativeModel` with `GenerationConfig` (temperature / max_output_tokens), structured output via `response_schema` / `response_mime_type`.

**Migration recipe:**

```python
# BEFORE
from vertexai.generative_models import GenerativeModel
model = GenerativeModel(name)
response = model.generate_content(
    contents=[prompt],
    generation_config={"temperature": 0.0, "response_mime_type": "application/json",
                       "response_schema": MyPydantic},
)

# AFTER
from google import genai
from google.genai import types

client = genai.Client()
response = client.models.generate_content(
    model=name,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=MyPydantic,  # SDK accepts Pydantic natively — DELETE _flatten_schema
    ),
)
```

Key diff: the new SDK accepts Pydantic models directly. `backend/agents/llm_client.py`'s `_flatten_schema` helper becomes dead code and must be removed in phase-11.3.

### Complex (orchestrator.py + llm_client.py as a pair)

Behavior: grounding via Vertex AI Search + Google Search; `Tool` + `grounding` submodule; `GenerationConfig` with thinking budget; 15-step pipeline with per-step different Gemini configs.

**Migration checklist for orchestrator.py + llm_client.py:**

1. `vertexai.init(...)` at `orchestrator.py:321` → delete. `google.genai.Client()` reads the same env vars (`GOOGLE_APPLICATION_CREDENTIALS` / `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`) by default; pass explicit `project=` + `location=` if environment handling differs.
2. `from vertexai.generative_models import Tool, grounding, GenerationConfig` → `from google.genai import types`; `types.Tool`, `types.GoogleSearchRetrieval`, `types.VertexAISearch` replace the Vertex submodules 1:1 but with different attribute names.
3. **Grounding protobuf hack** (`orchestrator.py:151-200` `_extract_grounding_metadata`) — new SDK returns `response.candidates[0].grounding_metadata` as a plain Pydantic model (not a protobuf message). Field access simplifies from `getattr(gm, "grounding_chunks", [])` to `gm.grounding_chunks`. The regex/protobuf walk becomes dead code; delete and replace with direct attribute reads.
4. **`llm_client.py` ThinkingConfig** (`:671-679`) — THIS IS THE SILENT BREAKAGE. See next section.
5. `GenerationConfig(response_mime_type="application/json", response_schema=SynthesisReport)` at `orchestrator.py:78+` → `types.GenerateContentConfig(response_mime_type=..., response_schema=SynthesisReport)`. Pydantic passthrough works; keep `_flatten_schema` out of the new path.

## ThinkingConfig silent-breakage mitigation (CRITICAL)

**The problem:** `backend/agents/llm_client.py:676-679` builds a dict like `generation_config={"thinking": {"type": "enabled", "budget_tokens": N}}`. In the new SDK, this key is ignored silently. No error, no warning, but `ThinkingConfig` is never applied — **extended thinking silently disabled on every judge agent** (Moderator, Critic, Risk Judge, Synthesis).

**The fix:** the new SDK requires an explicit `types.ThinkingConfig`:

```python
from google.genai import types

config = types.GenerateContentConfig(
    temperature=0.0,
    thinking_config=types.ThinkingConfig(
        include_thoughts=True,
        thinking_budget=N,
    ),
)
```

**Q/A grep pattern** (lock into phase-11.3 acceptance criteria):

```
# After migration, this grep must return ZERO hits (the old dict form):
grep -rn 'generation_config.*thinking.*budget_tokens\|thinking.*{.*budget_tokens' backend/ --include="*.py"
```

And this grep must return AT LEAST 4 hits (one per judge agent path):

```
grep -rn 'ThinkingConfig\|thinking_config=' backend/ --include="*.py"
```

Additionally add an explicit assertion in `llm_client.GeminiClient.generate_content` after migration:

```python
if thinking_requested and "thinking_config" not in kwargs.get("config", {}).__dict__:
    raise RuntimeError("ThinkingConfig did not land — check types.ThinkingConfig conversion")
```

This fails loudly the first time someone accidentally reintroduces the old dict form.

## API diff table (quick reference)

| vertexai surface | google-genai equivalent | Notes |
|---|---|---|
| `vertexai.init(project=..., location=..., credentials=...)` | `genai.Client(project=..., location=..., credentials=...)` | `Client()` is instance-per-caller, not global |
| `GenerativeModel(name)` | `client.models.generate_content(model=name, ...)` | Model is string-keyed per call |
| `model.generate_content(prompt)` | `client.models.generate_content(model=..., contents=prompt)` | Method lives on `.models` |
| `model.generate_content(contents=..., generation_config={...})` | `client.models.generate_content(..., config=types.GenerateContentConfig(...))` | Dict → typed config object |
| `GenerationConfig(response_mime_type="application/json", response_schema=M)` | `types.GenerateContentConfig(response_mime_type=..., response_schema=M)` | Pydantic native, no flattening |
| `generation_config={"thinking": {"budget_tokens": N}}` (dict) | `types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=N))` | **SILENT BREAKAGE without explicit conversion** |
| `Tool(grounding=Grounding(...))` | `types.Tool(google_search_retrieval=...)` or `types.Tool(vertex_ai_search=...)` | Rename + nested-config shape shift |
| Async: `GenerativeModel.generate_content_async(...)` | `client.aio.models.generate_content(...)` | New SDK splits sync/async namespaces |
| Streaming: `.generate_content(stream=True)` | `client.models.generate_content_stream(...)` | Separate method, not flag |
| Tool-use / function-calling | `types.FunctionDeclaration`, `types.Tool(function_declarations=[...])` | Names unchanged; imports shift |

## Authentication parity

- **Vertex AI SDK** reads: `GOOGLE_APPLICATION_CREDENTIALS` (ADC path), `gcp_project_id`, `gcp_location` via `vertexai.init(...)`.
- **google-genai** reads: `GOOGLE_GENAI_USE_VERTEXAI=true` + `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` OR explicit `genai.Client(vertexai=True, project=..., location=...)`.
- **pyfinagent settings flow**: `settings.gcp_project_id` + `settings.gcp_location` must be passed explicitly to `genai.Client(vertexai=True, project=..., location=...)` — do NOT rely on env-var passthrough because our service-account JSON is loaded from `settings.gcp_credentials_json` in some test paths.

## Dependency plan

Step 11.1 does:
- Add `google-genai==1.73.1` to `backend/requirements.txt`.
- Keep `google-cloud-aiplatform==...` pinned (still needed for BigQuery + Vertex AI Search non-generative APIs + `google.oauth2.service_account.Credentials`).
- Write `backend/agents/_genai_client.py` shim module: single `get_genai_client()` factory returning a cached `genai.Client(vertexai=True, project=..., location=...)`.

Step 11.4 does:
- Remove `vertexai` from `backend/requirements.txt` (grep-checked; the top-level `vertexai` package is the deprecated thing — `google-cloud-aiplatform` stays).
- Remove any `import vertexai` lines.
- Confirm `DeprecationWarning` no longer surfaces in `pytest` output.

## Per-step breakdown for phase-11.1 → 11.4

### 11.1 — Pin + shim
- Add `google-genai==1.73.1` to `backend/requirements.txt`.
- Write `backend/agents/_genai_client.py` exposing `get_genai_client() -> genai.Client`.
- Write `backend/tests/test_genai_client.py` with ≥3 tests (factory cache, project/location passthrough, env-var fallback).
- Immutable verify `python -c "from google import genai"` exit 0.
- Rollback: remove `_genai_client.py` + unpin.

### 11.2 — Migrate trivial + one moderate
- `evaluator_agent.py`: swap `GenerativeModel` to `get_genai_client().models.generate_content`.
- `skill_optimizer.py:_get_model`: same.
- `debate.py`: moderate, with structured output.
- Update `test_evaluator_agent.py` `VERTEX_AVAILABLE` patch to `GENAI_AVAILABLE`.
- Immutable verify: `pytest backend/tests/test_evaluator_agent.py -q` green.
- Rollback: `git revert` the three file edits.

### 11.3 — Migrate complex + ThinkingConfig
- `orchestrator.py`: full migration per the "Complex" recipe above. Delete `vertexai.init`, swap imports, rewrite `_extract_grounding_metadata` using direct attribute reads.
- `llm_client.py`: ThinkingConfig fix + `_flatten_schema` removal.
- `risk_debate.py`: moderate, with structured output.
- Add the assertion that fails loud on missing ThinkingConfig.
- Immutable verify: `pytest backend/tests/ -q` green AND the two greps above return 0 (old dict form) + ≥4 (new typed form).
- Rollback: Rainbow Deploys — flip selector back to the previous color.

### 11.4 — Remove vertexai
- Remove `vertexai` from `backend/requirements.txt` (keep `google-cloud-aiplatform`).
- Remove `import vertexai` lines.
- Verify `DeprecationWarning` gone from `pytest` output.
- Rollback: re-pin and re-add imports (trivial).

## Rainbow Deploys integration (phase-12.4)

The final step 11.4 cutover (removing the `vertexai` dep) is risky because:
- A missed call site silently falls back to an import error at runtime.
- The ThinkingConfig fix is behavior-preserving on paper but production-subtle.

**Coordination with phase-12.4** — the Rainbow Deploys step 12.4 is explicitly scoped as "First real migration using Rainbow: Vertex → google-genai". Concretely:
- Deploy `green` color with `google-genai` + `vertexai` both present.
- Shift 5% canary traffic to `green`; compare live SLOs (latency p95, error rate, thinking-tokens-emitted) against `blue` for 2 hours.
- If green SLOs match, promote green → blue; only then run step 11.4 (remove `vertexai`) on the new blue.
- Rollback plane: flip Service selector back to blue (the last known-good with vertexai still present) if anything tripped.

See phase-12 research gate (not yet run) for the detailed Rainbow runbook.

## Runbook: rolling back mid-migration

- **After 11.1 only** — no runtime behavior change; just remove `_genai_client.py` + unpin.
- **After 11.2** — revert the 3 file commits; callers fall back to vertexai (still installed).
- **After 11.3** — revert the commits; if state is incompatible, roll back via Rainbow to the prior color.
- **After 11.4** — Rainbow flip back to the previous color (vertexai still present there). Re-add `vertexai` pin only if the rollback color is also gone.

## References

Read in full via WebFetch on 2026-04-19:

1. [Vertex AI → Gen AI SDK Migration Guide](https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk) — official Google doc.
2. [googleapis/python-genai API reference](https://googleapis.github.io/python-genai/) — method-level reference.
3. [Migrating to the new Google Gen AI SDK (Python)](https://medium.com/google-cloud/migrating-to-the-new-google-gen-ai-sdk-python-074d583c2350) — practitioner walk-through.
4. [Good Bye Vertex AI SDK](https://leoy.blog/posts/good-bye-vertex-ai-sdk/) — community migration notes.
5. [google-genai PyPI](https://pypi.org/project/google-genai/) — version + dep info.
6. [Grounding with Vertex AI Search](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-with-vertex-ai-search) — grounding surface diff.
7. [Thinking | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/thinking) — ThinkingConfig reference.

Snippet-only:

- [googleapis/python-genai releases](https://github.com/googleapis/python-genai/releases)
- [google-cloud-aiplatform deprecation policy](https://cloud.google.com/python/docs/reference/aiplatform/latest)
- [pyfinagent `handoff/current/phase-11-research-brief.md`](../handoff/current/phase-11-research-brief.md) — full researcher envelope.
- [pyfinagent `.claude/masterplan.json` phase-11 + phase-12](../.claude/masterplan.json)
- [Brandon Dimcheff — Rainbow Deploys with Kubernetes (2018)](https://brandon.dimcheff.com/2018/02/rainbow-deploys-with-kubernetes/) — phase-12 reference.
- [pyfinagent CLAUDE.md](../CLAUDE.md) — harness protocol.
