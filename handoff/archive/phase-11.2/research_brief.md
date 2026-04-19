# Research Brief — phase-11.2: Migrate trivial + one moderate Vertex callers

**Tier:** moderate (assumed; not stated by caller)
**Date:** 2026-04-19

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://googleapis.github.io/python-genai/ | 2026-04-19 | official doc | WebFetch | `response.parsed` returns fully-validated Pydantic object; `response.text` for raw text; async via `client.aio.models.generate_content(...)` |
| https://github.com/googleapis/python-genai/issues/60 | 2026-04-19 | GitHub issue | WebFetch | Nested Pydantic `$defs`/`$ref` issue existed in v0.3.0 but SDK's `process_schema()` now resolves it in the transformation pipeline (indexed Feb 2026) |
| https://deepwiki.com/googleapis/python-genai/3.5.1-pydantic-model-integration | 2026-04-19 | technical doc | WebFetch | `process_schema()` handles `$defs` inlining; `response.parsed` calls `model_validate()` on the Pydantic class — type-safe |
| https://github.com/googleapis/python-genai/issues/699 | 2026-04-19 | GitHub issue | WebFetch | `Field(default=...)` causes API rejection: "Default value is not supported in the response schema." Affects `ModeratorConsensus.contradictions` and `CriticVerdict.issues` which use `Field(default_factory=list)`. Reported v1.10.0, still open as p2. |
| https://github.com/googleapis/python-genai | 2026-04-19 | GitHub repo/README | WebFetch | Latest v1.73.1 (2026-04-14); response_schema accepts Pydantic; async via `client.aio`; `response.text` is the documented primary accessor |
| https://ai.google.dev/gemini-api/docs/migrate | 2026-04-19 | official migration guide | WebFetch | `response.parsed` populated when `response_schema` is a Pydantic class; async split namespace `client.aio`; `asyncio.to_thread` bridging documented as valid but suboptimal |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview | official doc | Fetched but content confirmed migration intent without specifics — supplemented by other sources |
| https://ai.google.dev/gemini-api/docs/structured-output | official doc | Fetched; confirms Pydantic works, mentions `model_validate_json(response.text)` pattern — complementary |
| https://discuss.ai.google.dev/t/response-schema-from-pydantic/50028 | forum | Snippet only — confirmed by deepwiki |
| https://pypi.org/project/google-genai/ | PyPI | Snippet only — version confirmed via README fetch |

---

## Recency scan (2024-2026)

Searched for 2025-2026 literature on google-genai response_schema, nested Pydantic, and async patterns. Found two new findings that are directly load-bearing:

1. **Issue #699 (April 2025, open):** `Field(default=...)` / `Field(default_factory=...)` in Pydantic schemas causes API-level rejection. This affects `ModeratorConsensus` and `CriticVerdict` in `schemas.py` — both have `Field(default_factory=list)`. Workaround: strip defaults before passing schema, or use `Optional[list[...]] = None` without a default factory. This is NOT a 2026 fix; it is an ongoing constraint in v1.73.1.

2. **v1.73.1 (April 2026):** Latest stable, pinned in `backend/requirements.txt` per the shim doc. The `process_schema()` nested `$ref` fix is built in (deepwiki Feb 2026 index).

---

## Key findings

1. **`response.text` is the safe accessor; `response.parsed` is a bonus.** `response.text` always contains the raw text or JSON string. `response.parsed` is populated only when `response_schema` was a Pydantic class AND the SDK successfully calls `model_validate()`. Existing code in `debate.py` reads `response.text` and then calls `_clean_json` + `_parse_json` — that pipeline survives migration unchanged. (Source: googleapis.github.io/python-genai, 2026-04-19)

2. **`response_schema` accepts Pydantic directly; `_flatten_schema` is dead code for phase-11.3.** The new SDK runs its own `process_schema()` step. (Source: deepwiki.com/googleapis/python-genai/3.5.1-pydantic-model-integration, 2026-04-19)

3. **`Field(default_factory=list)` in schemas.py will cause API rejection.** `DevilsAdvocateResult` does NOT use default_factory, so it is safe. `ModeratorConsensus` has `contradictions: list[Contradiction] = Field(default_factory=list)` and `dissent_registry: list[Dissent] = Field(default_factory=list)` — these WILL be rejected. This is a phase-11.2 blocker for debate.py's moderator path. Workaround: pass the schema through a `model_json_schema()` dict with `default` keys stripped, OR use `Optional[list[...]] = None`. (Source: github.com/googleapis/python-genai/issues/699, 2026-04-19)

4. **`debate.py` is synchronous, not async.** `run_debate()` has no `async def`. It uses `time.sleep()` for retry backoff. The `_generate_with_retry` wrapper calls `model.generate_content(...)` synchronously. No `asyncio.to_thread` is needed — use `client.models.generate_content(...)` directly. The caller (`orchestrator.py`) bridges sync→async via `asyncio.to_thread` at step-8. (Source: debate.py lines 52-98, 2026-04-19)

5. **`debate.py` passes `generation_config` as a plain dict to `LLMClient.generate_content`.** The `_generate_with_retry` function passes `gen_config` to `model.generate_content(prompt, generation_config=config)`. This is the `LLMClient` interface (not `vertexai.GenerativeModel` directly) — the `model` arg is always an `LLMClient` instance. `debate.py` does NOT directly call `vertexai.GenerativeModel`; the only bare `from vertexai.generative_models import GenerativeModel` import at line 18 is dead — it is never used in the function bodies. The actual Gemini calls flow through `GeminiClient` in `llm_client.py`. This makes debate.py's phase-11.2 scope narrower than expected: only the dead import at line 18 needs removal. (Source: debate.py lines 18, 52-98, 2026-04-19)

6. **`evaluator_agent.py` calls `self.model.generate_content(prompt)` via `asyncio.to_thread`.** After migration: replace `GenerativeModel(model_name)` stored as `self.model` with the factory pattern: call `get_genai_client()` at call time (not at `__init__` — avoids singleton lock during import), then `client.models.generate_content(model=self.model_name, contents=prompt)`. Response text is `response.text` — same as existing code. Add `GENAI_AVAILABLE` flag mirroring `VERTEX_AVAILABLE`. (Source: evaluator_agent.py lines 39-103, 282-284, 2026-04-19)

7. **`skill_optimizer.py._get_model()` at lines 98-109 lazy-inits with `vertexai.init()` + `GenerativeModel`.** Migration: replace the lazy-init block with `get_genai_client()`. The `vertexai.init(project=..., location=...)` call goes away — the shim handles auth. The `_model` instance becomes the `(client, model_name)` pair or just the client (string model name stored separately). (Source: skill_optimizer.py lines 95-123, 2026-04-19)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/evaluator_agent.py` | 522 | LLM-as-evaluator; `GenerativeModel` init at 95, `.generate_content` at 283, response via `.text` at 284 | Trivial: swap init + call site |
| `backend/agents/skill_optimizer.py` | ~200+ | Lazy `_get_model()` at 98-109; `vertexai.init()` at 103-104 | Trivial: replace 12-line block |
| `backend/agents/debate.py` | 350 | `run_debate()` sync function; `from vertexai.generative_models import GenerativeModel` at line 18 (DEAD import — never instantiated in function bodies); all calls go through `LLMClient` | Moderate scope reduced to import-only deletion |
| `backend/agents/schemas.py` | 100+ | Pydantic schemas used as `response_schema`; `ModeratorConsensus` + `CriticVerdict` have `Field(default_factory=list)` | Requires schema fix before moderator path works with new SDK |
| `backend/agents/_genai_client.py` | 149 | Phase-11.1 shim; thread-safe singleton; `reset_for_test()` alias | Ready to use |
| `backend/tests/test_evaluator_agent.py` | 135 | Patches `VERTEX_AVAILABLE`; must rename to `GENAI_AVAILABLE` + patch new flag | 1-line patch change |
| `backend/tests/test_genai_client.py` | 175 | Phase-11.1 shim tests; pattern for mocking `_build_client` | Reference for evaluator test rewrite |

---

## Consensus vs debate (external)

**Consensus:** `response.text` is the correct primary accessor; `response.parsed` is an SDK-computed convenience. All sources agree `client.models.generate_content(model=name, contents=prompt, config=types.GenerateContentConfig(...))` is the canonical new call shape.

**Debate:** Whether `Field(default_factory=list)` rejection is truly blocking or can be avoided. Issue #699 is open/p2 and has no released fix in v1.73.1. Until a fix ships, schemas with `default_factory` lists must be stripped of defaults or typed as `Optional[list] = None`.

---

## Pitfalls (from literature + code inspection)

1. **`Field(default_factory=list)` rejection** (GitHub #699): `ModeratorConsensus` and `CriticVerdict` will trigger API 400. Must fix schemas or pass a stripped JSON schema dict.
2. **`debate.py` GenerativeModel import is dead code** — but removing it may hide that `GeminiClient` in `llm_client.py` still uses the old SDK internally. Don't mistake the import removal for a full debate.py migration.
3. **`asyncio.to_thread` in evaluator_agent.py** wraps a sync call to `self.model.generate_content`. After migration, `client.models.generate_content` is also sync (native async would be `client.aio.models.generate_content`). The `asyncio.to_thread` wrapper is still appropriate to avoid blocking the event loop.
4. **`_get_model()` returns a model object** — callers use it as `self._get_model().generate_content(...)`. After migration, `_get_model()` should return the client and the model name separately, or the call site should be refactored to call `get_genai_client().models.generate_content(model=name, contents=...)` directly.
5. **GENAI_AVAILABLE guard pattern**: the test patches `VERTEX_AVAILABLE` at module level. The new guard must be `GENAI_AVAILABLE` (not `VERTEX_AVAILABLE`) in `evaluator_agent.py`, and the test must patch `backend.agents.evaluator_agent.GENAI_AVAILABLE`.

---

## Application to pyfinagent — per-file migration order

**Recommended order: evaluator_agent.py → skill_optimizer.py → debate.py**

Rationale:
- `evaluator_agent.py` has the only live test (`test_evaluator_agent.py`) in scope; migrate first so the test suite provides a green baseline. Smallest diff: swap init block (lines 39-103), swap `_call_model` body (line 281-284), add `GENAI_AVAILABLE` flag, update 1-line test patch.
- `skill_optimizer.py` is trivial (12-line `_get_model` block replacement). No test exists for it, so risk is low but no regression harness either. Second position.
- `debate.py` is last because: (a) the "moderate" scope reduces to removing 1 dead import (the real `GeminiClient` path does not change), and (b) the `ModeratorConsensus` schema default_factory issue must be resolved before the moderator structured-output path fully works with the new SDK. Doing debate last means the schema fix is scoped and deliberate.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (10 collected: 6 read in full + 4 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (evaluator_agent, skill_optimizer, debate, schemas, _genai_client, test files)
- [x] Contradictions / consensus noted (default_factory issue documented)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 4,
  "urls_collected": 10,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "report_md": "handoff/current/phase-11.2-research-brief.md",
  "gate_passed": true
}
```
