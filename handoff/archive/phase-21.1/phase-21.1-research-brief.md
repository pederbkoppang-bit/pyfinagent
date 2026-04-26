# Research Brief: phase-21.1 -- Settings model propagation backend

Tier: simple, internal-only.

## Source of truth

- `backend/config/model_tiers.py` (resolve_model + 8 build-tier roles)
- `backend/config/settings.py:29-30` (gemini_model + deep_think_model fields)
- `backend/agents/agent_definitions.py:129/179/227/273` (4 MAS agent callers)

## Decisive findings

1. Single chokepoint for model assignment is `model_tiers.resolve_model(role)`. Adding override there propagates to all callers.
2. Settings already has `gemini_model` field (despite the misleading name -- it's the Standard model selector for any provider).
3. Two roles in `_BUILD_TIER` are Gemini-locked: `gemini_enrichment` (uses Vertex AI Search RAG) + `gemini_deep_think` (uses Vertex structured-output schemas). Per CLAUDE.md "Google Search Grounding is Gemini-only (degrades on Claude/OpenAI)".
4. Override must validate role FIRST (before checking the flag) so unknown roles still raise KeyError.

## Plan

3 file edits + 1 test file. Verify with `pytest test_apply_model_to_all_agents.py -v`.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "internal_files_inspected": 4,
  "gate_passed": true
}
```
