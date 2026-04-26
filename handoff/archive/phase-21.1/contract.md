---
step: phase-21.1
title: Settings-driven LLM model propagation -- backend
cycle_date: 2026-04-26
harness_required: true
verification: source .venv/bin/activate && python -m pytest backend/tests/test_apply_model_to_all_agents.py -v
research_brief: handoff/current/phase-21.1-research-brief.md
---

# Contract -- phase-21.1

## Step ID

`phase-21.1` -- "Settings-driven LLM model propagation: backend".

## Hypothesis

Adding `apply_model_to_all_agents: bool` to `Settings` and a single
override branch in `model_tiers.resolve_model()` propagates the
operator's Standard model selector to ALL Anthropic-routed agents
without touching Gemini-locked roles (RAG / Search Grounding /
Vertex structured output stay on `gemini-2.0-flash`).

## Immutable success criteria

```
source .venv/bin/activate && python -m pytest backend/tests/test_apply_model_to_all_agents.py -v
```

Expect 10/10 pass.

## Plan steps

1. `backend/config/settings.py` -- add `apply_model_to_all_agents: bool = False`.
2. `backend/config/model_tiers.py`:
   - Define `_GEMINI_LOCKED_ROLES = frozenset({"gemini_enrichment", "gemini_deep_think"})`
   - In `resolve_model()`: validate role first (so unknown roles always raise even with override on), then check the override flag and return `settings.gemini_model` for non-Gemini-locked roles.
3. New `backend/tests/test_apply_model_to_all_agents.py` (10 tests covering default off, override on, Gemini-lock skip, edge cases).
4. Verify pytest exit 0.

## References

- `backend/config/model_tiers.py` (resolve_model)
- `backend/config/settings.py` (gemini_model field at L29)
- `backend/agents/agent_definitions.py:129/179/227/273` (callers)

## Out of scope

- Frontend toggle UI (phase-21.2)
- Per-skill hooks (phase-21.3 doc; future cycle for impl)
- Gemini-only methods refactor (Vertex Search Grounding stays Gemini-locked by design)
