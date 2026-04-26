---
step: phase-22.1
title: Live model resolution in /api/agent-map + per-node Gemini-lock granularity
cycle_date: 2026-04-26
harness_required: true
verification: source .venv/bin/activate && python -m pytest backend/tests/test_agent_map_inventory.py backend/tests/test_agent_map_live_model.py -v
research_brief: handoff/current/phase-22.1-research-brief.md
---

# Contract -- phase-22.1

## Step ID

`phase-22.1` -- Live model resolution in `/api/agent-map` + per-node
Gemini-lock granularity (response to operator question "would Claude
run Layer-1 too?").

## Research-gate summary

Brief at `handoff/current/phase-22.1-research-brief.md` (18KB on disk
verified). 11 internal files inspected, 6 read in full, gate_passed=true.

**Direct answer to operator's question:** YES. Of the 28 Layer-1
enrichment skills:
- **1 hard-locked** (RAGAgent -- Vertex AI Search dependency at orchestrator.py:365-405)
- **4 grounding-dependent** (Market / Competitor / DeepDive / EnhancedMacro -- lose live web-search citations on Claude but still produce analysis text; orchestrator.py:603-748)
- **21 fully swappable** (all enrichment + debate + risk + synthesis + critic agents)
- **2 pure-Python** (BiasDetector, ConflictDetector -- no LLM at all)

## Hypothesis

Three changes give the operator both correctness AND clarity:

1. Add `gemini_locked: bool` + `grounding_dependent: bool` per-node fields in `_inventory.json` (default false; mark only the 1 hard-locked + 4 grounding-dependent nodes).
2. `/api/agent-map` injects `live_model` per node by calling `resolve_model(node_id)` -- so the operator sees the actual runtime model (which respects their Settings override).
3. New `tests/test_agent_map_live_model.py` covering live resolution + override propagation + lock-bypass.

## Immutable success criteria

```
source .venv/bin/activate && python -m pytest backend/tests/test_agent_map_inventory.py backend/tests/test_agent_map_live_model.py -v
```

Expect 16 existing + 8 new = 24 tests, all pass.

## Plan steps

1. Update `backend/agents/_inventory.json`:
   - Bump `version` 2 -> 3
   - Add `gemini_locked: true` to `rag_agent` node
   - Add `grounding_dependent: true` to market_agent, competitor_agent, deep_dive_agent, enhanced_macro_agent
   - All other nodes get the fields with default false (or omit, treated as false)
2. Update `backend/api/agent_map.py`:
   - For each node, attempt `resolve_model(node_id)` and add a `live_model` field; fall back to the static `model` field if no role mapping
   - Preserve all existing fields (backward-compat)
3. Extend tests:
   - 5 new tests in `test_agent_map_inventory.py` (version 3, lock flags accurate, count of locked = 1, count of grounding_dependent = 4, all other inventory tests still pass)
   - New file `test_agent_map_live_model.py` (5 tests: endpoint injects live_model, override propagates, locked bypass, gemini-locked stays gemini, no regression on the static `model` field)
4. Verify pytest exit 0.

## References

- Research brief findings (RAGAgent at orchestrator.py:365-405; grounding agents at 603-748; ClaudeClient schema injection at llm_client.py:748-759)
- Existing `_GEMINI_LOCKED_ROLES` in model_tiers.py (kept unchanged; this is per-role for the apply_model_to_all_agents path; new per-node flags are for display)

## Out of scope

- Frontend display of new fields (phase-22.2)
- Per-skill prompt variants (phase-21.3 doc; future cycle)
- Refactoring rag_client / grounded_client to also route via make_client (would be needed if we wanted RAG to ever swap; not requested)
