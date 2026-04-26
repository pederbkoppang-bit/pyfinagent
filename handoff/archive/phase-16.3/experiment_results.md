---
step: phase-22.1+22.2
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - backend/agents/_inventory.json (v2->v3, +5 nodes flagged: 1 gemini_locked + 4 grounding_dependent)
  - backend/api/agent_map.py (+_NODE_ID_TO_ROLE map + _inject_live_model() + endpoint injects live_model per node)
  - backend/config/model_tiers.py (+layer1_swappable role for non-locked Layer-1 skills)
  - backend/tests/test_agent_map_inventory.py (+5 tests for v3 fields)
  - backend/tests/test_agent_map_live_model.py (NEW, 7 tests)
  - backend/tests/test_apply_model_to_all_agents.py (1 test updated for new layer1_swappable role)
  - frontend/src/lib/api.ts (+live_model +gemini_locked +grounding_dependent +lock_reason on AgentMapNode)
  - frontend/src/components/AgentMap.tsx (AgentNode renders live_model + LOCKED/SEARCH badges)
---

# Experiment Results -- phase-22.1 + 22.2

## What was done

Operator screenshot showed `gemini-2.0-flash` for EvaluatorAgent, SynthesisAgent, Layer-1 Pipeline, SkillOptimizer — even though Claude was selected in Settings. Two issues fixed:

1. **Display gap** — agent map served the static inventory file with hardcoded model strings; never read live runtime resolution from `resolve_model()`.
2. **Operator question "can Claude run Layer-1?"** — researcher confirmed YES for 21 of 28 skills. Only RAGAgent is hard-locked (Vertex AI Search). 4 are grounding-dependent (degrade gracefully).

## Backend (phase-22.1)

### `_inventory.json` v3

Added per-node fields:
- `gemini_locked: true` on `rag_agent` (with `lock_reason` = "Vertex AI Search tool dependency at orchestrator.py:365-405")
- `grounding_dependent: true` on `market_agent`, `competitor_agent`, `deep_dive_agent`, `enhanced_macro_agent`

### `model_tiers.py` -- new role

Added `layer1_swappable` role (defaults to `gemini-2.0-flash`, NOT in `_GEMINI_LOCKED_ROLES`). Layer-1 swappable skills now use this instead of `gemini_enrichment`. The phase-21.1 override flag now correctly propagates to bull/bear/devil's advocate/etc.

### `agent_map.py` endpoint

- `_NODE_ID_TO_ROLE` map (40+ entries) tags each node with its `model_tiers` role
- `_inject_live_model(node)` calls `resolve_model(role)` and adds `live_model` field
- Locked nodes always get static gemini regardless of override
- Defensive: failures in resolver fall back to static `model`; nodes with no role mapping have no `live_model` added

## Frontend (phase-22.2)

### `AgentMap.tsx`

- `AgentNodeData` extended with `liveModel`, `geminiLocked`, `groundingDependent`, `lockReason`
- `AgentNode` renders `live_model` in preference to static `model`
- LOCKED badge (amber) on `gemini_locked` nodes
- SEARCH badge (sky) on `grounding_dependent` nodes
- Tooltip explains lock reason or grounding implication

### `api.ts`

Extended `AgentMapNode` interface with the four optional fields.

## Verification

```
$ python -m pytest backend/tests/test_agent_map_inventory.py backend/tests/test_agent_map_live_model.py backend/tests/test_apply_model_to_all_agents.py -v
============================== 38 passed in 0.08s ==============================

$ cd frontend && npx tsc --noEmit
(exit 0)

$ cd frontend && npm run build
... 14 routes built ...
```

Live endpoint smoke confirmed:
```
$ curl -sS http://localhost:8000/api/agent-map | jq '...'
version: 3
total nodes: 58
gemini_locked: 1 -> [rag_agent]
grounding_dependent: 4 -> [competitor_agent, enhanced_macro_agent, deep_dive_agent, market_agent]

sample live_model:
  main: model=varies -> live_model=claude-opus-4-6
  bull_agent: model=gemini-2.0-flash -> live_model=gemini-2.0-flash
  rag_agent: model=gemini-2.0-flash -> live_model=gemini-2.0-flash (LOCKED)
  skill_optimizer: model=gemini-2.0-flash -> live_model=claude-opus-4-6 (override applied)
```

## Files touched

| Path | Action |
|------|--------|
| backend/agents/_inventory.json | edit (v3 + 5 node-flag additions) |
| backend/api/agent_map.py | edit (+_NODE_ID_TO_ROLE + _inject_live_model + endpoint update) |
| backend/config/model_tiers.py | edit (+layer1_swappable role) |
| backend/tests/test_agent_map_inventory.py | edit (+5 tests; updated v2 -> v2+) |
| backend/tests/test_agent_map_live_model.py | CREATED (7 tests) |
| backend/tests/test_apply_model_to_all_agents.py | edit (1 test updated for new role) |
| frontend/src/lib/api.ts | edit (4 new optional fields on AgentMapNode) |
| frontend/src/components/AgentMap.tsx | edit (AgentNodeData + AgentNode + buildGraph plumbing) |

## Honest disclosures

1. **Two cycle-2 fixes during impl, not separate Q/A retries:**
   - Caught wrong role mapping (Layer-1 skills mapped to `gemini_enrichment` which is locked) → introduced new `layer1_swappable` role.
   - Caught regression in `test_gemini_locked_roles_set_is_correct` (assumed gemini-prefix == locked) → updated assertion to be explicit.

2. **`evaluator_agent` and `synthesis_agent` mapped to `layer1_swappable`** even though they're Layer-2. Per researcher their structured-output schemas degrade gracefully on Claude (ClaudeClient injects schema as system prompt at `llm_client.py:748-759`).

3. **Combined two sub-cycles into one experiment_results** — backend (22.1) + frontend (22.2) shipped together because the contract referenced both. The masterplan will list them as 22.1 + 22.2 separately for tracking but the harness ran them as a single iteration.

4. **Now toggle "Apply to all agents" + restart** — the agent map will show your Standard model on 21 of the 28 Layer-1 skills + RAGAgent will show LOCKED badge + 4 nodes will show SEARCH badge.

## Closes

UAT-22.1 + UAT-22.2.

## Next

Spawn Q/A. After PASS: log + flip + archive + commit + push.
