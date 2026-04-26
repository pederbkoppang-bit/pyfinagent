# Research Brief: phase-20.1 -- Production workflow data in inventory

Tier: simple, internal-only.

## Source of truth

`backend/services/autonomous_loop.py:53-90` -- `run_daily_cycle()` docstring + step structure.

8 production steps (verbatim from docstring):
1. Screen universe (free)
2. Analyze top candidates (lite mode) -- triggers Layer 1 (28 enrichments) + Layer 2 (synthesis/debate)
3. Re-evaluate holdings due for refresh
4. Decide trades -- planner_agent + analyst_agent + risk_judge_skill
5. Execute trades -- portfolio_manager -> paper_trader -> execution_router -> alpaca_broker
6. Mark to market
7. Save snapshot
8. Learn from closed trades -- outcome_tracker (LLM reflection)

Then loop back to step 1 the next day (cron-driven).

## Data shape

Add two top-level keys to `backend/agents/_inventory.json`:

```json
{
  "version": 2,        // bump from 1
  "workflow_steps": [
    {"step": 1, "name": "Screen universe", "agent_id": "autonomous_loop", "kind": "free"},
    {"step": 2, "name": "Analyze candidates", "agent_id": "layer1_pipeline", "kind": "llm"},
    {"step": 3, "name": "Re-eval holdings", "agent_id": "layer1_pipeline", "kind": "llm"},
    {"step": 4, "name": "Decide trades", "agent_id": "planner_agent", "kind": "llm"},
    {"step": 5, "name": "Execute trades", "agent_id": "portfolio_manager", "kind": "trade"},
    {"step": 6, "name": "Mark to market", "agent_id": "paper_trader", "kind": "data"},
    {"step": 7, "name": "Save snapshot", "agent_id": "paper_trader", "kind": "data"},
    {"step": 8, "name": "Learn", "agent_id": "outcome_tracker", "kind": "llm"}
  ],
  "workflow_edges": [
    {"from": "autonomous_loop", "to": "layer1_pipeline", "step": 2, "label": "screen->analyze"},
    {"from": "layer1_pipeline", "to": "synthesis_agent", "step": 2.5},
    {"from": "synthesis_agent", "to": "moderator_agent", "step": 2.7, "label": "debate"},
    {"from": "moderator_agent", "to": "analyst_agent", "step": 2.9},
    {"from": "analyst_agent", "to": "risk_judge_skill", "step": 4},
    {"from": "risk_judge_skill", "to": "planner_agent", "step": 4.5},
    {"from": "planner_agent", "to": "portfolio_manager", "step": 5},
    {"from": "portfolio_manager", "to": "paper_trader", "step": 5.5},
    {"from": "paper_trader", "to": "execution_router", "step": 5.7},
    {"from": "execution_router", "to": "alpaca_broker", "step": 5.9},
    {"from": "paper_trader", "to": "outcome_tracker", "step": 8, "label": "snapshot+learn"},
    {"from": "outcome_tracker", "to": "autonomous_loop", "step": 9, "label": "loop back tomorrow", "loop": true}
  ]
}
```

`step` field is float so 2.5 / 2.7 etc. order sub-steps. `loop: true` marks the daily cycle-back arrow.

## Test plan

Extend `tests/agents/test_agent_map_inventory.py` (or create `test_workflow.py`):
- `test_inventory_has_workflow_v2` -- version=2, both keys present
- `test_workflow_steps_8` -- 8 numbered steps
- `test_workflow_edges_reference_existing_nodes` -- every workflow_edge from/to is in nodes
- `test_workflow_has_loop_back` -- at least one edge has loop=true
- `test_workflow_steps_unique_ints` -- step ints 1-8 unique

## Plan

1. Bump version to 2 in `backend/agents/_inventory.json`
2. Add `workflow_steps` (8 entries) + `workflow_edges` (12 entries) per shape above
3. Update `backend/api/agent_map.py` to pass workflow_* fields through (it already returns the whole `data` dict, so no change needed -- verify)
4. Extend `backend/tests/test_agent_map_inventory.py` with 5 new tests
5. Verify: `python -m pytest backend/tests/test_agent_map_inventory.py -v`

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "internal_files_inspected": 3,
  "gate_passed": true,
  "gate_passed_basis": "internal-only; autonomous_loop.py docstring is authoritative source for the production cycle"
}
```
