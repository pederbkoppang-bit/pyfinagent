---
step: phase-20.1
title: Add production workflow data (workflow_steps + workflow_edges) to agent inventory
cycle_date: 2026-04-26
harness_required: true
verification: source .venv/bin/activate && python -m pytest backend/tests/test_agent_map_inventory.py -v
research_brief: handoff/current/phase-20.1-research-brief.md
---

# Contract -- phase-20.1

## Step ID

`phase-20.1` -- "Add production workflow data to inventory JSON".
First of 5 cycles delivering operator's "show production workflow on
agent map" + "settings model propagation" requests (acked).

## Research-gate summary

Internal-only brief. Source of truth: `backend/services/autonomous_loop.py:53-90`
`run_daily_cycle()` docstring. 8 production steps + cron-driven loop.

## Hypothesis

Adding `workflow_steps` + `workflow_edges` arrays alongside the existing
`nodes` + edge-derivation gives the frontend (phase-20.2) the data it
needs to render a directional workflow overlay without changing the
existing topology view. Backend stays pure-data.

## Immutable success criteria

```
source .venv/bin/activate && python -m pytest backend/tests/test_agent_map_inventory.py -v
```

## Plan steps

1. Update `backend/agents/_inventory.json`:
   - Bump `version` 1 -> 2
   - Add `workflow_steps` (8 entries per autonomous_loop docstring)
   - Add `workflow_edges` (12 entries with `step` ordering + `loop: true` on the cycle-back)
2. Confirm `backend/api/agent_map.py` passes the new fields through (it already returns the full `data` dict).
3. Extend `backend/tests/test_agent_map_inventory.py` with 5 new tests:
   - `test_inventory_version_2`
   - `test_workflow_steps_present`
   - `test_workflow_edges_reference_existing_nodes`
   - `test_workflow_has_loop_back`
   - `test_workflow_step_numbers_in_range`
4. Verify all (existing 11 + new 5 = 16 total) pass.

## References

- `backend/services/autonomous_loop.py:53-90` -- canonical 8-step daily flow
- `backend/agents/_inventory.json` -- existing schema (52 nodes)
- `backend/api/agent_map.py` -- the `_derive_edges` helper unchanged

## Out of scope

- Frontend rendering (phase-20.2)
- Live cycle telemetry (would need WebSocket)
- Settings model propagation (phase-21.x)
