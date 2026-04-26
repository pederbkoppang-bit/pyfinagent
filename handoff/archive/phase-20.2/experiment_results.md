---
step: phase-20.1
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - backend/agents/_inventory.json (v1 -> v2; +workflow_steps[8] +workflow_edges[12])
  - backend/tests/test_agent_map_inventory.py (+5 workflow tests; 11 -> 16 total)
---

# Experiment Results -- phase-20.1

Schema v2: added `workflow_steps` (8) + `workflow_edges` (12, one with `loop: true`)
to `backend/agents/_inventory.json` describing the production daily cycle from
`autonomous_loop.run_daily_cycle()` docstring.

## Verification

```
$ python -m pytest backend/tests/test_agent_map_inventory.py -v
============================== 16 passed in 0.07s ==============================
```

11 existing + 5 new workflow tests all pass.

## Files

- `backend/agents/_inventory.json` -- v=2, +21 lines (workflow_* keys before nodes[])
- `backend/tests/test_agent_map_inventory.py` -- +5 tests at end of file
- `handoff/current/{contract,experiment_results,phase-20.1-research-brief}.md`

NO API changes (`agent_map.py` already returns the full data dict so workflow_*
flow through unchanged), NO frontend changes (phase-20.2 scope), NO new deps.

## Honest disclosures

1. Skipped researcher subagent spawn -- the autonomous_loop.run_daily_cycle docstring is the canonical source and was already cached in my context. Internal-only brief documents this.
2. Cycle-2 not needed; first-pass clean.

## Closes

UAT-20.1. Next: phase-20.2 (AgentMap workflow overlay rendering).
