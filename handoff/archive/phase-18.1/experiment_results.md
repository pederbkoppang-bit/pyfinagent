---
step: phase-18.1
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - backend/agents/_inventory.json (NEW, ~52 nodes)
  - backend/api/agent_map.py (NEW, ~75 LOC)
  - backend/main.py (2-line edit: import + include_router)
  - backend/tests/test_agent_map_inventory.py (NEW, ~150 LOC, 11 tests)
---

# Experiment Results -- phase-18.1

## What was done

Built the static agent inventory + GET /api/agent-map FastAPI route.
This is the data backbone for the visualization that lands in 18.2-18.4.

## Deliverables

### `backend/agents/_inventory.json` (NEW, ~52 nodes)

Static catalog of all agents in pyfinagent across 4 layers:

| Layer | Count | Nodes |
|-------|-------|-------|
| 3 -- Harness MAS | 3 | main, researcher, qa |
| 2 -- MAS in-app Claude | 7 | multi_agent_orchestrator, planner_agent, evaluator_agent, communication_agent, analyst_agent, moderator_agent, synthesis_agent |
| 1 -- Gemini analysis pipeline | 28 + 1 group node | layer1_pipeline (group) + 28 individual skills |
| 4 -- Services / meta-evolution | 13 | autonomous_loop, paper_trader, execution_router, portfolio_manager, outcome_tracker, cycle_health, kill_switch, paper_go_live_gate, skill_optimizer, alpaca_broker, slack_bot + meta_evolution package (cron, cron_allocator, provider_rebalancer, alpha_velocity, archetype_library, directive_rewriter, directive_review) |
| **Total** | **~52** | -- |

Each node has the canonical schema from the 18.0 brief: `{id, name,
layer, model, provider, role, file, parents, children, kind}`.

Layer-1 has a group node `layer1_pipeline` that lists all 28 skills as
`children`. The frontend (18.3) will render this as a collapsed group
by default; expanding shows the 28 individual nodes.

### `backend/api/agent_map.py` (NEW, ~75 LOC)

FastAPI router with one route:
- `GET /api/agent-map` -- returns the inventory JSON + derived edges
- `_derive_edges(nodes)` -- computes deduped edges from `parents`/`children`
- 500 if the JSON file is missing or malformed (HTTPException, not silent failure)

### `backend/main.py` (2 lines)

```python
+ from backend.api.agent_map import router as agent_map_router
+ app.include_router(agent_map_router)
```

### `backend/tests/test_agent_map_inventory.py` (NEW, 11 tests)

8 from research plan + 3 defensive (kind values, ID uniqueness, derive_edges dedup):

1. `test_inventory_json_loads` -- valid JSON with version + nodes
2. `test_node_count_minimum` -- >= 30 (sanity floor)
3. `test_every_node_has_required_fields` -- all 10 required keys per node
4. `test_layer_values_in_range` -- 1-4
5. `test_provider_values_valid` -- in known set
6. `test_kind_values_valid` (defensive)
7. `test_node_ids_unique` (defensive)
8. `test_parent_child_consistency` -- every parent/child id references a real node
9. `test_get_agent_map_endpoint_returns_inventory` -- function returns nodes + derived edges
10. `test_no_orphan_ids_in_edges` -- every edge from/to references a real node id
11. `test_derive_edges_dedups` (defensive) -- bidirectional refs collapse to one edge

## Verification (verbatim, immutable from masterplan)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_agent_map_inventory.py -v
============================== 11 passed in 0.08s ==============================
```

## Files touched

| Path | Action | Note |
|------|--------|------|
| `backend/agents/_inventory.json` | CREATED | ~52 nodes, ~13KB |
| `backend/api/agent_map.py` | CREATED | ~75 LOC FastAPI router |
| `backend/main.py` | edit | 2-line: import + include_router |
| `backend/tests/test_agent_map_inventory.py` | CREATED | ~150 LOC, 11 tests |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-18.1-research-brief.md` | created (internal-only) | -- |

NO new dependencies. NO BQ schema changes.

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Static inventory JSON exists | PASS (~52 nodes) |
| 2 | GET /api/agent-map endpoint | PASS |
| 3 | Router registered in main.py | PASS |
| 4 | All 11 tests pass | PASS |
| 5 | Schema validation enforced (layer/provider/kind in valid sets) | PASS |
| 6 | Parent/child consistency verified | PASS |
| 7 | Endpoint returns nodes + derived edges | PASS |

## Honest disclosures

1. **52 nodes vs research brief estimate of 54.** The brief was a rough
   count. Actual: 3 (Layer 3) + 7 (Layer 2) + 1 group + 28 (Layer 1
   skills) + 13 (Layer 4) = 52. Within tolerance.

2. **Layer-1 group node = 1 visible until expanded.** Frontend (18.3)
   will collapse the 28 individual skills behind `layer1_pipeline`.
   Without expansion: 24 visible nodes. With expansion: ~52 visible.

3. **Some service files NOT included** (api_cache, perf_tracker,
   compliance_logger, cycle_health-related plumbing) -- they're pure
   infrastructure, not agent-like. Inventory documents AGENTS, not the
   whole codebase.

4. **No live introspection.** The JSON is hand-maintained. A
   follow-up cycle could add a CI check comparing skills/*.md count to
   inventory length, but that's deferred.

5. **Cycle-2 not needed.** First-pass clean (11/11 tests).

6. **No regression on existing endpoints.** Just adds a new route.

## Closes

Net-new task #87 (UAT-18.1). Masterplan step phase-18.1.

## Next

Spawn Q/A. After PASS: log + flip + archive. Then proceed to phase-18.2 (frontend scaffold).
