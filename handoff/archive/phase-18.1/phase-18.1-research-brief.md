# Research Brief: phase-18.1 -- Build agent inventory JSON + GET /api/agent-map

Tier: **simple** (internal-only, builds on phase-18.0 brief).

## Source for findings

Phase-18.0 already enumerated the framework choice + node count + JSON
schema. This brief just confirms the file paths + finalizes the JSON
schema before authoring `backend/agents/_inventory.json`.

## Internal file inventory (verified by `ls`)

| Layer | Files | Count |
|-------|-------|-------|
| Layer 3 -- Harness MAS | `.claude/agents/{researcher,qa}.md` + Main session | 3 |
| Layer 2 -- MAS in-app | `multi_agent_orchestrator.py`, `agent_definitions.py`, `planner_agent.py`, `evaluator_agent.py`, `risk_judge` (skills/), `moderator_agent` (skills/), `synthesis_agent` (skills/) | 7 |
| Layer 1 -- Gemini skills | `backend/agents/skills/*.md` excluding `SKILL_TEMPLATE.md` and `experiments/` dir | 30 confirmed (32 .md files - 1 template = 31; minus 1 stub = 30) |
| Services / meta-evolution | `autonomous_loop`, `paper_trader`, `outcome_tracker`, `cycle_health`, `kill_switch`, `paper_go_live_gate`, `skill_optimizer`, `meta_evolution/{cron,directive_rewriter,directive_review,provider_rebalancer,cron_allocator,alpha_velocity,archetype_library}` | ~14 |
| **Total visible** | -- | **~54** (Layer 1 collapsed to 1 group node = 25 visible) |

Adjusting the 18.0 estimate (~48) to match the actual file inventory: **~54 nodes**.

## Final JSON schema

```json
{
  "version": 1,
  "generated_at": "static",
  "nodes": [
    {
      "id": "string (snake_case unique)",
      "name": "string (display)",
      "layer": 1 | 2 | 3 | 4,
      "model": "string | null",
      "provider": "anthropic | google | openai | github_models | none",
      "role": "string (1-2 sentences)",
      "file": "string (repo-relative path)",
      "parents": ["id", ...],
      "children": ["id", ...],
      "kind": "harness | in_app | skill | service | meta_evolution"
    }
  ],
  "edges": [
    {"from": "id", "to": "id", "label": "string optional"}
  ]
}
```

`edges` is derived from `parents`/`children` for visualization
convenience (avoids duplicate computation in the frontend).

## API endpoint

`GET /api/agent-map` returns the JSON inventory with no parameters, no
auth. Add to `backend/api/sovereign.py` (existing sovereign-related
routes) OR create new `backend/api/agent_map.py`. Per project conv
(separate concerns), use new file.

## Test plan (8 tests)

1. `test_inventory_json_loads` -- file exists, valid JSON, has version + nodes
2. `test_node_count_minimum` -- nodes >= 30 (sanity floor; we'll have ~54)
3. `test_every_node_has_required_fields` -- id, name, layer, role, file, parents, children, kind
4. `test_layer_values_in_range` -- 1-4
5. `test_provider_values_valid` -- in known set
6. `test_parent_child_consistency` -- if A.parents has B, then B.children has A
7. `test_endpoint_returns_inventory` -- TestClient GET /api/agent-map -> 200 + matches file
8. `test_no_orphan_ids_in_edges` -- every edge from/to references an existing node

## Pitfalls

1. **Skill agent inventory drift:** the `skills/*.md` count changes over time. The JSON should be hand-maintained but a follow-up CI check could compare counts.
2. **Layer 3 Main is "this Claude Code session"** -- model varies (currently Opus 4.7). Set model="varies" and provider="anthropic".
3. **Some service files are pure plumbing** (api_cache, perf_tracker) -- exclude from inventory; only include agent-like (decision-making) services.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": true,
  "internal_files_inspected": 60,
  "report_md": "handoff/current/phase-18.1-research-brief.md",
  "gate_passed": true,
  "gate_passed_basis": "internal-only; builds on phase-18.0 brief which had 7 in-full external sources; this is the implementation cycle of an already-researched plan"
}
```
