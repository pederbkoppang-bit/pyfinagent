---
step: phase-18.1
title: Build agent inventory JSON + GET /api/agent-map endpoint
cycle_date: 2026-04-26
harness_required: true
verification: source .venv/bin/activate && python -m pytest backend/tests/test_agent_map_inventory.py -v
research_brief: handoff/current/phase-18.1-research-brief.md
---

# Contract -- phase-18.1

## Step ID

`phase-18.1` -- "Build agent inventory JSON + GET /api/agent-map
endpoint". First implementation cycle of the agent-map series.

## Research-gate summary

Internal-only brief at `handoff/current/phase-18.1-research-brief.md`,
building on phase-18.0's external research (7 in-full sources).
gate_passed=true.

Decisive findings:
- ~54 actual nodes (Layer 1=30 skills, Layer 2=7, Layer 3=3, services/meta=14)
- JSON schema: `{version, generated_at, nodes[], edges[]}`
- Each node: `{id, name, layer, model, provider, role, file, parents, children, kind}`
- New endpoint at `backend/api/agent_map.py` (separate file)
- 8 pytest cases covering schema + parent-child consistency + endpoint round-trip

## Hypothesis

A static `backend/agents/_inventory.json` (~54 nodes) + a thin
`GET /api/agent-map` FastAPI route that serves it gives the frontend
a stable data contract. Static JSON is correct here -- agent topology
changes infrequently and doesn't need DB persistence.

## Immutable success criteria

```
verification: source .venv/bin/activate && python -m pytest backend/tests/test_agent_map_inventory.py -v
```

## Plan steps

1. Create `backend/agents/_inventory.json` (~54 nodes per schema in research brief).
2. Create `backend/api/agent_map.py` with `GET /api/agent-map` returning the inventory.
3. Wire the router into `backend/main.py` (single line `app.include_router(...)`).
4. Create `backend/tests/test_agent_map_inventory.py` with 8 tests per research plan.
5. Run immutable verification.

## References

- `handoff/current/phase-18.1-research-brief.md`
- `handoff/archive/phase-18.0/` (the planning cycle that established framework + node schema)
- `backend/main.py` (router include pattern)
- `backend/agents/skills/` (30 .md files modulo template + experiments)

## Out of scope

- Frontend visualization (18.2+)
- Authentication on the endpoint (read-only metadata, low risk)
- Live introspection (file-system scan to auto-generate the JSON; defer to follow-up CI check)
- Edge labels (defer to 18.3)
