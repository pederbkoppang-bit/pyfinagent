---
step: phase-18.0
title: Plan agent-map work -- add masterplan steps 18.1-18.4
cycle_date: 2026-04-26
harness_required: true
verification: 'python3 -c "import json; m=json.load(open(''.claude/masterplan.json'')); ids=[s.get(''id'') for p in m[''phases''] for s in p.get(''steps'',[])]; assert ''18.1'' in ids and ''18.2'' in ids and ''18.3'' in ids and ''18.4'' in ids; print(''ok'')"'
research_brief: handoff/current/phase-18.0-research-brief.md
---

# Contract -- phase-18.0

## Step ID

`phase-18.0` -- "Plan agent-map work: inventory agents + propose
visualization framework + draft sub-step list". Net-new phase-18 in
masterplan. The deliverable of this cycle is the *new masterplan step
entries* (phase-18.1, 18.2, 18.3, 18.4) appended to
`.claude/masterplan.json` -- NOT the visualization itself.

User's two-phase instruction: (1) plan the steps first, (2) execute
each subsequent step under full harness.

## Research-gate summary

Spawned `researcher` (moderate tier). Brief at
`handoff/current/phase-18.0-research-brief.md`. Gate: 7 external
sources read in full (React Flow docs + dagre layout example, Anthropic
multi-agent research, arXiv 2502.02533 topology patterns, Kibana
production POC, MAS-pattern blogs), 15 URLs, recency scan, 12 internal
files inspected. `gate_passed: true`.

Decisive findings:
- Total agent count: ~48 (Layer 3 harness: 3, Layer 2 in-app Claude: 7, Layer 1 Gemini skills: ~28, services/meta-evolution: ~10)
- Recommended library: `@xyflow/react` (React Flow v12) + `dagre` (~40KB) for hierarchical TB layout
- Recommended data shape: static `backend/agents/_inventory.json` served via `GET /api/agent-map`
- Layer 1 must default to COLLAPSED (28-node clutter); expand on click
- Visual semantics: dashed borders for harness/external; solid for in-app; color by provider (Claude blue, Gemini green)

## Hypothesis

Adding 4 masterplan sub-steps to phase-18 (one per concern: inventory
JSON, scaffold component, wire data, page+nav) gives the operator a
complete dependency-ordered roadmap for the agent-map feature. Each
sub-step has a concrete, immutable verification command.

## Immutable success criteria

```
verification: python3 -c "import json; m=json.load(open('.claude/masterplan.json')); ids=[s.get('id') for p in m['phases'] for s in p.get('steps',[])]; assert '18.1' in ids and '18.2' in ids and '18.3' in ids and '18.4' in ids; print('ok')"
```

After this cycle: `phase-18.1`, `18.2`, `18.3`, `18.4` exist in masterplan.json.

## Plan steps

1. Create new `phase-18` block in `.claude/masterplan.json` with 4 sub-steps:
   - **phase-18.1** "Build agent inventory JSON + GET /api/agent-map endpoint"
     - Verification: `python -m pytest backend/tests/test_agent_map_inventory.py -v`
     - depends_on: []
   - **phase-18.2** "Scaffold AgentMap component (React Flow + dagre, mock data)"
     - Verification: `cd frontend && npm run build`
     - depends_on: [18.1]
   - **phase-18.3** "Wire real data + Layer-1 expand/collapse"
     - Verification: `cd frontend && npx tsc --noEmit && npm run build`
     - depends_on: [18.2]
   - **phase-18.4** "Page + sidebar nav entry (frontend/src/app/agent-map/page.tsx)"
     - Verification: `cd frontend && npm run build`
     - depends_on: [18.3]

2. Run immutable verification (the 4 sub-step IDs must exist in JSON).

## References

- `handoff/current/phase-18.0-research-brief.md`
- `backend/agents/orchestrator.py:536-795` (Layer 1 pipeline)
- `backend/agents/agent_definitions.py:122-396` (Layer 2 configs)
- `backend/agents/multi_agent_orchestrator.py:124-220` (orchestrator + model assignments)
- `backend/agents/skills/` (~28 skill agent prompts)
- `.claude/agents/{researcher,qa}.md` (Layer 3 harness MAS)
- React Flow v12 docs + Dagre example

## Out of scope

- Any actual code (visualization, API endpoint, JSON inventory) -- those land in 18.1-18.4
- Choosing the exact node visual style (rectangles vs pills) -- left to 18.2 design discretion
- Filter UI (provider / layer dropdowns) -- 18.3 scope
- Wiring into the broader sidebar nav structure -- 18.4 scope
