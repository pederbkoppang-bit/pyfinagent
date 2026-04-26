---
step: phase-18.2
title: Scaffold AgentMap component (React Flow + dagre, mock data)
cycle_date: 2026-04-26
harness_required: true
verification: cd frontend && npm run build
research_brief: handoff/current/phase-18.2-research-brief.md
---

# Contract -- phase-18.2

## Step ID

`phase-18.2` -- "Scaffold AgentMap component". Installs `@xyflow/react` + `dagre`, creates the component with mock data + dark theme + dagre TB layout. Real data wiring + expand/collapse defer to 18.3.

## Research-gate summary

Internal-only brief. Builds on phase-18.0's framework choice. gate_passed=true.

## Hypothesis

A scaffolded `AgentMap.tsx` with mock data + dagre layout + dark theme proves the rendering chain works end-to-end before wiring real data. Avoids debugging two unknowns at once.

## Immutable success criteria

```
verification: cd frontend && npm run build
```

## Plan steps

1. `cd frontend && npm install @xyflow/react dagre @types/dagre`
2. Create `frontend/src/components/AgentMap.tsx`:
   - 3-node mock data
   - Custom AgentNode component with icon + name + model badge
   - Dagre TB layout (memoized via `useMemo`)
   - Dark theme via React Flow's `colorMode="dark"`
   - 600px x 400px wrapper for the smoke test
3. Verify `npm run build` exits 0

## References

- phase-18.0 brief (framework + schema)
- @xyflow/react v12 docs
- dagre docs

## Out of scope

- Real data fetch (18.3)
- Expand/collapse (18.3)
- Filter controls (18.3)
- Page route (18.4)
