# Research Brief: phase-18.2 -- Scaffold AgentMap component

Tier: **simple** (internal-only, builds on phase-18.0 framework choice).

## Source

phase-18.0 chose `@xyflow/react` v12 + `dagre`. This cycle installs them
+ scaffolds the component.

## Plan

1. `npm install @xyflow/react dagre @types/dagre` in `frontend/`
2. Create `frontend/src/components/AgentMap.tsx` with:
   - Custom node component showing icon + name + model badge
   - Dark theme (`colorMode="dark"` prop)
   - Dagre TB (top-to-bottom) layout
   - Mock data (3 nodes for smoke test) -- real data wired in 18.3
   - `data-testid="agent-map"` for future e2e tests
3. Verify `cd frontend && npm run build` exits 0

## Pitfalls

- Import from `@xyflow/react`, NOT deprecated `react-flow-renderer`
- React Flow needs explicit `width` + `height` on parent wrapper
- Dagre layout must be called once (or memoized) -- not on every render
- Phosphor icons must come from `@/lib/icons` per ESLint rule

## JSON envelope

```json
{"tier": "simple", "external_sources_read_in_full": 0, "internal_files_inspected": 4, "gate_passed": true, "gate_passed_basis": "phase-18.0 has the 7 in-full external sources"}
```
