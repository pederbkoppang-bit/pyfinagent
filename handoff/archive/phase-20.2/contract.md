---
step: phase-20.2
title: AgentMap workflow overlay rendering (static topology vs production flow toggle)
cycle_date: 2026-04-26
harness_required: true
verification: cd frontend && npx tsc --noEmit && npm run build
research_brief: handoff/current/phase-20.1-research-brief.md
---

# Contract -- phase-20.2

## Step ID

`phase-20.2` -- "AgentMap workflow overlay rendering". Frontend half of
operator's "show production workflow on agent map" request. Reuses
phase-20.1's brief (same scope; data + UI together).

## Hypothesis

Adding `workflowMode` toggle + alternative edge-rendering path lets the
operator switch between static topology (existing) and the directional
production daily-cycle (new) without re-mounting the component or
fetching different data.

## Immutable success criteria

```
cd frontend && npx tsc --noEmit && npm run build
```

## Plan steps

1. `frontend/src/lib/api.ts`: add `AgentMapWorkflowStep` + `AgentMapWorkflowEdge` types; add optional `workflow_steps` + `workflow_edges` fields to `AgentMapResponse`.
2. `frontend/src/components/AgentMap.tsx`:
   - Add `inWorkflow` field to `AgentNodeData`.
   - Add `workflowMode` to `BuildArgs`; in workflow mode, render `data.workflow_edges` instead of `data.edges` with: step number labels, animated style for forward edges, dashed orange `step` type for `loop: true` edges, cyan stroke.
   - In `AgentNode`, add cyan ring-2 highlight when `data.inWorkflow` is true.
   - Add `workflowMode` state + toolbar button toggling it.
3. `npx tsc --noEmit` + `npm run build` clean.

## Out of scope

- Live cycle telemetry (websocket-driven highlighting of active step) -- separate cycle
- Phase-21 settings model propagation
