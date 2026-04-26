# Research Brief: phase-18.3 -- Wire real data + Layer-1 expand/collapse + filters

Tier: simple, internal-only.

## Plan

1. Replace AgentMap mock with fetch from `/api/agent-map`
2. Add Layer-1 expand/collapse toggle: by default show only `layer1_pipeline` group node; on click, show its 28 children
3. Add provider/layer filter dropdowns
4. Re-run dagre layout when toggling expand/collapse
5. Add an api.ts function `fetchAgentMap()`

JSON envelope: tier=simple, gate_passed=true, internal-only.
