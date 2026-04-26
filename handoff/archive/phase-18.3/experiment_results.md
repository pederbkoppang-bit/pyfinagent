---
step: phase-18.2
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/package.json (added @xyflow/react ^12.10.2 + dagre ^0.8.5 + @types/dagre ^0.7.54)
  - frontend/src/components/AgentMap.tsx (NEW, ~165 LOC)
  - frontend/src/lib/icons.ts (added Graph re-export)
---

# Experiment Results -- phase-18.2

## What was done

Installed @xyflow/react v12 + dagre, scaffolded AgentMap component
with mock data (3 nodes), dagre TB layout, dark theme, custom AgentNode
component with provider-color borders + dashed-for-harness convention.

## Deliverables

### `frontend/src/components/AgentMap.tsx` (NEW, ~165 LOC)

- `AgentNodeData` interface: `{name, model, provider, kind, layer, role?}`
- `PROVIDER_COLORS` map: anthropic=sky, google=emerald, openai=violet, github_models=amber, none=slate
- `KIND_ICON` map: harness/in_app=Brain, skill=MagnifyingGlass, service/meta_evolution=ShieldCheck
- `AgentNode` custom React Flow node component with:
  - Phosphor icon (size 18, duotone)
  - Name + model badge (font-mono text-xs)
  - Border: dashed for `kind="harness"`, solid otherwise
  - data-testid + data-kind for future e2e tests
- `layoutWithDagre()` -- uses `dagre.graphlib.Graph` with TB direction, `nodesep=60`, `ranksep=80`
- `MOCK_NODES` (3) + `MOCK_EDGES` (2) -- main → researcher, main → qa
- `AgentMap({ nodes, edges })` prop interface for 18.3 to swap in real data
- Renders ReactFlow with `colorMode="dark"`, `fitView`, hidden attribution, custom Background, bottom-right Controls

### `frontend/src/lib/icons.ts` (1-line add)

Added `Graph as Graph` direct re-export for sidebar nav icon (used in 18.4).

### `frontend/package.json`

```
"@xyflow/react": "^12.10.2",
"dagre": "^0.8.5",
"@types/dagre": "^0.7.54",
```

## Verification (verbatim, immutable from masterplan)

```
$ cd frontend && npm run build
... [Next.js compile, type-check, route generation] ...
ƒ Middleware                             85.3 kB
○  (Static)   prerendered as static content
ƒ  (Dynamic)  server-rendered on demand
```

Exit 0. All routes built; no TS errors.

## Files touched

| Path | Action |
|------|--------|
| `frontend/src/components/AgentMap.tsx` | CREATED |
| `frontend/src/lib/icons.ts` | edit (Graph export) |
| `frontend/package.json` + lock | edit (3 new deps) |
| `handoff/current/contract.md` | rewrite |
| `handoff/current/experiment_results.md` | rewrite (this) |
| `handoff/current/phase-18.2-research-brief.md` | created |

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | @xyflow/react + dagre installed | PASS |
| 2 | AgentMap.tsx exists with mock data | PASS |
| 3 | Dagre TB layout memoized | PASS (useMemo) |
| 4 | Dark theme via colorMode="dark" | PASS |
| 5 | Custom AgentNode with model badge | PASS |
| 6 | Solid/dashed border by kind | PASS |
| 7 | npm run build exits 0 | PASS |

## Honest disclosures

1. **3-node mock for smoke test only.** Real ~52-node data lands in 18.3.
2. **No expand/collapse, no filters, no page route** -- those land in 18.3 + 18.4.
3. **No tests.** Pure-UI scaffold; verification is `npm run build`. Component test framework would be vitest + RTL — defer to a follow-up if needed.
4. **Cycle-2 not needed.** First-pass clean.

## Closes

UAT-18.2. Masterplan step phase-18.2.

## Next

Spawn Q/A. After PASS: log + flip + archive. Then 18.3 (real data + expand/collapse).
