# Contract — phase-80.3

**Step id:** `80.3` (phase-80, **P0**, `harness_required: true`) — *[P0 — WHOLE PAGE
NON-FUNCTIONAL] `/agent-map` renders ZERO edges and a broken layout.*
Date 2026-07-25. Wave 2. **Tier T3** (Opus 5 `xhigh`) — a P0, but frontend-only with no
money path.

## 1. Research gate — PASSED

`handoff/current/research_brief_80.3.md`: `gate_passed: true`, **9** sources read in full,
22 URLs, recency scan, 11 internal files. Installed: **`@xyflow/react` 12.10.2** (measured
in `node_modules`, not inferred from `package.json`).

**The three defects, each with a mechanism proven from source rather than guessed:**

**(A) Zero edges.** Custom nodes must render their own handles — built-ins ship them,
custom ones do not. `AgentNode` (`:82-141`) renders none and `Handle` is not imported
(`:19-27`), so `getEdgePosition` returns `null` for every edge
(`system/index.js:1378-1385`) and drops it. **The `"null"` in error 008 is just the un-set
`sourceHandle` interpolated into the message — it is NOT bad edge data**, so no
edge-object change is needed. Confirmed independently: `grep -n 'Handle'
src/components/AgentMap.tsx` → no matches.

**(B) Clipping — and the header is not lying.** Measured by re-running `buildGraph` +
dagre against the live API: **29 nodes / 24 edges / a 4238×490px graph**. `fitView` wants
zoom **~0.23** (measured 0.2301), but **`minZoom` defaults to 0.5** and `clamp()` raises it — so 2119px is
rendered into a ~1120px canvas and clips off both sides. **The "29 of 58" header is
CORRECT** (58 nodes total, 29 layer-1 children hidden). So (B) is **one clipping defect,
not a count/filter bug** — which is the opposite of what the step text implies.

> **This is the finding that changes the fix.** Lowering `fitViewOptions.minZoom` alone
> would fit correctly and then **snap back to 0.5 on the operator's first scroll**,
> because d3's `scaleExtent` re-clamps from the prop. The **`minZoom` PROP** must be
> lowered.

**(C) Blank on resize.** `fitView` (the boolean prop) is **one-shot on mount**. React
Flow's container `ResizeObserver` only records width/height and never re-fits.
`useReactFlow()` must be called from a **CHILD** of `<ReactFlow>` or it throws error 001.

## 2. Immutable success criteria — VERBATIM from `.claude/masterplan.json`

> 1. AgentNode renders both a <Handle type="target"> and a <Handle type="source">, and Handle is imported from @xyflow/react
> 2. A fresh load of /agent-map produces ZERO '[React Flow]' console warnings (browser_console_messages level=warning returns none from reactflow)
> 3. Edges are visibly drawn between nodes in a Playwright screenshot -- assert on the rendered graph, not just on the absence of warnings
> 4. All nodes are inside the canvas bounds on a fresh 1440x900 load (nothing clipped at either horizontal edge), and the node count shown matches the header count
> 5. After a browser_resize the graph re-fits and still renders nodes+edges (the blank-canvas-after-resize case is explicitly re-tested)

**Immutable verification command:**
`cd frontend && npx tsc --noEmit -p tsconfig.json && grep -n 'Handle' src/components/AgentMap.tsx`

**live_check:** `handoff/current/live_check_80.3.md` — Playwright screenshots BEFORE/AFTER
at 1440×900 showing edges drawn and no clipping, plus verbatim `browser_console_messages`
showing 0 React Flow warnings, plus one capture after a resize.

## 3. Plan

1. **(A)** Add `Handle` to the `@xyflow/react` import and render two **unnamed** handles
   inside `AgentNode`'s root div, matching the dagre **TB** direction already set at
   `:165-166`: `<Handle type="target" position={Position.Top} />` and
   `<Handle type="source" position={Position.Bottom} />`. `Position` is already imported.
   No `updateNodeInternals` needed — the handles are static, not added programmatically.
   Style them unobtrusively **without** `display:none` (a hidden handle still needs
   layout to bind edges); verify edges still attach.
2. **(B)** Add **`minZoom={0.1}`** *and* `fitViewOptions={{ padding: 0.15 }}` to
   `<ReactFlow>`.
3. **(C)** Render a small child component **inside** `<ReactFlow>` that calls
   `useReactFlow()`, attaches a `ResizeObserver` to the flow container, and calls
   `fitView()` from a `requestAnimationFrame` callback. Must not loop (a re-fit changes
   the transform, not the container size).
4. **Tests.** Per the gate, **Playwright IS the test** for criteria 3/4/5 — jsdom cannot
   assert drawn edges without mocking `DOMMatrixReadOnly`, `offsetWidth/Height`,
   `SVGElement.getBBox` and a firing `ResizeObserver`, and would then be asserting on
   mocked geometry. So:
   - one **narrow vitest guard** that `AgentNode` renders two `.react-flow__handle`
     elements (mutation-resistant against the handles being deleted again) — this does
     not need measurement;
   - **Playwright** for edges drawn, no clipping, no warnings, and the resize case.
5. **Mutation-test** both: remove the handles → the vitest guard must fail; and confirm
   the guard cannot pass on a do-nothing render.

## 4. Explicitly OUT of scope — queued

**Readable node text at the default view.** At the required ~0.23 zoom the nodes are ~51px
wide, so the graph will read as *structure*, not as *text*; the `<Controls>` let the
operator zoom in. Criterion 4 ("all nodes inside the canvas bounds") is fully satisfied by
the zoom fix, but **a 29-node dagre rank at 220px each cannot be legible in 1120px** — that
is a layout problem (rank splitting, collapsing, or a different direction) and belongs in
its own step rather than being smuggled in here.

**An honest limit on (C), carried from the gate:** the retained-transform mechanism is
proven from source and fully explains a *mis-fit* after resize; combined with a graph
already ~2× the canvas width it also plausibly explains a *fully blank* canvas. But the
exact blank trigger was **not** isolated from source. A second candidate: `updateDimensions`
returns early when `checkVisibility?.()` is false and substitutes a 500×500 fallback when a
dimension measures zero (error 004) — and the filter bar at `:332` is `flex-wrap`, so
narrowing the window re-wraps it and changes canvas height mid-resize. The fix cures both
mechanisms, but **criterion 5 must be VERIFIED LIVE, not argued.**

## 5. DO-NO-HARM

- **Frontend-only, confirmed.** `AgentMap`'s only consumers are
  `app/agent-map/page.tsx:29` and the fetcher in `lib/api.ts:832-880`.
  `backend/api/agent_map.py` is read-only over a static JSON inventory and is not touched.
  Nothing here touches a trading, money or backend path.
- No `.env` edit, no flag, no optimizer run, `historical_macro` FROZEN;
  kill-switch/stops/sector-caps/DSR/PBO untouched.
- **`.claude/rules/frontend.md` binding:** Phosphor icons only (no emoji), navy/slate
  palette (never zinc), JIT-safe literal class strings.
- **The resize handler must not loop.** A `fitView()` changes the viewport transform, not
  the container size, so the observer must not re-trigger itself — verify live.
- Playwright on the isolated skip-auth `:3100` rig only; never the operator's `:3000`.
  Restore `tsconfig.json` + `next-env.d.ts` afterwards.
- `git add -An` before the flip.

## 6. Evidence

`experiment_results_80.3.md` · `live_check_80.3.md` (+ `captures_80.3/`) ·
`evaluator_critique_80.3.md` · `harness_log.md` append **before** the flip.
