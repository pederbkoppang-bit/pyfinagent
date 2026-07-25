# Research Brief -- step 80.3 (agent-map React Flow: zero edges, bad fit, blank-on-resize)

> **CORRECTION APPENDED BY MAIN, 2026-07-26 (phase-80.3 cycle 7).** Do not rely on the
> zoom figures below without reading this. The Q/A re-derived them three independent ways
> — live `GET /api/agent-map` + dagre + the **installed** `getViewportForBounds` — and
> measured **0.2301** for the collapsed 29-node graph (4237.5x490px) and **0.1000** for
> the expanded one. The `0.220` and `0.091` in the table at `:58`, the `0.220` at `:147`
> and in the JSON envelope `summary` at `:400`, and the `~48px` node width at `:293`
> (correct value **~51px**, `220 x 0.2301 = 50.62`) are **wrong as written** — not
> superseded by a later change, simply inaccurate, in a table this brief labels
> "Measured, not inferred."
>
> They are left in place rather than edited, because this is the dated research-gate
> record and rewriting it would misrepresent what the gate produced. The corrected figures
> live in `contract_80.3.md`, `experiment_results_80.3.md` and `live_check_80.3.md`.
>
> **The gate's load-bearing conclusion is unaffected**: the required zoom is far below
> React Flow's `minZoom` default of 0.5, the fit is clamped, and the graph renders ~2119px
> into a ~1120px canvas. That is why the fix had to lower the `minZoom` **prop**.
> I originally argued these numbers were correct-when-written and later superseded; the
> Q/A showed that premise was false, and this note records the correction rather than
> quietly repairing the artifact.



Tier: **moderate**. `coverage.audit_class = false`. Accessed 2026-07-25.

**Installed version measured, not assumed:** `frontend/package.json:26` declares
`"@xyflow/react": "^12.10.2"`; `frontend/node_modules/@xyflow/react/package.json`
reports `"version": "12.10.2"`. This is **React Flow v12** (`@xyflow/react` scope),
NOT v11 (`reactflow`). All citations below are v12.

---

## Headline answers (the two the caller asked to lead with)

### 1. The exact `<Handle>` pattern for v12.10.2

Built-in node types ship handles; **custom node types do not**. Verbatim:

> "our built-in nodes include one source and one target handle, but you can
> customize your nodes with as many different handles as you need."
> -- https://reactflow.dev/learn/customization/handles (accessed 2026-07-25)

Documented minimal pattern (verbatim, same page):

```javascript
import { Position, Handle } from '@xyflow/react';

export function CustomNode() {
  return (
    <div className="custom-node">
      <div>Custom Node Content</div>
      <Handle type="source" position={Position.Top} />
      <Handle type="target" position={Position.Bottom} />
    </div>
  );
}
```

`id` is **optional for a single unnamed handle**: "For a single unnamed handle, no
`id` is required. However, when using multiple source or target handles, you need
to specify each handle with a unique `id`." Every edge in `AgentMap.tsx` is built
WITHOUT `sourceHandle`/`targetHandle` (measured -- see inventory), so **one unnamed
source + one unnamed target handle per node is exactly right and no edge object
needs to change.**

`sourcePosition`/`targetPosition` (`AgentMap.tsx:165-166`) only say WHERE a handle
would sit; they do not create one. The caller's diagnosis is correct.

For the TB layout the correct pairing is `<Handle type="target" position={Position.Top} />`
+ `<Handle type="source" position={Position.Bottom} />` -- matching `:165-166`.

### 2. Defect (B): the layout coordinates DO exceed the viewport, and `minZoom` blocks the fit

**Measured, not inferred.** I re-ran the component's own `buildGraph` +
`layoutWithDagre` logic in Node against the live API payload:

| State | nodes | edges | dagre graph size | zoom fitView WANTS | zoom it GETS | rendered width |
|---|---|---|---|---|---|---|
| Layer-1 collapsed (default) | 29 | **24** | **4238 x 490 px** | 0.220 | **0.500** (clamped) | 2119px in a ~1120px canvas |
| Layer-1 expanded | 58 | 57 | 10228 x 1050 px | 0.091 | **0.500** (clamped) | 5114px in a ~1120px canvas |

The **24 edges** in the default collapsed state matches the operator's "24 unique
edges x 5 render passes = 120 warnings" exactly -- independent confirmation the
replication is faithful.

Root cause: **`minZoom` defaults to `0.5`** and `fitView` clamps to it:

- `node_modules/@xyflow/react/dist/esm/index.js:3598` -- `ReactFlow({ ..., minZoom = 0.5, maxZoom = 2, ... })`
- `node_modules/@xyflow/react/dist/esm/index.js:3132` -- `getInitialState({ ..., minZoom = 0.5, maxZoom = 2, ... })`
- `node_modules/@xyflow/system/dist/esm/index.js:431` --
  `getViewportForBounds(bounds, width, height, options?.minZoom ?? minZoom, options?.maxZoom ?? maxZoom, options?.padding ?? 0.1)`
- `node_modules/@xyflow/system/dist/esm/index.js:747` -- `const clampedZoom = clamp(zoom, minZoom, maxZoom);`

`fitView` is working as designed: it computes 0.22, `clamp()` raises it to the 0.5
floor, the graph renders ~2x the canvas width, centred -> clipped ~500px on EACH
side. Exactly the reported symptom.

**The header count is CORRECT -- (B) is ONE defect, not two.** `AgentMap.tsx:383-385`
renders `{nodes.length} of {data.nodes.length}`, where `nodes` is the POST-filter
array actually handed to `<ReactFlow>`. The live API returns 58 nodes and
`layer1_pipeline` has 29 children, hidden while collapsed -> 58 - 29 = **29
rendered**. "29 of 58" is accurate. All 29 nodes ARE in the DOM; the operator saw
~12 because the rest are clipped outside the canvas. No filter bug, no count bug.

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
| --- | --- | --- | --- | --- |
| https://reactflow.dev/learn/customization/handles | 2026-07-25 | official doc | WebFetch (full) | "our built-in nodes include one source and one target handle"; hide handles with `visibility: hidden` or `opacity: 0`, "instead of `display: none`" |
| https://reactflow.dev/error | 2026-07-25 | official doc | WebFetch (full) | Error 008 = "Couldn't create edge for source/target handle id"; cause = "a handle with the referenced ID doesn't exist in your custom node" |
| https://reactflow.dev/learn/troubleshooting | 2026-07-25 | official doc | WebFetch (full) | "The React Flow parent container needs a width and a height to render the graph."; edges-not-displaying causes incl. "missing handles in custom nodes" |
| https://reactflow.dev/learn/advanced-use/testing | 2026-07-25 | official doc | WebFetch (full) | "React Flow needs to measure nodes in order to render edges and for that relies on rendering DOM elements"; "we recommend to use Cypress or Playwright"; Jest needs 4 mocks |
| https://reactflow.dev/learn/troubleshooting/migrate-to-v12 | 2026-07-25 | official doc | WebFetch (full) | v12 moves measured dims to `node.measured`; `width`/`height` become inline-style inputs; adds controlled `viewport` + `onViewportChange` |
| https://reactflow.dev/learn/customization/custom-nodes | 2026-07-25 | official doc | WebFetch (full) | The built-in `TextUpdaterNode` example renders NO handle; "To enable your custom node to connect with other nodes, check out the Handles page" |
| https://reactflow.dev/whats-new/2025-03-27 | 2026-07-25 | vendor release notes | WebFetch (full) | 12.5.0: "Fix fitView not working immediately after adding new nodes" (#5067); **fitView still does not auto re-run**; padding gains `'25px'` / `'10%'` / directional forms |
| https://reactflow.dev/whats-new/2024-09-26 | 2026-07-25 | vendor release notes | WebFetch (full) | 12.3.1: "#4653 Calculate viewport dimensions in `fitView` instead of using stored dimensions"; "#4670 Improve `fitView` to respect clamped node positions based on `nodeExtent`" |
| https://github.com/xyflow/xyflow/issues/4801 | 2026-07-25 | issue tracker `[ADVERSARIAL]` | WebFetch (full) | Counter-evidence that fitView is reliable: "Sometimes the `fitView()` function fails to properly fit flow nodes to the view" on 12.3.4; manual timing "is also unreliable" |
| https://reactflow.dev/api-reference/components/handle | 2026-07-25 | official doc | WebFetch (nav-only) | Props table is JS-rendered; yielded only `HandleProps` export name. Superseded by the installed `.d.ts` + bundle reads below. |

Ten unique URLs read via WebFetch; **9 yielded substantive content** (the Handle
API-reference page is nav-only -- see Pitfalls). Two further API-reference pages
(`types/fit-view-options`, `hooks/use-react-flow`) were curl-extracted and returned
navigation chrome only, consistent with the known JS-rendered-docs failure mode;
they are listed snippet-only and their content was obtained from the installed
bundle instead.

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://reactflow.dev/api-reference/types/fit-view-options | official doc | curl returned nav tree only (JS-rendered); read the installed bundle instead |
| https://reactflow.dev/api-reference/hooks/use-react-flow | official doc | same JS-render failure |
| https://reactflow.dev/api-reference/react-flow | official doc | prop tables render client-side; empty of values |
| https://github.com/xyflow/xyflow/issues/4369 | issue | "Can't create edges on react flow v12" -- duplicate of the handles finding |
| https://github.com/wbkd/react-flow/issues/2904 | issue | "Allow custom edges with undefined handles" -- v11-era feature request |
| https://github.com/xyflow/xyflow/issues/3946 | issue | `setNodes` + `fitView` uncontrolled-flow bug; not our controlled setup |
| https://github.com/xyflow/xyflow/discussions/2849 | discussion | "How to zoom out or fitView to show 1,000s of nodes" -- same minZoom class |
| https://github.com/xyflow/xyflow/issues/4652 | issue | "Calculate viewport dimensions in fitView" -- shipped in 12.3.1 |
| https://reactflow.dev/examples/layout/dagre | official example | canonical dagre wiring; ours already matches |
| https://reactflow.dev/learn/layouting/layouting | official doc | layout overview |
| https://ncoughlin.com/posts/react-flow-dagre-custom-nodes | blog | community-tier; corroborates the width/height-must-match-dagre rule |
| https://reactflow.dev/learn/troubleshooting/common-errors | official doc | superset already covered by `/error` |

**Search-query composition** (three variants per `.claude/rules/research-gate.md`):
current-year -- `"react flow dagre layout too wide fitView minZoom padding large graph 2026"`;
last-2-year -- `"xyflow react flow blank empty canvas after window resize fitView 2025"`;
year-less canonical -- `"React Flow custom node handles required edges not rendering"`
and `"React Flow fitView minZoom clamped nodes clipped outside viewport"`.

## Recency scan (2024-2026)

Performed. **Two findings that complement (none that supersede) the canonical docs:**

1. **12.3.1 (2024-09-26)** -- "#4653 Calculate viewport dimensions in `fitView`
   instead of using stored dimensions." Relevant: `fitView` reads live container
   dimensions, so a re-fit triggered on resize will use the NEW size. Our fix is
   sound on 12.10.2.
2. **12.5.0 (2025-03-27)** -- "Fix fitView not working immediately after adding new
   nodes" (#5067), removing the old `setTimeout`/`requestAnimationFrame` hacks.
   Crucially, the release notes do **not** add auto-re-fit: fitView still does not
   re-run on its own. So defect (C) is not a version bug we can upgrade away from.

`[ADVERSARIAL]` counter-source: issue #4801 documents that `fitView()` was
*independently* flaky on 12.3.4 regardless of handles or zoom. This qualifies my
confidence: some fitView flakiness in the wild is a library bug, not a caller bug.
It does **not** explain our case -- our numbers show a deterministic clamp (0.220 ->
0.500), not intermittent failure -- but it is why criterion 5 must be re-tested
live rather than reasoned about.

No 2024-2026 change to the custom-node-must-render-its-own-Handle rule.

## Key findings

1. **Error 008 fires, and the edge is DROPPED, when the node has no handles.** Exact
   trigger measured at `@xyflow/system/dist/esm/index.js:1365-1385`:
   ```js
   const sourceHandle = getHandle$1(sourceHandleBounds?.source ?? [], params.sourceHandle);
   ...
   if (!sourceHandle || !targetHandle) {
     params.onError?.('008', errorMessages['error008'](!sourceHandle ? 'source' : 'target', {...}));
     return null;   //  <-- edge is never rendered
   }
   ```
   With no `<Handle>` in `AgentNode`, `handleBounds.source` is empty, the lookup
   returns undefined, and `getEdgePosition` returns `null`. This is why **every**
   edge fails, not some.

2. **The `"null"` in the message is NOT bad edge data.** The message template
   (`system/dist/esm/index.js:15`) interpolates `params.sourceHandle` directly:
   `` `Couldn't create edge for ${handleType} handle id: "${...sourceHandle...}", edge id: ${id}.` ``
   Our edges never set `sourceHandle`, so it interpolates the absent value as
   `"null"`. It means "the edge asked for the *default* handle and none exists" --
   **not** "someone set sourceHandle to null." Correcting the caller's open
   question: no edge-construction change is needed.

3. **`fitView` is a one-shot on mount.** `fitViewQueued` is seeded from
   `fitView ?? false` (`react/index.js:3198`) and cleared once nodes are measured
   (`:3277-3282`). The prop only re-queues when its **value changes**
   (`:314-315`, inside a `useEffect` whose deps are the tracked prop values) -- ours
   is a constant `true`, so it never re-queues. There is no auto-re-fit.

4. **React Flow DOES ship a container ResizeObserver -- but it only records size.**
   `useResizeHandler` (`react/index.js:1246-1271`) registers `window.addEventListener('resize', ...)`
   plus `new ResizeObserver(() => updateDimensions())`, and `updateDimensions` does
   only `store.setState({ width, height })`. It never calls `resolveFitView`. So on
   resize the store learns the new size while the viewport transform (x, y, zoom)
   stays computed for the OLD size. There is no built-in `refitOnResize` prop.

5. **`fitViewOptions.minZoom` is applied WITHOUT d3 clamping -- but the first user
   scroll snaps it back.** `fitViewport` calls `panZoom.setViewport(...)` ->
   `setTransform` -> `d3ZoomInstance.transform(...)` (`system/index.js:2900-2907`),
   which bypasses `scaleExtent`. But the d3 instance is constructed
   `zoom().scaleExtent([minZoom, maxZoom])` (`system/index.js:2887`), so any wheel /
   dblclick / Controls interaction re-clamps to 0.5. React Flow's own guidance is
   that fitView's minZoom "should match the component props to avoid viewport
   jumps." **Therefore the `minZoom` PROP must be lowered, not just `fitViewOptions`.**

6. **`useReactFlow()` cannot be called from `AgentMap` as currently structured.**
   `<ReactFlow>` self-wraps in a `ReactFlowProvider` only for its *children*
   (`Wrapper`, `react/index.js:3579-3588` -- `const isWrapped = useContext(StoreContext)`).
   `AgentMap` renders `<ReactFlow>`, so it sits OUTSIDE that provider; a
   `useReactFlow()` call there throws error 001 ("Seems like you have not used
   zustand provider as an ancestor", `system/index.js:7`). The resize logic must
   live in a small child component rendered inside `<ReactFlow>` (alongside
   `<Background>`/`<Controls>`), **or** `AgentMap` must be wrapped in an explicit
   `<ReactFlowProvider>`. This is the single most likely way the fix fails on first
   attempt.

7. **Zero jsdom test coverage exists and a meaningful one is not cheaply achievable.**
   No `AgentMap.test.tsx` exists (30 test files enumerated; none covers it). The docs
   are explicit: "React Flow needs to measure nodes in order to render edges and for
   that relies on rendering DOM elements", and recommend "Cypress or Playwright".
   `frontend/vitest.setup.ts:6-17` provides only a **no-op ResizeObserver shim**
   whose callback never fires, and no `DOMMatrixReadOnly`, `offsetWidth/offsetHeight`,
   or `SVGElement.getBBox` mocks. So nodes never get measured -> edges never render
   -> a jsdom assertion on drawn edges is not achievable without adding all four
   mocks, and even then it would assert on mock-fed geometry.

## Internal code inventory (real line numbers, measured 2026-07-25)

| File | Lines | Role | Status |
| --- | --- | --- | --- |
| `frontend/src/components/AgentMap.tsx` | 419 total | the whole defect surface | **3 defects** |
| -- import block | `:19-27` | pulls `ReactFlow, Background, Controls, Position` + types | **`Handle` MISSING** |
| -- `AgentNode` | `:82-141` | custom node; returns a plain `<div>` at `:98-139` | **renders NO `<Handle>`** |
| -- `NODE_TYPES` | `:143` | `{ agent: AgentNode }` | correct |
| -- `NODE_W`/`NODE_H` | `:145-146` | `220` / `70` | matches dagre input `:157` -- correct |
| -- `layoutWithDagre` | `:148-171` | dagre TB, `nodesep: 50, ranksep: 70` (`:155`) | hand-rolled; produces 4238px width |
| -- `sourcePosition`/`targetPosition` | `:165-166` | `Position.Top` / `Position.Bottom` | set, but no handle to place |
| -- node construction | `:224-244` | `type: "agent"`, `position: {x:0,y:0}` pre-layout | correct |
| -- edge construction (workflow) | `:249-265` | sets `id/source/target/label/style/type` | **no `sourceHandle`/`targetHandle`** (correct) |
| -- edge construction (topology) | `:267-275` | sets `id/source/target/animated/style` | **no `sourceHandle`/`targetHandle`** (correct) |
| -- header count | `:382-386` | `{nodes.length} of {data.nodes.length}` | **CORRECT (29 of 58)** |
| -- `<ReactFlow>` element | `:401-412` | props: `nodes`, `edges`, `nodeTypes`, `colorMode="dark"`, `fitView` (`:406`), `proOptions`, `onNodeClick` | **no `minZoom`, no `fitViewOptions`, no resize handling** |
| -- canvas wrapper | `:389` | `flex-1 min-h-[400px] rounded-xl border ...` | has height -- error 004 not implicated at rest |
| `frontend/src/app/agent-map/page.tsx` | 34 | only consumer; standard two-zone shell | OK, no change needed |
| `frontend/src/lib/api.ts` | `:832-880` | `AgentMapNode/Response` types + `getAgentMap()` at `:878` | OK, no change needed |
| `backend/api/agent_map.py` | 183 | serves `_inventory.json` + derived edges (`_derive_edges` `:32-55`) + `live_model` injection | **healthy -- HTTP 200, 25919 bytes** |
| `frontend/vitest.setup.ts` | `:6-17` | no-op `ResizeObserver` shim only | insufficient for React Flow |
| `frontend/vitest.config.ts` | 19 | `environment: "jsdom"`, `setupFiles: ["./vitest.setup.ts"]` | -- |
| _(absent)_ | -- | `AgentMap.test.tsx` | **DOES NOT EXIST** |

**Live API measurement** (`curl localhost:8000/api/agent-map` -> HTTP 200):
`nodes: 58`, `edges: 57`, `workflow_edges: 12`, `layer1_pipeline.children: 29`,
layer distribution `{1: 30, 2: 7, 3: 3, 4: 18}`.

**Is there filtering between API and render?** Yes -- three filters at
`AgentMap.tsx:199-212`: layer-1 collapse (`:199-203`), `providerFilter` (`:208`),
`layerFilter` (`:209`). All default to showing everything except collapsed layer-1
children. Edges are then filtered to visible endpoints (`:250`, `:268`). This is
already reflected in the header number, so it does **not** explain "12 visible vs
29 claimed" -- clipping does.

**Layout algorithm:** hand-rolled dagre call (`dagre` npm, `:28`, `:148-171`), NOT
elk. Coordinates are absolute pixels in an unbounded space, and they genuinely
exceed a 1440x900 viewport by ~3.8x. Confirmed by measurement, not inference.

**Verification-command baseline:** `cd frontend && npx tsc --noEmit -p tsconfig.json`
currently exits **0 with zero output**. So the immutable command is a clean
baseline; any new tsc error would be attributable to this step.
`grep -n 'Handle' src/components/AgentMap.tsx` currently returns **nothing**
(the string does not occur), which is exactly why it is a good gate.

**Binding conventions from `.claude/rules/frontend.md`:**
- "Icons: `src/lib/icons.ts` -- Phosphor icon aliases (never use emoji in UI)".
  `AgentMap.tsx:29` already imports from `@/lib/icons` -- keep it.
- "Use the project's navy + slate palette, NOT Tailwind's default zinc palette."
  Any handle styling must use navy/slate tokens, not zinc.
- "Tailwind JIT-safe class strings" -- no template-string class construction.
- Live-UI verification: "Never touch the operator's :3000 instance"; start
  `LIGHTHOUSE_SKIP_AUTH=1 npx next dev --port 3100`, capture with Playwright MCP,
  "Kill the :3100 server after capture" and verify :3000 still answers. The bypass
  is real: `frontend/src/middleware.ts:27-29` documents `LIGHTHOUSE_SKIP_AUTH=1` as
  an explicit opt-in for "the :3100 skip-auth Playwright rig".
- Also binding (`feedback_second_next_dev_breaks_operator_3000`): do not let the
  second dev server share or clear `.next`.

## Application to pyfinagent

**(A) Zero edges -- `AgentMap.tsx:19-27` + `:82-141`.** Add `Handle` to the import
block and render two unnamed handles inside `AgentNode`'s root `<div>` (which begins
at `:98`), matching the TB direction set at `:165-166`:
`<Handle type="target" position={Position.Top} />` and
`<Handle type="source" position={Position.Bottom} />`. `Position` is **already
imported** at `:23`. No edge-object change; no `updateNodeInternals` needed (handles
are static, not added programmatically). This alone satisfies criteria 1, 2 and 3.

**(B) Clipping -- `AgentMap.tsx:401-412`.** Add `minZoom={0.1}` (or lower) to
`<ReactFlow>` AND `fitViewOptions={{ padding: 0.15 }}`. Per finding 5, lowering the
**prop** is required -- `fitViewOptions.minZoom` alone would fit correctly and then
snap back to 0.5 on the operator's first scroll. At the required 0.22 zoom the nodes
are ~48px wide, so the graph will be legible as *structure* but not as *text*; the
`<Controls>` at `:411` let the operator zoom in. **Flagging honestly:** criterion 4
("all nodes inside the canvas bounds") is fully satisfiable by the zoom fix, but if
the operator wants readable node text at the default view, that is a separate layout
problem (a 29-node rank at 220px each cannot be legible in 1120px) and belongs in
its own masterplan step rather than being smuggled into 80.3.

**(C) Blank-on-resize -- new code near `:401-412`.** There is no built-in re-fit.
Render a small child component INSIDE `<ReactFlow>` (finding 6 -- calling
`useReactFlow()` in `AgentMap` itself throws error 001), which attaches a
`ResizeObserver` to the flow container and calls `fitView()` from a
`requestAnimationFrame` callback.

Honest limit on (C): the retained-transform mechanism (finding 4) is proven from
source and fully explains a *mis-fit* after resize; combined with the graph already
rendering ~2x the canvas width, it also plausibly explains a *fully blank* canvas.
But I did not isolate the exact blank trigger from source alone. A second candidate
worth keeping in mind: `updateDimensions` returns early when
`domNode.current.checkVisibility?.()` is false (`react/index.js:1249-1251`) and
substitutes a 500x500 fallback when a dimension measures zero, emitting error 004
(`:1255-1257`) -- and the filter bar at `AgentMap.tsx:332` is `flex-wrap`, so
narrowing the window re-wraps it and changes the canvas height mid-resize. The
proposed fix cures both mechanisms, but **criterion 5 must be verified live, not
argued** -- which is exactly what the live_check already requires.

**Testing.** Per finding 7, Playwright IS the test here. A vitest file could
meaningfully assert only the static contract (that `AgentNode` renders two
`.react-flow__handle` elements with `data-handleid`/`data-handlepos`, via
`@testing-library` on the node component in isolation) -- that is worth having as a
mutation-resistant guard against the handles being deleted again, and it does not
require node measurement. Asserting *drawn edges* in jsdom is not achievable without
adding `DOMMatrixReadOnly`, `offsetWidth`/`offsetHeight` and `SVGElement.getBBox`
mocks plus a firing ResizeObserver, and even then it would assert on mocked
geometry. Recommend: one narrow vitest guard on handle presence + Playwright for
criteria 3, 4, 5.

## Risk / do-no-harm

- **Frontend-only, confirmed.** The only consumers of `AgentMap` are
  `frontend/src/app/agent-map/page.tsx:29` and the type/fetcher block in
  `frontend/src/lib/api.ts:832-880`. `backend/api/agent_map.py` is read-only over a
  static JSON inventory and needs no change. Nothing in the fix touches a trading,
  order, sizing, or scheduler path.
- **Yes, handles are visible by default -- measured.**
  `node_modules/@xyflow/react/dist/style.css` sets `.react-flow__handle { position: absolute;
  pointer-events: none; min-width: 5px; min-height: 5px; width: 6px; height: 6px;
  background-color: ...; border: 1px solid ...; border-radius: 100%; }`. With
  `colorMode="dark"` (`:405`) the defaults resolve to `--xy-handle-background-color-default: #bebebe`
  on `--xy-handle-border-color-default: #1e1e1e` -- i.e. a light-grey 6px dot at the
  top-centre and bottom-centre of every node. That is a real, if small, visual change
  to the node design.
- **An invisible handle still binds edges, with one hard constraint.** The docs are
  explicit: "you must use `visibility: hidden` or `opacity: 0` instead of
  `display: none`" -- because React Flow measures the handle's box to compute edge
  endpoints. `display: none` removes the box and the edges break again (re-creating
  defect A). Safe options: `className="!opacity-0"` or `style={{ opacity: 0 }}`.
  **Do not** use `hidden`, `display:none`, or `w-0 h-0`.
  Recommendation: ship the handles **visible** first (it is the library default and
  reads as intentional on a topology graph), and treat hiding them as an operator
  design call -- if hidden is preferred, `opacity: 0` is the safe idiom.
- **Render-loop risk is low but real.** A `ResizeObserver` callback that calls
  `fitView()` mutates only the CSS transform on `.react-flow__viewport`, a *child* of
  the observed container -- it does not resize the observed element, so there is no
  true feedback loop. The residual risk is the browser's "ResizeObserver loop
  completed with undelivered notifications" warning when the callback writes layout
  synchronously. Mitigate by deferring the `fitView()` into a
  `requestAnimationFrame` and cleaning up the observer on unmount. Note this warning
  would be browser-emitted, not `[React Flow]`-prefixed, so it would not by itself
  break criterion 2 -- but it should not be shipped either.
- **Criterion 2 nuance worth pre-empting:** it demands zero `[React Flow]` warnings.
  Error 004 ("The React Flow parent container needs a width and a height") is
  emitted through the same channel; if the capture is taken while the container is
  mid-collapse it could appear. Capture after the page settles.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (9 substantive of 10 fetched)
- [x] 10+ unique URLs total (22: 10 fetched + 12 snippet-only)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (component, page, api client, backend route, test config)
- [x] Contradictions noted (issue #4801 `[ADVERSARIAL]`: fitView independently flaky on 12.3.4)
- [x] All claims cited per-claim
- [~] Gap disclosed: the exact trigger for the *fully blank* canvas after resize was
  not isolated from source alone; two mechanisms identified, both cured by the
  proposed fix, and criterion 5 requires live re-test.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 12,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "React Flow v12.10.2 measured in node_modules. (A) Custom nodes must render their own handles -- built-ins ship them, custom ones do not; AgentNode (AgentMap.tsx:82-141) renders none and Handle is not imported (:19-27), so getEdgePosition returns null for every edge (system/index.js:1378-1385) and drops it. The 'null' in error 008 is just the un-set sourceHandle interpolated, not bad edge data -- no edge-object change needed. (B) Measured by re-running buildGraph+dagre on the live API: 29 nodes / 24 edges / 4238x490px graph; fitView wants zoom 0.220 but minZoom defaults to 0.5 and clamp() raises it, rendering 2119px in a ~1120px canvas -> clipped both sides. The '29 of 58' header is CORRECT (58 nodes, 29 layer1 children hidden), so (B) is ONE clipping defect, not a count/filter bug. Fix requires lowering the minZoom PROP, not just fitViewOptions, or d3 scaleExtent re-clamps on first scroll. (C) fitView is one-shot on mount; React Flow's container ResizeObserver only records width/height and never re-fits. useReactFlow() must be called from a CHILD of <ReactFlow> or it throws error 001. No AgentMap test exists and jsdom cannot meaningfully assert drawn edges -- Playwright is the test.",
  "brief_path": "handoff/current/research_brief_80.3.md",
  "gate_passed": true
}
```
