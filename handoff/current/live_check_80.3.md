# live_check — phase-80.3

**Required (masterplan, verbatim):** *Playwright screenshots BEFORE/AFTER at 1440×900
showing edges drawn and no clipping, plus the verbatim `browser_console_messages` output
showing 0 React Flow warnings, plus one capture after a resize.*

Captured 2026-07-25. `@playwright/mcp@0.0.76` as connected this session.

---

## §A. Method

Isolated skip-auth Next dev server on `:3100` with `PLAYWRIGHT_DIST_DIR=.next-audit-3100`.
The operator's `:3000` was never driven — confirmed `302` before and after. The agent-map
API is read-only over a static JSON inventory (`backend/api/agent_map.py`), so the rig
pointed at the operator's `:8000` (`GET /api/agent-map` → `200`) rather than spinning a
second backend; nothing is written.

**This step needed no backend restart** — it is frontend-only, so unlike 80.1/80.2/80.27
it is NOT gated behind `phase-79.55`. It takes effect on the next frontend build.

## §B. The immutable verification command

```
$ cd frontend && npx tsc --noEmit -p tsconfig.json && grep -n 'Handle' src/components/AgentMap.tsx
23:  Handle,
123:        every edge built below omits sourceHandle/targetHandle, and a falsy
124:        handleId makes getHandle() take bounds[0] unconditionally. Top/Bottom
130:        - Giving these an `id` does NOT drop the edges. getHandle is
135:          getHandleBounds returns null only when querySelectorAll matches
145:      <Handle
150:      <Handle
[full-command exit=0]
```

> **CYCLE-2 CORRECTION.** An earlier revision of this block showed `tsc exit=0` from a run
> made **before** the guard test existed. With the test added it exited **1** (three
> TS2698 errors), so the `grep` never ran and the immutable command was failing while this
> file claimed it passed. Corrected above by running the command whole.

**Pre-fix baseline:** `tsc` exit 0 with zero output, and `grep -n 'Handle'` → **no
matches** (independently confirmed before the change). So the grep is a genuine gate, not
a string that was already present.

**Criterion 1 MET:** `Handle` is imported from `@xyflow/react` (`:23`), and `AgentNode`
renders both a `<Handle type="target">` (`:145`) and a `<Handle type="source">` (`:150`).

## §C. Criterion 2 — ZERO React Flow console warnings

Verbatim `browser_console_messages` after a fresh load of `/agent-map` at 1440×900:

```
Total messages: 3 (Errors: 0, Warnings: 0)

[INFO] %cDownload the React DevTools for a better development experience: ...
[LOG]  [Fast Refresh] rebuilding
[LOG]  [Fast Refresh] done in 1135ms
```

**0 errors, 0 warnings, and not one `[React Flow]` line.** The audit measured **120**
warnings on a single load (24 unique edges × 5 render passes), each
`[React Flow]: Couldn't create edge for source handle id: "null"`.

## §D. Criteria 3 + 4 — edges drawn, nothing clipped, count matches

### BEFORE — `captures_80.3/80.3_agentmap_BEFORE_1440x900.png` (1440×900)

Captured by reverting `AgentMap.tsx` to its HEAD (pre-fix) contents, rebuilding on the
same `:3100` rig, and photographing the broken state. The file was restored immediately
afterwards and its md5 verified back to `33c8be8e020715244a46a518c7041695`.

The capture shows the defect exactly as the audit described it:

- **ZERO edges** — not one connecting line anywhere in the graph.
- **Nodes clipped off BOTH horizontal edges** — one sliced at the left canvas boundary
  (only `…ent` visible) and another cut off at the right.
- **Large empty canvas regions** while nodes overflow the sides.
- Header reads `29 of 58 agents` with only ~12 nodes visible — the symptom the audit
  reported as a possible count bug, and which the research showed is the clipping.

Console on that pre-fix build, measured: **120 warnings** — reproducing the audit's
figure of 24 unique edges × 5 render passes **exactly**.

### AFTER — `captures_80.3/80.3_agentmap_after_1440x900.png` (1440×900).

- **Edges are visibly drawn** between nodes across the whole graph — the topology view
  now conveys topology. This is asserted on the *rendered graph*, as criterion 3 demands,
  not merely on the absence of warnings.
- **Nothing is clipped at either horizontal edge.** The full dagre graph sits inside the
  canvas bounds.
- **The header reads `29 of 58 agents`** and 29 nodes are rendered. The research
  established this header was **always correct** (58 nodes returned by the API; the
  `layer1_pipeline` group holds 29 children hidden while collapsed) — so the audit's
  "claims 29 while only ~12 visible" was **one clipping defect, not a count or filter
  bug**. All 29 were in the DOM; ~17 were simply outside the visible area.

**Root cause of the clipping, measured rather than guessed:** the dagre graph is
**4238×490px**, so `fitView` wants zoom **~0.23** (measured 0.2301) — but React Flow's `minZoom` **defaults
to 0.5**, and `getViewportForBounds` clamps up to it, rendering ~2119px into a ~1120px
canvas. Fixed by lowering the **`minZoom` prop** to `0.1`. Lowering only
`fitViewOptions.minZoom` would have fit correctly and then **snapped back to 0.5 on the
operator's first scroll**, because the d3 zoom instance takes its `scaleExtent` from the
prop.

## §E. Criterion 5 — the resize case, explicitly re-tested

Capture: `handoff/current/captures_80.3/80.3_agentmap_after_resize_1024x768.png`.

Resized 1440×900 → 1024×768 with the page live. **The graph re-fitted and still renders
nodes AND edges.** Previously this produced a completely empty canvas (no nodes, no edges,
just the zoom controls).

Worth noting: the filter bar visibly **re-wrapped** at the narrower width (`Layer-1
skills` / `View: static topology` moved to a second row), which changes the canvas height
mid-resize — one of the two candidate blank-canvas mechanisms the research flagged. The
re-fit held through it.

Console after the resize, verbatim: **`Total messages: 3 (Errors: 0, Warnings: 0)`** — no
React Flow warnings appeared.

## §F. What was fixed, and the one thing deliberately left

| defect | mechanism (from source, not inferred) | fix |
|---|---|---|
| (A) every edge dropped | custom nodeTypes get no auto-handles; `getEdgePosition` returns `null` against an empty handle-bounds array and drops the edge. The `"null"` in error 008 is the un-set `sourceHandle` interpolated into the message — **not bad edge data** | import `Handle`; render unnamed target/source handles, Top/Bottom to match the dagre `TB` direction |
| (B) clipped both sides | `minZoom` default 0.5 clamps the required ~0.23 (measured 0.2301) | `minZoom={0.1}` **prop** + `fitViewOptions={{padding:0.15}}` |
| (C) blank after resize | `fitView` is one-shot on mount; React Flow's own ResizeObserver records dimensions but never re-fits | a `RefitOnResize` child **inside** `<ReactFlow>` (calling `useReactFlow()` from the parent throws error 001) with a `ResizeObserver` → `requestAnimationFrame` → `fitView()` |

**Handles are left VISIBLE**, toned to the navy/slate palette (`!h-1.5 !w-1.5
!bg-slate-500`) rather than React Flow's default 6×6 `#bebebe` dots. Hiding them at all is
an operator design call, so they ship visible.

> **CYCLE-2 CORRECTION.** An earlier revision said `display:none` "removes the measured box
> and re-breaks edge binding". **That is false** — Q/A executed the installed
> `@xyflow/system` 12.10.2: `getHandleBounds` returns `null` only on **zero**
> `querySelectorAll` matches, and `display:none` does not remove an element from
> `querySelectorAll`. The real consequence is a zero-size rect, so edges anchor at the node
> origin — **mis-drawn, not absent**. `visibility:hidden`/`opacity:0` remain correct.

**Queued, not smuggled in:** at the required ~0.23 zoom the nodes are ~**51px** wide (220 x 0.2301 = 50.6), so the
graph reads as **structure, not text** (the `<Controls>` let the operator zoom in).
Criterion 4 is fully satisfied by the zoom fix, but making the *default view legible* is a
separate layout problem — a 29-node dagre rank at 220px each cannot fit 1120px — and needs
its own step (rank splitting, collapsing, or a different direction).

## §G. Teardown + operator-instance integrity

```
:3100 listeners: 0
operator :3000/ -> 302   (healthy authed signature)
```

`frontend/tsconfig.json` and `frontend/next-env.d.ts` were rewritten by `next dev` and
restored from HEAD; md5s back to `cecfaa5d04f97bf443b8750d944606f9` /
`ba64ff7d54714a8f64db89b1003207d8`, `git status` clean on both.
