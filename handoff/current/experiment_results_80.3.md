# Experiment Results — phase-80.3

**Step:** `80.3` (P0) — `/agent-map` rendered ZERO edges and a broken layout.
Date 2026-07-25. Contract: `contract_80.3.md`. Gate: `research_brief_80.3.md`
(`gate_passed: true`, 9 sources in full, 22 URLs, recency scan, 11 internal files).

## 1. What was built

One file: `frontend/src/components/AgentMap.tsx`. Three defects, three mechanisms, each
proven from the installed `@xyflow/react` **12.10.2** source rather than guessed.

**(A) Zero edges — the whole page's purpose.** Custom nodeTypes get **no** automatic
handles; only React Flow's built-ins ship them. `AgentNode` rendered a plain `<div>` and
`Handle` was not imported, so `getEdgePosition()` looked up `sourceHandleBounds?.source ??
[]`, found nothing, returned `null`, and **every** edge was dropped — 24 of them, ×5
render passes = the 120 measured warnings.

> **A premise the research corrected:** the `"null"` in
> `Couldn't create edge for source handle id: "null"` is just the **un-set `sourceHandle`
> interpolated into the message string**. It is *not* bad edge data. So no edge object
> needed changing, and no `updateNodeInternals` was needed (the handles are static).
> Fixing the node fixed all 24 edges.

Fix: import `Handle`; render two **unnamed** handles (correct, because every edge omits
`sourceHandle`/`targetHandle`), `Position.Top` / `Position.Bottom` to match the dagre `TB`
layout already set on the node objects.

**(B) Clipped off both edges.** Measured by re-running `buildGraph` + dagre against the
live API: **29 nodes / 24 edges / a 4238×490px graph**. `fitView` wants zoom **~0.23** (measured **0.2301** for the collapsed 29-node graph — the default view, and the one this step is judged on);
React Flow's `minZoom` **defaults to 0.5** and `getViewportForBounds` clamps up to it →
~2119px rendered into a ~1120px canvas.

> **The finding that changed the fix:** lowering `fitViewOptions.minZoom` alone would fit
> correctly and then **snap back to 0.5 on the operator's first scroll**, because the d3
> zoom instance is constructed `scaleExtent([minZoom, maxZoom])` from the **prop**. The
> prop is what had to change.

> **And the header was never lying.** `29 of 58` is correct — 58 nodes from the API, 29
> `layer1_pipeline` children hidden while collapsed. All 29 were in the DOM; ~17 were
> outside the visible area. So (B) is **one clipping defect, not a count/filter bug**,
> contrary to what the step text implies.

Fix: `minZoom={0.1}` + `fitViewOptions={{ padding: 0.15 }}`.

**(C) Blank canvas after resize.** `fitView` (the boolean prop) is **one-shot on mount** —
queued once, only re-queued when the prop *value* changes, and ours is a constant `true`.
React Flow's own container `ResizeObserver` records width/height and **never re-fits**. So
the transform stayed computed for the old canvas while the graph already overflowed it.

Fix: a `RefitOnResize` child rendered **inside** `<ReactFlow>` — calling `useReactFlow()`
from the component that *renders* `<ReactFlow>` throws error 001, because the provider
only wraps its children — with a `ResizeObserver` → `requestAnimationFrame` → `fitView()`.
Loop-safe: `fitView` changes the transform on a *child* of the observed element, not the
observed element's own box.

### Files

| File | Δ |
|---|---|
| `frontend/src/components/AgentMap.tsx` | `Handle` + `useReactFlow` imports; two handles in `AgentNode`; `AgentNode` exported for the guard; `RefitOnResize`; `minZoom` + `fitViewOptions` on `<ReactFlow>` |
| `frontend/src/components/AgentMap.handles.test.tsx` | **new**, 3 tests |

## 2. Verification (verbatim in `live_check_80.3.md`)

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

$ npx vitest run src/components/AgentMap.handles.test.tsx
Test Files  1 passed (1)      Tests  3 passed (3)
```

> **CYCLE-2 CORRECTION (Q/A finding, FAIL-grade).** The cycle-1 version of this block
> showed `tsc exit=0`. **It did not reproduce.** I ran `tsc` *before* creating
> `AgentMap.handles.test.tsx`, then never re-ran the command after adding it — and the
> test's `{...({} as never)}` spread produced three **TS2698** errors, so `tsc` exited 1
> and the `grep` half of the immutable command **never executed**. The product fix was
> fine; the gate was red and my artifact said otherwise. Fixed by typing the test through
> a narrow local alias (`const Node = AgentNode as unknown as (props: { data: unknown })
> => React.ReactElement`) instead of a `never` spread. The block above is the whole
> command, re-run end to end.

Live, `@playwright/mcp@0.0.76`, isolated `:3100` rig:

```
console after fresh load  -> Total messages: 3 (Errors: 0, Warnings: 0)   [was 120 warnings]
console after resize      -> Total messages: 3 (Errors: 0, Warnings: 0)
```

## 3. Criteria → evidence

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | AgentNode renders target + source handles; `Handle` imported | verification command output above; `grep` was a genuine gate (no matches pre-fix) | **MET** |
| 2 | ZERO `[React Flow]` console warnings on a fresh load | live_check §C — `Errors: 0, Warnings: 0`, not one `[React Flow]` line (was 120) | **MET** |
| 3 | Edges **visibly drawn** in a screenshot — assert on the rendered graph | `captures_80.3/80.3_agentmap_after_1440x900.png` | **MET** |
| 4 | All nodes inside canvas bounds at 1440×900; node count matches header | same capture — nothing clipped either side; header `29 of 58`, 29 rendered. Research established the header was always correct | **MET** |
| 5 | After resize the graph re-fits and still renders nodes+edges | `captures_80.3/80.3_agentmap_after_resize_1024x768.png` — 1440×900 → 1024×768, graph re-fit, nodes AND edges present (was blank) | **MET** |

## 4. Mutation matrix — 8/8 killed, plus the cycle-3/4 discriminators

The vitest guard exists so the handles cannot be silently deleted again. Mutation-tested
because a guard that cannot fail does not count.

| # | Mutation | Result |
|---|---|---|
| H1 | remove BOTH handles (the original defect) | **KILLED** |
| H2 | remove only the SOURCE handle | **KILLED** |
| H3 | hide handles with inline `display:none` (zero-size rect -> edges anchor at the node origin, mis-drawn) | **KILLED** |
| H4 | both handles become `type="target"` (no source handle at all -> error 008, edges dropped) | **KILLED** |
| H5 | hide via a Tailwind `hidden` class — and, since cycle 4, `!hidden` (important modifier) too | **KILLED** (both) |
| H6 | positions become `Left`/`Right` (wrong for a dagre `TB` layout) | **KILLED** |
| H7 | delete the `AgentNode` export the guard imports | **KILLED** |
| **M6** | **give the handles `id="a"` / `id="b"`** — Q/A's survivor. Hygiene, **not** a binding break — see the correction below | **KILLED (cycle 2)** |
| W3a | swap the type/position pairing (`target`@bottom, `source`@top) — still binds, so only a pairing assertion catches it | **KILLED (cycle 3)** |
| W3c-a | `!hidden` (Tailwind important modifier) | **KILLED (cycle 4)** |
| W3c-b | `overflow-hidden` — benign, **must NOT fire** | **passes, correctly** |

> **CYCLE-2 CORRECTION — two of my rationales above were FALSE.** Q/A executed the
> installed `@xyflow/system` 12.10.2 rather than reasoning about it, and both of my stated
> mechanisms are wrong. The guards are sound and stay; the *explanations* were not, and
> they had been written into the **shipped source comments** where a maintainer would read
> them as fact. Corrected in `AgentMap.tsx`, the test comments, and here:
>
> - **Naming the handles does NOT drop the edges.** `getHandle` is
>   `(!handleId ? bounds[0] : bounds.find(d => d.id === handleId)) || null` — a falsy
>   `handleId` takes `bounds[0]` **unconditionally, id or no id**. Measured: unnamed and
>   `id="a"/"b"` produce an *identical* edge position with zero errors. Unnamed is kept as
>   **hygiene** — the simplest configuration that provably binds — not because naming
>   re-breaks binding.
> - **`display:none` does NOT produce a zero-edge state.** `getHandleBounds` returns
>   `null` only when `querySelectorAll` matches **zero** elements, and `display:none` does
>   not remove an element from `querySelectorAll`. The real consequence is a zero-size
>   rect, so edges anchor at the node origin — **mis-drawn, not absent**.
>   `visibility:hidden`/`opacity:0` remain the correct way to hide a handle.
>
> This is the same species as cycle 1's FAIL: a claim that was plausible when written and
> never executed. Third instance this session.

**M6 is the one Q/A found and I had missed** — the guard now asserts `data-handleid` is
null/empty on both handles. Worth keeping despite the corrected rationale: it pins the
configuration the rest of the file's edge construction assumes.

## 5. Scope honesty

**What the vitest guard does NOT do, stated plainly.** It guards the **static contract
only** — two handles, right types, right positions. It does **not** assert edges are
drawn, and it cannot: React Flow measures real DOM boxes and jsdom has no layout.
`vitest.setup.ts` ships a **no-op ResizeObserver whose callback never fires**, and there
is no `DOMMatrixReadOnly` / `offsetWidth` / `getBBox`. A jsdom "edges are drawn" assertion
would be asserting on mocked geometry. **Playwright IS the test** for criteria 2–5, which
is why the live_check carries screenshots and verbatim console output rather than a unit
test standing in for them.

**Queued, not smuggled in:** at the required ~0.23 zoom the nodes are ~**51px** wide (220 x 0.2301 = 50.6) — the
graph reads as **structure, not text**. Criterion 4 is fully satisfied by the zoom fix,
but making the *default view legible* is a separate layout problem (a 29-node dagre rank
at 220px each cannot fit 1120px) and needs its own step.

**Handles ship VISIBLE**, toned to the navy/slate palette rather than React Flow's default
`#bebebe`. Hiding them at all is an operator design call, not mine to make silently.

**An honest limit carried from the gate:** the retained-transform mechanism fully explains
a *mis-fit* after resize; combined with a graph ~2× the canvas width it also plausibly
explains a *fully blank* canvas — but the exact blank trigger was not isolated from source.
A second candidate was `updateDimensions` bailing on `checkVisibility()` / substituting
500×500 on a zero measurement, plausible because the `flex-wrap` filter bar re-wraps
mid-resize. **Both are cured by the fix, and criterion 5 was verified live rather than
argued** — the capture shows the filter bar re-wrapping and the re-fit holding through it.

## 6. DO-NO-HARM

| Item | Status |
|---|---|
| Live book | **Cannot move.** Frontend-only; `AgentMap`'s only consumers are `app/agent-map/page.tsx:29` and the fetcher in `lib/api.ts:832-880` |
| Backend | `backend/api/agent_map.py` is read-only over a static JSON inventory — untouched |
| `.env` / flags / optimizer | No edit, no flag, no run; `historical_macro` FROZEN |
| Kill-switch / stops / sector caps / DSR / PBO | Not in the diff |
| Restart gating | **None** — frontend-only, so unlike 80.1/80.2/80.27 this is not blocked by `phase-79.55` |
| Resize loop | `fitView` changes the transform on a child of the observed element; verified live across a real resize |
| Operator `:3000` | `302` before and after; never driven |
| `tsconfig.json` / `next-env.d.ts` | Restored, md5s back to baseline, `git status` clean |
| `.claude/rules/frontend.md` | Phosphor icons only, no emoji, navy/slate palette, JIT-safe literal classes — all respected |

## 7. Tier ledger

RESEARCH **T3** (Agent-tool `researcher`, Opus 5 / max) — GENERATE **T3** (Main, Opus 5
`xhigh`) — EVALUATE **T3** (fresh Q/A, Opus 5 / max). Fable not spent: frontend-only, no
money path, and the design was fully determined by the gate.
