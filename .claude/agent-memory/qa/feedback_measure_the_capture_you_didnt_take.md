---
name: measure-the-capture-you-didnt-take
description: When qa.md §1c forces the degraded path (grading Main's Playwright PNGs), measure the images numerically instead of eyeballing them, and hunt the gitignored .playwright-mcp/ session logs for the raw console evidence
metadata:
  type: feedback
---

On the degraded §1c path (rig already torn down, `lsof -ti tcp:3100` → 0, and Q/A must
never start a server), a screenshot Main produced can still be graded as *measurement*
rather than as testimony. Two techniques, both proven on phase-80.3 cycle 6:

**1. Recover the render transform from the pixels.** Load the PNG with PIL, scan one row
across the canvas, segment it against the modal background colour, and read the box
**pitch** (centre-to-centre). Pitch ÷ the graph-coordinate pitch = the live zoom. On
`/agent-map` the dagre pitch is `NODE_W + nodesep = 220 + 50 = 270`:
- AFTER capture → pitch 62.0 → zoom **0.2296**, matching the artifact's 0.2301 and giving
  node width 50.6px (the claimed `~51px`).
- BEFORE capture → pitch 135.0 → zoom **exactly 0.500**, i.e. React Flow's `minZoom`
  default clamp — which is what *proves* the capture is genuinely the pre-fix build, not a
  re-photograph of the fixed one.
Clipping is measurable the same way: a box truncated to 46px where its siblings are 101px,
starting exactly at the canvas x-origin, IS the left-edge clip. Use `git`-independent
content bounds (`content x 361..1334` inside `canvas 292..1405`) for "nothing clipped".

**2. `.playwright-mcp/console-*.log` is the raw console record.** It is **gitignored**
(`.gitignore:71`) so it never appears in `git status`, but the MCP server appends every
console line there per session, named by session-start UTC timestamp — match it to the
capture's mtime. On 80.3 c6 the BEFORE-window log held exactly **120**
`[React Flow]: Couldn't create edge` lines across **24 unique edge ids** (= 5 render
passes), reproducing an "exactly 120" claim that had no pasted output; the AFTER-window
log held **zero**. `page-*.yml` in the same directory is the accessibility snapshot and
confirms header text.

**Why:** the §1c degraded path is where a verdict is weakest — the author supplied the
evidence. These two moves convert "I read Main's PNG" into three independent
confirmations (live API + dagre + installed `getViewportForBounds`; rendered pixels; raw
console log), which is what lets a criterion be graded MET rather than MET-degraded.

**How to apply:** any UI-touching step where you did not take the capture. Do it BEFORE
writing the verdict, and cite the numbers. Related:
[[rederive-the-label-not-just-the-number]], [[killed-mutant-needs-differential-too]].
