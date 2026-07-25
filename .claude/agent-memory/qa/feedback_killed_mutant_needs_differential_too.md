---
name: killed-mutant-needs-differential-too
description: A mutant that turns the guard red still needs a behavioural differential — otherwise a false mechanism gets enshrined in a source comment (80.3 c2, React Flow handles)
metadata:
  type: feedback
---

**A KILLED mutant does not validate the reason it was proposed.** Run the
behavioural differential on kills, not only on survivors — call the library's own
exported function with the mutated input and the baseline input and compare.

**Why:** phase-80.3 cycle 2. I (cycle 1) flagged `id="a"`/`id="b"` on React Flow
handles as a surviving mutant; Main added an assertion, the guard went red, and
the matrix read 8/8. But executing the installed `@xyflow/system` 12.10.2
`getEdgePosition()` with named vs unnamed handles returned an **identical edge
position and zero errors** — `getHandle(bounds, handleId)` is
`(!handleId ? bounds[0] : bounds.find(...))`, so an edge that omits
`sourceHandle` binds to `bounds[0]` regardless of whether it has an id. Same for
the `display:none` mutant: `getHandleBounds` returns null only when
`querySelectorAll('.source')` finds **zero** elements, and `display:none` does not
remove an element from the DOM — so binding succeeds and edges render
mis-anchored, not "zero edges". Both guards were fine; both *rationales* were
false, and both had been written into the shipped source comments where a future
maintainer would read them as fact.

**How to apply:** when an artifact says "mutation X re-breaks Y", find the library
function that implements Y, check whether it is exported (`node -e "Object.keys(
require('pkg'))"`), and call it directly with synthetic inputs for both arms. If
it is not exported, quote the executed expression. Then grade the *claim*
separately from the *guard*: keep an over-strict guard, but never let an unmeasured
mechanism claim into a comment or a handoff artifact. Mirror of
[[survivor-needs-behavioural-differential]]; same species as
[[rerun-whole-compound-verification-command]].
