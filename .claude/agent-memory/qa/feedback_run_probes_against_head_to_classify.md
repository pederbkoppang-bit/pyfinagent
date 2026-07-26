---
name: run-probes-against-head-to-classify
description: Run your regression probe against the HEAD component too — it separates a NEW regression from pre-existing behaviour, and after a fix it exposes siblings the author patched only at the named instance
metadata:
  type: feedback
---

When a probe fails on the shipped code, re-run the **same probe file against the
HEAD version** of the module before calling it a regression. Three-way outcome:
fails on both = pre-existing (not this diff's fault); passes on HEAD, fails now =
NEW; fails on HEAD, passes now = the fix working.

And after a remediation, sweep every **sibling handler with the same shape**, not
just the one you named last cycle.

**Why:** phase-80.5 cycle 2. Cycle 1 found `onFocus={() => setHoverIdx(i)}` let a
pending grace-timer wipe a focus-opened tooltip. Main fixed exactly that line and
added a guard for exactly that line. Six probes against shipped code showed three
failures — but only the HEAD differential told them apart: two (focus + unrelated
hover ending; blur wiping a mouse-held tooltip) failed on HEAD too, so they were
the pre-existing single-`hoverIdx` focus/hover conflation, NOT introduced. The
third — pointer leaving the TOOLTIP while focus still holds it — **passed on HEAD
and failed on shipped**, because the diff newly added `onMouseLeave={scheduleClose}`
to the tooltip itself. Same WCAG 1.4.13 Persistent class, new instance, unguarded.
Without the HEAD run I'd have reported three regressions (two wrong) or dismissed
all three as pre-existing (one wrong).

**How to apply:** keep the probe file component-agnostic (public props only) so you
can swap the implementation underneath via the same mutation/restore harness
(`git show HEAD:<path>`). Confirm the mechanism by mutating the suspected line away
and checking the probe flips — that names WHICH line, rather than inferring it. See
[[killed-mutant-needs-differential-too]] and
[[mutate-the-flag-read-not-just-the-guard]].
