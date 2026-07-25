---
name: feedback-stale-figure-in-gate-artifact
description: Ruling on stale numbers left in a dated research-gate artifact -- annotate, never rewrite, and first check whether the figure was refined or simply wrong
metadata:
  type: feedback
---

When a later measurement supersedes a number that lives in a **dated gate
artifact** (`research_brief_*.md`), the remedy is an **append-only dated
correction note**, not a rewrite and not silence. Rewriting falsifies the
record of what the gate actually found; leaving it unmarked ships a false
number in a committed artifact.

**Before ruling, re-derive the superseded figure and classify it:**

- *Refinement* (same quantity, better instrument) -- low severity, annotate.
- *Error* (produced by a method the artifact misdescribes) -- higher severity,
  because the artifact is asserting something it never measured.

Worked instance, phase-80.3 cycle 7: `research_brief_80.3.md:58` carried
`0.220` under a table labelled "**Measured, not inferred.**" I reproduced it as
`1120 / (4237.5 * 1.2)` -- the researcher's own `CW/(w*(1+2*PAD))` formula,
which React Flow does not use. The installed `getViewportForBounds` gives
`0.2407` at the pre-fix default padding and `0.2301` shipped. So it was an
*error*, not a refinement -- which the author's "a later measurement superseded
it" framing had wrong. Same formula produced the brief's `0.091` expanded
figure (real value ~0.0952-0.0997).

**Severity wiring:** blocking only if a forward-looking consumer still reads
the wrong number. Check the executor-facing record specifically (in this repo,
the queued-defect list in `cycle_block_summary.md`). If every forward-looking
artifact carries the corrected figure and only the dated gate record does not,
that is WARN-level -- do not block a correct P0 fix over it.

**Why:** six cycles of phase-80.3 were spent on prose defects while the product
code was correct every time. An evaluator that cannot distinguish "a wrong
number a future executor will act on" from "a wrong number in a historical
record" converts the harness into a logging loop.

**How to apply:** on any cycle where the author discloses a stale figure and
asks for a ruling, (1) re-derive it to classify refinement-vs-error, (2) sweep
repo-wide with a derived scope, (3) grade severity by whether a forward-looking
consumer reads it, (4) rule -- do not defer back to the author.

Related: [[feedback-rederive-the-label-not-just-the-number]],
[[feedback-measure-dont-assert-claims]], [[feedback-recheck-prior-remediation-list]].
