---
name: recheck-prior-remediation-list
description: On a re-spawn cycle, re-derive the PRIOR cycle's full "what unblocks PASS" list from the critique file; never accept the follow-up's count of how many findings there were
metadata:
  type: feedback
---

On every cycle-N re-spawn, read the previous cycle's critique in full and enumerate its
remediation list yourself. Do NOT take the author's follow-up table as the scope of what
was asked. The structured `violated_criteria` array is usually shorter than the prose
"What unblocks PASS" section, and a follow-up that answers only the structured entries
will read as complete ("All three accepted") while leaving named items open.

**Why:** phase-80.3 cycle 4 (2026-07-25). Cycle 3's §14 listed **six** numbered
PASS-unblocking items; `violated_criteria` held **three**. Main's follow-up table closed
R1/R2/R3 verbatim and correctly, opened with "All three accepted", and never mentioned
items 3, 5, 6 — all three of which were still open (`AgentMap.tsx:240` `zoom 0.220`, the
`!hidden` regex gap, the un-queued defect bullet). The narrowing was invisible from the
follow-up alone.

Two corollaries, both earned in the same cycle:
- **Watch which mutants the follow-up re-runs.** Cycle 4 re-ran the two W3c directions
  cycle 3 had explicitly ruled "weren't the harmful one" and presented them as
  confirmation, while omitting the mutant cycle 3 named as surviving. Re-run the named
  survivor first ([[mutate-the-flag-read-not-just-the-guard]], qa.md §4c).
- **Trace a non-reproducing number to its generating script, not just to a better
  measurement.** `0.220` came from the author's own `dagre_measure.js` using
  `CW/(w*(1+2*PAD))`; React Flow's real `getViewportForBounds` is
  `(width - p.x)/bounds.width`. Provenance explains the whole family of the error and
  survives re-measurement disputes.

**Third instance, and the subtlest — the SET SUBSTITUTION (cycle 6, same step).** Cycle 5
named three files holding a stale `~48px`: `experiment_results:162`, `live_check:122` and
`cycle_block_summary.md:100` (flagged in bold as "the record a future executor acts on").
Cycle 6's follow-up row restated the finding as "in three artifacts" and then verified
"no `~48px` remains in `contract`, `experiment_results` or `live_check`" — a set of three
that ADDS the one file cycle 5 had explicitly ruled exempt (the PLAN snapshot) and DROPS
the one it flagged. Cardinality matched, membership did not. **When a follow-up echoes
your count, diff the MEMBERS, not the number** (qa.md §4b symmetric-difference rule) — an
`ls`/`grep` over the file list you named, run yourself, settles it in one command.

**Fourth instance — the UNION DOUBLE-COUNT (phase-80.4 cycle 3, 2026-07-26).** The
remediation ADDED the evaluator's four independently-authored mutations to the matrix and
re-totalled `9/9 -> 13/13, measured`. But three of the added rows (delete `onopen`; reset
the failure budget in `onopen`; `onopen` sets the wrong status) were the SAME mutations
cycle 1 had already run as F1/F2/F3, so `7 (c1) + 6 (c2)` counted them twice — 10 distinct,
not 13. The same summary line also relabelled cycle 1's 7 as "7 backend" when its own
itemisation two lines above read "Backend (c1, 4) / Frontend (c1, 3)". **When a re-total
appears, recover the PRIOR total's itemisation from git (`git show <wip-commit>:<artifact>`)
and set-difference it against the new rows before accepting the sum** — re-running a
mutation is re-verification, not a new mutation.

**How to apply:** before grading a cycle-N≥2 handoff, grep the critique for the prior
cycle's numbered remediation section, list every item AND every file:line it names, and
mark each closed/open by execution. State the open ones by number in your verdict so the
count cannot drift again. Related: [[tightened-guard-opens-false-negative]],
[[rerun-whole-compound-verification-command]], [[rederive-the-label-not-just-the-number]].
