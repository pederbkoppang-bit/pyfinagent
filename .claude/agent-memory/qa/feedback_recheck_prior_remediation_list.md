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

**Fifth instance — the NOTE TIER VANISHES (phase-75.11.4 cycle 4, 2026-08-17).** Cycle 2's
critique carried a 5-item fix list: (1)(2) BLOCKING guard defects on criteria 5/7/9, (3) a
WARN, and (4)(5) NOTEs. Cycle 3 closed (1)(2)(3) beautifully — I re-ran all four cycle-2
survivors in my own harness and every one was KILLED — and (4)(5) simply **disappeared**:
not fixed, not annotated, not queued as residuals, not mentioned. They were the cheapest
items on the list. `_move` still `mkdir`s before its `if dry_run:` return, and the three
empty dirs that dry run created (`handoff/archive/phase-80.5`, `-81.1`, `-82.23`, mtime
18:42:23Z) are STILL on the live tree — and I classified them: all three score
`no_contract`, so the step's own dry run added 3 to the 845-dir denominator the step
reports. Likewise "19 files are held back" still reads 19 in two artifacts; re-derived
read-only with the script's own `_masterplan_referenced_names` + `_is_rolling_keep`, the
live answer is **20**. **A tier is not a disposition.** When you grade a finding WARN or
NOTE, the next cycle reads that as "optional" unless the artifact says fixed-or-queued —
so enumerate the prior critique's items by TIER and check the low tiers first, because the
blocking ones always get done. Corollary: the "SURVIVORS: none" line cycle 2 asked to be
scoped is still unscoped, and my own battery found two more (an emptied
`ROLLING_KEEP_PREFIXES` archives a done step's `evaluator_critique_<sid>.json`, restoring
the 81.0 verdict-gate-dark defect; `_safe_target` returning `dest` clobbers prior archived
evidence) — which is exactly what an unscoped global claim under an N-cell matrix hides.

**How to apply:** before grading a cycle-N≥2 handoff, grep the critique for the prior
cycle's numbered remediation section, list every item AND every file:line it names, and
mark each closed/open by execution. State the open ones by number in your verdict so the
count cannot drift again. Related: [[tightened-guard-opens-false-negative]],
[[rerun-whole-compound-verification-command]], [[rederive-the-label-not-just-the-number]].
