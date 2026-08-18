---
name: swapping-the-operand-leaves-the-arithmetic
description: 86.79 c4 -- a doc fix repointed a comparison at the CORRECT field and kept the off-by-one; and the same cycle added 5 checks without raising the cardinality floor, so the new block became silently skippable
metadata:
  type: feedback
---

When a fix **repoints** something -- a comparison's operand, a threshold's input,
a guard's subject -- re-derive the ARITHMETIC around it. Swapping in the right
field does not make the expression right, and adding checks does not make the
floor that counts them right.

**Why (both measured in one cycle, step 86.79 cycle 4, 2026-08-17):**

1. **The operand moved; the off-by-one stayed.** `qa.md:715` was corrected from
   "if `records_retained` > the ledger's verdict count, the ledger is STALE" to
   "if `attempt_number` > ...". Both operands are INCLUSIVE of the current
   spawn, and the ledger can only ever hold rows for PRIOR attempts (the
   in-flight verdict is written caller-side after return). So for a *perfectly
   current* ledger the rule fires ALWAYS. Measured on the step grading itself:
   `attempt_number=4 > rows=3` -> "STALE", while `prior_attempts=3 > 3` is
   False -> the ledger was exactly current (3 rows for 3 prior spawns). The
   sentence was one of the two sites the step's own §8 claimed it had fixed.
2. **The checks grew; the cardinality floor did not.** `EXPECTED_CHECKS` stayed
   `53` while the gate went 55 -> 60. Commenting out the whole 5-check block
   cycle 4 had just added left `checks run : 55` and **exit 0**; commenting out
   the entire 7-check tail left exactly `53` and still exit 0. The constant's
   own comment says it was raised "to sit just under the current total so a
   silently-skipped block is caught rather than absorbed" -- after a prior Q/A
   found 12 checks of slack. Slack reopened the moment checks were added.

**How to apply:** for every repointed comparison, write out both sides' UNITS
(inclusive/exclusive, per-attempt/per-verdict) and evaluate it on the live
numbers before accepting it -- see [[feedback-normalization-rule-must-be-stated-with-the-ratio]].
For every checker with an `EXPECTED_CHECKS`-style floor, run the mutant that
**comments out the block the cycle just added** and read the exit code; the
floor is only a guard while `total - new_block < floor`. Relatedly
[[feedback-the-guard-carries-the-defect-it-guards]]: here the guard against
stale doc-state was itself calibrated to the pre-change state.

Same cycle, worth pairing: the new doc pins were whole-file byte-presence
checks, so moving the pinned text into a `//` or `<!-- -->` comment while
INVERTING the live sentence left the gate green -- see
[[feedback-byte-presence-pin-is-satisfied-by-a-comment]]. A natural revert was
killed 6/6, so this is WARN, not vacuity.
