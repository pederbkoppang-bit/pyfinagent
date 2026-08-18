---
name: carried-forward-residuals-go-stale
description: A residual list copied from the prior cycle's prose gets stale when a SIBLING step closes the item; and a +N delta attributed to a commit that PREDATES the baseline is always false. Settle both by title-set diff plus commit dates.
metadata:
  type: feedback
---

A "residuals still queued" list and a "+N came from X" attribution are both
CLAIMS ABOUT THE TREE, not history you may copy forward from the prior cycle's
prose. Re-derive each one against the tree, at the commit you are grading.

**Why:** step 86.37 cycle 4 (2026-08-17) re-captured its checker at today's tree
(121 -> 124) and made two prose claims, both false, both from copying rather
than deriving:

1. *"the +3 are phase-86.28's cycle-5 additions to the same file"* — phase-86.28's
   three commits to that checker were at 10:06 / 10:22 / 10:46 on 2026-08-10,
   i.e. BEFORE the step's own first commit (17:34) and before the cycle-3
   baseline commit (18:03). **They were already inside the 121.** A commit that
   predates the baseline can never explain a delta measured from that baseline —
   checking the candidate's DATE against the baseline's is a one-line refutation.
   The real +3 were a *different step's* work landed 4 days later (a retry loop
   added by phase-86.81).
2. *"(b) a driver-level happy-path assertion remains queued"* — one of those very
   +3 assertions IS a driver-level happy path (`drive(...)` then
   `gate_passed === true`). The residual had been closed by a sibling step and
   the artifact re-queued it. Proven load-bearing by mutating
   `STAGE1_MAX_ATTEMPTS 3 -> 1`, which fails exactly that assertion.

**How to apply:** when an artifact reports a count delta between two trees,
extract the assertion/test TITLES at both commits and compare by symmetric
difference — `ADDED` and `REMOVED`, not the two counts (equal counts can cover
different members). Then `git log --format='%h %ad' -- <file>` and check whether
the credited commit is even in the window. When a cycle carries a residual list
forward, grep the tree for each item before accepting "still queued": residuals
are closed by whoever touches the file next, and that is often not this step.
Companion checks: [[regenerated-label-is-a-claim-check-the-diff]],
[[rederive-the-label-not-just-the-number]],
[[a-later-step-bolts-a-mode-on-with-no-guard]].
