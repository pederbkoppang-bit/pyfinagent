---
name: a-bound-on-a-universal-claim-is-a-new-census
description: When a fix REPLACES a falsified universal claim with a bounded one ("falsified by exactly those two"), the bound is itself a completeness claim -- census the whole population, not the members the prior verdict happened to name
metadata:
  type: feedback
---

A remediation that narrows "every X reproduces" to "every X in scope S reproduces,
falsified by exactly those two" has not retired the completeness claim -- it has
issued a NEW one, over a smaller set, with a cardinality attached. Grade it the same
way: enumerate the whole population yourself and test each member.

**Why:** 86.21 cycle 8 bounded a falsified universal reproduces-claim to the two
figures the cycle-7 verdict had named. A census of all 9 fenced blocks in
`experiment_results_86.21.md` found a THIRD non-reproducing block (a `--step 86.21`
capture showing 5 verdicts / consecutive 2, live 6 / 3) -- and it had aged for the
same reason the other two had, in the same window, under another step's commits. The
author fixed the list; the class had one more member. Same shape as
[[count-the-class-not-your-list]] and [[guard-from-instance-not-class]], arriving
inside the fix for it.

**How to apply:** parse the artifact for every fenced block (a 20-line script), print
`section | line range | first line`, then reproduce each one. Cheap, and it converts
"the two the verdict named" into a derived scope. Two refinements that mattered:
- **A historical section is not automatically a defect.** The third member sat under
  a "DISPOSITION -- PARKED" heading describing a past state, and it UNDER-claimed.
  That is a queue-able residual, not a criterion miss -- say which, explicitly.
- **Annotation vs regeneration is settled by a line-diff, not by preference.** The
  same cycle annotated two stale figures in place where the verdict had said "re-run
  and paste it whole". Stripping the labelled annotation and diffing gave 43/43 lines
  identical to a fresh run, so the annotation was substantively equivalent AND
  carried more provenance (which commit of which OTHER step aged the figure). Do the
  diff before crediting or faulting either choice.
