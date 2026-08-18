---
name: same-source-recount-cannot-see-upstream
description: A cross-check that recomputes an aggregate from the SAME list it was computed from kills hardcodes and nothing else -- mutate the COLLECTOR, not the aggregation site, and the guard stays green with the number wrong
metadata:
  type: feedback
---

The standard repair for "this counter is asserted by nothing" is
`if stored != recount_from(source): problems.append(...)`. It is a real guard, but
its reach is exactly one shape: **a hardcode or drift at the aggregation site.**
Because both sides read the same list, any defect UPSTREAM of that list moves both
operands together and the check stays green while the reported number is wrong.

**Why:** 86.84 cycle 11. The fix recomputed `erased_unclassified` /
`_post_removal` inside `verify()` from `remediation["erased_transcripts"]`, and the
author's cells S17/S18 hardcoded each to 0 -- both KILLED (`stored 0 != recount 42`,
`stored 0 != recount 1`). My mutation went one seam earlier: make the role
classifier never yield `None` (a default-role defect). Stored 0, recount 0,
`verify_ok=True, problems=[]`, true value 41. The guard did precisely what its
comment promised and could not have failed.

Two things decided severity, and both are worth copying:
1. **Is the wrong number still PRINTED?** After the same cycle's render fix the
   default report moves `41 -> 0`, so an ORACLE-mode cell (whole-report diff
   against the control) WOULD catch it -- the harness had that mode already and
   simply had no cell for this shape. A missing cell is weaker than a blind guard.
2. **Does a conservation law help?** Here it does not: per-role sum 44 +
   unclassified 0 == 44 still balances under the mutant. Check before proposing one.

**How to apply:** whenever you grade a "recomputed from the source" cross-check,
write down where the source is BUILT and mutate there -- classifier defaults,
filters, globs, parse fallbacks. Ask "which side of this equality does the defect
enter?"; if the answer is "both", the check is a consistency invariant, not a
correctness one. Name that in the finding rather than calling the guard vacuous:
it kills a real class, just a narrower one than a reader assumes.
Related: [[feedback_byte_presence_pin_is_satisfied_by_a_comment]],
[[feedback_assert_the_output_not_its_feed]],
[[feedback_a_fingerprint_that_includes_its_own_source]],
[[feedback_matrix_oracle_inherits_selftest_blindspots]].
