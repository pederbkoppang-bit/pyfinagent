---
name: run-status-is-not-agent-outcome
description: A binary `status == "failed"` predicate over a multi-valued run status buckets every other value as SUCCESS, and a retry layer hides agent-level failures inside runs that completed -- census the status values and look for the failure signature on the success side
metadata:
  type: feedback
---

When a probe classifies outcomes as `dropped = record["status"] == "failed"`,
run `Counter(status)` over the whole corpus before believing either side. Two
distinct leaks, both measured on 86.84 (2026-08-14):

1. **A third status value.** The corpus had `completed` 520, `failed` 46,
   **`killed` 6**. The predicate's complement was labelled "completed" / "ok"
   throughout the output, so 10 externally-aborted spawns were counted as
   successes and sat in the denominator of the detector control (1257/**1277**).
   `not failed` is not `completed`.
2. **A retry absorbing the agent-level failure.** The run-level predicate can
   only see runs. Two research-gate runs had `status: completed` while
   containing a researcher spawn at exactly its cap that never emitted the
   schema call -- the phase-86.81 retry re-spawned and the second attempt
   carried the run. The true failure-signature population was 50, not 48.

**Why:** the cheap check that finds both is to look for the FAILURE SIGNATURE on
the SUCCESS side -- here, "completed spawns that never emitted StructuredOutput"
(20 of them), which exposed the killed-run contamination and the retry-absorbed
exhaustions in one query. A clean binary split over a field you never censused
is exactly the shape of [[feedback_suspect_the_clean_check]].

**How to apply:** on any run/record-level classification, (a) census the raw
field's distinct values, (b) name what the complement actually contains, and
(c) query the success bucket for the failure signature. In this instance both
leaks pointed the SAME way as the author's thesis, so they were WARN-level, not
a refutation -- state the direction of an accounting error before grading it.
Related: [[feedback_count_the_class_not_your_list]],
[[feedback_survivor_needs_behavioural_differential]].
