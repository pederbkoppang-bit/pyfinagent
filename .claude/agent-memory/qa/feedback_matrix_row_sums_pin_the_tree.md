---
name: matrix-row-sums-pin-the-tree
description: failed+passed on every mutation row sums to the suite size the matrix RAN against; compare it to the shipped count to catch a matrix run before the last tests landed
metadata:
  type: feedback
---

Add `failed + passed` on each row of a mutation matrix. That sum is the test
count of the tree the matrix actually ran against. Compare it to the suite size
you measure yourself. A mismatch means the matrix is not evidence about what
shipped, however "verbatim" the block is labelled.

**Why:** 86.88's post-fix matrix declared `CONTROL: 69 passed` and every row
summed to 69 (1+68, 2+67, 15+54, 14+55) while the shipped suite is **72** --
the matrix predated the last three tests, which were precisely the criterion-6
route tests that assert the new behaviour. Corroborated two ways: my equivalent
of their M4 cell killed 4 tests at HEAD where they reported 1, and the test-file
mtime plus the commit time both postdate the matrix run. The same artifact
printed "72 passed" nine lines further down, so the file contradicted itself and
nobody had added the numbers.

**How to apply:** do the arithmetic on every matrix before reading its verdicts;
also do it on any pytest block (progress dots vs the summary count). When the
sums are stale, re-run the load-bearing cells yourself at HEAD rather than
rejecting the finding -- in 86.88 the cells still killed, so the defect was in
the EVIDENCE, not the product, and that distinction sets the severity.
Related: [[feedback_regenerated_label_is_a_claim_check_the_diff]],
[[feedback_recheck_head_before_returning_a_scoped_grade]],
[[project_verbatim_paste_drift_arithmetic]].
