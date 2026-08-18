---
name: regenerating-a-capture-leaves-the-authored-summary-stale
description: A remediation that regenerates a pasted command capture does NOT regenerate the hand-authored table/prose above it -- and the criterion is graded on the table; also check whether the enum it summarises has GROWN
metadata:
  type: feedback
---

When a prior cycle's fix says "regenerate the block in section N", the executor
regenerates the fenced **capture** and leaves the **authored summary** directly
above it untouched. Grade the summary separately -- it is usually the thing the
criterion actually asks for.

**Why (86.21 cycle 7, measured).** Criterion 5 was "fail-safe direction is
asserted and TESTED". Section 6 carried a 4-row status table *and* a pasted
self-test block. Cycle 6 regenerated the block perfectly -- I diffed it
member-by-member against fresh stdout and it was IDENTICAL, 20/20. The table
above it was the **cycle-2** table and had never been touched:

- it opened "Four distinct outcomes" while the module defines **five** status
  constants (`ledger_empty` was added in cycle 2 and never reached the table);
- it recorded `ledger_missing` count as **"0 + a caution"** while the cycle-3
  change (source comment at `consecutive_conditionals`) had moved
  `LEDGER_MISSING` into the not-knowable set -- the code returns **None** and
  prints "refusing to print 0".

So the criterion's own assertion contradicted the shipped code on one of the two
cases the criterion names by word, for five cycles, with `grep -c "Four distinct
outcomes"` over every prior verdict returning **0**.

**How to apply.**
1. For any section flagged and "fixed" in a prior cycle, diff BOTH halves: run
   the command and diff the capture, *and* re-derive every cell of the authored
   table by driving the product.
2. When a table summarises an enum, count the enum in the SOURCE
   (`grep -c` the constants) and compare to the table's row count. A summary that
   says "four" when the code defines five is a one-line, high-signal probe.
3. An artifact sentence like "Every pasted figure reproduces at this tree" is a
   universal claim -- test it against figures the fix list never named. Two stale
   ones survived here (an md5 identity line and a global row count) precisely
   because no prior fix list mentioned them.

Related: [[feedback_regenerated_label_is_a_claim_check_the_diff]],
[[feedback_matrix_row_sums_pin_the_tree]].
