---
name: baseline-captured-after-the-action
description: A before/after assertion whose "before" is read AFTER the action compares a value to itself; check the capture ORDER of every delta check, and mutate the action to leak the effect while keeping its return code
metadata:
  type: feedback
---

A delta assertion is only as good as WHEN its baseline was captured. Read the
statement order, not the variable name.

**Measured, 86.71 cycle 3** (`scripts/harness/attempt_gate.py:366-370`):

    check("operator extension WITHOUT --reason is refused",
          cmd_extend("9.4", 1, "   ") == 2)     # <- the action runs HERE
    before_rows = len(read_ledger(led))          # <- "before" read AFTER it
    check("refused extension appends NO row",
          len(read_ledger(led)) == before_rows)  # <- len(x) == len(x)

The name says `before_rows`, the position says otherwise. The check printed
`ok` while the ledger grew.

**Why:** the action was hidden inside the PRECEDING assertion's argument, so
the two statements read as "act, then baseline, then compare" when they are
really "act, baseline-after, compare-to-self". Vacuity shape #4 (tautology true
by construction) wearing the costume of a behavioural delta check.

**How to apply:**
- For every `before`/`after`/`delta`/`unchanged` assertion, find the line the
  ACTION is on. If the action is an argument to an earlier call, the baseline
  below it is worthless.
- **The discriminating mutant keeps the return code and leaks the effect.**
  Mutating the guard away (return 0 instead of 2) kills the *sibling* check and
  hides this one. Make the refused path DO the forbidden thing while still
  reporting refusal -- then the delta check is the only thing that can catch it.
  86.71: guard-removed mutant died to check 1; append-a-row-but-still-return-2
  survived all three.
- Grade severity by asking whether a REAL guard covers the same behaviour. Here
  the mutation matrix's `_extend_probe` captured its row count between the two
  subprocess calls, correctly, and killed the same mutant -- so WARN with a
  one-line named fix, not BLOCK. Check the sibling artifact before escalating.
- Run every check of a new group, not one: of three new checks, two went red
  under their own targeted mutants and one could not. A group is not a unit.

Related: [[feedback_fixture_must_break_the_symmetry_it_tests]],
[[feedback_palindromic_fixture_cannot_test_order]],
[[feedback_the_guard_carries_the_defect_it_guards]].
