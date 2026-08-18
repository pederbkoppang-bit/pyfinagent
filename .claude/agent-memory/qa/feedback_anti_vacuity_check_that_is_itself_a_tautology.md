---
name: anti-vacuity-check-that-is-itself-a-tautology
description: A guard added to prove a FIXTURE is non-degenerate can be a predicate over string literals -- constant-true, zero free names, so the fixture can drift right back and every box stays green
metadata:
  type: feedback
---

When a cycle answers "your guard was vacuous" by adding a guard *about the
fixture*, check whether the new guard reads the fixture at all.

86.85 cycle 12 shipped, in two places:

    check("filter fixture is prefix-related (anti-vacuity for the filter axis)",
          "99.40".startswith("99.4") and "99.40" != "99.4")          # self-test
    assert "4.10".startswith("4.1") and "4.10" != "4.1"              # pytest

Both predicates are over STRING LITERALS. Executed proof, cheap and decisive:
compile the expression and `eval` it with `{"__builtins__": {}}` as globals, and
walk the AST for `ast.Name` nodes -- **free names = [], value = True**. No program
state can falsify them.

**Why it matters, measured.** Drift the fixture back to the prefix-UNRELATED pair
*consistently* (append `99.2`, reverse-query `99.2`), leave the "anti-vacuity"
check verbatim, and apply the mutant it exists to catch: **rc=0, SELF-TEST
PASSED, all three filter checks green**, on BOTH oracles (self-test and pytest).
The exact defect the previous cycle FAILED for is fully restorable with every box
green.

**Why:** the same file already had the correct pattern ~100 lines above. The
DATE-axis anti-vacuity check was caught as this identical tautology one cycle
earlier (`len({f"2026-08-1{i}" for i in range(3)}) == 3`) and fixed by deriving
from the ROWS ON DISK:

    len({r["date"] for r in read_rows(p) if r.get("step_id") == "99.4"}) == 3

The ORDER-axis sibling was real because it reads `ordered`, a value derived from
the fixture. A meta-guard is real exactly when its predicate names something the
fixture produced.

**How to apply:** for every check whose NAME contains "anti-vacuity", "fixture",
"not degenerate", "distinct", "non-palindromic" -- extract the predicate, count
its free names, and evaluate it in an empty namespace. Zero free names = it
cannot fail. Then run the two-part mutation: (1) break the fixture only, and
confirm the meta-guard goes red; (2) break the fixture AND apply the production
mutant, and confirm the suite goes red. If (1) stays green the meta-guard is
inert. Severity: the *axis* usually still has genuine behavioural coverage
(here MUT-A/MUT-B/MUT-C all killed real checks), so this is WARN-level per
qa.md 4c -- but the meta-property itself has NO other coverage, so name it.

Related: [[feedback_palindromic_fixture_cannot_test_order]],
[[feedback_fixture_must_break_the_symmetry_it_tests]],
[[feedback_the_guard_carries_the_defect_it_guards]].
