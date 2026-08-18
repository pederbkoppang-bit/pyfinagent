---
name: mutate-into-the-shape-the-criterion-forbids
description: When a criterion names a forbidden shape ("must not special-case id X"), build THAT mutant against the guard; 86.91's replay guard fixture used only the one id the criterion forbade hardcoding, so the hardcoded mutant survived every assertion
metadata:
  type: feedback
---

When an immutable criterion NAMES an anti-pattern, that named shape is the first
mutant to construct. The author reads it as a rule to obey; you read it as a
mutation to execute against the guard.

**Why:** 86.91 criterion 2 read *"a fix that special-cases 86.86 or any single
step id rather than the CLASS fails this criterion"*. The production guard `[1]`
honoured it -- it drives ids `9.99` and `12.7` in unrelated phases. The SIBLING
replay guard `[5]`/`[6]` did not: every fixture used the single id `86.86`.
Narrowing the shipped predicate to `... is _ABSENT and s == "86.86"` (anchor
unique) left ALL FOUR `[5]` assertions green -- extractable/runnable, True-arm
`['86.86']`, False-arm `[]`, "the two arms genuinely DISAGREE" -- so the mutant
SURVIVED the whole 31-green checker, while returning `[]` for `9.99`. And
`newly_done_ids` is the instrument that produces the step's three headline
counts, so a narrowed predicate would have under-reported them with the guard
green.

Two compounding traps found in the same section:
- **A cycle-2 remediation converts the guards the finding NAMED and leaves its
  siblings.** `[5]` had three checks; two became driven, the corpus-pin check
  stayed a substring scan (`"CORPUS_UNTIL" in SRC and "CORPUS_UNTIL = None" not
  in SRC`). Stripping the one line that USES the pin kept every literal, kept
  the guard green, and measurably unpinned the corpus.
- **A "class" fixture that instantiates the class ONCE is a single-instance
  fixture.** Cardinality of ids, not the word "class" in the assertion name, is
  what makes it a class test.

**ADDING FIXTURE VALUES MOVES THE BOUND, IT DOES NOT CLOSE IT (86.91 cycle 3).**
The remediation went from one fixture id to two (`86.86`, `12.7`), and the 1-id
mutant now correctly dies. But I whitelisted the predicate to *exactly the
fixture's two ids* -- `s in ("86.86","12.7")` on the replay, and the 5 fixture
ids on the hook -- and BOTH survived all 34 assertions. Any N-id fixture is
defeated by an N-id whitelist, so this is unwinnable by adding exemplars. The
fix that actually closes it is a **runtime-generated id that appears in no
source literal** (property-style input), or an explicit statement of the bound.
Grade the undisclosed version as a WARN: the fix is cheap and the artifact
claimed coverage it does not have.

**How to apply:** before grading a guard, re-read the criteria for any named
negative ("not a single id", "not by reading the source", "not hand-edited",
"rather than asserted") and build that exact mutant. Then count the DISTINCT
fixture values behind every assertion whose name contains CLASS / EVERY / ALL --
if the count is 1, the assertion is about an instance; if it is N, try the N-id
whitelist before crediting the fix. Related:
[[feedback_palindromic_fixture_cannot_test_order]],
[[feedback_count_the_class_not_your_list]],
[[feedback_recheck_prior_remediation_list]].
