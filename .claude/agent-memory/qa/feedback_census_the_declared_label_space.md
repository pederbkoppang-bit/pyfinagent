---
name: census-the-declared-label-space
description: When an artifact declares a labelled finding space ("15 issues R1-R15"), grep-count EVERY label across all artifacts; zero-mention labels are findings with no disposition anywhere
metadata:
  type: feedback
---

When an artifact declares a labelled finding space -- "the adversarial verification found 15
issues (`R1`-`R15`)", "M1-M8 mutations", "F1-F12 audit items" -- do not read the prose summary.
Run a per-label census across **every** artifact and treat any label with **zero** mentions as an
undisposed finding:

```bash
for r in R1 R2 ... R15; do
  n=$(grep -oE "\b${r}\b" handoff/current/*.md | wc -l); echo "$r : $n"
done
```

Also check whether the label's only hit is the range token itself (`R1`-`R15` matches `R15`);
that is a zero, not a mention.

**Why:** phase-36.7 cycle 2 (2026-07-26). `experiment_results_36.7.md:85` declared 15 adversarial
findings on P0 kill-switch code. The census returned R10:0, R13:0, R14:0, and R15:1 -- where the
single R15 hit was the `R1`-`R15` range string. Four findings on safety-critical code were neither
fixed, nor queued as steps, nor dismissed, anywhere in seven artifacts. Prose read as complete
because the fixed and queued items were each described at length; only counting exposed the hole.
The same section carried two more counts that failed re-derivation: "Three were cheap ... and
fixed" above **four** `###` fix subsections, and "Six defects ... are queued" above an enumeration
of **eight** items (seven R-labels plus one unlabelled). One paragraph, three numbers, none
reproducing.

**How to apply:** any time an artifact names a bounded finding set. Derive the label space from the
artifact's own declaration, never from the items it chose to describe -- that is the same
derive-the-scope discipline as [[derived-scope-lint-use-xargs]] applied to prose. Cardinality
agreement is not enough (see [[recheck-prior-remediation-list]]): compare MEMBERS. A count that
disagrees with its own adjacent enumeration is a `Contradiction`; a declared-but-never-mentioned
label is a `Missing_Assumption` -- and on P0 code it is a real gap, not a typo, because an operator
cannot tell whether the item was triaged or lost. Relates to
[[measure-the-capture-you-didnt-take]] and the qa.md §4b claim-auditing mandate.
