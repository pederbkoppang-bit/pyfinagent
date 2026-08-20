---
name: phase-tag-in-ui-91-9
description: Step 91.9 -- a naive `grep phase-` on frontend/src over-reports the UI-leak defect ~20:1 (60 raw -> 1 real); a "grep returns zero" criterion would be permanently unsatisfiable
metadata:
  type: project
---

Step 91.9 (leaking internal phase-tracking labels into user-facing UI copy).
Measured 2026-08-20 over all 174 source files under `frontend/src`.

**The funnel, measured:**
- `grep -rnE 'phase-[0-9]'` raw: ~60 hits
- after stripping `/* */` and `//` comments: **38**
- after excluding `*.test.tsx` / `*.test.ts` (Vitest `describe()` names): **3**
- after excluding a `console.error` string and a product-cycle string: **1**

The single real defect is `frontend/src/app/observability/page.tsx:115` --
`Per-table age + SLA bands across the warehouse (phase-25.C7)` rendered as the
Tier-1 page subtitle.

**Why:** the ~57 excluded hits are `//` and `{/* */}` provenance comments that the
repo's own conventions REQUIRE (`observability/page.tsx:9-12` is one). A criterion
worded "grep for `phase-` in frontend/src returns zero hits" therefore demands
deleting legitimate provenance and can never go green. The verification command
must comment-strip AND test-exclude, or scope to rendered JSX text nodes.

**How to apply:** whenever a step's criterion is "no X anywhere in <tree>", check
whether the naive matcher's denominator includes a legitimate class. Related:
[[feedback_immutable_criteria_must_be_green_able]], [[feedback_count_the_class_not_your_list]].

**Two adjacent instances, deliberately OUT of the stated scope** -- name them in
the contract rather than leaving a reader to wonder:
- `frontend/src/app/sovereign/page.tsx:61` -- phase tag inside `console.error`
  (devtools-visible, not UI copy)
- `frontend/src/app/settings/page.tsx:961` -- `cycles 1+2+3+5; cycle 4` is rendered
  help text, but refers to the alpha-overlay PRODUCT cycles, a different class from
  a masterplan phase id

**Non-obvious external result:** OWASP WSTG-INFO-05 covers this territory but
assigns NO severity and ships NO remediation section (its examples are leaked DB
passwords and SQL). A targeted 2024-2026 search for a documented incident of
phase/sprint ids shipping in UI found NOTHING. So the security framing is
supporting context only -- a criterion asserting security impact would over-claim.
The separation pattern already exists in the defect's own file at
`observability/page.tsx:22-27` (`BAND_LABEL` maps internal enum -> user words), so
the fix is a one-line copy edit, NOT an i18n catalog.
