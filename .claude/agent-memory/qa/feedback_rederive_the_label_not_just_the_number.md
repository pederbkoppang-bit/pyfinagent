---
name: rederive-the-label-not-just-the-number
description: When an author adopts the evaluator's corrected number, re-derive its LABEL too across every config -- 80.3 c5 relabelled 0.2404 "expanded" when expanded measures 0.1000
metadata:
  type: feedback
---

A number you handed the author in a prior cycle can come back **correct but with a
fabricated derivation label**. Re-run the derivation across EVERY configuration the
label could name, not just the one you originally measured.

**Why:** phase-80.3 cycle 5. Cycle 4 measured `0.2404` as *the collapsed graph at the
pre-fix default padding 0.1* and told Main to fix `0.220`. Main fixed the number and
wrote `experiment_results_80.3.md:29` as "(measured 0.2301 collapsed / **0.2404
expanded**)". I re-ran `getViewportForBounds` over all four view states from the live
API: collapsed=0.2301 (pad .15) / 0.2405 (pad .10), **expanded=0.1000**, workflow
collapsed=0.1919, workflow expanded=0.1000. `0.2404` occurs in NO expanded state. The
digit was right; the provenance was invented to fill the sentence.

Watch the same shape one derivation downstream: `~48px` node width survived in three
artifacts because `220 x 0.22 = 48.4`, while `220 x 0.2301 = 50.6`. Correcting a number
does not correct what was computed FROM it.

**How to apply:** for any parenthetical of the form "(measured X <condition>)", enumerate
the conditions the system actually has and compute X for each. If X appears under a
different condition than the one named, that is a `Contradiction` finding. Also grep for
every figure derived from the retired number. Related: [[measure-dont-assert-claims]],
[[recheck-prior-remediation-list]].
