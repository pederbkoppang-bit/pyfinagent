---
name: the-instrument-that-closes-a-channel-opens-one
description: A visibility counter added to close a prior false-negative channel had its own silent-zero -- its role classifier keyed on prompt LITERALS emitted by two other checked-in files, with nothing pinning the coupling; mutate the marker, not just the counter
metadata:
  type: feedback
---

When a cycle's remedy is a NEW INSTRUMENT that exists to make a previously
invisible loss visible, mutate the instrument's INPUT COUPLING, not only its
body. Ask: what does this classifier read, who writes that, and what pins them
together?

**Why:** 86.84 cycle 9. Cycle 8 FAILED because a re-dispatch silently erased
run-record entries, so two real non-emitters were invisible to an entry-keyed
floor. Cycle 9's fix swept ORPHAN TRANSCRIPTS and printed
`erased=2(non-emit 2)` -- correct, reproduced, and the two orphans independently
corroborated (`API Error: 529 Overloaded` in both). But the sweep decides an
orphan's ROLE by looking for the literals `"IMMUTABLE SUCCESS CRITERIA"` and
`"OBJECTIVE:"` in the transcript's first user message, and orphans that match
neither are collected then dropped by the per-role filter. Those literals are
emitted by `.claude/workflows/qa-verdict.js` and `research-gate.js`; nothing
asserts the coupling. Executed on temp copies: control `erased qa=(2,2)
verify_ok=True`; marker drifted one word -> `erased qa=(0,0) verify_ok=True
problems=[]`. The census showed 41 of 44 orphans already classify as role=None
(all pre-removal, so currently harmless) -- the channel is live, just empty.

The tell that made it a finding rather than a gripe: the SAME mutation cell
(S14) explicitly disclosed a *sibling* fragility -- "this cell's discriminating
signal lives in the rotating corpus; when those transcripts age out (~30d) the
cell degrades to equivalent". The author had the disclosure discipline and
applied it to the TEST's fragility while the PRODUCTION classifier's went
unstated, in code and in both artifacts.

**How to apply:** on any cycle whose remedy adds a counter/sweep/classifier,
(1) run a CONTROL, (2) mutate the string/shape it keys on -- not the code that
consumes it -- and (3) check whether the immutable command stays GREEN with the
signal at zero. If it does and no artifact says so, that is a `Missing_Assumption`
WARN with a one-line fix (assert the literal, add a marker-mutation cell, or
disclose). Related: [[feedback_guards_stop_one_seam_short]],
[[feedback_the_guard_carries_the_defect_it_guards]],
[[feedback_a_fix_verifier_can_be_vacuous_too]].
