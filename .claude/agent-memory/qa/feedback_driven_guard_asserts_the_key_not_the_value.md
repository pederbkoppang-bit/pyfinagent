---
name: driven-guard-asserts-the-key-not-the-value
description: When a lexical/source-scan guard is upgraded to a DRIVEN behavioural one, check that the driven assertion covers the VALUE, not just the key's presence and type -- a hollow container passes
metadata:
  type: feedback
---

When an author discharges a "your guard is a source scan / lexical decoy is possible"
finding by replacing it with a **DRIVEN** check, verify the driven assertion covers the
**contents**, not merely that the key exists and has the right `typeof`.

**Why:** 86.78 cycle 7. The capping WARN was QM3 -- a string-literal decoy defeating a
first-match source locator (the 3rd lexical form after `//` and `/* */`). The fix was
genuinely the right class-closing move: drive the whole script with a stubbed judge and
assert the returned object's keys. I built the decoy myself and confirmed it: the lexical
check SURVIVES my mutant, the DRIVEN check KILLS it **by property** (the failure detail
printed a real key list, so the drive completed -- not a build-failure false kill). Real
discharge.

But the assertion was
`hasOwnProperty('escalation') && driven.escalation && typeof driven.escalation === 'object'`.
Two mutants I wrote SURVIVED the entire 57-check suite at exit 0:
`escalation: {}` and `escalation: { note: 'x' }` at the merge -- every computed field
(`would_auto_fail`, `consecutive_conditionals`, `burden_on`, `override`) silently
discarded while the container kept the right shape. `null` and a string both redden
(falsy / wrong typeof); the **empty object** is the hole.

**How to apply:**
- After the container assertion passes, mutate the container to EMPTY and to a
  decoy-keyed object. `typeof {} === 'object'` and `{}` is truthy, so both slip through.
- Always run a **no-op control mutant** first (a harmless comment) so you can say the
  redness is not "any edit reddens" -- mine stayed green at exit 0, which is what makes
  the other rows mean something.
- Distinguish **kill by property** from **kill by throw**: read the failure detail. If it
  prints a real observed value the drive completed and the assertion did the work; if it
  prints "drive threw", the mutant may simply not build (see
  [[a-mutant-that-cannot-build-scores-as-a-kill]]).
- Severity: this is NOT the illusory-guard heuristic -- the detection question ("name the
  mutation that makes this fail") has answers, so the guard is real and its label is
  accurate. It is a **coverage gap in a real guard**, evidence-quality, especially when the
  product state is independently proven elsewhere (there, 73 live workflow envelopes).
  Related: [[boundary-on-elements-not-the-container]], [[assert-the-output-not-its-feed]].
