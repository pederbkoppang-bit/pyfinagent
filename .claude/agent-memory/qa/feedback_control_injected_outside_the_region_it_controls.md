---
name: control-injected-outside-the-region-it-controls
description: A positive control that injects a marker to prove a transform is live must inject it INSIDE that transform's input domain; a marker placed just before a slice anchor is never scanned, so the control passes unconditionally (86.92)
metadata:
  type: feedback
---

When a checker adds a **positive control** to prove some transform (a comment
stripper, a sanitiser, a filter) is live rather than decorative, verify the
injected marker actually lands **inside the transform's input domain** — then
neuter the transform and confirm the control goes RED.

**Why:** phase-86.92 shipped a control that read:

```js
const poisoned = src.replace('function enforceGate',
  '// verification.__bogusProseOnlyField__ appears only in prose here\nfunction enforceGate')
const naive    = /verification\.__bogusProseOnlyField__/.test(poisoned)      // whole source
const stripped = verificationFieldsRead(poisoned).fields.includes('__bogus…') // sliced region
check('the stripper rejects a comment-only field', naive && !stripped)
```

The scanner slices **from** `indexOf('function enforceGate')`, and the poison was
inserted **immediately before** that anchor — measured at index 33473 against a
slice start of 33524. So `stripped` is false whether the stripper works or not,
and `naive && !stripped` is true unconditionally. Mutating both strip operations
to an inert `.replace(/__NEVER_MATCHES__/g,'')` left the checker at 94 passed / 1
failed with **both control lines still `ok`**. The in-source comment claimed the
opposite in so many words: "A control that cannot fail is not a control."

Two tells that generalise:
1. **The code did not implement the comment.** The prose described a
   scan-vs-scan differential ("the un-stripped scan sees it, the stripped scan
   rejects it"); the code compared a raw-string regex against a region scan.
   When a comment describes a differential, check both sides are the same
   instrument.
2. **The transform had no current effect anyway.** `fields(stripped)` ==
   `fields(unstripped)` on the real source — identical arrays — so nothing in
   the tree exercised it. A guard whose subject is inert today is where a
   vacuous control hides longest.

**How to apply:** for every control cell, (a) print the index of the injected
marker and the boundary of the region under test and assert containment, and
(b) run the mutation that makes the controlled transform inert. Placing the same
comment *inside* the region discriminated immediately (ON -> false, OFF -> true),
so the remedy is usually one line. Related: [[feedback-a-fix-verifier-can-be-vacuous-too]],
[[a-mutant-that-cannot-build-scores-as-a-kill]],
[[feedback-a-slicing-checker-cannot-cover-what-it-slices]].
