---
name: reference-structured-output-schema-floors-not-enforceable
description: Anthropic structured outputs STRIPS minimum/maximum/minLength/maxLength and caps minItems at 1 — numeric floors in a workflow schema are comments, not constraints; assert them in JS
metadata:
  type: reference
---

A JSON-schema numeric floor placed in a Claude structured-output / Workflow
`agent({schema})` call is **not enforced**. Per
https://platform.claude.com/docs/en/build-with-claude/structured-outputs
(read in full 2026-08-09):

- **NOT supported / stripped:** `minimum`, `maximum`, `multipleOf`,
  `minLength`, `maxLength`, pattern (limited), recursive schemas, external `$ref`.
  SDK helpers remove them from the wire schema and append them to the field
  *description*, then validate client-side.
- **`minItems` supports only the values 0 and 1** — you cannot require an array
  of >= 5 items.
- **Supported:** `type`, `enum`, `const`, `required`, `additionalProperties`
  (must be `false`), `anyOf`/`allOf`, `$ref`/`$defs` (internal), `default`,
  string formats.

So `{minimum: 5}` on a count field is a comment. Any threshold must be
**asserted in the calling script**, and the script should RECOMPUTE the
pass/fail field rather than trust the returned one.

**The `const` trap:** `const: true` IS supported, so it is tempting for a gate
flag (`recency_scan_performed`, `gate_passed`). Do not use it. It makes the
field unfalsifiable and destroys the honest-failure path — a field that cannot
report failure is not a measurement.

Supporting evidence that schema conformance buys form, not truth:
- "The schema constrains form, not truthfulness" (Anthropic, same doc).
- The Constraint Tax, arXiv:2605.26128v1 (2026-05-20): constrained decoding
  moved schema validity 61.5%->100% but answer accuracy 19.7%->11.0%, and
  **wrong-valid-schema rate 49.5%->88.9%**. Constraining makes wrong answers
  *more* likely to look right.
- EviBound, arXiv:2511.05524: prompt-level self-reflection alone left 100% (8/8)
  of agent claims hallucinated; a post-hoc gate querying the artifact store cut
  it to 25%; dual gates reached 0%. "A governance layer that refuses to promote
  any claim without machine-checkable proof."
- Building to the Test, arXiv:2606.28430: with a checkable oracle in-loop,
  agents scored 221+/222 while 11 of 12 runs shipped a dead or absent library —
  so keep cross-checks cheap-to-satisfy-honestly, or they get optimized.

**How to apply:** whenever writing a `.claude/workflows/*.js` schema that
carries a threshold (source counts, coverage, test counts), put the threshold in
the `description` (it still instructs the model) AND assert it in JS after the
`await agent(...)`. Cross-check any self-reported count against the artifact on
disk. Treat an empty/errored return as a FAILED gate, never a pass. Related:
[[project-research-gate-discipline]], [[feedback-measure-dont-assert-claims]].
