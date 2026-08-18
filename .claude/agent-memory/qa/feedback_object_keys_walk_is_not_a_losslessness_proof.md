---
name: object-keys-walk-is-not-a-losslessness-proof
description: To attack a "lossless-or-throw" JSON renderer, use non-enumerable props, a non-enumerable toJSON, and a non-deterministic getter -- an Object.keys walk misses all three and the value renders lossily without throwing (86.90)
metadata:
  type: feedback
---

When a fix claims **"lossless-or-throw"** (render faithfully, otherwise throw), the
validator is almost always a recursive walk over `Object.keys` + `typeof`. That walk
proves less than the claim. Attack it with these, in this order:

1. `Object.defineProperty(o, 'hidden', {value: X, enumerable: false})` -- `Object.keys`
   never sees it, `JSON.stringify` never emits it. **Silently dropped.**
2. A **non-enumerable `toJSON`** -- same blind spot, but far worse: `JSON.stringify`
   calls it and the WHOLE value is replaced. Measured on 86.90: a `{real_evidence, more}`
   object rendered as the single string `"REPLACED"`, no throw. That is a placeholder
   substitution reached through a guard written to forbid placeholder substitution.
   (An *enumerable* `toJSON` is caught -- it looks like a function-valued key.)
3. A **non-deterministic getter** -- the walk reads each property once and
   `JSON.stringify` reads it AGAIN. Validated value != rendered value (TOCTOU).
4. An **array with a non-index own property** (`a = ['x']; a.extra = 'gone'`).
5. Controls that must behave: enumerable `toJSON` -> THROWS; `Object.create(null)` ->
   renders losslessly (proto `null` is legitimately allowed).

**Why:** phase-86.90 shipped exactly this renderer and Main explicitly asked me to try to
break criterion 5. All five landed. But **reachability decided the severity**: the args
reach the script only via `classifyArgs`, which either `JSON.parse`s a string or passes
the runtime's structured object through -- and a JSON-derived object has no
non-enumerables, no getters and no `toJSON`. So these are WARN-level residuals, not a
criterion miss. The chargeable part is the **in-code absolute** ("THE RULE IS
LOSSLESS-OR-THROW") being broader than what was measured.

**How to apply:** run the probe against the SHIPPED renderer (slice the module at its
driver boundary and import it), not against your reading of it. Then, before grading,
trace backwards to whether a real caller can construct the shape -- see
[[survivor-needs-behavioural-differential]]. Report the residual AND its reachability;
an unreachable lossy corner is a claim defect, not a live defect.
