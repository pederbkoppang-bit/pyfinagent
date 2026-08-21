---
name: input-space-family-90-14
description: Step 90.14 -- the 4-shape probe family holds VERDICT fixed at CONDITIONAL, so criterion 6 clause 2 relocates a FIFTH time; measured survivor V1; necessity wrappers are gated on their own shape
metadata:
  type: project
---

Step 90.14 asked for coverage parameterised over a FAMILY of probe inputs. Three things
measured 2026-08-21 that a future cycle should not re-derive from scratch:

1. **The work was already committed BEFORE the research gate** -- `d4d27b50 phase-90.14:
   ... (PROVISIONAL, pre-research)`. The checker passes its own `--self-test`: 105 checks,
   0 failed, 24 cells, N0 SURVIVED + M1 KILLED + QX ERROR on the same run.
2. **The family varies arity and details-shape but pins `verdict` to CONDITIONAL on all
   four members** (`familyShapes()` at `scripts/qa/verify_severity_routing_90_2.mjs:211-237`,
   every shape calls `V('CONDITIONAL', ...)`). MEASURED in a shadow tree
   (`/tmp/shadow90_14_a`, control GREEN first): a verdict-gated drop
   `const derivedOnly = verdict.verdict === 'FAIL' ? derived.map(d=>d.severity).slice(0,-1)
   : derived.map(d=>d.severity)` SURVIVES all 105 checks and is NON-EQUIVALENT -- it
   differs on 6 of the checker's own 24 real fixture returns (all FAILs), route unchanged
   so every route assertion is blind. The UNGATED form of the same drop is KILLED loudly,
   which is the discriminating control: survival is due to the verdict GATE.
   So the family is 1-way over its own input model {arity} x {details-shape} x {verdict}.
3. **The criterion-4 necessity proof is weaker than the published one.**
   `attributionMutants()` (:242-286) authors one wrapper per shape, each gated on exactly
   the condition its own shape produces, then `shapesThatCatch()` asserts caught.length===1.
   The literature's test (Gopinath 2016) is leave-one-out against the REAL cells: drop shape
   S_i, re-score M15-M23, show >=1 flips KILLED->SURVIVED. Against the real cells S3 may be
   subsumed by S2 (M15/M16 fire on any `emitted.length > 1`).

**Why:** the same criterion (6 clause 2, "a mutant silently dropping ANY reported finding
must be KILLED") has now relocated across derived_severities -> governing_severities ->
emitted_severities -> input-shape -> verdict. Each fix closed one DIMENSION and the next
relocation crossed to a new one.

**How to apply:** when a step claims a coverage surface is closed, ask which FACTORS the
probe set holds fixed, not which fields it enumerates. The defensible claim is "t-way
complete over a NAMED input model", never "complete" (NIST SP 800-142; arXiv:2605.17437).
Related: [[project-severity-routing-90-2]], [[criterion-shape-90-9]].
