---
name: fixture-rot-dead-gate-86-92
description: 86.92 -- the step title named the wrong artifact; a PURE validator cannot be affected by the file its fixture's path string points at; fixture rot silently DISABLED a mutation cell
metadata:
  type: project
---

`verify_workflow_args_boundary.mjs` was RED (84 passed / 3 failed, exit 1) and
the step -- filed by the 86.90 Q/A -- blamed `handoff/current/research_brief_86.17.md`
for lacking a `brief_status` marker. **That premise is mechanically false.**

**Why:** `enforceGate` in `.claude/workflows/research-gate.js` is documented PURE
("no I/O, no Node APIs") and the checker never opens that brief -- every
`readFileSync` reads one of the two workflow scripts. The brief path inside the
fixture is an inert **string**. The real cause is the hand-written `verification`
object literal at `verify_workflow_args_boundary.mjs:179` (cloned at `:319`),
which omits 3 of the 9 fields `BRIEF_VERIFICATION_SCHEMA` has required since
phase-86.28 / 86.37. Driven directly: a FAKE brief path gives a **byte-identical**
violation array; adding the 3 fields gives `gate_passed=true`, 0 violations.
Fixing the file the message names would have left the checker RED.

Three transferable pieces:

1. **A path string in a fixture is not a dependency.** Before accepting "the
   validator failed because file X is stale", check whether the validator can
   read X at all. A pure function's inputs are its arguments.
2. **Fixture rot can DISABLE a mutation cell rather than just fail it.** Cell
   `[4] drop-blind-violation` asserts a mutant lets a blind run PASS. The same
   stale literal held `gate_passed=false` regardless of the mutation, so the cell
   reported FAIL whether the guard was present or absent -- a false alarm masking
   a dead cell. After a fix, re-score the cells; exit 0 is not evidence.
3. **The sentinel vs the message.** `-1 distinct URLs` looked like an arithmetic
   bug. `const n = v => (typeof v === 'number' && Number.isFinite(v) ? v : -1)`
   (`research-gate.js:632`) is a deliberate absent-value sentinel, used
   identically for `sources` and `urls`. The defect is the **rendering** at `:740`
   interpolating it into prose that reads as a count -- which is plausibly how the
   filing reached the wrong artifact.

**How to apply:** when a step's own title states a cause, treat it as a claim to
be reproduced, not a premise. See [[feedback_measure_dont_assert_claims]] and
[[a_correct_observation_can_credit_the_wrong_mechanism]]. Also note the
second-order trap: 86.92's immutable verification command is
`node --check <the checker>` -- a parse check that exits 0 with the defect fully
present, so the step that fixes a red checker can be verified without running it.
