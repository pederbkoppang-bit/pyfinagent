# Contract -- phase-86.17

**Step:** 86.17 (P1) -- the Layer-3 Workflow rail silently runs a BLIND gate when
its `args` do not parse, and a blind gate can still return `gate_passed: true`.
**Date:** 2026-08-10. **Cycle:** 197.
**Research gate:** PASSED -- `handoff/current/research_brief_86.17.md`
(10 sources read in full >= floor 5, 54 URLs >= floor 10, recency scan
performed, 9 internal files inspected, `gate_passed: true`). **The gate was run
in a previous session and is NOT re-run**: the brief is on disk, its envelope is
intact, and re-running it would burn a `max`-effort session to re-derive a
conclusion that has not changed.

---

## 1. Research-gate summary

**The defect, confirmed at source.** Both `.claude/workflows/research-gate.js`
and `.claude/workflows/qa-verdict.js` carry the same block: parse `args`, and on
ANY failure `catch (_e) { a = {} }`. Every field then falls back
(`a.step_id || a.stepId || 'UNSPECIFIED'`), so the gate runs with no step id, no
topic and no scope -- writing `research_brief_UNSPECIFIED.md`, a name that
collides across every step that ever hits this path. Nothing in the return
value, the log line or the checker reports that the run was blind.

**The brief REFUTED the first explanation and I am recording that rather than
quietly dropping it.** "args must be an object; a string is unsupported" is
FALSE -- a well-formed JSON string parses fine. The defect is the silent catch
plus the fallbacks. The brief measured **12 input shapes** and found **9 that
silently default** (the caller had reported 5 of 7), including a
**double-encoded JSON string that parses SUCCESSFULLY** into a string -- a shape
no amount of catch-hardening would cover, because nothing throws.

**The trap that would have broken the naive fix.** The empty `catch` is
**load-bearing**: `scripts/qa/verify_research_gate_workflow.mjs` imports the
slice with `args` UNBOUND, so a bare `args === undefined` raises
`ReferenceError` and kills all 40 green checks. The fix must use
`typeof args === 'undefined'`. This is not merely a checker artifact -- a real
no-args launch leaves `args` genuinely unbound, so it binds production too.

**External findings, cited per claim in the brief.** RFC 9413 (virtuous
intolerance: fail fast so the CALLER gets fixed) and CWE-392 (missing report of
an error condition) support throwing on unusable input; Shore's assertion-message
rule shapes what the throw must NAME; Saltzer's fail-safe defaults and
EviBound's "no evidence, no claim" settle the second question below; and the
existing in-repo idiom already says *"do not repair near-misses"*, which forbids
re-parsing a double-encoded string a second time.

**The question the brief answers explicitly, and it is the heart of this step:
may an absent-args (dry-run) run return `gate_passed: true`? NO. NEVER.** A dry
run has no step, no topic and no criteria -- there is no subject to certify, so
a `true` is a certificate with no referent. **The legitimacy of the dry run is
about not THROWING; it is not a licence to PASS.** The current code conflates the
two.

## 2. Hypothesis

Classifying `args` into ABSENT / UNUSABLE / INCOMPLETE and giving each its own
outcome -- run-but-cannot-pass, throw, throw -- removes every path on which a
gate can report a result under an identity it does not have, while preserving
the documented dry run. Because a schema cannot make a self-report true, the
blind state must ALSO be forced into the gate result, so that any future path
that bypasses the throw still cannot pass.

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. "The defect is REPRODUCED FIRST and recorded verbatim, for BOTH .claude/workflows/research-gate.js and .claude/workflows/qa-verdict.js, across at least these eight args shapes: a plain object; a VALID JSON string; a malformed JSON string; a JSON string containing a raw newline inside a string value; an array; a non-object scalar; absent/undefined; and a double-encoded JSON string. For each shape record what stepId resolved to under the CURRENT code"
2. "args that are PRESENT but unusable (malformed JSON, array, scalar) make the script FAIL LOUD -- it throws, and the thrown message names the received type and size. Asserted for BOTH scripts. Silently defaulting to {} is no longer reachable on this path"
3. "args PRESENT but missing step_id throws rather than resolving to 'UNSPECIFIED'. Asserted for BOTH scripts. No brief and no verdict may ever be written under an UNSPECIFIED identity"
4. "ABSENT args still runs WITHOUT throwing -- the documented dry-run mode must not break -- but the run is marked blind in the RETURN VALUE and cannot yield gate_passed:true (research-gate) or a PASS verdict (qa-verdict). Assert BOTH halves: that it does not throw, AND that it cannot pass"
5. "No regression: verify_research_gate_workflow.mjs still passes every pre-existing check. Assert the count as a measured DELTA against the recorded 40-passed/0-failed baseline, and state the new total rather than asserting a bare 'all green'"
6. "MUTATION-TEST every new guard: reverting each one individually to the old behaviour (restore `catch (_e) { a = {} }`; restore `|| 'UNSPECIFIED'`; remove the blind-run marking) must make the specific new check FAIL. Mutate BOTH scripts. A guard whose mutant survives does not count and must be rewritten"
7. "The qa-verdict.js:28-29 comment declaring the silent fallback deliberate is either corrected to match the new behaviour or removed -- a stale comment asserting the opposite of the code is how this survived review"

**Verification command (immutable):**
`bash -c 'node scripts/qa/verify_research_gate_workflow.mjs && node scripts/qa/verify_workflow_args_boundary.mjs'`

**live_check (immutable):** "Verbatim output of the new args-boundary checker showing, per script and per args shape, the resolved stepId and the throw-or-run outcome; plus the verbatim thrown message for one malformed-args case on each script; plus the verbatim pre-existing-checker totals proving no regression against the 40/0 baseline."

## 4. Design, and the three classes

| Class | Detection (ORDER MATTERS) | Behaviour |
|---|---|---|
| **A. ABSENT** -- the documented dry run | `typeof args === 'undefined'` (covers the UNBOUND identifier AND explicit `undefined`) or `args === null` | **Do NOT throw.** Run in explicit `dryRun` mode, and **force the run to be unable to pass**: `gate_passed:false` with a named violation for `research-gate`; `verdict: null` with `dry_run: true` for `qa-verdict`, so Main's transcribe-VERBATIM rule has nothing verdict-shaped to copy. |
| **B. PRESENT BUT UNUSABLE** | not class A, and does not reduce to a PLAIN OBJECT (`typeof x === 'object' && x !== null && !Array.isArray(x)`), re-checked **AFTER** `JSON.parse` so a double-encoded string is caught | **THROW**, naming the received `typeof`, `Array.isArray`, length and a truncated preview. |
| **C. PRESENT BUT INCOMPLETE** | class-B check passes, but no `step_id`/`stepId` | **THROW.** A present args object proves the caller INTENDED to parameterise, so this is unambiguously a caller bug, not a dry run. |

**Defence in depth, and it is deliberate:** the throw handles B and C, and the
forced-false handles anything that ever slips past the throw. Saltzer's complete
mediation -- the gate must fail closed even against a regression in this fix.

**`''` (empty string) is the ONE case the brief could not settle from inside the
script**, because it depends on how the Workflow tool represents "no args".
**This contract requires it to be MEASURED against a live no-args launch before
the classification is locked**, and to be kept as a separately named case either
way so the decision stays visible. It is currently provisionally class B.

**Do NOT repair near-misses.** No second `JSON.parse` on a double-encoded
string, no coercion of an array's first element, no synthesising a step id from
a brief path.

## 5. Plan

1. **[done]** Research gate -- PASSED (previous session), `research_brief_86.17.md`.
2. **[this file]** Contract, written BEFORE any code.
3. **MEASURE the `''` case** against a real no-args launch and record it.
4. **Reproduce FIRST** (criterion 1): a new checker
   `scripts/qa/verify_workflow_args_boundary.mjs` drives BOTH scripts across the
   eight named shapes and records what `stepId` resolves to under CURRENT code.
   ESM caches by URL, so each shape needs a fresh temp module URL -- the brief
   measured that a module-scope free variable cannot be varied otherwise.
5. **Implement** the A/B/C classification in both scripts, plus `input_health`
   in the research-gate return and the blind-run WARNING log.
6. **Correct the `qa-verdict.js` comment** that declares the silent fallback
   deliberate (criterion 7) -- a stale comment asserting the opposite of the
   code is how this survived review.
7. **Criterion 5 as a measured DELTA** against the 40-passed/0-failed baseline,
   re-derived at fix time, stating the new total rather than "all green".
8. **Mutation-test every new guard in BOTH scripts** (criterion 6), reverting
   each one individually.
9. Q/A via the Workflow rail; transcribe verbatim; append `harness_log.md`; flip.

## 6. Traps this step must not fall into (measured, from the brief)

- **`typeof args === 'undefined'`, NEVER a bare `args === undefined`.** The
  checker imports the slice with `args` unbound; a bare comparison throws
  `ReferenceError` and kills all 40 checks.
- **Workflow-runtime constraints, confirmed at source:** no `fs`, no `process`,
  no static `import` of any form, and **exactly one `export`**
  (`export const meta`). `node --check` will NOT catch a violation and neither
  will the immutable command's first half -- the existing checker asserts all
  four, so run it.
- **A double-encoded JSON string PARSES SUCCESSFULLY.** Catch-hardening alone
  does not cover it; the plain-object check must be re-applied AFTER the parse.
- **`enforceGate` must stay PURE** -- no I/O. Thread `inputHealth` in as a
  parameter; the driver logs, not the gate.
- **Do not weaken the research floors** (>=5 sources read in full, >=10 URLs)
  to simplify anything here. They are enforced in JS precisely because the wire
  schema strips `minimum`/`minLength`.
- **The dry run must keep working.** Breaking it to fix the blindness trades one
  defect for another; criterion 4 asserts BOTH halves.

## 7. References

- `handoff/current/research_brief_86.17.md` (10 read in full, 54 URLs).
- RFC 9413 (Maintaining Robust Protocols) -- https://www.rfc-editor.org/rfc/rfc9413.html
- CWE-392 Missing Report of Error Condition -- https://cwe.mitre.org/data/definitions/392.html
- Saltzer & Schroeder, fail-safe defaults -- http://web.mit.edu/Saltzer/www/publications/protection/
- Internal: `.claude/workflows/research-gate.js`, `.claude/workflows/qa-verdict.js`,
  `scripts/qa/verify_research_gate_workflow.mjs`, `.claude/rules/research-gate.md`.
