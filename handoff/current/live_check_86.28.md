# live_check -- step 86.28

Generated 2026-08-10 by Main (session pyfinagent-06). Every block below is
verbatim command output or a verbatim workflow return value.

## 1. Immutable verification command -- BEFORE any edit (baseline)

```
$ node scripts/qa/verify_research_gate_workflow.mjs   # captured before the first edit
ALL GREEN: 40 passed, 0 failed
```

## 2. Immutable verification command -- AFTER (re-run just now)

```
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 61 passed, 0 failed
```

## 3. New checks, verbatim (phase-86.28 sections)

```
[6b] phase-86.28 -- an UNSUPPORTED tier fails closed; an ABSENT tier does not
  ok   UNSUPPORTED tier => gate_passed false (refuses to certify a standard never applied)
  ok   the violation names the requested tier and what actually ran
  ok   refusal path (env=null, unsupported tier) reports the TIER as the cause
  ok   refusal path does NOT claim an agent returned null (no agent was asked to run)
  ok   refusal path still fails closed
  ok   empty return on a supported tier still reports empty_or_errored_return
  ok   ABSENT tier still PASSES (defaulting is legitimate when the caller named nothing)
  ok   a SUPPORTED tier passes
  ok   absent opts.tier behaves exactly as before (pre-86.28 call sites unaffected)

[6c] phase-86.28 -- the two formerly-uncorroborated self-reports are checked against the brief
  ok   recency_scan_performed=true with NO recency section in the brief => gate_passed false
  ok   recency corroboration PASSES when the brief carries the section
  ok   recency_scan_performed=false still fails via the original check, not the corroboration
  ok   urls_collected over-claim (99 claimed, 25 in the brief) => gate_passed false
  ok   urls_collected within what the brief carries PASSES
  ok   absent verification still fails via fail-closed ONLY (corroboration does not double-fire)

[7] criterion 6 MUTATION-TEST -- weakening a floor in the SOURCE must break the check enforcing it
```

## 4. Mutation output -- every mutant, verbatim (criterion 5)

```
[7] criterion 6 MUTATION-TEST -- weakening a floor in the SOURCE must break the check enforcing it
  ok   mutant "FLOOR_SOURCES 5 -> 1" is KILLED [let-a-bad-envelope-through]
  ok   mutant "FLOOR_URLS 10 -> 1" is KILLED [let-a-bad-envelope-through]
  ok   mutant "recency check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "audit-class dry check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "over-claim check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "fail-closed on absent verification removed" is KILLED [threw: Cannot read properties of null (reading 'brief_exists')]
  ok   mutant "tier_unsupported check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "recency corroboration removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "urls corroboration removed" is KILLED [let-a-bad-envelope-through]

[8] structural -- no stripped schema keywords, no forbidden runtime imports, riders intact
```

## 5. Structural + ordering assertions (criterion 8 riders intact)

```
[8] structural -- no stripped schema keywords, no forbidden runtime imports, riders intact
  ok   no `minimum:` in the schema (stripped on the wire -- would be false assurance)
  ok   no `minItems:` in the schema (capped at 1 on the wire)
  ok   gate_passed is NOT const:true (honest failure must be representable)
  ok   additionalProperties:false on the envelope
  ok   NO static imports of ANY form (found 0) -- the Workflow runtime parses only dynamic import()
  ok   agentType is 'researcher' (needs Write for write-first)
  ok   model is 'opus' (rider-trap R4)
  ok   no Monitor/watchdog (rider-trap R11)
  ok   exactly ONE export (`export const meta`) -- a trailing export list is unlaunchable
  ok   driver REFUSES TO SPAWN on an unsupported tier
  ok   the refusal is placed BEFORE the researcher spawn (else it saves no tokens)
  ok   the refusal path returns gate_passed:false
  ok   enforceGate is pure -- no fs/process use in its body

ALL GREEN: 61 passed, 0 failed
```

## 6. LIVE spawn evidence (criterion 9)

`node --check` passes on scripts that cannot launch (measured 2026-08-09,
three independent ways -- see the header of `research-gate.js`). So the
gate was exercised LIVE through the Workflow runtime.

### Run 1 -- `wf_4da39b31-695`, BEFORE the ordering fix. Recorded a real defect.

```json
{
  "agentCount": 0,
  "violations": ["empty_or_errored_return"],
  "checks": ["empty_or_errored_return: the agent returned null"],
  "tier_requested": "deep",
  "tier_applied": "moderate",
  "tier_supported": false
}
```

This says "the agent returned null" for a run in which **no agent was ever
asked to run**. Misleading in exactly the way this step exists to prevent.
No then-existing check caught it; only the live spawn did.

### Run 2 -- `wf_23d9ed4b-22c`, AFTER the fix

```json
{
  "step_id": "86.28-LIVETEST-2-unsupported-tier",
  "gate_passed": false,
  "agent_self_reported_gate_passed": null,
  "self_report_disagreed": false,
  "violations": ["tier_unsupported: the caller requested tier \"deep\" which this rail does not implement (supported: simple, moderate, complex). Ran at \"moderate\". Refusing to certify a standard that was never applied -- pass a supported tier, or implement the requested one."],
  "checks": [],
  "input_health": {"status": "ok", "blind": false},
  "tier_requested": "deep",
  "tier_applied": "moderate",
  "tier_supported": false,
  "brief_path": null,
  "brief_verification": null,
  "envelope": null,
  "reason": "UNSUPPORTED TIER: the caller named a tier this rail does not implement. No researcher was spawned."
}
```

Workflow log line, verbatim:

```
research-gate 86.28-LIVETEST-2-unsupported-tier: REFUSING TO SPAWN -- caller requested tier "deep" which this rail does not implement (supported: simple, moderate, complex). Zero agents spawned. Pass a supported tier, or implement the requested one.
```

`agentCount: 0`, `totalTokens: 0`, `durationMs: 5`. No brief was written
under the test identity:

```
$ ls handoff/current/*LIVETEST*
zsh: no matches found: handoff/current/*LIVETEST*
```

### Run 3 -- this step's OWN research gate, `wf_60de95f7-5dc` (pre-change, supported tier)

The full stage-1 + stage-2 path, exercised live on the PRE-change script:
`gate_passed: true`, 7 sources read in full, 34 URLs, brief 41,652 chars,
all 7 claimed sources independently confirmed present, `self_report_disagreed:
false`.

### DISCLOSED GAP

The FULL path (stage-1 researcher + stage-2 verifier) was **not re-run
live after these changes** -- both post-change live runs take the refusal
branch and spawn nothing. The two new stage-2 fields are declared
`required` in `BRIEF_VERIFICATION_SCHEMA`, so constrained decoding should
supply them, but that is reasoning, not measurement. Failure direction is
safe: if stage 2 omitted a required field, the gate fails CLOSED. The next
real research gate on any step exercises the path. Not claimed as verified.

## 7. Operator decision owed -- the deep-tier divergence (criterion 3)

`.claude/agents/researcher.md:204,206-273` documents a `deep` tier with
materially stricter conditions (>=20 sources read in full vs 5, >=1
`[ADVERSARIAL]` source, explicit multi-pass structure). This rail does not
implement it.

That divergence is now **loud** -- a caller gets `tier_supported: false`,
a refusal, and zero spend -- but it is **not resolved**, deliberately.
`researcher.md:248-263` makes deep's fourth requirement a MULTI-SUBAGENT
PRODUCER FORK, so implementing the tier means first deciding the
producer-fan-out question that audit `wf_d61fef3b-25c` left open (both
adversarial refuters returned `refuted: true` on the recommendation's
evidence base).

Two options, for the operator:

- **(a) Implement `deep`** -- requires deciding fan-out first, and would
  need per-branch brief paths, cross-branch URL de-duplication before
  floors are applied, one stage-2 verification per branch, and a merge
  stage. None of that exists today.
- **(b) Mark `deep` as not-implemented in `researcher.md`** -- cheap,
  honest, and removes the divergence without pre-deciding fan-out.

Not decided here.

## 8. Separation of duties

`.claude/agents/researcher.md` was edited this cycle (the `agentType`
doc-drift line). Per CLAUDE.md's separation-of-duties rule this is flagged
for Peder's review. The edit is a one-line factual correction so the doc
matches shipped code (`research-gate.js:419`, asserted by the checker at
`verify_research_gate_workflow.mjs:271`); it changes no behaviour of the
role.
