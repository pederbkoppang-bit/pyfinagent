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

`.claude/agents/researcher.md` (grep the "### `deep` tier" heading)
documents a `deep` tier with
materially stricter conditions (>=20 sources read in full vs 5, >=1
`[ADVERSARIAL]` source, explicit multi-pass structure). This rail does not
implement it.

That divergence is now **loud** -- a caller gets `tier_supported: false`,
a refusal, and zero spend -- but it is **not resolved**, deliberately.
`researcher.md`'s "Multi-subagent fork option" makes deep's fourth listed
element a CONDITIONAL MULTI-SUBAGENT PRODUCER FORK, so implementing the tier means first deciding the
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
matches shipped code (the stage-1 `agent()` call in `research-gate.js` --
grep `agentType: 'researcher'` -- asserted by the checker's
`"agentType is 'researcher'"` assertion); it changes no behaviour of the
role. Line numbers omitted deliberately: cycle 1 of this step cited
`:419` and `:271`, both of which were accurate before this cycle's own
edits moved them. Measured at cycle 2: the pin is at `research-gate.js:584`
and the assertion at `verify_research_gate_workflow.mjs:399`, and those
numbers will move again -- grep the symbol.

---

# CYCLE 2 -- after the CONDITIONAL (Q/A run `wf_10c6cbd2-cad`)

**Sections 2 and 5 above are the CYCLE-1 measurement (61 passed).** They
are left as recorded. The current state is below.

## 9. Immutable command -- cycle 2

```
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 64 passed, 0 failed
```

40 (baseline) -> 61 (cycle 1) -> 64 (cycle 2). Nothing removed or weakened.

## 10. W1 FIXED -- the ordering guard now fails when its subject is broken

The Q/A defeated the cycle-1 guard with mutant M5 (comment token before
the spawn + the real refusal relocated after it) and measured
`ALL GREEN: 61 passed, 0 failed`. The guard now strips `//` lines before
indexing and matches the refusal as a block reaching its `return {`.

Standing vacuity tests (verbatim):

```
  ok   driver REFUSES TO SPAWN on an unsupported tier
  ok   the refusal is placed BEFORE the researcher spawn (else it saves no tokens)
  ok   M5 genuinely defeats the ORIGINAL naive guard (else it probes nothing)
  ok   ordering guard REJECTS the M5 comment-token + relocation defeat
  ok   ordering guard REJECTS a refusal relocated AFTER the spawn
```

The first assertion is deliberate: if M5 ever stops defeating the ORIGINAL
naive predicate, M5 has stopped reproducing the defect and the guard below
it would be probing nothing. It prevents the vacuity test itself from
going quietly stale.

### Independent behavioural reproduction (the Q/A's own method)

A copy of the repo with M5 applied to `research-gate.js`, run against the
UNMODIFIED checker:

```
$ node scripts/qa/verify_research_gate_workflow.mjs      # M5-mutated copy
FAILED: 62 passed, 2 failed
  - the refusal is placed BEFORE the researcher spawn (else it saves no tokens)
  - M5 genuinely defeats the ORIGINAL naive guard (else it probes nothing)
```

Cycle 1 measured `ALL GREEN: 61 passed, 0 failed` on this same mutation.
**M5 is now KILLED.**

## 11. W2 FIXED -- and the CLASS was audited, not just the 3 named instances

The Q/A named three stale citations. I grepped every `file:line` citation
written this cycle and found the class is larger:

| Location | Cited | Measured | Action |
|---|---|---|---|
| `CLAUDE.md` | `research-gate.js:419` | `:584` | replaced with the symbol |
| `.claude/agents/researcher.md` | `research-gate.js:419` | `:584` | replaced with the symbol |
| `live_check` §8 | `verify_..._workflow.mjs:271` | `:399` | replaced with the symbol |
| `experiment_results` "What was built" | `research-gate.js:419` | `:584` | **NOT flagged by the Q/A** -- found by auditing the class; fixed |
| `experiment_results` "Not done" | `researcher.md:248-263` | `:255-267` | fixed (symbol) |
| `live_check` §7 | `researcher.md:204,206-273` | `:211,213-...` | fixed (symbol) |

Every `researcher.md` citation was stale by ~7 lines because THIS cycle's
own edit to that file shifted them -- including the Q/A's own citation of
`researcher.md:253` for "Multi-subagent fork option", which measures at
`:255`. That is the systemic point: in this repo a line number is stale the
moment anyone edits above it, so present-tense claims now cite the SYMBOL.

Deliberately NOT rewritten:

- `contract_86.28.md` -- a PLAN records what was planned; its citations
  were accurate when written. Annotated with a cycle-2 note instead.
- `research_brief_86.28.md` -- another agent's evidence artifact. Its
  citations were accurate at research time. Rewriting another agent's
  evidence is worse than a stale line number.
- `evaluator_critique_86.28.md` -- verbatim Q/A transcription. Editing it
  would break the no-self-eval guarantee.

## 12. Q/A note N2 accepted -- I was overstated

The Q/A observed that I called the multi-subagent fork deep's "FOURTH
REQUIREMENT" while `researcher.md` titles it an "option" conditioned on
caller request or >=3 separable sub-questions. **Correct; I overstated
it.** It is deep's fourth listed element and it is conditional. Corrected
in `experiment_results` and above. This does not change criterion 3, which
independently mandates not adding `deep`, nor the reason for refusing: a
conditional fork on an N=1 artifact rail is still a fork the rail cannot
support.

## 13. Q/A note N1 -- queued, not patched

The `n()` sentinel renders an omitted count as "only -1 distinct URLs
appear in the brief". It fails closed correctly; the message is confusing.
Cosmetic, outside the frozen criteria, and the tree is under evaluation --
queued rather than patched mid-grade.

---

# CYCLE 3 -- after the SECOND CONDITIONAL (Q/A `wf_d0934c91-70b`)

Sections 2, 5 (cycle 1) and 9 (cycle 2) above are historical measurements.
Current state below.

## 14. Immutable command -- cycle 3

```
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 73 passed, 0 failed
```

40 (baseline) -> 61 (c1) -> 64 (c2) -> 73 (c3). Nothing removed or weakened.

## 15. W3 FIXED -- the guard is now BEHAVIOURAL, not a source scan

Two Q/A passes defeated a source scan (`//` token, then `/* */` block
comment). The Q/A named the terminal fix: observe the property instead of
pattern-matching it. Verbatim:

```
[6d] phase-86.28 cycle 3 -- BEHAVIOURAL: does the driver actually spawn? (replaces the source scan)
  ok   RECORDER WORKS: a SUPPORTED tier really does spawn (known-positive)
  ok   the first spawn is the stage-1 researcher (agentType researcher)
  ok   UNSUPPORTED tier spawns ZERO agents (measured, not scanned)
  ok   UNSUPPORTED tier returns gate_passed:false with the tier reported
  ok   UNSUPPORTED tier does NOT claim an agent returned null
  ok   BLIND run spawns ZERO agents (86.17 property, measured)
  ok   B1 (block-comment decoy + relocated refusal) IS CAUGHT behaviourally

```

The known-positive check is first on purpose: without proving the recorder
can see a spawn that DOES happen, a reading of zero proves nothing.

### Independent reproduction of the Q/A's B1 mutant

```
$ node scripts/qa/verify_research_gate_workflow.mjs      # B1-mutated repo copy
FAILED: 68 passed, 3 failed
  - UNSUPPORTED tier spawns ZERO agents (measured, not scanned) -- recorded 2 agent() call(s) -- the refusal did not prevent the spawn
  - ordering guard REJECTS the M5 comment-token + relocation defeat
  - ordering guard REJECTS a refusal relocated AFTER the spawn
```

The Q/A measured `ALL GREEN 64 passed, 0 failed` on this same mutation.
B1 is KILLED. The source scan additionally now strips block comments and
is demoted to cheap-secondary; section [6d] is the authority.

## 16. W4 FIXED -- three stale claims in the file my audit missed

| Site in `research-gate.js` | Was | Measured | Now |
|---|---|---|---|
| deep-tier reference | `researcher.md:204,206-273` | deep section at `:213` | symbol (grep the heading) |
| fork reference | `researcher.md:248-263` + "fourth requirement" | fork at `:255`; it is conditional | symbol + "fourth LISTED ELEMENT ... CONDITIONAL" |
| implementation proof | "\`grep -c deep\` returns 0" | returns **8** | two enforced checks (below) |

The grep claim defeated itself by containing the word. Replaced with a
claim that cannot, and ENFORCED rather than asserted:

```
  ok   VALID_TIERS does not contain 'deep' (the tier is documented but NOT implemented here)
  ok   every 'deep' occurrence in the file is a COMMENT, never code
```

Mutation-tested -- adding `'deep'` to VALID_TIERS in a repo copy:

```
FAILED: 68 passed, 5 failed
  - UNSUPPORTED tier spawns ZERO agents ... recorded 2 agent() call(s)
  - UNSUPPORTED tier returns gate_passed:false with the tier reported
  - UNSUPPORTED tier does NOT claim an agent returned null
  - VALID_TIERS does not contain 'deep' -- VALID_TIERS = ['simple', 'moderate', 'complex', 'deep']
```

### Why cycle 2's audit missed it

I derived the scope from the union of MY OWN commits -- and
`research-gate.js` is not in them, because the peer session's `git add -A`
swept it into `cad38647`. A scope that looked derived was still wrong.
Cycle 3 derives from the step BASE (`089726f9..HEAD`), deliberately
over-inclusive. That rerun also exposed a zsh trap in my first attempt:
`for f in $SCOPE` does not word-split in zsh, so the loop audited nothing
and printed a clean result.

## 17. Cycle-2 false sentence corrected

"strips \`//\` comment lines before indexing, so a comment cannot stand in
for code" was measurably false for block comments. Corrected to `//`
comment, with a pointer to the behavioural replacement.
