# live_check -- step 86.28 (CURRENT STATE)

> **Full six-cycle history:** `handoff/current/live_check_86.28_history.md`
> (719 lines, byte-identical copy of this file before compaction, md5
> 750975dcb174d9e081d30f4135c1bde2). Nothing deleted. Moved out of the
> mandatory read path because two Q/A spawns dropped while reading it
> (`wf_01c83c86-09d` 197,091 tokens; `wf_9c55b720-ef3` 184,753 tokens).
> Every block below is captured command output, never typed.

## 1. Immutable verification command

Baseline before any edit of this step (re-derived by two Q/As at the base
commit `089726f9`): `ALL GREEN: 40 passed, 0 failed`.

Now:

```
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 92 passed, 0 failed
```

Ladder 40 -> 61 -> 64 -> 73 -> 78 -> 92. Zero checks removed or renamed at
any point (verified by symmetric difference of check names, not totals).

## 2. Behavioural spawn guard -- section [6d]

The property "no researcher is spawned on an unsupported tier" is COUNTED,
not pattern-matched: the whole script is loaded as a drivable function and
run with a recording `agent()` stub. Two earlier source-scan versions were
defeated by a `//` token and then a `/* */` block comment; this reads no
source at all.

```
[6d] phase-86.28 cycle 3 -- BEHAVIOURAL: does the driver actually spawn? (replaces the source scan)
  ok   RECORDER WORKS: a SUPPORTED tier really does spawn (known-positive)
  ok   the first spawn is the stage-1 researcher (agentType researcher)
  ok   UNSUPPORTED tier spawns ZERO agents (measured, not scanned)
  ok   UNSUPPORTED tier returns gate_passed:false with the tier reported
  ok   UNSUPPORTED tier does NOT claim an agent returned null
  ok   BLIND run spawns ZERO agents (86.17 property, measured)
  ok   ABSENT tier still SPAWNS (the converse of the refusal -- Q/A mutant Q1)
  ok   ABSENT tier raises NO tier_unsupported violation
  ok   ABSENT tier reports tier_requested null and applied moderate
  ok   a SUPPORTED non-default tier is APPLIED, not silently downgraded
  ok   a SUPPORTED non-default tier still spawns
  ok   TIER_ABSENT fixture matches the driver (supported:false for an absent tier)
  ok   TIER_ABSENT fixture matches the driver on the BRANCH-STEERING fields
  ok   TIER_UNSUPPORTED fixture matches the driver (supported:false)
  ok   TIER_UNSUPPORTED fixture matches the driver on the BRANCH-STEERING fields
  ok   fidelity check REJECTS the cycle-4 fixture shape (supported:true for absent)
  ok   enforceGate emits the tier_absent_defaulted_ok label for an ABSENT tier
  ok   ...and does NOT emit it for an UNSUPPORTED tier
  ok   B1 (block-comment decoy + relocated refusal) IS CAUGHT behaviourally

```

The known-positive leads deliberately: without proving the recorder sees a
spawn that DOES happen, a reading of zero proves nothing.

## 3. Mutation matrices

`[7]` mutates the source and probes through `enforceGate`. `[7b]` mutates
the source and RE-DRIVES the module, because every `[6d]` check is
end-to-end driver behaviour that `[7]` structurally cannot reach.

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

[7b] phase-86.28 cycle 6 -- DRIVER-level mutants (the [7] matrix drives enforceGate ONLY)
  ok   driver-mutant "driver reports the APPLIED tier as tier_requested (main return)" anchor is UNIQUE (1)
  ok   driver-mutant "driver reports the APPLIED tier as tier_requested (main return)" is KILLED [guard: ABSENT tier reports tier_requested null and applied moderate]
  ok   driver-mutant "refusal path claims the tier WAS supported" anchor is UNIQUE (1)
  ok   driver-mutant "refusal path claims the tier WAS supported" is KILLED [guard: TIER_UNSUPPORTED fixture matches the driver / UNSUPPORTED returns the tier]
  ok   driver-mutant "supported tier silently downgraded to the default" anchor is UNIQUE (1)
  ok   driver-mutant "supported tier silently downgraded to the default" is KILLED [guard: a SUPPORTED non-default tier is APPLIED, not silently downgraded]
  ok   fixture-mutant "TIER_ABSENT reverted to cycle-4 supported:true" is KILLED

[7b] phase-86.28 cycle 6 -- DRIVER-level mutants (the [7] matrix drives enforceGate ONLY)
  ok   driver-mutant "driver reports the APPLIED tier as tier_requested (main return)" anchor is UNIQUE (1)
  ok   driver-mutant "driver reports the APPLIED tier as tier_requested (main return)" is KILLED [guard: ABSENT tier reports tier_requested null and applied moderate]
  ok   driver-mutant "refusal path claims the tier WAS supported" anchor is UNIQUE (1)
  ok   driver-mutant "refusal path claims the tier WAS supported" is KILLED [guard: TIER_UNSUPPORTED fixture matches the driver / UNSUPPORTED returns the tier]
  ok   driver-mutant "supported tier silently downgraded to the default" anchor is UNIQUE (1)
  ok   driver-mutant "supported tier silently downgraded to the default" is KILLED [guard: a SUPPORTED non-default tier is APPLIED, not silently downgraded]
  ok   fixture-mutant "TIER_ABSENT reverted to cycle-4 supported:true" is KILLED

```

## 4. Structural / rider assertions

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
  ok   M5 genuinely defeats the ORIGINAL naive guard (else it probes nothing)
  ok   ordering guard REJECTS the M5 comment-token + relocation defeat
  ok   ordering guard REJECTS a refusal relocated AFTER the spawn
  ok   VALID_TIERS does not contain 'deep' (the tier is documented but NOT implemented here)
  ok   every 'deep' occurrence in the file is a COMMENT, never code
  ok   enforceGate is pure -- no fs/process use in its body

ALL GREEN: 92 passed, 0 failed
```

## 5. LIVE spawn evidence (criterion 9)

`node --check` passes on scripts that cannot launch (three independent ways,
recorded in the header of `research-gate.js`), so the gate was exercised
LIVE through the Workflow runtime.

`wf_23d9ed4b-22c` (post-fix, unsupported tier), verbatim return:

```json
{
  "step_id": "86.28-LIVETEST-2-unsupported-tier",
  "gate_passed": false,
  "violations": ["tier_unsupported: the caller requested tier \"deep\" which this rail does not implement (supported: simple, moderate, complex). Ran at \"moderate\". Refusing to certify a standard that was never applied -- pass a supported tier, or implement the requested one."],
  "tier_requested": "deep",
  "tier_applied": "moderate",
  "tier_supported": false,
  "brief_path": null,
  "envelope": null,
  "reason": "UNSUPPORTED TIER: the caller named a tier this rail does not implement. No researcher was spawned."
}
```

`agentCount: 0`, `totalTokens: 0`, `durationMs: 5`. `wf_4da39b31-695` is the
pre-fix run that recorded the misleading `"the agent returned null"`.
`wf_60de95f7-5dc` is this step's own research gate (full stage-1+stage-2
path, `agentCount: 2`, gate PASSED).

A prior Q/A closed the load-bearing half of criterion 9 deterministically:
cycle 3's commit was proven comment-only (0 executable-line changes) and
HEAD is byte-identical to it, so those live runs cover the shipped
executable behaviour.

## 6. Disclosed gaps and residuals

- **Criterion 9 residual:** the full stage-1 + stage-2 path has NOT been
  re-run live post-change; both post-change live runs take the refusal
  branch. A Q/A measured the failure direction as safe (omitted stage-2
  fields -> gate fails CLOSED, never a false pass). Disclosed, not claimed.
- **`coverage.dry`** is left uncorroborated on purpose: "dry" is K executed
  search rounds with no new findings, a property of executed discovery, not
  of a file. A file-derived proxy would be false assurance (EBTE).
- **`opts.floors`** left unwired: zero callers, and its only consumer would
  be tier-aware floors, which depend on the unresolved deep-tier decision.
- **`n()` `-1` sentinel** renders an omitted count as "only -1 distinct
  URLs". Fails closed correctly; message is confusing. Queued, not patched.

## 7. OPERATOR DECISION OWED -- the deep-tier divergence

`.claude/agents/researcher.md` (grep the "### `deep` tier" heading)
documents a `deep` tier with materially stricter conditions (>=20 sources
read in full vs 5, >=1 `[ADVERSARIAL]` source, explicit multi-pass
structure). This rail does not implement it.

The divergence is now LOUD -- a caller gets `tier_supported: false`, a
refusal, and zero spend -- but deliberately NOT resolved: researcher.md's
"Multi-subagent fork option" makes deep's fourth listed element a
CONDITIONAL multi-subagent producer fork, so implementing the tier means
first deciding the producer-fan-out question that audit `wf_d61fef3b-25c`
left open.

- **(a) Implement `deep`** -- needs the fan-out decision first, plus
  per-branch brief paths, cross-branch URL de-duplication before floors,
  one stage-2 verification per branch, and a merge stage. None exists.
- **(b) Mark `deep` not-implemented in `researcher.md`** -- cheap, honest,
  removes the divergence without pre-deciding fan-out.

## 8. Separation of duties

`.claude/agents/researcher.md` was edited this step (a one-line factual
correction of `agentType` to match shipped code). Flagged for Peder's
review per CLAUDE.md; it changes no behaviour of the role.
