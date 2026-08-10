# Experiment results -- step 86.28

**Phase**: GENERATE
**Date**: 2026-08-10
**Driver**: Main (session `pyfinagent-06`)

---

## What was built

Four changes across four files. Nothing was added to `VALID_TIERS`; no
producer fan-out was implemented.

### 1. ABSENT tier vs UNSUPPORTED tier (`.claude/workflows/research-gate.js`)

Replaced the single `tierDefaulted` flag -- which collapsed two cases
that differ in kind -- with an explicit classification:

```js
const tierRequested = a.tier || null
const tierAbsent = !tierRequested
const tierSupported = !tierAbsent && VALID_TIERS.includes(tierRequested)
const tierUnsupported = !tierAbsent && !tierSupported
const tier = tierSupported ? tierRequested : 'moderate'
```

- **ABSENT** -> default to `moderate`, no violation. Unchanged behaviour.
- **UNSUPPORTED** -> the gate refuses. See 2.

The prompt string at `TIER:` previously asserted "NOT passed by the
caller" even when a tier HAD been passed. It is now conditioned on
`tierAbsent` alone, so it can no longer state something false.

### 2. Refuse to spawn, and report in the RESPONSE

Two mechanisms, deliberately both:

- **Driver early-return before the spawn.** An unsupported tier is a
  decided outcome, so spawning a max-effort researcher first would burn a
  full session and deposit a brief filed under a standard nobody asked
  for. This applies the file's own args-boundary reasoning (`:102-106`)
  and mirrors the 86.17 blind-run refusal.
- **`enforceGate` violation retained** as complete mediation, so any
  future path reaching the gate with an unsupported tier still fails
  closed. Same pattern 86.17 used for the blind run.

The return value now carries `tier_requested`, `tier_applied` and
`tier_supported`. This is the RFC 7240 `Preference-Applied` shape from
the research: a preference may be ignored only because the response says
so. Previously the substitution reached the agent PROMPT and nothing
else -- payload, not response.

### 3. Corroborating two self-reports (same file, plus stage 2)

`BRIEF_VERIFICATION_SCHEMA` gained `recency_section_present` and
`distinct_urls_in_brief`; the stage-2 prompt asks for both. `enforceGate`
now rejects:

- `recency_scan_performed: true` while the brief carries no dedicated
  recency-scan section (the rules require the section "even when empty");
- `urls_collected` exceeding the distinct URLs observable in the brief.

Both checks live INSIDE the branch that already has an independently-read
brief, so the fail-closed path when stage 2 did not run is untouched, and
a dedicated check asserts they do not double-fire there.

**Naming discipline.** These are STRUCTURAL checks and are named so.
`recency_section_present` says a section exists; it does not claim the
scan was substantive. Per the research (EBTE / Proof-or-Stop, "structural
is not semantic"), an unverifiable process claim should be demoted rather
than given a fake proxy -- which is exactly why `coverage.dry` was left
alone (see "Not done").

### 4. Doc drift (`CLAUDE.md`, `.claude/agents/researcher.md`)

Both said `agentType:'general-purpose'`; the shipped code pins
`'researcher'` on the stage-1 `agent()` call (grep `agentType:
'researcher'`) and the checker asserts it. Line number deliberately
omitted -- the cycle-1 text cited `:419`, which this same cycle's edits
moved to `:584`.
Corrected in both, with the reason (write-first needs `Write`).
CLAUDE.md's internal contradiction is called out in the corrected text.

### 5. Checker (`scripts/qa/verify_research_gate_workflow.mjs`)

- Fixture brief is now compliant with what the rules require: read-in-full
  table, snippet-only table, dedicated recency section. Previously it had
  neither snippet table nor recency section, so neither new check could
  have been exercised.
- Fixture is internally consistent: 8 read-in-full + 17 snippet-only = 25
  distinct URLs, matching `goodEnvelope()`'s `urls_collected: 25` and
  `snippet_only_sources: 17`.
- `verifyBrief` (the checker's faithful re-implementation of stage 2) now
  produces the two new fields.
- `makeGate` passes `opts` through, without which no probe could reach the
  tier check at all.
- New assertions in `[6b]`, `[6c]`, and ordering assertions in `[8]`.
- **Three new mutants**, one per new check.

---

## File list

| File | Change |
|---|---|
| `.claude/workflows/research-gate.js` | tier classification, refuse-to-spawn, tier violation, 2 corroboration checks, stage-2 schema + prompt, 3 new return fields |
| `scripts/qa/verify_research_gate_workflow.mjs` | compliant fixtures, stage-2 simulation updated, `opts` passthrough, new assertions, 3 new mutants, ordering checks |
| `CLAUDE.md` | `agentType` corrected to `'researcher'`; self-contradiction noted |
| `.claude/agents/researcher.md` | same correction |
| `handoff/current/contract_86.28.md` | PLAN artifact |

---

## Verification output (verbatim)

### Immutable command, BEFORE any edit (baseline)

```
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 40 passed, 0 failed
```

### Immutable command, AFTER

```
$ node scripts/qa/verify_research_gate_workflow.mjs
ALL GREEN: 61 passed, 0 failed
```

40 -> 61. No pre-existing check was deleted or weakened; the 21 added are
new assertions, 3 new mutants and 3 ordering checks.

### New mutants, all KILLED

```
  ok   mutant "tier_unsupported check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "recency corroboration removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "urls corroboration removed" is KILLED [let-a-bad-envelope-through]
```

Pre-existing mutants all still KILLED:

```
  ok   mutant "FLOOR_SOURCES 5 -> 1" is KILLED [let-a-bad-envelope-through]
  ok   mutant "FLOOR_URLS 10 -> 1" is KILLED [let-a-bad-envelope-through]
  ok   mutant "recency check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "audit-class dry check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "over-claim check removed" is KILLED [let-a-bad-envelope-through]
  ok   mutant "fail-closed on absent verification removed" is KILLED [threw: Cannot read properties of null (reading 'brief_exists')]
```

---

## A defect in this change, found by the live test and fixed

The first implementation put the tier check AFTER `enforceGate`'s
empty-envelope guard. The refusal path calls `enforceGate` with
`env === null` on purpose (no agent ran), so the live run returned:

```json
"violations": ["empty_or_errored_return"],
"checks": ["empty_or_errored_return: the agent returned null"]
```

That describes a failure of an agent that was never asked to run, and it
hid the actionable cause. It is the same class of defect this step exists
to fix, reproduced inside the fix. **No check that existed at the time
would have caught it** -- the checker drove `enforceGate` directly with
good envelopes, and `node --check` passes on scripts that cannot even
launch. Only the live spawn (criterion 9) surfaced it.

Fixed by computing the tier classification BEFORE the empty guard, and
suppressing `empty_or_errored_return` when the tier was the reason no
agent ran. Two regression checks were added, plus a converse check that an
empty return on a SUPPORTED tier still reports `empty_or_errored_return`.

---

## Live evidence (criterion 9)

Run 1 `wf_4da39b31-695` (before the ordering fix) recorded the defect
above. Run 2 `wf_23d9ed4b-22c`, after the fix:

```json
{
  "step_id": "86.28-LIVETEST-2-unsupported-tier",
  "gate_passed": false,
  "violations": ["tier_unsupported: the caller requested tier \"deep\" which this rail does not implement (supported: simple, moderate, complex). Ran at \"moderate\". Refusing to certify a standard that was never applied -- pass a supported tier, or implement the requested one."],
  "checks": [],
  "tier_requested": "deep",
  "tier_applied": "moderate",
  "tier_supported": false,
  "brief_path": null,
  "envelope": null,
  "reason": "UNSUPPORTED TIER: the caller named a tier this rail does not implement. No researcher was spawned."
}
```

`agentCount: 0`, `totalTokens: 0`, `durationMs: 5`. No brief was written
under the test identity -- confirmed, `handoff/current/` contains no
`*LIVETEST*` artifact.

**GAP, disclosed rather than papered over:** the FULL path (stage 1
researcher + stage 2 verifier) was NOT re-run live after these changes.
Both post-change live runs take the refusal branch and spawn nothing. The
new stage-2 fields are declared `required` in the schema, so constrained
decoding should supply them, but "should" is not "measured". The next
real research gate on any step exercises it, and if stage 2 omits a
required field the gate fails CLOSED rather than passing wrongly -- the
failure direction is safe. Cost was the reason: a full exercise means a
max-effort researcher session.

---

## Not done, deliberately

| Item | Why |
|---|---|
| Add `'deep'` to `VALID_TIERS` | `researcher.md`'s "Multi-subagent fork option" (grep the heading; line numbers move) makes deep's fourth listed element a CONDITIONAL MULTI-SUBAGENT PRODUCER FORK ("2-3 parallel deep-tier researcher subagents", "~1 Claude Max 5-hour rolling window per subagent"). Enabling it ships producer fan-out onto an N=1 artifact rail and pre-empts an open operator decision. **Disclosed for the operator, not resolved here.** |
| Corroborate `coverage.dry` | "Dry" is K consecutive EXECUTED search rounds with no new findings -- a property of executed discovery, not of a file. No read-only file check establishes it. A proxy would be false assurance (EBTE). Left honest and uncorroborated. |
| Wire `opts.floors` | Zero callers pass it. Its only consumer would be tier-aware floors, which depend on the deep decision. A change with no behaviour behind it. |
| Change the envelope `tier` enum | The research flagged that `enum` is NOT stripped on the wire, so an honest `'deep'` is unrepresentable there. But the agent reports the tier it ACTUALLY operated at, always a supported value, so the enum is not lying. The requested-vs-applied distinction is the SCRIPT's to report, and it now does. Disclosed, not actioned. |

---

## Operator decision owed

`.claude/agents/researcher.md` documents a `deep` tier that this rail does
not implement. That divergence is now **loud** (callers get
`tier_supported: false` and a refusal) rather than silent, but it is not
resolved. The options are (a) implement `deep`, which means deciding the
producer-fan-out question first, or (b) remove/mark the deep tier in
`researcher.md` as not-yet-implemented. **Not decided here.**

Note also: `.claude/agents/researcher.md` was edited this cycle (doc-drift
line). Per CLAUDE.md's separation-of-duties rule this is flagged for
review; the edit is a one-line factual correction to match shipped code,
not a behavioural change to the role.

---

# Follow-up -- cycle 2 (after CONDITIONAL)

Q/A run `wf_10c6cbd2-cad` returned **CONDITIONAL** with two WARN findings.
Both were real, both are fixed. Neither impugned shipped behaviour; the
Q/A found all 9 criteria met on their literal wording and reproduced every
numeric claim independently.

## W1 -- the ordering guard was an illusory guard (FIXED)

The Q/A executed a mutant (M5) that inserted

```
// harmless note: if (tierUnsupported) { we would refuse here }
```

before the spawn and relocated the REAL refusal block to AFTER it. The
cycle-1 checker still printed `ALL GREEN: 61 passed, 0 failed`. The guard
used a bare `src.indexOf('if (tierUnsupported) {')` over raw source, so a
COMMENT satisfied it while production did the opposite. That is qa.md §4c
shape #2 (source scan defeated by moving the scanned text) and #8 (comment
token) -- and it is precisely the "guards stop one seam short" class.

**Fix.** The predicate is extracted as `refusalPrecedesSpawn(src)` and:

- strips `//` comment lines before indexing, so a comment cannot stand in
  for code;
- matches the refusal as a BLOCK reaching its `return {`, not as a bare
  opening token.

**And it is now watched failing.** Three standing vacuity tests replace
the one-time live observation:

```
  ok   M5 genuinely defeats the ORIGINAL naive guard (else it probes nothing)
  ok   ordering guard REJECTS the M5 comment-token + relocation defeat
  ok   ordering guard REJECTS a refusal relocated AFTER the spawn
```

The first is deliberate: if M5 ever stops defeating the naive predicate it
has stopped reproducing the defect, and the guard below it would be
probing nothing. A vacuity test that cannot itself go stale-silent.

**Independent behavioural reproduction**, the way the Q/A did it -- a repo
copy with M5 applied, checker unmodified:

```
$ node scripts/qa/verify_research_gate_workflow.mjs      # against the M5-mutated copy
FAILED: 62 passed, 2 failed
  - the refusal is placed BEFORE the researcher spawn (else it saves no tokens)
  - M5 genuinely defeats the ORIGINAL naive guard (else it probes nothing)
```

Cycle 1 measured `ALL GREEN: 61 passed, 0 failed` on this same mutation.
The mutant is now KILLED.

## W2 -- I shipped three false line-number citations (FIXED)

The doc-drift fix asserted the pin was at `research-gate.js:419` in BOTH
`CLAUDE.md` and `.claude/agents/researcher.md`, and credited the checker's
assertion to `verify_research_gate_workflow.mjs:271` in the live_check.

**Measured**: the pin is at `:584`; `:419` is now a bare `}`. The
assertion is at `:399`; `:271` is a comment.

**Root cause, and it is not a typo.** `:419` was CORRECT at the base
commit `089726f9`. My own +208-line edit to that file moved it. I wrote
the citation before the edit and never re-derived it -- while CLAUDE.md
carries a standing warning about this exact class two sections above
("Re-derive the line number before citing it again -- it has moved
twice"). I walked into a trap the file explicitly documents.

**Fix**: cite the SYMBOL, not the line, in all three places -- `grep -n
"agentType: 'researcher'"` and the checker's `"agentType is 'researcher'"`
assertion. A symbol cannot go stale under an edit to an unrelated part of
the file. The measured line numbers are recorded once in the live_check
with an explicit note that they will move again.

## Checker total

40 (baseline) -> 61 (cycle 1) -> **64 (cycle 2)**, 0 failed. The three
added are the ordering-guard vacuity tests. No check was removed or
weakened.

## Q/A notes accepted but NOT actioned this cycle

- **N1** -- the `n()` `-1` sentinel renders as "only -1 distinct URLs
  appear in the brief" when a field is omitted. Fails closed correctly;
  the message is confusing. Cosmetic, out of the frozen criteria, and the
  tree is being graded. Queued, not patched.
- **N2** -- experiment_results called the multi-subagent fork deep's
  "FOURTH REQUIREMENT"; `researcher.md` titles it an "option" conditioned
  on caller request or >=3 separable sub-questions. **The Q/A is right and
  I was overstated.** It is deep's fourth *listed element*, and it is
  conditional. This does not change criterion 3, which independently
  mandates not adding `deep`, nor the reasoning for refusing (a
  conditional fork on an N=1 artifact rail is still a fork the rail
  cannot support). Corrected here rather than silently.
