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

### Immutable command, AFTER -- **CYCLE-1 MEASUREMENT**

> Everything from here to the cycle-2 follow-up is the CYCLE-1 record and
> is left as measured. **Current total is 73** (cycle 3); the file list and
> mutant count below are likewise cycle-1 and were extended in cycles 2-3.
> Read the follow-up sections for current state.

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

- strips `//` comment lines before indexing, so a **`//` comment** cannot
  stand in for code. **CORRECTED cycle 3**: this sentence originally said
  "a comment cannot stand in for code", which was measurably false -- the
  next Q/A defeated it with a `/* */` block comment. See the cycle-3
  section for the behavioural replacement;
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

---

# Follow-up -- cycle 3 (after the SECOND CONDITIONAL)

Q/A `wf_d0934c91-70b` verified both cycle-1 fixes genuinely landed (M5
reproduced KILLED at exactly 62/2; `research-gate.js` md5-identical to
cycle 1) and then found **two NEW** WARNs. Both real. Both fixed.

## W3 -- the hardened guard was STILL defeatable, and I stopped patching regexes

The Q/A defeated the cycle-2 guard with a `/* */` block comment:
`stripLineComments` filtered only `^\s*//`, so the block comment survived
and the first-match regex anchored inside it. Measured `ALL GREEN: 64
passed, 0 failed` while the production refusal sat AFTER the spawn.

**It explicitly told me a third regex patch was not the ask.** The named
terminal fix is to make the check BEHAVIOURAL, because the property is
"was `agent()` called?" -- and that is this step's own research finding F6
(EBTE: *structural is not semantic*) applied to my own guard. It was right.

**What I built** (`[6d]` in the checker):

- `loadDriver()` -- wraps the WHOLE script in an async function so the
  driver can run outside the Workflow runtime (legalising its top-level
  `return`/`await`, exactly as the runtime does). `loadModule()` could not
  do this: it slices the file at `phase('Research')` and keeps only the
  definitions.
- `driveRecording()` -- runs it with a RECORDING stub for `agent()`.
- The property is then **counted, not pattern-matched**.

**Known-positive first.** The section leads with a check that a SUPPORTED
tier really does spawn. Without it, "0 spawns" on the unsupported path
would be exactly the vacuous pass this section exists to eliminate -- the
instrument has to be shown working before its null reading means anything.

```
  ok   RECORDER WORKS: a SUPPORTED tier really does spawn (known-positive)
  ok   the first spawn is the stage-1 researcher (agentType researcher)
  ok   UNSUPPORTED tier spawns ZERO agents (measured, not scanned)
  ok   UNSUPPORTED tier returns gate_passed:false with the tier reported
  ok   UNSUPPORTED tier does NOT claim an agent returned null
  ok   BLIND run spawns ZERO agents (86.17 property, measured)
  ok   B1 (block-comment decoy + relocated refusal) IS CAUGHT behaviourally
```

**Independent reproduction of B1**, the Q/A's own method -- repo copy,
unmodified checker:

```
$ node scripts/qa/verify_research_gate_workflow.mjs      # B1-mutated repo copy
FAILED: 70 passed, 3 failed
  - UNSUPPORTED tier spawns ZERO agents (measured, not scanned) -- recorded 2 agent() call(s) -- the refusal did not prevent the spawn
  - the refusal is placed BEFORE the researcher spawn (else it saves no tokens)
  - M5 genuinely defeats the ORIGINAL naive guard (else it probes nothing)
```

The Q/A measured `ALL GREEN 64/0` on that same mutation. **B1 is KILLED**
-- by the behavioural spawn count and the two ordering probes. An earlier
revision of this block named different checks and a total (68+3=71) this
73-check suite cannot emit; it was typed, not captured. Corrected above.

Note what that run also showed: the *source scan* still printed `ok`
under B1. A check that says "ok" during a real breach is worse than no
check, so the scan now strips block comments too and is explicitly
demoted to cheap-secondary, with `[6d]` named as the authority.

## W4 -- my "class was audited" claim excluded the file I edited

The Q/A re-derived the scope from git instead of from my hand-made table
and found three survivors in `.claude/workflows/research-gate.js`:

1. `researcher.md:204,206-273` -- stale (deep section is at `:213`), staled
   by **this cycle's own edit to researcher.md**: the identical mechanism
   W2 named, and the identical string sitting in my own "FIXED" column.
2. `researcher.md:248-263` -- stale (fork at `:255`), **and** still said
   "fourth requirement", the exact N2 wording I accepted as overstated and
   then corrected only in the artifacts, not in the source.
3. `` `grep -c deep` on this file returns 0 `` -- **measures 8 now**. The
   comment defeated its own count by containing the word.

All three fixed: symbols instead of lines, "fourth LISTED ELEMENT
... CONDITIONAL" instead of "fourth requirement", and the self-defeating
grep replaced with a claim that cannot defeat itself -- now **enforced**
rather than asserted:

```
  ok   VALID_TIERS does not contain 'deep' (the tier is documented but NOT implemented here)
  ok   every 'deep' occurrence in the file is a COMMENT, never code
```

Mutation-tested, because a guard nobody has watched fail is not a guard.
Adding `'deep'` to `VALID_TIERS` in a repo copy:

```
$ node scripts/qa/verify_research_gate_workflow.mjs
FAILED: 68 passed, 5 failed
  - UNSUPPORTED tier spawns ZERO agents (measured, not scanned) -- recorded 2 agent() call(s) -- the refusal did not prevent the spawn
  - UNSUPPORTED tier returns gate_passed:false with the tier reported
  - UNSUPPORTED tier does NOT claim an agent returned null
  - VALID_TIERS does not contain 'deep' (the tier is documented but NOT implemented here) -- VALID_TIERS = ['simple', 'moderate', 'complex', 'deep']
  - every 'deep' occurrence in the file is a COMMENT, never code -- found in code: ["const VALID_TIERS = ['simple', 'moderate', 'complex', 'deep']"]
```

**Why my cycle-2 audit missed it, stated plainly.** I derived the scope
from the union of MY OWN commits -- and `research-gate.js` is not in them,
because the peer session's `git add -A` swept it into `cad38647`. So a
scope that looked derived was still wrong. Cycle 3 derives from the step
BASE (`089726f9..HEAD`), deliberately over-inclusive, since over-coverage
is the safe direction for a completeness claim. That scan also caught my
own zsh trap: `for f in $SCOPE` does not word-split in zsh, so the first
attempt silently audited nothing and printed a clean result.

## Also corrected

The cycle-2 sentence "so a comment cannot stand in for code" was
measurably false. It now says `//` comment, with a pointer to the
behavioural replacement.

## Checker

40 (baseline) -> 61 (c1) -> 64 (c2) -> **73 (c3)**, 0 failed.

---

# Follow-up -- cycle 4 (after the FAIL)

Q/A `wf_e262facc-cdc` returned **FAIL**. It confirmed all 9 criteria MET,
attacked the new behavioural guard with four mutants of its own (all
killed), and verified cycle 3 was comment-only via comment-stripped md5
identity. The FAIL was on **evidence integrity**, not on the work.

## The defect: I typed a transcript instead of capturing one

`live_check` §15 carried `FAILED: 68 passed, 3 failed` inside a block
formatted as shell output. The suite emits a fixed 73 checks, so 68+3=71
is impossible. I had spliced an old summary line (from when the suite had
71 checks) with failing-check names from a different mutant run. Two of
the three names were wrong. §16 listed 4 failure lines under a summary
saying 5.

There is no charitable reading of this. A step whose thesis is "never
certify an uncorroborated self-report" shipped remediation evidence that
its own tool could not have produced, and the test that catches it --
does passed+failed equal the suite size? -- costs one line of shell.

## Fixed

- Both blocks REGENERATED by re-running the mutants and piping stdout.
  Real values: B1 = `70 passed, 3 failed`; VALID_TIERS-gains-deep =
  `68 passed, 5 failed` (the fifth line, previously omitted, is the
  `every 'deep' occurrence` check).
- Kill attribution corrected: B1 dies by the behavioural spawn count and
  the two ordering probes, not by the checks I credited.
- Arithmetic self-check added to `live_check` §20 and run over every
  transcript block in both artifacts. All totals now reconcile.
- `live_check` §8's `:584`/`:399` corrected to `:598`/`:495` -- the third
  staling inside one step. No live citation carries a line number now.

## Class audit, again

I grepped EVERY `FAILED:`/`ALL GREEN:` line in both artifacts and totalled
each. Exactly two were fabricated; the rest reconcile against the suite
size at their cycle (40 / 61 / 64 / 73). The check is now written down so
it is re-runnable rather than dependent on me noticing.

---

# Follow-up -- cycle 5 (after the CONDITIONAL on Q1)

Q/A `wf_5a217e41-9b9` confirmed the cycle-4 evidence fix (both transcripts
reproduced byte-exactly under its own mutant rebuilds) and found one new
defect.

## Guarding one direction of a two-way distinction is half a guard

Mutant **Q1** (`tierUnsupported = !tierAbsent && !tierSupported` ->
`= !tierSupported`) survived at `ALL GREEN 73/0` while an ABSENT tier
stopped spawning and returned `tier_unsupported: ... tier "null"`. Every
caller that omits `tier` -- the common case -- would have broken silently.

Every check I wrote for the UNSUPPORTED half asserts that **nothing
happens**. None asserted the ABSENT half still **works**. The fixture made
it invisible: `TIER_ABSENT` had `supported: true`, which the driver never
produces (`tierSupported = !tierAbsent && ...` is false when absent), so
enforceGate-level probes were testing a state production cannot reach.

## Fixed

- `TIER_ABSENT.supported` -> `false`, matching the driver.
- Three driven ABSENT-tier checks in `[6d]`: still spawns; no
  `tier_unsupported` violation; reports `tier_requested: null` /
  `tier_applied: moderate`.
- **Fixture fidelity asserted against the running driver** by two new
  checks. The comment claiming the fixtures were "the shape the driver
  builds" was false on that field; a comment is now a test.

Q1 and Q5 both KILLED -- captures in `live_check` §26, arithmetic
reconciled in §27.

Checker: 40 -> 61 -> 64 -> 73 -> **78**, 0 failed.

---

# Follow-up -- cycle 6 (criterion 5: three checks without mutants)

Q/A `wf_344395f1-4ac` reproduced Q1/Q5 byte-exactly (corroborating that the
cycle-4 transcript regeneration was genuine), read the trajectory as
converging, and blocked on criterion 5: 3 of the 5 cycle-5 checks had no
demonstrated mutant. It built all three itself and all three were killed --
so the checks were sound and the gap was in the evidence.

## Fixed properly, not by pasting its captures

Criterion 5 requires the mutant to live IN the checker, so the three are
now standing tests in a new `[7b]` DRIVER-level matrix. The `[7]` matrix
could never have covered them: it probes through `enforceGate`, while every
`[6d]` check is end-to-end driver behaviour.

## My own mutant was the defective part

The `tier_requested` mutant SURVIVED at first. `tier_requested:
tierRequested,` occurs twice -- refusal path and main return -- and a
first-match replace hit the branch the ABSENT probe never runs. A weak
mutant is indistinguishable from a weak check until you look. Fixed with a
unique anchor, and every driver-mutant now carries an **anchor-uniqueness
assertion** so this fails loudly rather than producing a false survivor.

## Both WARNs closed

- Fixture fidelity now pins `unsupported`/`absent` -- the branch-steering
  fields -- not just `supported`, which `enforceGate` never reads.
- The known-positive is driven at `complex` instead of `moderate`, closing
  a hole where a supported tier could be silently downgraded: the very
  defect this step exists to fix, previously reintroducible without any
  check noticing. That mutant is in `[7b]` and killed.
- The `tier_absent_defaulted_ok` label is asserted present for ABSENT and
  absent for UNSUPPORTED.

Checker: 40 -> 61 -> 64 -> 73 -> 78 -> **92**, 0 failed, nothing removed.
`research-gate.js` remains untouched since cycle 3.
