# Experiment results -- step 86.37

**Step**: `86.37` (phase-86, **P1**) | **Phase**: GENERATE | **Date**: 2026-08-10
**Operator-directed**: "make sure to fix the researcher!"

## 0. What changed

A dropped research gate no longer destroys the run. Two edits, one per cause:

| file | change |
|---|---|
| `.claude/workflows/research-gate.js` | stage 1 wrapped in try/catch (mirroring stage 2); `rail_dropped` surfaced as its own return field; the brief's born-inert marker made a HARD gate in `enforceGate`, failing closed on a missing value |
| `.claude/agents/researcher.md` | the envelope is written into the brief EARLY carrying `brief_status: "INCOMPLETE"`, updated as sources land, flipped to `COMPLETE` as the final act |
| `scripts/qa/verify_research_gate_workflow.mjs` | +13 assertions; spawn locator widened; wrapper assertion re-pinned by proximity; fixture teaches the marker |

## 1. Contract-before-generate: SATISFIED, and provable

| artifact | mtime |
|---|---|
| `contract_86.37.md` | **17:25:58** |
| `.claude/workflows/research-gate.js` (first edit) | after |
| `experiment_results_86.37.md` | last |

I checked `git diff --stat` on both production files immediately after writing
the contract and it was **empty**. This is the ordering 86.30 breached hours
earlier; I am not asking for credit, only recording that the check now passes on
evidence rather than assertion.

## 2. Research gate: REUSED, disclosed, judge it

**No new researcher was spawned.** The rail being fixed is the rail that would
run the gate, and it dropped 25 minutes earlier at 181,082 tokens. Instead the
contract cites `research_brief_86.31.md` -- run `wf_3cce0af1-102`, **12 sources,
64 URLs, gate PASSED** -- whose subject is "how does a Layer-3 rail survive a
drop", i.e. this step's subject on the other rail. Its F4 (born inert), F5
(crash-only), F6 (a partial must never read as the verdict) and F7 (termination
unaffected by context budget) each decide a design point here. **If an evaluator
judges the reuse illegitimate, the remedy is to require a fresh gate.**

## 3. Criterion-by-criterion

| # | Criterion (abridged) | Evidence | Status |
|---|---|---|---|
| 1 | reproduce that a stage-1 failure kills the run; show it returning after | mutation R1 on a **syntactically valid** unwrap -> 2 assertions red | MET |
| 2 | a dropped run returns `gate_passed:false` ALWAYS, even with a brief that clears every floor | assertion "a DROPPED stage 1 (null envelope) fails the gate even with a COMPLETE brief on disk"; mutation R2 | MET |
| 3 | the dropped return CARRIES a recovery report + a distinct drop flag | `rail_dropped` field; `brief_verification` still computed; mutation R5 | MET |
| 4 | born-inert marker, and a caller shown checking it | stage-2 schema + prompt read `brief_status_in_brief`; `enforceGate` hard-gates it; mutations R3/R4 | MET |
| 5 | floors and anti-trust discipline unchanged | immutable command **110 passed, exit 0** (was 97) | MET |
| 6 | mutation-tested, incl. reverting the try/catch and making the drop pass | **5 cells, all KILLED**, each naming an assertion | MET |

**`gate_passed:false` on a drop is decided by the EXISTING fail-closed logic**,
not a new special case. That is deliberate: a special case is something a later
edit can quietly invert, and mutation R2 exists to catch exactly that.

## 4. Three defects in my own work, found and fixed during the step

**(a) The marker gate was born DEAD.** My first version tested only the three
known values, so a verification object that OMITTED the field matched none of
them and the check silently did nothing -- green and blind. Now it fails closed,
and two assertions plus mutation R4 pin it.

**(b) My first R1 mutation was a mis-attributed kill.** Removing `try {` while
leaving `} catch` made the file a SyntaxError, so it "killed" by not parsing
rather than by any assertion detecting the missing wrapper. Redone as a
syntactically valid unwrap -- which then revealed (c).

**(c) The wrapper assertion was too loose to fire.** It was
`/try\s*\{[\s\S]*?envelope = await agent\(PROMPT[\s\S]*?\}\s*catch/`, which any
EARLIER `try {` (`classifyArgs` has one) plus any LATER `catch` (stage 2 has one)
satisfies. It stayed green against a valid unwrap. Re-pinned by **proximity** --
nearest `try {` within 200 chars before the spawn, a `catch` within 600 after --
and it now fires.

## 5. A checker locator I widened, and the proof I did not weaken it

My try/catch removed the literal `const envelope = await agent(PROMPT` that two
assertions used as their landmark, so `spawnAt` became -1 and both failed -- not
because the ordering broke, but because the locator lost its landmark. I widened
it to match both forms.

**Widening a checker to go green is exactly the move that deserves suspicion, so
I mutated a REAL breach**: relocating the tier-refusal to genuinely after the
spawn turns **3 assertions red**. My first attempt at that probe inserted the
block *before* the spawn and wrongly reported SURVIVED -- caught by the probe's
own sanity assertion (`refusal AFTER spawn`), which is why that assertion is
there.

## 6. Verbatim

```
$ bash -c 'node --check .claude/workflows/research-gate.js && node scripts/qa/verify_research_gate_workflow.mjs'
ALL GREEN: 110 passed, 0 failed          exit=0

mutation (hermetic mini-repo, repo tree never written):
  CONTROL              ALL GREEN 110 passed
  R1-UNWRAP-VALID      KILLED  2 failed  (wrapper proximity + rail_dropped)
  R2-DROP-PASSES       KILLED  1 failed  (drop path must not assign gate_passed)
  R3-MARKER-INERT      KILLED  2 failed  (INCOMPLETE must fail the gate)
  R4-FAIL-OPEN-MARKER  KILLED  4 failed  (missing marker must fail closed)
  R5-DROP-FIELD-GONE   KILLED  1 failed  (rail_dropped must be its own field)
```

## 7. Scope and what I cannot verify

- **This does not make the rail drop less often.** F7 measured premature
  termination unaffected by context budget; the fix makes a drop survivable.
- **No live drop was induced** -- the drop path is exercised by mutation and by
  driving `enforceGate` with a null envelope. A real drop is not summonable.
- **`researcher.md` binds the next Agent-tool spawn**, not this session's roster;
  the Workflow rail reads it from disk at runtime, so it is live there now.
- **86.29 stays gate-failed.** This step does not re-run it.
