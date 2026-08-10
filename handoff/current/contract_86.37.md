# Contract -- step 86.37

**Step**: `86.37` (phase-86, **P1**, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-10 | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written**: BEFORE any code this time. The mtime ordering is the evidence.

---

## 1. Research gate -- REUSED, and the reuse is the first thing to judge

**No new researcher was spawned, deliberately, and this is disclosed rather than
buried.** Two reasons:

1. **The rail being fixed is the rail that would run the gate.** It dropped 25
   minutes ago on 86.29 after 181,082 tokens. Spending another ~180K on a run
   with a measured ~7.5% drop rate, to research how to survive that same drop,
   is a poor trade the operator should not have to pay for.
2. **A PASSED gate already covers this exact question.**
   `handoff/current/research_brief_86.31.md` -- run `wf_3cce0af1-102`, tier
   `complex`, **12 sources read in full, 64 URLs, gate PASSED**, cross-checked
   by the shipped script. Its subject is *"how does a Layer-3 rail survive a
   drop"*, which is this step's subject on the other rail.

### What that brief already establishes, and how it decides this design

| finding | bearing on 86.37 |
|---|---|
| **F4 -- BORN INERT, then one small atomic commit act.** SQLite's journal page-count starts at zero so a torn record is *inert*, not *ambiguous*. Atomic rename fixes torn VISIBILITY but not semantic INCOMPLETENESS. | The envelope must be written EARLY carrying `STATUS: INCOMPLETE`, and flipped as the final act. A brief that stops mid-loop must be unreadable as complete. |
| **F5 -- CRASH-ONLY: re-run, never salvage.** A crashed process's partial output is INFORMATION, never its RESULT. | A recovered brief is **evidence for the re-run**, never a gate pass. `gate_passed` stays `false` on every drop. |
| **F6 -- the verdict-shopping hazard.** A partial artifact must never itself be readable as the verdict. | Same shape: a partial brief must never be readable as `gate_passed: true`. |
| **F7 -- premature termination is measured UNAFFECTED by context budget** (`arXiv:2606.20724`), matching this project's own falsified compaction experiment. | Do NOT attempt to fix the drop by shrinking the prompt or the loop. Make the drop survivable, not rarer. |

**If an evaluator judges this reuse illegitimate, the correct remedy is to say so
and require a fresh gate before GENERATE is accepted.** I am not claiming a gate
I did not run; I am citing one that passed, on the same question, and naming the
cost that made me reuse it.

## 2. The defect, measured

**Script side.** `.claude/workflows/research-gate.js` awaits stage 1 bare:

```js
const envelope = await agent(PROMPT, { ...agentType: 'researcher'... })
```

When that throws, the workflow dies. No `enforceGate`, no `brief_verification`,
no return -- the caller gets an exception. **The file already solves this for its
OTHER agent call**: stage 2 is wrapped and sets
`verification = null // fail closed in enforceGate`. The asymmetry is the bug;
the remedy is the pattern already in the file.

**Agent side.** `.claude/agents/researcher.md` §"Output JSON envelope" says emit
it *"at the tail of every brief"*. A run that drops mid-loop never reaches its
tail. Confirmed on the live 86.29 artifact: 25,359 bytes, 15 sources, **no
envelope, no completion marker**, stopping inside *"rounds 5-6 (audit-class loop
continues)"*. The research was largely done and none of it was assessable.

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. REPRODUCE FIRST: demonstrate that a stage-1 agent failure currently kills the whole workflow and yields NO return value -- by simulation if a real drop cannot be induced on demand -- and show the same scenario after the fix returning a structured object. State which method was used.
2. A DROPPED stage-1 run returns `gate_passed: false` -- ALWAYS, with no input under which a drop yields true. Prove it by driving the drop path with a brief on disk that WOULD satisfy every floor, and showing the result is still false. An errored return is a FAILED gate and this step must not create an exception to it.
3. The dropped-run return CARRIES a recovery report: the brief's on-disk verification (existence, size, the URL cross-check) plus an explicit flag naming the drop and its error text, so a caller can distinguish 'nearly complete, cheap re-run' from 'nothing usable'. The flag must be a distinct field, not folded into gate_passed or violations.
4. The researcher writes a BORN-INERT envelope into the brief early and flips it as its final act, so a dropped brief is self-describing. Demonstrate the marker semantics: an INCOMPLETE brief must not be readable as complete, and a caller that checks it must be shown checking it.
5. The floors and the anti-trust discipline are UNCHANGED: >=5 sources, >=10 URLs, the recency scan, `enforceGate` still RECOMPUTING rather than trusting the self-report, and the claimed-URL-vs-brief cross-check all still enforced. Prove by re-running `node scripts/qa/verify_research_gate_workflow.mjs` and showing it green, and by an assertion that an over-claiming self-report is still rejected.
6. MUTATION-TESTED: revert the stage-1 try/catch and prove a NAMED assertion goes red; and mutate the drop path to return `gate_passed: true` and prove that is caught. A guard that has not been observed failing does not count.

**Verification command** (immutable):
`bash -c 'node --check .claude/workflows/research-gate.js && node scripts/qa/verify_research_gate_workflow.mjs'`

## 4. Plan

**P1 -- wrap stage 1, mirroring stage 2.** `try { envelope = await agent(...) }
catch (e) { envelope = null; railDropped = {error: String(e)} }`. Then let stage
2 and `enforceGate` run as they already do. `enforceGate(null, verification, …)`
already fails closed, so `gate_passed` is `false` by the existing logic -- **not
by a new special case**, which matters: a special case could be mutated away.

**P2 -- surface the drop as its own field.** `rail_dropped` in the return, with
the error text. Distinct from `gate_passed` and from `violations`, so a caller
can tell "failed the floors" from "the rail died" from "never had a subject"
(the existing `input_health`).

**P3 -- born-inert envelope in the brief.** `researcher.md` requires the envelope
block written within the first few tool calls carrying
`"brief_status": "INCOMPLETE"`, updated as sources land, flipped to `"COMPLETE"`
as the final act. Mirrors 86.31's marker deliberately.

**P4 -- a caller that checks it.** The brief verification must READ that marker
and report it, so an INCOMPLETE brief is visibly incomplete rather than merely
short.

**P5 -- extend `verify_research_gate_workflow.mjs`** with the drop path, the
always-false property, and the marker semantics.

**P6 -- mutation**: revert the try/catch; make the drop path return `true`;
neutralise the marker read.

### Explicitly NOT doing

- **Not** lowering any floor, and **not** letting a drop pass the gate.
- **Not** making `enforceGate` trust the self-report; the URL cross-check stays.
- **Not** changing `agentType:'researcher'` (the role needs `Write`).
- **Not** shrinking the prompt/loop to reduce drops -- F7 measured that
  premature termination is unaffected by context budget.
- **Not** re-running the 86.29 gate as part of this step; 86.29 stays gate-failed
  until a fresh run of its own.

## 5. References

- `handoff/current/research_brief_86.31.md` (gate PASSED, `wf_3cce0af1-102`) --
  F4 born-inert, F5 crash-only, F6 partial-never-a-verdict, F7 termination
  unaffected by context budget
- `.claude/rules/research-gate.md` (an errored return is a FAILED gate)
- `.claude/workflows/research-gate.js` (stage-2 try/catch precedent),
  `.claude/agents/researcher.md`, `scripts/qa/verify_research_gate_workflow.mjs`
- phase-86.31 as the shipped precedent on the Q/A rail
