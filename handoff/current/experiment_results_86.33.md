# Experiment results -- step 86.33

**Step**: `86.33` (phase-86, P2) | **Phase**: GENERATE (PARTIAL -- P0 only)
**Date**: 2026-08-11 | **Driver**: Main (`pyfinagent-06`)

**STATUS: P0 SHIPPED, P1-P4 BLOCKED ON DATA THAT DOES NOT EXIST YET.** This is
not a pause for convenience; the step's own plan says measure before redesigning,
and the measurement has not been possible yet. Details in §3.

## 1. What shipped (P0)

`.claude/hooks/qa-write-guard.sh` now captures and logs `agent_id`, a second
PreToolUse identity field the guard had **never read** (`grep -c agent_id` → 0
before, commit `8a9a4293`).

**LOG-ONLY BY DESIGN.** The value is recorded and used in **no** decision. The
whole finding of this step is that authorization currently rests on `agent_type`,
a string the **spawner chooses**; changing policy on a *second* unmeasured field
would repeat that mistake one level down.

Verified rather than asserted, because this guard is **fail-open** and one
apostrophe disabled it entirely on 2026-08-10:

```
embedded python: 0 apostrophes, compiles           bash -n: OK
prove_qa_write_separation_86_31.py : 15/15, exit 0  (behaviour unchanged)
verify_qa_write_first_86_31.py     : green,  exit 0
real hook driven: qa -> backend/main.py rc=2, agent_id captured
```

## 2. The measured basis for the design (re-derived by me, one number corrected)

| claim | research gate | my measurement | verdict |
|---|---|---|---|
| distinct `agent_type` values | 72 | **72** | exact |
| `general-purpose` evaluator-critique writes | 15 | **15** | exact |
| `workflow-subagent` legitimate writes | "~50" | **82** | **corrected (undercount)** |

`workflow-subagent`'s 82 Write/Edit events are all outside the qa memory dir and
all plainly legitimate GENERATE work (`kill_switch.py` ×12, its rotation test ×7,
`KillSwitchPanel.tsx` ×5, `paper_trading.py` ×4). **Widening the predicate to
match it would break GENERATE**, which is what the guard's docstring says and
what the naive reading of this step would have done.

**The asymmetry that reshapes the step:** `general-purpose` wrote **15
`evaluator_critique_*.md` events** — the artifact Main is contractually the
verbatim scribe for, the same class as the `qa-80-2` breach behind 86.31. So the
two unmatched identities are **not** one bucket: one has a track record of
writing the protected artifact, the other of writes that must not be blocked.

## 3. P1 IS BLOCKED, AND THE REASON IS A CONTAMINATION TRAP I ALMOST FELL INTO

P1 (invert to a narrow permit) depends on whether `agent_id` actually
discriminates. First reading of the post-change log looked decisive:

```
post-change invocations         : 150
with a NON-EMPTY agent_type     : 145   <- "subagent calls"
distinct agent_id values        : {''}  <- "the runtime never populates it"
```

**That conclusion would have been false.** Attributing the records:

```
attributable to MY OWN prover   : 112 of 150
  (prove_qa_write_separation_86_31.py sends {agent_type, tool_name, tool_input}
   and NO agent_id -- so the empty value is MY payload's shape, not the runtime's)
remainder                       : 38
  agent_type='qa'                 30   <- ambiguous: my prover uses that literal too
  agent_type='<empty>'             5   <- Main's own calls, which carry neither field
  research-82-0 / res-78-1 / researcher-80-4-death : 3  <- the other checker
```

**Zero genuine runtime subagent calls have hit the guard since the change.** The
log cannot answer the question, and every "empty `agent_id`" in it is explained
by my own harness not sending the field.

This is the third time in two days that this log's contamination has nearly
produced a wrong conclusion (2026-08-10: a substring rule mis-scoring 118
traversal paths; the "156 organic writes" figure withdrawn as unrecoverable).
**The log records DECISIONS from every source, including test drives, and any
population derived from it must be attributed before it is used.**

**What unblocks P1:** one real Layer-3 spawn writing its verdict file, which
fires the guard with the runtime's own payload. Both sessions run Q/As routinely,
so this arrives on its own — it is not worth manufacturing.

## 4. What I cannot verify

- **Whether the runtime populates `agent_id` at all.** Open. If it is always
  empty for subagents, P1's premise collapses and the plan must change to a
  non-identity control — that possibility is live and is not assumed away.
- **No Q/A has run on this step.** No verdict is claimed.
- **The research gate ran ZERO of 3 mandated search variants** (WebSearch
  200/200 exhausted at spawn; the researcher self-reported `gate_passed: false`
  and the script's floors overrode it). The 9 full reads are canonical and
  reachable without search, but **no 2025-2026 development is covered**. Any
  GENERATE claim resting on current best practice must be re-gated.
- **Not touched**: `archive-handoff.sh` and anything of 86.29's (peer's).
