# 86.33 -- the blocking measurement, taken on REAL runtime traffic

**Status: this UNBLOCKS 86.33's P1. It does not close the step** -- 86.33 still
needs its own research gate, contract, GENERATE and Q/A. This file is evidence for
whoever executes it, not a substitute for the cycle.

**The blocker, verbatim from the binding goal:** *"P1 BLOCKED: the post-change log
is 112/150 my own prover, so it cannot say whether the runtime populates the field.
One real spawn unblocks it."*

Three real Q/A spawns (86.41 cycles 1-3, via the Workflow rail) plus researcher
runs happened on 2026-08-11 after the P0 shipped. That is the real traffic the step
was waiting for.

---

> **CORRECTED 2026-08-11 (~15:4x CEST). THE 70% IN THIS FILE IS AN ARTIFACT OF MY
> OWN FILTER.** I counted rows by `agent_type in (qa, researcher)` without asking
> whether each came from a real spawn or from my own prover
> (`scripts/qa/prove_qa_write_separation_86_31.py`), which drives the hook directly
> with fabricated payloads. **All 10 rows I counted as "missing" are prover rows** --
> `/tmp/evil.md`, `/tmp/x.md`, `../../../etc/x`, `backend/main.py`,
> `qa/MEMORY.md`. The 86.33 research gate independently measured **63/63 real
> subagent writes carry `agent_id`, and 0/77 of Main's**.
>
> **The correct figure is 100%, not 70%.** Everything below about field SHAPE
> (absent-not-empty, joins to a transcript) stands; only the ratio was wrong.

## Answer: YES, the runtime populates `agent_id`

**RULE -- the denominator is rows written AFTER the field existed.** The P0
(`8a9a4293`) shipped at **2026-08-11T07:55:14 UTC**. Rows before that cannot carry
the field, and including them measures the deploy, not the runtime.

```
rows with UTC ts >= 2026-08-11T07:55:14 : 78
  with agent_id    : 23  (29.5% of all 78)
  without          : 55

  agent_type               HAS    NO
  (empty)                    0    45
  qa                        12     8
  researcher                11     2
```

**Among rows that carry a role (`qa` / `researcher`): 23 of 33 = 70% populated.**
The 45 unpopulated rows all have an **empty `agent_type`** and never carry an
`agent_id` -- consistent with main-session writes rather than subagent writes, since
the guard logs every Write.

Field shape, measured:

```
keys on a POPULATED row: ['agent_id', 'agent_type', 'file_path', 'tool_name', 'ts']
keys on an EMPTY row   : [            'agent_type', 'file_path', 'tool_name', 'ts']
```

The key is **absent**, not empty-string, when the runtime does not supply it.

Sample real ids (truncated): `ab3ff92edf47e42e5` -- this is the 86.41 **cycle-1
Q/A**, cross-checkable against its transcript
`subagents/workflows/wf_f819502b-c1e/agent-ab3ff92edf47e42e5.jsonl`. So the id is a
genuine runtime identity that joins to a transcript on disk, not an opaque token.

## TWO WRONG DENOMINATORS I HIT GETTING HERE -- both would have been reported

1. **Whole-log denominator: "24 of 7,635 (0.3%)".** Meaningless -- 7,611 of those
   rows predate the field. The correct denominator is post-change rows.
2. **A timezone comparison that produced a confident ZERO.** I cut at
   `2026-08-11T09:55:14` from `git log`, which is **local (CEST)**, against log
   timestamps in **UTC**. The log's last entry is 09:45 UTC, so the filter returned
   **0 rows** and I nearly recorded "the runtime never populates it" -- the exact
   inverse of the truth. Correct cut is 07:55:14 UTC.

Both are the same defect the goal's TRAPS section names: a clean-looking result that
is a property of the query rather than of the world.

## What 86.33's executor still has to do

- **Explain the 10 role-typed rows with no `agent_id`** (8 `qa`, 2 `researcher`).
  Is the field absent for a whole spawn PATH (Agent-tool vs Workflow rail), or
  intermittently within one? That distinction decides whether `agent_id` can be a
  load-bearing predicate. **Do not assume it is the rail split** -- that hypothesis
  is untested here.
- **Confirm `agent_id` is unforgeable** before any guard decides on it. It is
  currently LOG-ONLY by design (`qa-write-guard.sh:53-55`), and the P0 commit
  message says "decide nothing on it yet".
- Recall-test any predicate against the **whole class** of spawn names. The prior
  defect (`reference_agent_type_is_the_spawn_name`) was a NAME match that missed 27
  named Q/As -- the named types in this very log (`qa_86_31`, `qa-80-2-c2`,
  `QA-80-2`, `researcher-80-4-death`, ...) are that class, and a rule keyed on
  `agent_type == "qa"` would miss every one of them.
