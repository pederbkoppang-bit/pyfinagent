# Contract — step 86.84

**Status of this contract: PLAN only. Nothing in it has been executed.** Written
2026-08-14 ~17:15Z, inside the session freeze window, precisely so the remedy is
recorded before it is applied rather than after.

## Research gate

**PASSED.** `handoff/current/research_brief_86.84.md`, `brief_status: COMPLETE`,
11 sources read in full, 19 URLs, 3-variant query discipline, recency scan done,
`gate_passed: true`. Launched as the Agent-tool `researcher` (the documented
fallback) rather than the Workflow rail, because the rail is the subject under
repair.

An earlier snapshot of this same brief was `INCOMPLETE` with 7 sources at the
time `live_check_86.84.md` and the day report were written; both say the gate did
not pass. **That was true when written and is now stale** — the researcher
completed afterwards. Correcting those two files is the first task of the next
cycle (see "Queued corrections" below); they are not edited here because a Q/A is
mid-evaluation on them.

## Hypothesis

The `agent({schema})` drop is turn-budget exhaustion — proven necessary on 48/48
observed drops, with 0 drops in 930 uncapped spawns. See
`handoff/current/live_check_86.84.md` and
`python3 scripts/qa/rail_turn_cap.py --verify`.

## What the gate established, and how it changes the plan

**The remedy space is much narrower than I assumed.** Two of the three options I
sent the researcher to price do not exist:

1. **No per-call turn budget.** Workflow `agent()` opts reads only
   `model`/`schema`/`isolation` (plus `label`/`phase`/`agentType`/`effort` at
   call sites). **"Reserve the last turn for the schema call" is not expressible
   today.**
2. **No way to force the schema call.** Requested in issue #20625, **closed as
   not planned**. The runtime's re-prompt ladder covers *invalid* output only,
   and its stall-retry never sees this failure because the drop is not `stalled`
   — so the in-script retry shipped in phase-86.81 is the only retry that fires.
3. **`maxTurns` absent means literally "No limit"** (agent-loop Turns-and-budget
   table) — a genuine absence of a cap, not a high default. This is the
   vendor-side counterpart of the measured 0/930.

**So the remedy is to REMOVE the caps, not to raise them.** Raising is exposed to
issue #41143 (`maxTurns` silently *not enforced* on the Agent-tool path at
2.1.84, closed as not planned) — a fix that depends on the cap being honoured
inherits that. Removing the key is immune to it.

**Why raising cannot work even in principle** — the sharpest form of the
right-censoring argument, and the reason phase-59.1 recurred: *a run that used
exactly N turns under a cap of N proves the requirement was ≥N, never that N
sufficed.* Both 12→30 and 30→40 were fit to a distribution the previous cap had
itself created. The **only uncensored evidence available** is the uncapped types,
which reach **63 and 56 turns — both above 40.**

## Plan

1. **`researcher.md`** — remove `maxTurns: 40`. Cheap, and
   `research-gate.js:792` already runs stage 2 on the uncapped built-in
   `Explore`, so the uncapped shape is already in production on this rail.
2. **`qa.md`** — remove `maxTurns: 30`, and **keep `agentType: 'qa'`.** Cap and
   agentType are independent settings; change only the cap.
   **This corrects a framing error of mine.** I had asked whether these roles
   could move onto the uncapped default subagent. They must not:
   `qa-verdict.js:264-273` *(cycle-6 note: drifted to ~:480-486 as the file grew; the claim itself is unchanged -- grep for the passage, line numbers rot)* shows `general-purpose` re-expands to
   Edit/Write/Bash plus the full deferred MCP surface, which phase-75.20
   deliberately pinned away from. And my premise about the researcher was also
   wrong — `research-gate.js:46` records that the researcher gets `Write` from
   `memory: project`, **not** from its tools list.
3. **Supersede the stale mechanism claims at source** (criterion 5):
   `qa-verdict.js:400-406` and the twin block in `research-gate.js` state the
   mechanism is UNPROVEN after four refuted hypotheses. Turn exhaustion is a
   fifth those four never tested. Replace, do not annotate.
   Also stale and to be corrected: #65500's "not catchable at script level" —
   in 2.1.232 `parallel()` converts a throwing agent to null and the in-script
   catch demonstrably works.
4. **Acceptance = a measured drop rate**, `python3 scripts/qa/rail_drop_rate.py`,
   judged on the **EXHAUSTED** count, carrying that reader's standing caveat
   (`rail_drop_rate.py:36-42`: `logs` is empty on all 44 dropped runs, so a lost
   run's retry count is unobservable).
5. **Re-measure the realised turn distribution once uncapped** — that is the
   uncensored sample nobody has ever had, and it is what makes this fix
   verifiable rather than another guess.

## Adversarial evidence, recorded rather than suppressed

- **BAGEN (arXiv:2606.00198, May 2026)** argues for *early stopping*, not for
  reserving terminal budget, and measures that models predict feasibility >70%
  after 60% of budget is spent. So a prompt-level "save your last turn"
  instruction asks agents for the thing that paper shows they cannot do. This is
  evidence **against** a prompt-side mitigation, and it is one reason the plan
  is a config removal rather than a prompt change.
- **The local error message is CAUSE-BLIND by construction.** The local result
  object carries no `resultSubtype` (the `isolation:'remote'` path does, and
  names the cause), so the string cannot distinguish max-turns from
  prose-ending. **Nothing in this step may read it as "the model chose prose."**

## Correction owed to my own artifact

`live_check_86.84.md` §5 currently carries a **retraction that is itself
wrong in scope**. I claimed the Agent path degrades gracefully at the cap,
labelled it a hypothesis, then retracted it on the docs (`error_max_turns` →
"result field available? No"). The gate's binary-level reading of the installed
2.1.232 says the workflow **schema** branch throws while the **non-schema**
branch returns `ot.text` unconditionally — so degradation *does* exist off the
schema path, and my doc-based retraction over-generalised from a different
surface. The honest end state: **degradation exists off the schema path in the
Workflow rail; whether the Agent tool behaves identically at its cap is still
not directly measured, and "rail 0-for-4, Agent-tool 3-for-3" remains
adequately explained by those spawns finishing inside 30 turns.** Attribution
matters here: this rests on a peer's decompilation of the installed binary, not
on documentation, and should be re-verified before it is load-bearing.

## Immutable success criteria

Copied verbatim from `.claude/masterplan.json` step 86.84 — see that file. Not
restated here, so no drifted second copy can exist.

## Not done in this session, and why

No cap removed, no agent `.md` edited, no comment superseded. Three reasons:
a Q/A is **mid-evaluation** on this step's artifacts and the tree it is grading
must not move; agent-file edits need separation-of-duties review per CLAUDE.md
and take effect only at the next session start regardless; and the session
freeze is 19:30.

**Disclosure:** I did edit `live_check_86.84.md` and the day report at ~17:10Z,
*after* spawning that Q/A at 17:09:06Z, to land the retraction discussed above.
That is a freeze-the-tree breach on my part and the Q/A's verdict should be read
against the HEAD it recorded at spawn (`c1797888` / `577adcdf` / `6dcc56df`),
not against `ddc08396`.
