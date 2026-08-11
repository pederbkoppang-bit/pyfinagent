# Contract -- step 86.36

**Step**: `86.36` (phase-86, P2, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-11 (08:4x CEST) | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written BEFORE any code.** `git diff` on `scripts/qa/qa_wip.py`,
`.claude/workflows/qa-verdict.js`, `.claude/agents/qa.md` and
`.claude/hooks/qa-write-guard.sh` is empty at this moment.

**Concurrency**: peer session `pyfinagent-51` owns 86.25 / 86.34 / 86.29 / 86.38
and has confirmed it will not touch 86.36. I flip only this step.

---

## 1. Research gate

**PASSED** -- `wf_5dc835fa-de3`, tier `moderate`, brief
`handoff/current/research_brief_86.36.md` (24,395 chars). Enforced by the script,
not self-reported: **10 sources read in full** (floor 5), **18 URLs** (floor 10),
recency scan present, all 10 claimed URLs verified present in the brief,
`urls_collected_corroborated: 18 <= 18`, `brief_status_in_brief: COMPLETE`,
`rail_dropped: null`, and `gate_passed` **recomputed** with the self-report
agreeing.

**Disclosed limitation, carried forward rather than buried:** WebSearch quota
killed **2 of the 3 mandated query variants**, and the run took ~30 tool calls
against `moderate`'s 18. So the external survey is thinner than the tier implies.
The 10 full reads are real and the internal measurement is first-hand; treat the
*breadth* of the external scan as the weak leg.

### The finding that decides the design

**BORN-INERT INVERTS THE TIMING, AND THE SAFETY PROPERTY IS THE HAZARD.**
phase-86.31 requires the Q/A to write its record **immediately on spawn**, before
any analysis, so a drop leaves evidence. At a fixed path that same first write is
what **destroys the previous attempt**. The write that makes a crash survivable
is the write that erases the last crash's testimony. This is not a bug in 86.31 --
it is the direct consequence of pairing write-first with a shared name.

**Measured first-hand by the researcher, live, between two of its own tool
calls** (not reconstructed afterwards):

```
verdict_wip_86.34.md   4,921 bytes (WRITTEN 06:27:15Z)  ->  796 bytes (WRITTEN 06:40:32Z)
```

and the replacing file's own header reads *"This file OVERWRITES the DROPPED
run's WIP"*. Two other sink files changed concurrently in the same window. That
is a **second, independent** instance beyond the 86.31 cycle-4 destruction I
caused myself, and it happened in a peer session -- so the defect is not an
artifact of one session's habits.

### What the external work says (10 read in full)

- **`rename()` buys VISIBILITY, not durability.** POSIX specifies atomicity of
  the name change and says nothing about persistence; durability needs
  `fsync(file)` **and** `fsync(dir)`. So "write temp + rename" alone does not
  make a record survive a crash -- it makes the swap non-torn.
- **SQLite names a derived/shared path as a corruption mode outright**
  (`howtocorrupt.html`), which is exactly the shape here: two writers, one
  deterministic name.
- **Every production system surveyed solves it with a PATH COMPONENT or a
  rename-aside, never with locking**: Airflow `attempt={try_number}.log`,
  journald `.journal~`, k8s retaining exactly one prior container log.
- **Retention is BOUNDED everywhere.** Nobody keeps every attempt. k8s keeps one.
- **Crash-only design** (Candea/Fox): recovery evidence must be written by the
  crashing process, not reconstructed by the successor -- which is what makes
  destroying it at spawn the wrong trade.

### Two internal facts that constrain the fix

- **A cycle-suffixed name needs NO `qa-write-guard.sh` change.** The allowlist
  keys on the DIRECTORY (`.claude/agent-memory/qa/`), so any new filename inside
  it is already permitted. This keeps 86.36 disjoint from 86.33 and satisfies
  criterion 5 by construction rather than by argument.
- **Git is NOT a retention channel.** The gate checked: `HEAD` already holds the
  POST-destruction stub. The cycle-1 record referenced in the step text survives
  only because a commit happened to catch it mid-window. Do not propose "commit
  the sink" as the remedy.

## 2. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. REPRODUCE FIRST: drive two spawns for one step id and show, with byte counts, that the second destroys the first's record under the CURRENT design. Use a faithful simulation if a real double-drop cannot be induced; state which was used.
2. After the fix, a second cycle's record does NOT destroy the first: demonstrate two cycles' records coexisting and readable, with each carrying its own WRITTEN/COMPLETED stamps, and show the byte counts of both. Whatever naming is chosen must keep the artifacts OUT of audit_memory.py's non-recursive top-level glob -- prove it by running that auditor before and after and showing the output is unchanged.
3. `scripts/qa/qa_wip.py` resolves the record for a GIVEN cycle, and its `--spawned-at` staleness logic still works: STALE and IDENTITY_UNKNOWN must still fire, proven by the existing assertions in verify_qa_write_first_86_31.py continuing to pass (or their successors, if the resolver's signature changes -- in which case say so explicitly rather than quietly dropping them).
4. No record is EVER readable as a verdict: `report()` still carries no `verdict` key and `is_verdict: false`, for every cycle's artifact including the newest. Assert it, do not state it.
5. The guard is UNCHANGED or strictly tightened: show `git diff` on .claude/hooks/qa-write-guard.sh is empty, or that the change denies strictly more. The qa-role predicate added in 86.31 must still deny all of production code, tests, .claude/masterplan.json and other steps' handoff artifacts -- re-run verify_qa_write_first_86_31.py and show it green.
6. MUTATION-TESTED: revert the retention change and prove a NAMED assertion goes red; and mutate the retention so it keeps only ONE record and prove the coexistence assertion fires. A guard that has not been observed failing does not count.

**Verification command** (immutable):
`bash -c 'source .venv/bin/activate && python scripts/qa/verify_qa_write_first_86_31.py'`
-- note this is **86.31's** checker, so it proves the separation did not regress
and says **nothing** about retention. Criterion 6's mutation and criterion 2's
coexistence carry the real evidence. Stated here so a green command is not
mistaken for a green step.

## 3. Plan

**P1 -- A UNIQUE PATH COMPONENT PER RUN, not a counter.** The external
consensus is a path component; the open question is which. A **cycle counter is
rejected**: deriving `c<N>` requires read-then-write on the directory, and with
two live sessions that is a race that silently collides. The component must be
unique without coordination. Leading candidate:
`verdict_wip_<step>__<UTC-compact-timestamp>.md`, decided in GENERATE against
these stated requirements:
  (a) no change to any caller -- `qa.md` says "write it", not "name it";
  (b) two concurrent spawns for one step cannot collide;
  (c) the reader can still answer "the record for THIS spawn" from
      `--spawned-at`, which is criterion 3's existing contract.

**P2 -- BOUNDED RETENTION, and the bound is stated not implied.** Keep the
current record plus the **K most recent prior** (K decided in GENERATE, default
2 on the k8s/journald precedent). Pruning happens at write time, oldest first.
An unbounded sink is a slow leak in a directory the memory system reads.

**P3 -- REPRODUCE BEFORE FIXING (criterion 1).** Drive two spawns at one step id
against the CURRENT code and capture the byte counts. **A real instance already
exists** (4,921 -> 796 above, plus 86.31's 6,239 -> stub) but criterion 1 says
*drive it*, so it gets driven in a scratch sink and both are reported, with the
real one labelled as corroboration rather than as the driven reproduction.

**P4 -- `qa_wip.py` resolves BY SPAWN, keeping the existing semantics
(criterion 3).** `--spawned-at` already distinguishes current from stale; with
multiple records it selects the one belonging to that spawn and reports the
others as prior attempts. STALE and IDENTITY_UNKNOWN must keep firing. If the
signature changes, say so **loudly** in the live_check -- the criterion
explicitly forbids quietly dropping those assertions.

**P5 -- criterion 4 by ASSERTION over every record**, not just the newest: a
loop over all retained artifacts asserting `is_verdict is False` and `"verdict"
not in report()`. The step text is explicit that stating it is not enough.

**P6 -- criterion 2's auditor proof.** Run `audit_memory.py` before and after and
diff the output. The sink is already a SUBDIRECTORY and the glob is
non-recursive, so the expected result is "unchanged" -- but that is a prediction,
and I have been wrong this week predicting a hook's behaviour from reading it, so
it gets run.

**P7 -- criterion 6 mutation, two cells, both required:** revert the path
component (retention collapses to one file) and require a NAMED assertion to go
red; and cap retention at exactly one and require the coexistence assertion to
fire. Green control first, so neither cell can kill on an already-red suite.

### Explicitly NOT doing

- **Not** touching `.claude/hooks/qa-write-guard.sh`. Criterion 5 is satisfied by
  a `git diff` that is EMPTY. Its `workflow-subagent` / `general-purpose` gap is
  86.33's and is adjacent to the peer's hook work.
- **Not** committing the verdicts sink to git as a retention mechanism -- the
  gate measured that `HEAD` already holds a post-destruction stub.
- **Not** adding locking or coordination between sessions. No surveyed system
  does it, and it would couple two independent Claude sessions.
- **Not** changing the Q/A's criteria, judgement, effort or output schema
  (86.31 criterion 7 still binds).

### Risk

`qa_wip.py` and the write-first directive are live Layer-3 harness
infrastructure that BOTH sessions are exercising right now -- the peer has Q/As
in flight this hour. A change that breaks the writer silently loses crash
evidence, which is the opposite of this step's purpose. Every change must
preserve: write-first timing, the guard's allowlist, and `is_verdict: false`.
Because the peer is live, **GENERATE works in a scratch sink** and the real
sink is touched only when the change is proven.

## 4. References

- `handoff/current/research_brief_86.36.md` (gate PASSED, `wf_5dc835fa-de3`)
- SQLite atomic-commit + howtocorrupt; POSIX `rename()`; LWN 457667 / 191059;
  systemd journald file format; Airflow task logging; k8s cluster logging;
  Candea & Fox, *Crash-Only Software*
- `scripts/qa/qa_wip.py`, `scripts/qa/verify_qa_write_first_86_31.py`,
  `.claude/agents/qa.md` (write-first), `.claude/hooks/qa-write-guard.sh` (read-only here)
