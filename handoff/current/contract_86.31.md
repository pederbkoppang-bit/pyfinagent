# Contract -- step 86.31

**Step**: `86.31` (phase-86, P1, `harness_required: true`)
**Phase**: PLAN
**Date**: 2026-08-10
**Driver**: Main (session `pyfinagent-06`), Opus / effort max

> Line-number citations below were measured 2026-08-10 and WILL go stale.
> Grep the named symbol. This project staled three citations inside one step
> on this same day.

---

## 1. Research gate summary

**PASSED** -- run `wf_3cce0af1-102`, tier `complex` (requested and applied).

| Field | Value |
|---|---|
| sources read in full | 12 (floor 5) |
| URLs collected | 64 (floor 10) |
| snippet-only | 52 |
| internal files inspected | 12 |
| brief | `research_brief_86.31.md`, 62,381 chars, independently read |
| cross-check | 12/12 claimed sources present; recency section present; `urls_collected_corroborated: 64 <= 65` |

Worth noting: `tier_supported_ok: "complex"` and the two corroboration
checks in that envelope are **86.28's shipped work running in production**.

### Findings that decide the design

**F1 -- THE OBVIOUS DESIGN IS A CONFUSED DEPUTY. Verified independently by
Main, not taken on report.** The natural sink for a Q/A verdict is
`handoff/current/evaluator_critique_<sid>.json`. That path is
`verdict_gate.py::resolve_verdict_source`'s **first** candidate, and
`auto-commit-and-push.sh` reads that gate to decide whether to HOLD the
push. Allowlisting it would let the evaluator **authorise its own push** by
writing `PASS`. The step's own P1 sketch was heading there.

**F2 -- THE Q/A ALREADY HAS `Write`.** `qa-verdict.js` documents it in
source: "the loader injects Write/Edit into the qa type past its frontmatter
allowlist (probe-proven; `disallowedTools` is silently ignored)". So the
question was never *whether* to grant a write, only *which sink*.

**F3 -- A SAFE SINK ALREADY EXISTS, and no guard change is required.**
`qa-write-guard.sh` denies `Write`/`Edit` for `agent_type == "qa"` *unless*
the path is under `.claude/agent-memory/qa/`. Measured: that directory is
tracked (41 files, not gitignored, so writes survive a drop and are visible
to Main), and **no hook or gate reads it** -- `grep` over `.claude/hooks/`
returns only the guard's own definition of the constant. No authority path,
therefore no confused deputy.

**F4 -- BORN INERT, then one small atomic commit act** (SQLite atomic-commit:
the journal's page-count starts at zero, so a torn record is *inert*, not
*ambiguous*). Atomic rename (LWN) fixes torn *visibility* but not semantic
*incompleteness* -- a fully-flushed half-analysis is still a half-analysis.

**F5 -- CRASH-ONLY: re-run, never salvage.** A crashed process's partial
output is INFORMATION, never its RESULT.

**F6 -- THE VERDICT-SHOPPING HAZARD.** `qa.md`'s no-auto-PASS rule survives
only if a partial artifact is never itself readable as a verdict; otherwise
a post-drop respawn quietly becomes verdict-shopping.

**F7 -- THE FALSIFIED FIX IS CORROBORATED EXTERNALLY.** Premature
termination is measured **unaffected by context budget**
(`arXiv:2606.20724`), matching this project's own failed compaction
experiment. MAST puts the failure at 7.82%, with structural verification
worth +15.6pts. The Constraint-Tax paper is deterministic/open-weight-only
and does **not** explain this symptom.

---

## 2. Hypothesis

A dropped Q/A destroys a *completed* evaluation only because the role has no
write-first discipline -- not because it lacks permission. Giving it the same
write-first habit the researcher already has, aimed at the sink it may
already use, converts a total loss into a recoverable one **without touching
the guard, the criteria, or the judgement**.

**Design, revised by F1/F3 away from the step's own sketch:**

- Sink: `.claude/agent-memory/qa/verdict_wip_<step_id>.md`. Allowed by the
  existing guard, read by no gate, tracked so it survives.
- **No allowlist is added and no deny is removed.** The blast radius is a
  prompt/doc change plus tests.
- Born inert (F4): the file carries an explicit `STATUS: INCOMPLETE` marker
  from its first byte; the final act flips it to `COMPLETE`. A truncated
  file is therefore *inert*, not *ambiguous*.
- Main's contract (F5/F6): a recovered WIP is **evidence for the next
  spawn**, never a verdict. `errored return is NO VERDICT, NEVER PASS`
  is restated unchanged.

---

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. the Q/A can WRITE its verdict artifact for the step under evaluation and NOTHING else -- demonstrated by driving the guard with both an allowed path and at least four denied paths (production code, a test file, .claude/masterplan.json, and another step's handoff artifact), with the deny decisions captured verbatim
2. the no-self-eval guarantee is preserved and shown to be preserved: prove the Q/A still cannot modify the work under evaluation, and state which mechanism enforces it now that the blanket deny is gone
3. a DROPPED Q/A run leaves a recoverable partial verdict on disk. Demonstrate with a real interrupted run or a faithful simulation, and record what is recoverable versus lost. If a partial verdict can be mistaken for a complete one, that is a defect -- the artifact must carry an explicit completion marker that a caller checks
4. Main MUST NOT treat a recovered partial verdict as a verdict. The recovery path is documented as evidence-for-the-next-spawn only, exactly as it was used on 2026-08-10, and the rule that an errored return is NO VERDICT NEVER PASS is restated unchanged
5. the guard change is MUTATION-TESTED: prove the allowlist rejects a path outside it, and prove the deny path still fires, with output recorded. A guard that has not been observed denying is not a guard
6. the intermittent-drop measurements are recorded as a table (run id, outcome, tokens, tool uses) so a future reader can see the populations overlap and does not re-try the falsified volume hypothesis
7. no change to the Q/A's criteria, judgement, effort, or output schema beyond adding a completion marker if criterion 3 requires one

**Verification command** (immutable):
`bash -c 'test -f .claude/hooks/qa-write-guard.sh || test -f .claude/hooks/lib/qa_write_guard.py; echo guard-present=$?'`

**live_check**: `live_check_86.31.md` -- verbatim guard decisions for the
allowed path and four denied paths, mutation output showing the allowlist
and the deny path each observed failing, the recoverability demonstration,
and the run-outcome table.

### A note on criterion 2's premise

Criterion 2 says "now that the blanket deny is gone". Under the revised
design **the blanket deny is NOT gone** -- nothing is removed. The criterion
is answered as written: the enforcing mechanisms are unchanged (the guard's
memory-dir restriction, `qa.md` prose, and Main's post-verdict `git status`
cleanliness check), and the answer must state that no deny was removed
rather than quietly satisfying a premise that no longer applies.

---

## 4. Plan

**P1 -- Write-first for the Q/A.** Add to `qa.md` and the `qa-verdict.js`
prompt: create `.claude/agent-memory/qa/verdict_wip_<step_id>.md` within the
first few tool calls, append findings as they are established, and flip the
status marker to `COMPLETE` as the final act before returning. Mirrors the
researcher's write-first wording deliberately.

**P2 -- Born-inert marker.** First line `STATUS: INCOMPLETE -- not a
verdict`; final act rewrites it to `STATUS: COMPLETE`. A reader that does
not see `COMPLETE` must treat the file as notes.

**P3 -- Main-side recovery contract.** Document in
`docs/runbooks/per-step-protocol.md` §4: on an errored/empty return, read
the WIP as EVIDENCE for the next spawn, never transcribe it into
`evaluator_critique_*`, and never let it satisfy the verdict gate. Restate
`NO VERDICT, NEVER PASS` verbatim.

**P4 -- Tests.** A re-runnable checker under `scripts/qa/` that drives the
guard with one allowed and four denied paths, mutation-tests the deny path,
and asserts the marker semantics (an INCOMPLETE file is not readable as a
verdict). Criterion 5 requires the deny path be OBSERVED failing.

**P5 -- The drop table** (criterion 6), from today's measured runs.

### Explicitly NOT doing

- **Not** allowlisting `evaluator_critique_<sid>.json` or any path
  `verdict_gate.py` consumes (F1).
- **Not** removing or loosening any deny in `qa-write-guard.sh`.
- **Not** touching the Q/A's criteria, judgement, effort, or output schema
  beyond the marker (criterion 7).
- **Not** salvaging a partial into a verdict (F5/F6).

### Discovered defects to QUEUE, not fix here

Found while verifying F3; each is independent of write-first and none is
introduced by this step:

1. `qa-write-guard.sh` uses `os.path.normpath`, not `realpath` -- a symlink
   inside `.claude/agent-memory/qa/` pointing outside would pass (CWE-59).
2. No project-root anchor on the path comparison.
3. The Bash gap is real but **already documented in the guard's own header**
   with a named covering control (Main's post-verdict `git status` check).

---

## 5. References

- `handoff/current/research_brief_86.31.md` (gate PASSED, 12 sources)
- SQLite atomic commit; LWN atomic-rename; Saltzer & Schroeder; CWE-367
  (TOCTOU); CWE-59 (symlink); crash-only software (EPFL);
  `arXiv:2606.20724` (termination unaffected by context budget); MAST
- `.claude/hooks/qa-write-guard.sh`; `.claude/hooks/lib/verdict_gate.py`
  (`resolve_verdict_source`); `.claude/hooks/auto-commit-and-push.sh`
  (verdict gate block); `.claude/workflows/qa-verdict.js`; `.claude/agents/qa.md`
