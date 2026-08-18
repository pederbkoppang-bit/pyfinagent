---
name: write-first-collision-86-43
description: step 86.43 (PENDING) -- concurrent sessions truncate each other's step-scoped handoff artifacts; measured 24,904 -> 1,164 bytes; nothing locks handoff/current/ and the only repo lock is fail-open
metadata:
  type: project
---

Step **86.43** is `status: pending` and is the standing concurrency exposure on
`handoff/current/`. Read it before planning anything that writes or moves files
there.

**The measured incident.** Two Claude sessions run on this repo routinely. The
researcher's brief path is STEP-SCOPED (`handoff/current/research_brief_<sid>.md`),
and write-first makes creating it the FIRST tool call. When a second session
spawned a gate on a step the first already owned, the new researcher's first call
truncated the owner's brief: `research_brief_86.21.md` went **24,904 bytes ->
1,164 bytes** and stayed wrong ~4 minutes (2026-08-11 ~07:42Z-07:46Z). It was
recoverable ONLY because it had been committed.

**This is NOT phase-86.36.** 86.36 was a RETRY overwriting a PRIOR ATTEMPT at a
run-scoped path, fixed with a per-run filename stamp. 86.43 is a DIFFERENT SESSION
overwriting a LIVE artifact at a step-scoped path, where a run stamp cannot help
because the artifact is deliberately one-per-step.

**THIRD VARIANT, measured 2026-08-17 on step 86.69 -- a SAME-STEP RE-SPAWN.** No
concurrency at all: 86.69's first evaluation returned CONDITIONAL and the step was
PARKED, then the research gate was re-spawned on the same objective. The cycle-2
researcher's mandatory write-first opening call overwrote a **40,219-byte
COMPLETE cycle-1 brief** at the same step-scoped path. So the exposure is wider
than "two sessions": **any re-entry into RESEARCH on a step that already has a
brief** hits it -- park/re-spawn, retry-after-FAIL, or a research-on-demand leg.
Recovered ONLY because it was committed (`de895f25`), which is the same rescue as
the 86.21 incident -- twice now the protocol did not save it and `git commit` did.

**The cheap countermeasure that needs no protocol change:** run
`git log --oneline -- <brief_path>` (or `ls -la`) **before** the opening write. If
it returns a commit, restore with `git show <sha>:<path> > <path>` and APPEND a
labelled cycle-2 section instead of starting a new file. Keep exactly one
AUTHORITATIVE `brief_status` marker and state marker precedence in the header --
`enforceGate`'s stage-2 reader looks for "the brief's OWN envelope", so two
markers must not disagree.

**Not yet established (the step must establish, not assume):** whether
`contract_<sid>.md`, `experiment_results_<sid>.md`, `live_check_<sid>.md` share
the exposure (all are written by one role at a shared path); whether a
refuse-if-exists-and-larger precondition breaks legitimate re-runs; whether the
Q/A verdict sink is exposed now that 86.36 made it run-scoped.

**Explicit prohibition in the step text:** "DO NOT 'FIX' THIS BY WEAKENING
WRITE-FIRST" -- a partial brief surviving a drop is what makes the research gate
auditable.

**Why it matters mechanically:** there is NO lock on `handoff/current/`. The only
lock in the hook tree is `.claude/hooks/auto-commit-and-push.sh:299-301`
(`.git/pyfinagent-auto-commit.lock.d`, 20s wait, 120s stale) and it is
**fail-open by design** because the hook must never break a masterplan write.

**How to apply:** any step proposing a bulk move/cleanup of `handoff/current/`
(e.g. [[layout-invariant-86-105]]) intersects this open exposure -- check
`git status` for live modified artifacts there first, and prefer scoping the work
to `handoff/` root rather than touching `current/`.
