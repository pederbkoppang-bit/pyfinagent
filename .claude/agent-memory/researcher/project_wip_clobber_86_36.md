---
name: project-wip-clobber-86-36
description: Step 86.36 durable-WIP research -- born-inert makes the clobber EARLIER not later; I measured a live destruction event mid-session; the write-guard does not intercept Bash mv; git is not a retention channel
metadata:
  type: project
---

Research for step 86.36 (durable checkpoint files for crash-prone workers).
Four findings that are NOT derivable by re-reading the code.

**1. Born-inert INVERTS the timing of the damage on a fixed path.** The
phase-86.31 discipline says a retry's FIRST act is to write a tiny
`STATUS: INCOMPLETE` stub. On a path that is fixed per step
(`.claude/agent-memory/qa/verdicts/verdict_wip_<sid>.md`), that stub lands ON TOP
of the prior attempt before any analysis happens. So the safety property (a torn
file is inert, not ambiguous) and the hazard (the retry destroys the previous
attempt's evidence) are the SAME write. Do not reason about them separately.

**Why:** measured first-hand 2026-08-11 during the 86.36 research session, ~4
minutes apart: `verdict_wip_86.34.md` went 4,921 bytes / `WRITTEN 06:27:15Z` ->
796 bytes / `WRITTEN 06:40:32Z`. The replacing file says so in its own header:
"This file OVERWRITES the DROPPED run's WIP that sat...". Two other files changed
in the same window (86.29: 628->3,926; 86.25: 3,473->8,543 INCOMPLETE->COMPLETE)
from at least two concurrent Q/A sessions -- concurrency in this sink is routine,
not hypothetical.

**How to apply:** never propose "atomic rename fixes it". Atomic rename fixes
torn VISIBILITY; it does nothing about a semantically complete overwrite by a
peer. The precedent fix is a PATH COMPONENT (Airflow `attempt={try_number}.log`)
or rename-aside-on-open (journald `.journal~`), not another marker.

**2. `qa-write-guard.sh` matches `Write|Edit` only -- Bash subprocess writes are
NOT intercepted** (its own comment says so). So an agent-performed `mv` rotation
sits entirely outside the hook. Prefer having the launcher or a helper own the
rotation over trusting agent prose. A cycle-suffixed filename needs NO guard
change, which keeps 86.36 disjoint from 86.33 (the guard-modification step).

**3. Git is not a retention channel for this sink.** It is tracked and NOT
gitignored, but `git show HEAD:` for the clobbered path already returned the
796-byte stub -- HEAD captured the post-destruction state. `handoff/harness_log.md`
:33484 records an earlier 6,239-byte artifact that "was never committed, so it
survives ONLY because I hand-copied it". Commits capture whatever is on disk when
a peer runs `git add -A`.

**4. Retention is BOUNDED in every precedent.** Kubernetes keeps exactly one
terminated container (`kubectl logs --previous`, `containerLogMaxFiles=5`);
journald archives + vacuums. Nobody keeps zero; nobody keeps all forever. N=1 or
small N is the defensible design.

Related: [[project_phase86_31_qa_write_first]], [[reference_macos_lock_primitives]].
