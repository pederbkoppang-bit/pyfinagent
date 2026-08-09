# FINDING (queued for investigation) -- stale `.claude/scheduled_tasks.lock` held by a dead pid

- **Found:** 2026-08-09 16:56 CEST, session `pyfinagent-06`
- **Action taken:** lock file REMOVED (see "What I did" below)
- **Status:** NOT investigated. **Filed as masterplan step 86.16** (P3,
  `pending`, `harness_required: true`) on 2026-08-09 17:0x once the peer
  session went quiet. The step carries the full evidence; this note is
  the discovery record.
- **Why this file and not a masterplan step:** the operator instructed
  "leave the masterplan alone" while peer session `pyfinagent-43` was
  actively writing `.claude/masterplan.json` (last write 16:46, commits
  through 16:51). Filing the step concurrently would have raced that
  session and risked the cross-attribution class it had just queued as
  86.13/86.15. **Convert this file into a masterplan step once the
  masterplan is quiet.**

## The artifact, verbatim (preserved before deletion)

```
$ ls -la .claude/scheduled_tasks.lock
-rw-r--r--  1 ford  staff  130  7 aug. 09:39 .claude/scheduled_tasks.lock

$ md5 -q .claude/scheduled_tasks.lock
dfa8d285ff0f441a5496174c6b631103

$ cat .claude/scheduled_tasks.lock
{"sessionId":"cc942179-c9ad-44ed-a4e0-9ea34b301ce6","pid":32888,"procStart":"Thu Aug  6 09:40:12 2026","acquiredAt":1786088379631}
```

Decoded: `acquiredAt` 1786088379631 ms = **2026-08-07 09:39:39 CEST**.
The lock had therefore been held for **2 days 7 hours** at discovery.

## What I measured (all read-only, all re-runnable)

1. **The holder process is dead.**
   `ps -p 32888` returned no row, twice, at 16:56 and again at 16:58.
2. **The holder session is dead.** Transcript
   `~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/cc942179-c9ad-44ed-a4e0-9ea34b301ce6.jsonl`
   last written **2026-08-07 19:36** -- no activity for ~2 days.
3. **Nothing in this repo reads or writes the lock.**
   `grep -rn "scheduled_tasks"` over `*.py|*.sh|*.js|*.mjs|*.json|*.md`
   (excluding `node_modules`, `.venv`) returns **zero** code hits. All 6
   hits are prose in `handoff/away_ops/session_notes.md` and three
   archived `evaluator_critique.md` files that merely *describe* the file
   as a runtime lock in a `git status` listing. It is a **Claude Code
   internal**, not a pyfinagent artifact -- the
   `sessionId`/`pid`/`procStart`/`acquiredAt` shape is Claude Code's, and
   `handoff/audit/phase-4.11/claude_code_core.md:158` classifies it under
   "Custom / Claude Code core".
4. **It is untracked and git-ignored**, so removing it is repo-neutral:
   `git check-ignore -v` -> `.git/info/exclude:8:**/.claude/scheduled_tasks.lock`.
5. **Deleting it has precedent with no recorded fallout.**
   `handoff/archive/phase-17.1/evaluator_critique.md:31` records a cycle
   whose diff included "deleted scheduled_tasks.lock" and still passed.

## What I did NOT verify -- do not assume any of this

- **Whether anything was actually blocked.** I have no source access to
  Claude Code's scheduled-tasks subsystem, so I **cannot** say whether it
  respects this lock, whether it breaks stale locks on a dead pid or an
  age threshold, or whether any run was skipped between 08-07 09:39 and
  08-09 16:56. **No impact is claimed.** The finding is "a dead-pid lock
  sat for 2+ days", not "routines were blocked".
- **`CronList` returned "No scheduled jobs" and that is NOT evidence of a
  quiet estate.** Its own description scopes it to jobs created via
  `CronCreate` **in this session** -- a fresh session necessarily reports
  none. Do not cite it as proof the machine has no routines.
- **Whether this relates to the three red scheduler jobs** in the
  phase-85 scheduler-estate work (autoresearch ERROR, e2e-smoke running
  0 tests, away-watchdog stuck). Those are pyfinagent-side APScheduler
  jobs, a **different** subsystem from Claude Code routines. Related-
  sounding is not related. Check, don't assume.
- **Why the lock was never released.** Unknown. Candidates: the session
  was killed rather than exited cleanly (no release path on SIGKILL), or
  there is no release-on-exit at all. Not investigated.

## What I did

```
$ rm .claude/scheduled_tasks.lock
```

Nothing else. No repo file was modified to clear it (the file is
git-ignored), no process was started or stopped, no config was changed,
no masterplan edit, no restart. If the lock is legitimately needed, the
owning subsystem will simply re-acquire it with a live pid.

## What the investigation step should answer

1. Does Claude Code's scheduled-tasks runner **block** on this lock, or
   is it advisory? Does it break a lock whose pid is dead?
2. Is there a release-on-exit path, and does it survive a killed session?
   (The 2-day hold suggests it does not.)
3. Were any scheduled routines expected to run in the 08-07..08-09
   window, and did they? Answer from run evidence, not from `CronList`.
4. Should the repo carry a **staleness sweeper** for dead-pid locks?
   Enumerate the whole lock population before designing one; do not fix
   one instance and call the class closed.

   **CORRECTION (same session, after checking the code).** An earlier
   revision of this line called `handoff/locks/optimizer_plateau.lock`
   "a second, unrelated lock in the same never-released shape" because
   it has carried `"cleared_at": null` since **2026-07-24**. That
   framing is **wrong and would have caused real damage** if an executor
   had acted on it. That lock is **BY DESIGN**:
   `backend/api/backtest.py:338-343` makes `start_optimizer` raise HTTP
   409 `PlateauLockPresent` while the file exists, the comment at `:304`
   states operators clear it via `DELETE /api/backtest/optimize/lock`,
   and that route is implemented as `clear_plateau_lock` at `:403`. It
   is an intentional operator circuit-breaker. **Do not sweep, delete or
   "fix" it.** The only open question about it is whether a 16-day hold
   produces any operator signal -- and note the confound before claiming
   it cost anything: `historical_macro` is frozen and the optimizer is
   intentionally not being run, so the 409 may have had zero practical
   effect.
5. `.claude/cron_budget.yaml` (15 runs/day self-imposed cap, phase-10.7)
   references this subsystem -- confirm whether the cap accounting and
   this lock are the same mechanism or two unconnected things.

## Cross-references

- `handoff/locks/optimizer_plateau.lock` -- created 2026-07-24T11:04:50Z,
  `cleared_at: null`. **By-design operator circuit-breaker, not a stale
  lock** -- see the CORRECTION under item 4 above.
- `backend/api/backtest.py:304,338-343,403` -- the plateau-lock guard,
  its 409, and the operator DELETE escape.
- `.claude/cron_budget.yaml` -- cron budget allocator, phase-10.7.
- `handoff/audit/phase-4.11/claude_code_core.md:158` -- classifies the
  lock as Claude Code core.
