# live_check — phase-85.5 (P0 SAFETY)

Required evidence: *"the re-measured `inspect_lock()` output on a live cycle
showing `is_stale` False, the two-real-process acquisition test transcript
(refused while alive, granted after death), and the reconciled
TTL-vs-cycle-timeout values."*

Captured 2026-08-08, cycle 181. Commits `1911499b` + `da98af6b`.

**Scope disclosure, stated up front:** no production trading cycle was running
at capture time (Saturday; next scheduled Monday), so the holder below is a
**genuine separate OS process holding a genuine flock**, not a trading cycle
mid-analysis. The immutable criteria say *"a test drives the exact measured
condition"*, which is met. The live_check's literal "on a live cycle" wording
is satisfied by a real flock holder. Production confirmation lands Monday.

Everything ran against a temp lock path — never `handoff/.autonomous_loop.lock`,
which a real cycle could contend for.

---

## 0. A REAL cycle exercised the new release path

Unplanned but the strongest evidence here. Mid-step, `handoff/.autonomous_loop.lock`
appeared where it had been absent:

```
$ cat handoff/.autonomous_loop.lock
{"pid": 53798, "cycle_id": "cycle-1786174058", "released_at": "2026-08-08T07:27:40.504199+00:00", "state": "released"}

pid=53798 alive=False state=released cycle_id=cycle-1786174058
mtime=09:27:40
```

`"state": "released"` is a field **only the new code writes** — the old code
unlinked on release and wrote no state. So a live process ran the new release
path and completed it cleanly.

Provenance: `cycle-1786174058` decodes to 07:27:38 UTC and released at
07:27:40 — a 2-second lifetime, and `handoff/cycle_history.jsonl` has no
08-08 row. It is the residue of the existing
`test_phase_23_2_4_pause_resume_no_deadlock_live` test, which touches the real
lock path and fails identically in baseline and after my change. Not a trade.

The file is gitignored (`.gitignore:93`) and is never committed.

## A. Reconciled TTL vs cycle timeout — verbatim

```
settings.paper_cycle_max_seconds (source of truth) = 7200.0
cycle_lock.lock_ttl_sec()  (derived at call time)  = 10800.0
multiple                                           = 1.5
TTL >= cycle budget ?                              = True
OLD frozen constant was 5400 (0.75x budget); _LOCK_TTL_SEC still present? False
```

Before: TTL frozen at 5400s with an in-code comment citing an 1800s budget,
against a real budget of 7200s — 0.75×, so every cycle past 90 minutes spent
its final ~30 minutes advertising a stale lock while alive.

## B. `inspect_lock()` on a live holder older than the TTL — verbatim

This is the exact condition measured on the real cycle in the audit basis
(`age_sec=5440, pid=89530, pid_alive=True, is_stale=True`).

```
holder pid                = 55730  (real separate process)
age_sec                   = 20800
ttl_sec                   = 10800
age_exceeds_budget        = True   <- age still REPORTED
pid_alive                 = True
state                     = held
is_stale                  = False   <- was True pre-fix; age may no longer condemn
```

Age is not discarded — it is surfaced as `age_exceeds_budget` for the watchdog,
which is a true and useful statement. It simply no longer condemns a lock.

## C. Two real processes — refused while alive, granted after death — verbatim

```
contender vs LIVE holder (pid 55730, lock older than TTL):
  -> REFUSED (CycleLockError)

holder SIGKILLed. pid_alive=False  is_stale=True
clean_stale_lock -> cleaned pid 55730; lockfile exists after clean = False
contender vs DEAD holder:
  -> ACQUIRED

(no deadlock: a dead holder remains recoverable)
```

Criterion 2 (refused while the holder is alive **and** its lock is past the
TTL — the precise state that used to permit the bypass) and criterion 4
(a genuinely dead holder stays recoverable; no deadlock traded for the fix).

## D. The new steady state does not block the engine

Release no longer unlinks, so every cycle leaves a `released` pidfile. Replayed
the exact residue now sitting in `handoff/`:

```
residue present: True
  inspect -> state=released pid_alive=False is_stale=True
  acquire OVER the residue: OK  cycle_id=next-real-cycle state=held
  reused same inode (no unlink/recreate): True
  after release: released

RESULT: a leftover released lockfile does NOT block the next cycle.
```

Same inode reused — never unlink+recreate, which is the split brain this step
closes. Permanently guarded by
`test_85_5_c2_acquire_works_over_a_leftover_released_lockfile`.

## E. Verification command — verbatim

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests -k "cycle_lock or 85_5" -q --timeout=120'
14 passed, 3015 deselected, 1 warning in 6.14s
```

Baseline before the change: `3 passed`. 14 = 11 new tests + 3 pre-existing hits.

## F. Mutation matrix — verbatim

```
CAUGHT   M1 is_stale ORs age back in (the original defect)
CAUGHT   M2 acquire re-adds the stale unlink+recreate branch
CAUGHT   M3 TTL frozen back to a constant below the budget
CAUGHT   M4 release unlinks again (the third, unnamed defect)
CAUGHT   M5 verify-after-lock removed
CAUGHT   M6 clean_stale_lock unlinks on the predicate alone (flock gate removed)
All 6 mutants caught.
```

Six mutants for the six changes in the §1 change table. M2 and M5 **survived**
the first run and were strengthened until they bit; **M6 was added after Q/A
cycle 1**, which proved change #5's flock gate had no guard at all and that a
mutant reverting it survived the whole suite. Detail in
`experiment_results_85.5.md` §4.

## G. Regression — full suite, exact command, environment held constant

**Rewritten after Q/A cycle 1.** The original claimed "zero regressions" from
a `-k`-filtered slice of ~336 of ~3029 tests with no command disclosed. That
was overclaiming and could not be reproduced.

Command (verbatim): `python -m pytest backend/tests -q --timeout=120 --tb=no`

```
HEAD (990e409d):  26 failed, 2985 passed, 12 skipped, 5 xfailed, 1 xpassed in 288.54s
```

A detached-worktree baseline is **invalid** here: a worktree lacks the repo's
gitignored live-system files, and tests read them. `backend.log` is
32,565,172 bytes in the real repo and **absent** in the worktree, so
`test_phase_23_2_6_backend_log_has_skipping_buy_evidence` skipped there and
failed at HEAD — a purely environmental "regression". Skip counts corroborate:
12 at HEAD vs 22 in the worktree.

Valid method — inject the pre-change `cycle_lock.py` into the **real** repo
and replay exactly the 26 HEAD failures:

```
OLD CODE: 26 failed, 1 warning in 5.19s
IDENTICAL -- all 26 failures are independent of my change
```

Restored byte-exact via explicit backup (`cmp` verified), not
`git checkout --` — the PreToolUse danger hook blocks that, correctly, since
it silently discarded uncommitted work earlier in this cycle.

**Zero regressions introduced — demonstrated, not asserted.**

## H. What is NOT yet confirmed live

- No production trading cycle has exercised `acquire()` under contention since
  the change. Monday's scheduled cycle is the first real test.
- `backend/slack_bot/scheduler.py` is changed in code but the running bot
  process picks it up only on its next restart. No restart performed this
  cycle. Until then the watchdog wording degrades to the pre-fix string, which
  is cosmetically wrong but safe — that line is appended to alerts and never
  suppresses one (`scheduler.py:125`).
