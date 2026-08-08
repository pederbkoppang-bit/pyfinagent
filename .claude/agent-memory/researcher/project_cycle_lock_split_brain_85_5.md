---
name: project-cycle-lock-split-brain-85-5
description: phase-85.5 cycle_lock research — a THIRD defect on the normal release path; the -k gate misses the real regression suite; macOS spawn makes a child test hit the REAL lockfile
metadata:
  type: project
---

Research gate for phase-85.5 (P0, `backend/services/cycle_lock.py`), 2026-08-08.
Findings that are NOT re-derivable from the step premise or the code alone.

**A THIRD defect the step premise does not name.** The step cites (1) the OR'd
`is_stale` predicate at `:79` and (2) the stale-path unlink+reopen at `:119-125`.
There is a third: the `finally` at `:151-159` unlinks (`:153`) **before** `LOCK_UN`
(`:157`). MEASURED in a scratch dir (macOS 26.5, Py 3.14.4): A holds; B opens the
same inode and is refused; A unlinks-then-unlocks; B re-locks the now-orphaned
inode; C creates a new inode at the path and locks it — `BOTH BELIEVE THEY HOLD
THE CYCLE LOCK: True`, `SAME INODE B vs C: False`. This fires on the **normal
successful release path** with no TTL expiry and no `clean_stale_lock`. The loser's
pidfile write lands on the orphaned inode and is invisible to `inspect_lock()`, so
the observable state shows one holder while two run. Fixing only the predicate
leaves it open.

**Why:** the step was scoped from two proven measurements; this is Flohr's
"first unlink, then unlock" variant, which nobody had run against the release path.

**How to apply:** flag it to the operator as a scope question — don't silently
widen the step, don't silently omit it.

**The verification command is a watermelon.** `pytest backend/tests -k "cycle_lock
or 85_5"` collects exactly **3** tests (all green). One is a false friend —
`test_zero_gemini_25_literals_outside_model_tiers[backend/services/cycle_lock.py]`
matches only because the **parametrize id** contains the file path. The real
regression suite, `backend/tests/test_phase_38_6_restart_survivable.py` (8 tests),
matches NEITHER token in its path or test names, so it is **not collected**. The
gate can be green while 85.5 breaks the primitive's own suite.

**Two of those 8 pin the superseded design** and must be rewritten, not routed
around: `test_phase_38_6_ttl_constant_is_90_minutes` (`:148-153`) hard-asserts
`_LOCK_TTL_SEC == 5400`; `test_phase_38_6_acquire_writes_pid_and_cycle_id_then_unlinks`
(`:55`) asserts the lockfile IS unlinked after release — i.e. it pins the exact
behaviour the literature calls split-brain. Replace the literal TTL assert with a
tracking assert (monkeypatch the setting, assert the derived TTL moves).

**macOS `spawn` sends a child test at the REAL lockfile.** Measured:
`multiprocessing.get_start_method() -> 'spawn'` (default on macOS since 3.8; Py
3.14 moved POSIX to `forkserver`). The autouse fixture in
`test_phase_38_6_restart_survivable.py:33-42` monkeypatches `cycle_lock._LOCK_PATH`
to `tmp_path` — a spawned child **re-imports the module and gets
`handoff/.autonomous_loop.lock`**. Criteria 2 and 4 must inject the path via
`env`/`argv` (prefer `subprocess` over `multiprocessing`), or the test suite itself
will touch live trading state.

**The TTL must be derived lazily.** `settings.paper_cycle_max_seconds`
(`backend/config/settings.py:33`, `7200.0`) is the source of truth *and is
runtime-mutable* — `backend/api/settings_api.py:171` allows `300 ≤ x ≤ 21600` via
`PUT /api/settings`. An import-time constant reintroduces the same staleness class.
`autonomous_loop.py:439`'s fallback literal is a stale `1800.0`; don't copy it.

**External canon (for reuse, not re-research):** the failure class is **lock
split-brain via lockfile inode swap**. Named + fixed in portalocker 4.0.0 issue
#115 — "acquire now re-verifies, after locking, that its handle still names the
current path, and retries within the timeout." `filelock/_unix.py` carries the
invariant in source: "Unlinking a locked file on Unix splits waiters across inodes
and breaks mutual exclusion." `trbs/pid` has **no TTL term at all** (staleness =
`_pid_exists` only). `flufl.lock` is the lone TTL design and it is a *refreshed
lease* (`refresh()`), which a write-once pidfile is not. Decisive one-liner:
"the kernel releases the process's advisory lock the moment it dies, **file present
or not**" (portalocker) — so `BlockingIOError` already proves a live holder and no
age test can improve on it.

Related: [[project_backend_restart_safety]], [[project_cron_scheduler_control_topology]],
[[project_observability_ops_residuals_60_4]].
