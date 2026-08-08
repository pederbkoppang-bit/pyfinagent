# Research Brief — phase-85.5 (P0 SAFETY): cycle_lock stale-predicate + unlink-recreate flock defeat

Tier: **moderate** (caller-set). Audit-class: **false**. Date: 2026-08-08.
Scope note: the caller's ask has four parts (A external / B testing / C internal / D macOS)
plus a design recommendation — that is complex-tier shaped. I exceeded the moderate
tool-call and word budgets to cover it in full rather than truncate a P0 safety brief.
Disclosed, not hidden.

READ-ONLY on project code was honored. The only files written are this brief and one
scratch script under the session scratchpad. `handoff/.autonomous_loop.lock` was never
touched; no cycle started; no service restarted.

---

## Search queries run (three-variant discipline)

- **Year-less canonical:** `flock lockfile unlink race condition delete while locked new inode`;
  `pidfile stale lock detection fstat st_ino compare after acquiring flock`;
  `pytest testing file lock multiple processes multiprocessing subprocess deterministic no sleep`
- **Current-year (2026):** `file lock stale detection best practice 2026 lease heartbeat versus TTL distributed lock fencing`
- **Last-2-year (2025):** `"lock file" pidfile race 2025 python fcntl flock inode verification retry loop implementation`

The year-less variant carried this topic, exactly as `.claude/rules/research-gate.md`
predicts for decades-old prior art: the decisive sources (Flohr, flock(2), the library
implementations) are all year-less hits.

---

## Read in full (10; floor is 5 — counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://www.guido-flohr.net/never-delete-your-pid-file/ | 2026-08-08 | authoritative blog (systems) | WebFetch | The canonical treatment. "Deleting a file has two effects. The visible effect is that the directory entry vanishes. And under the hood, the kernel also decrements the _link count_." Two race variants — *unlock-then-unlink* and *unlink-then-unlock* — both end with two processes holding locks on **different inodes**: "B gets the lock for the file descriptor pointing to A's file. And C gets it for the one it has created itself after A has unlinked the other one." Remedy: **never delete the file**. Also: writing PIDs is "a bogus concept" because recycled PIDs create false associations; the **lock**, not the filename, is the protection. |
| https://portalocker.readthedocs.io/en/latest/troubleshooting.html | 2026-08-08 | official library docs | WebFetch | **The single most on-point source.** Names the defect and ships the exact fix: "4.0.0 fixed a related but distinct race in the same area, tracked as issue #115: a competing acquirer could open and lock the very inode a releaser was in the middle of unlinking, so both processes believed they held the lock at once (**split-brain**). acquire now **re-verifies, after locking, that its handle still names the current path, and retries within the timeout** if a race is detected." Also kills the cleanup rationale: "The kernel releases the process's advisory lock the moment it dies, file present or not, so the next acquire on the same path succeeds immediately; the leftover file is **inert litter, not a lock that needs clearing by hand**." And on pidfiles: `read_pid` "only ever says who last wrote the file, never whether that process is still alive." |
| https://raw.githubusercontent.com/tox-dev/filelock/main/src/filelock/_unix.py | 2026-08-08 | library source | WebFetch | `fcntl.flock(fd, LOCK_EX\|LOCK_NB)`. Release = `LOCK_UN` + close, **no unlink**, with the in-source rationale: "We leave the lock file in place after release. **Unlinking a locked file on Unix splits waiters across inodes and breaks mutual exclusion** for processes that coordinate via the same path." |
| https://raw.githubusercontent.com/trbs/pid/master/pid/base.py | 2026-08-08 | library source | WebFetch | Order = `open("a+")` → `_flock(fileno)` → `check()` → write pid. **Never unlinks a pidfile it does not own**; a failed acquire calls `close(cleanup=False)`. Stale detection is `_pid_exists(pid)` **only — there is no age/TTL term at all**. Unlink happens solely in `close()` when `cleanup=True` and `_need_cleanup` (i.e. this instance created it). |
| https://man7.org/linux/man-pages/man2/flock.2.html | 2026-08-08 | official man page | WebFetch | "Locks created by flock() are associated with an **open file description**" — dup/fork share one lock. Decisive for this bug: "If a process uses open(2) (or similar) to obtain **more than one file descriptor for the same file, these file descriptors are treated independently by flock()**." Advisory only. |
| https://man.freebsd.org/cgi/man.cgi?query=flock&sektion=2 | 2026-08-08 | official man page (BSD/Darwin lineage) | WebFetch | "Locks are on **files**, not file descriptors. That is, file descriptors duplicated through dup(2) or fork(2) do not result in multiple instances of a lock, but rather multiple references to a single lock." `LOCK_NB` → `EWOULDBLOCK`. "The flock(), fcntl(2), and lockf(3) locks are compatible." Note the wording differs from Linux but the operational consequence is identical. |
| https://flufllock.readthedocs.io/en/stable/using.html | 2026-08-08 | official library docs | WebFetch | The **only** library surveyed that has a TTL — and it is a *lease*, not a start-timestamp. `lifetime` (default 15s) is "the maximum length of time the process expects to retain the lock"; the holder calls **`refresh()`** to extend: "if the process holding the lock refreshes it, it will hold it ... for as long as it needs." Without refresh, "the lock is stolen from the parent process even if the parent never unlocks it." Trade-off stated verbatim: "Too long and other processes will hang; too short and you'll end up **trampling on existing process locks**." |
| https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html | 2026-08-08 | authoritative blog (named researcher) | WebFetch | `[ADVERSARIAL to any TTL design]` "Clients 1 and 2 now both believe they hold the lock." "If the GC pause lasts longer than the lease expiry period, and the client doesn't realise that it has expired, it may go ahead and make some unsafe change." Time-based expiry is unsafe alone; the remedy is a monotonic **fencing token** enforced at the resource. Efficiency-vs-correctness split: correctness locks protect against "corrupted file, data loss, permanent inconsistency." |
| https://docs.python.org/3.14/library/multiprocessing.html | 2026-08-08 | official docs | WebFetch | "Changed in version 3.8: On **macOS, the _spawn_ start method is now the default**. The _fork_ start method should be considered unsafe as it can lead to crashes of the subprocess as macOS system libraries may start threads." "Changed in version 3.14: On POSIX platforms the default start method was changed from _fork_ to _forkserver_." Spawned children re-import; `__main__`-local targets fail with `AttributeError`. |
| https://pypi.org/project/pytest-timeout/ | 2026-08-08 | official package docs | WebFetch | **signal** method is the default where `SIGALRM` exists (POSIX): "the pytest process is not terminated and the test run can complete normally"; it `pytest.fail()`s the item. **thread** method "will terminate the whole process", via `os._exit()` → "no teardown, JUnit XML output etc." Per-test override `@pytest.mark.timeout(60, method=...)`. |

## Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/gluster/glusterfs/issues/3153 | issue tracker | Fetched, but corroborating only: "stat(\"lockfile\") and fstat(fd) supposedly references the same file, but it doesn't ... processes can believe they have the lock, whilst they don't actually." Kept out of the gate count as community-tier. |
| https://py-filelock.readthedocs.io/en/latest/index.html | official docs | Nav/landing page only; the substance is in `_unix.py`, which I read instead. |
| https://portalocker.readthedocs.io/en/latest/index.html | official docs | Nav-only landing page; troubleshooting page read instead. |
| https://github.com/trbs/pid | README | Rendered README was thin; read `pid/base.py` source instead. |
| https://man7.org/linux/man-pages/man1/flock.1.html | man page | Read; contains **no** unlink guidance ("flock does not detect deadlock"; NFS/CIFS caveat). A negative result, recorded as such. |
| https://copyprogramming.com/howto/flock-removing-locked-file-without-race-condition | community | SEO-farm content; hierarchy tier 5. |
| https://www.postgresql.org/message-id/E1TyTI2-0004b2-Gb%40gemulon.postgresql.org | mailing list | pg_upgrade stale `postmaster.pid`; corroborates pid-liveness-only staleness. |
| https://pypi.org/project/pidlockfile | package | Confirms "if a daemon process is terminated unexpectedly, the lock is automatically released" — same point as portalocker. |
| https://github.com/pytest-dev/pytest-xdist/issues/668 | issue | Parallel-test lock interference; not applicable (suite is not xdist here). |
| https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/ | blog | fork-vs-spawn primer; superseded by the CPython docs. |
| https://hackernoon.com/the-fencing-gap-why-your-distributed-lock-isnt-safe-and-how-to-fix-it | blog | Fencing-token restatement of Kleppmann. |
| https://singhajit.com/distributed-systems/lease/ | blog | Lease pattern primer. |

**URLs collected: 22 unique** (10 read in full + 12 snippet-only).

---

## Recency scan (last 2 years, 2024–2026)

Performed. **One materially new finding**, and it is the best source in the brief:

- **portalocker 4.0.0 / issue #115** documents that this exact defect was found and fixed
  in a maintained Python library, names it **split-brain**, and states the remedy as
  "acquire now re-verifies, after locking, that its handle still names the current path,
  and retries within the timeout." This *confirms* the caller's hypothesised
  `fstat`/`stat` comparison + bounded retry, from a shipped implementation rather than
  folklore.
- **CVE-2024-41012** (surfaced in the 2025 query) is a *kernel-side* `fcntl`/`close`
  lock-removal race — adjacent, not the same bug, and not actionable here.
- **Python 3.14** changed the POSIX default start method `fork` → `forkserver`; macOS has
  defaulted to `spawn` since 3.8. Directly load-bearing for criteria 2 and 4 (see §B).

Nothing in the window supersedes Flohr (2015-era), flock(2), or Kleppmann (2016). The
canonical guidance is stable; only the *worked example* is new.

---

## Key findings

1. **The failure has a name: lock split-brain** (via lockfile inode swap). It is a TOCTOU
   on *lock identity*: the flock is bound to an inode, the coordination contract is bound
   to a path, and unlink decouples the two. — "a competing acquirer could open and lock the
   very inode a releaser was in the middle of unlinking, so both processes believed they
   held the lock at once (split-brain)" (portalocker troubleshooting).

2. **The remedy has two tiers, and the strong one is "don't unlink at all."** filelock
   states it as an invariant in source: "Unlinking a locked file on Unix splits waiters
   across inodes and breaks mutual exclusion for processes that coordinate via the same
   path." Flohr's conclusion is identical and unqualified.

3. **The weak (defence-in-depth) tier is exactly what the caller proposed**: after
   `flock` succeeds, `fstat(fd)` and `stat(path)`, compare `(st_dev, st_ino)`; on
   mismatch or `FileNotFoundError`, close and retry in a **bounded** loop. This is
   portalocker's shipped 4.0.0 fix, and I verified it works on this machine (§probe).

4. **Liveness (holding the flock) is strictly stronger than an age heuristic.** Three
   independent supports:
   - The kernel is the authority and it is exact: "The kernel releases the process's
     advisory lock the moment it dies, **file present or not**" (portalocker). A dead
     holder therefore *cannot* hold the flock — so `BlockingIOError` already means "a
     live process holds it," and no age test can improve on that.
   - The libraries agree by omission: **filelock, portalocker and trbs/pid have no TTL at
     all.** trbs/pid's staleness test is `_pid_exists(pid)` and nothing else.
   - Where a TTL *does* exist (flufl.lock), it is a **renewable lease**: the holder calls
     `refresh()`. flufl.lock names the failure mode of getting it wrong — "you'll end up
     trampling on existing process locks."

5. **A start-timestamp TTL is not a lease and cannot be made safe by tuning.**
   `cycle_lock` writes the pidfile once at acquire and never rewrites it, so `age_sec` is
   *elapsed cycle duration*, not *time since last proof of life*. Raising the constant
   moves the cliff; it does not remove it. Kleppmann is the general statement:
   "Clients 1 and 2 now both believe they hold the lock" whenever expiry is decided by a
   clock rather than by the resource. His fencing-token remedy is the right *concept*
   (only the current holder may act) but the wrong *mechanism* here — pyfinagent has a
   real kernel lock on one host, which is a stronger primitive than a lease.

6. **`os.kill(pid, 0)` is a weak liveness signal and the pid field is weak evidence.**
   Flohr: recycled PIDs "create false associations"; portalocker: the pid "only ever says
   who last wrote the file, never whether that process is still alive." It is fine as
   *diagnostic metadata* (which is how the Slack watchdog uses it) and unfit as a
   *gate on destructive action*.

---

## Scratch-dir probe (macOS 26.5 arm64, Python 3.14.4, project venv)

Mirrors `acquire()`'s exact syscall order in a `tempfile.mkdtemp()`; no project file
touched. Script: `<scratchpad>/race_probe.py`.

```
A locked   inode=10334977
B opened   inode=10334977  lock_ok=False  (expect False)
A released (unlink -> LOCK_UN -> close), path now absent: True
B RE-LOCK  inode=10334977  ok=True   (orphaned inode, no dir entry)
C locked   inode=10334978  ok=True

SAME INODE B vs C: False
BOTH BELIEVE THEY HOLD THE CYCLE LOCK: True
on-disk pidfile at path says: {"pid": 2222} <- B's write is INVISIBLE (orphan inode)

D verified-acquire: verified attempt=1 inode=10334979
E verified-acquire after someone unlinked under D: verified attempt=1 inode=10334980
D re-verify still-mine: False (False => D can self-abort)
```

**Two things this establishes.**

**(i) A THIRD defect the step premise does not name.** The `finally` block at
`cycle_lock.py:151-159` unlinks (`:153`) *before* `LOCK_UN` (`:157`) — Flohr's
"first unlink, then unlock" variant. This produces a double-acquire on the **normal,
successful release path**, with no stale lock, no TTL expiry and no `clean_stale_lock`
involvement. It needs only a second actor that opened the path before the release. The
losing writer's pidfile write lands on an orphaned inode and is invisible to
`inspect_lock()`, so the observable state shows one holder while two are running.
Whoever executes 85.5 should treat the release-path unlink as in scope: fixing only
the `is_stale` predicate leaves this open.

**(ii) The proposed remedy works here.** The `(st_dev, st_ino)` post-lock comparison
detected the swap in both directions — the new acquirer verified cleanly, and the
displaced holder detected that its fd no longer names the path (`still-mine: False`),
which is enough to self-abort rather than keep trading.

---

## Internal code inventory

| File:line | Role | Status |
|---|---|---|
| `backend/services/cycle_lock.py:41` | `_LOCK_TTL_SEC = 90*60`, comment "1.5x paper_cycle_max_seconds (1800s)" | **WRONG** — real budget is 7200s (below) |
| `backend/services/cycle_lock.py:79` | `is_stale = (age_sec > TTL) or (not pid_alive)` | **DEFECT 1** (proven by caller) |
| `backend/services/cycle_lock.py:83-98` | `clean_stale_lock` → `unlink` | destructive; gated only on `is_stale` |
| `backend/services/cycle_lock.py:119-125` | `BlockingIOError` → clean → close → **reopen** → flock | **DEFECT 2** — new inode (proven by caller) |
| `backend/services/cycle_lock.py:151-159` | `finally`: `unlink` (:153) **then** `LOCK_UN` (:157) | **DEFECT 3 — unnamed in the step premise; proven above** |
| `backend/services/autonomous_loop.py:302-330` | the ONE `acquire()` call site; `CycleLockError` → `{"status":"skipped","reason":"already_running_file_lock"}` | fail-safe; a stricter predicate degrades to a *skip*, never a crash |
| `backend/services/autonomous_loop.py:439` | `_cycle_timeout = float(getattr(settings, "paper_cycle_max_seconds", 1800.0))` | the **only** consumer of the cycle budget; note the stale `1800.0` fallback literal |
| `backend/config/settings.py:33` | `paper_cycle_max_seconds: float = Field(7200.0, ...)` | **the source of truth for criterion 3** |
| `backend/api/settings_api.py:123,171,308,383` | `7200.0` default; `ge=300.0, le=21600.0`; env `PAPER_CYCLE_MAX_SECONDS`; operator-mutable via `PUT /api/settings` | budget is **runtime-mutable** — a TTL must be *derived at call time*, not frozen at import |
| `backend/main.py:265-276` | lifespan startup → `clean_stale_lock(reason="startup_recovery")`, fail-open | second `unlink` site; harmless *if* the predicate is liveness-based |
| `backend/slack_bot/scheduler.py:120-141` | `_cycle_state_line()` reads `is_stale` for **alert text only** — "APPENDED to alerts, never used to suppress them" | read-only consumer; a predicate change flips watchdog wording from "backend looks DOWN" to "cycle IN PROGRESS" — an *improvement*, but it is a user-visible string change |
| `backend/api/cron_dashboard_api.py:519-529` | `POST /jobs/{id}/trigger` → delegates to `run_now()` (the triple guard) | comment-only reference; **the manual double-fire vector** |

**Blast radius is small and one-directional:** exactly one `acquire()` call site, one
startup `clean_stale_lock()`, one read-only Slack consumer. No API endpoint or frontend
reads `is_stale`.

### Existing tests + the verification command (measured, not asserted)

`pytest backend/tests -k "cycle_lock or 85_5"` today collects **3 tests, all passing**
(exit 0, 3.65s):

```
backend/tests/test_book_safety_69.py::test_cycle_lock_failed_acquire_keeps_live_pidfile
backend/tests/test_phase_38_6_1_wiring.py::test_phase_38_6_1_autonomous_loop_imports_cycle_lock
backend/tests/test_phase_75_5_2_model_pins.py::test_zero_gemini_25_literals_outside_model_tiers[backend/services/cycle_lock.py]
```

Three problems with that gate, all measured:

- The third hit is a **false friend** — a `gemini-2.5` literal sweep that matches only
  because the *parametrize id* contains the file path. It tests nothing about locking.
- **The real regression suite is NOT collected.** `backend/tests/test_phase_38_6_restart_survivable.py`
  (8 tests, green in 0.02s) matches neither `cycle_lock` nor `85_5` in its path or test
  names. So `-k "cycle_lock or 85_5"` can be green while 85.5 breaks the primitive's own
  regression suite. **Watermelon risk.** Either name the new file/tests to widen the `-k`,
  or have the executor run the 38.6 suite alongside and record both.
- **Two existing 38.6 tests will go RED under the canonical fix**, and this is a genuine
  design collision, not an oversight to route around:
  - `test_phase_38_6_ttl_constant_is_90_minutes` (`:148-153`) hard-asserts
    `_LOCK_TTL_SEC == 5400`. Criterion 3 makes this false by construction.
  - `test_phase_38_6_acquire_writes_pid_and_cycle_id_then_unlinks` (`:55`) asserts
    `not fake_lock.exists()` after release — i.e. it **pins the exact behaviour the
    literature says causes split-brain**.
  - `test_phase_38_6_simulated_kill_then_startup_cleans` depends on `clean_stale_lock`
    unlinking (survivable if the *predicate* changes but the unlink stays for a
    provably-dead pid).

  Per `feedback_immutable_criteria_must_be_green_able`, these should be updated as part
  of 85.5 with the reason recorded — they encode a superseded design, and leaving them
  green would mean the fix was not applied.

---

## macOS / BSD specifics (question D)

Measured on this host: `macOS-26.5-arm64`, Python **3.14.4**, APFS.

1. **`multiprocessing` default start method is `spawn`** (measured:
   `get_start_method() -> 'spawn'`; available `['spawn','fork','forkserver']`). This is
   the single biggest trap for criteria 2 and 4: a child **will not inherit** a
   `monkeypatch.setattr(cycle_lock, "_LOCK_PATH", tmp_path/...)`. The autouse fixture in
   `test_phase_38_6_restart_survivable.py:33-42` works only because every existing test is
   single-process. A spawned child re-imports `cycle_lock` and gets the **real
   `handoff/.autonomous_loop.lock`** — which would violate the caller's own "do not touch
   the lockfile" constraint from inside the test suite. Pass the path explicitly (env var
   or argv) and have the child set it before use.
2. **Linux vs BSD wording differs, semantics do not.** Linux: locks attach to the "open
   file description"; FreeBSD/Darwin: "Locks are on files, not file descriptors ... dup/fork
   ... multiple references to a single lock." Both yield the property this bug turns on:
   a **fresh `open()` produces an independent lock reference**. No Linux-derived fix is
   wrong here on that axis.
3. **Darwin offers atomic open-and-lock that Linux lacks** — `os.O_EXLOCK` is present on
   this interpreter (measured: `hasattr(os,'O_EXLOCK') is True`, value `32`; `O_SHLOCK`
   too). Flohr notes BSD/macOS have it and dismisses it *for portability*. My
   recommendation does **not** use it: the deployment is macOS-only today, but a
   `O_EXLOCK`-dependent design would be silently wrong if anything ever runs in a Linux
   container, and the verify-after-lock pattern is portable and independently sourced.
   Worth knowing it exists; not worth depending on.
4. **PID reuse is real** (`kern.maxproc: 4000` concurrent; Darwin PIDs wrap well under
   100k). `_is_pid_alive` can return True for an unrelated recycled pid. Another reason
   to demote pid to metadata.
5. **APFS reuses inode numbers.** A `(st_dev, st_ino)` comparison is sound while the file
   is *never deleted*; if you keep an unlink path, the reuse window is narrow but nonzero.
   This argues for the strong remedy (don't unlink) over relying on verification alone.
6. **`pytest-timeout` is installed**; on POSIX the default method is **signal**, which
   `pytest.fail()`s the item without killing the run. Important consequence: **a timeout
   does not reap your child processes.** `pytest.ini` (repo root) sets only the
   `requires_live` marker — no `timeout =` key — so the `--timeout=120` the caller cites
   comes from the invocation, not from config.

---

## Testing patterns (question B)

- **Prefer `subprocess` over `multiprocessing`** for criteria 2 and 4. Under `spawn` the
  two are nearly equivalent in cost, and `subprocess` makes the "real second process"
  claim self-evident to a reviewer, gives a real exit code to assert on, and makes the
  lock-path injection explicit (`env=` / `argv`) instead of relying on inheritance that
  macOS does not provide. Write the child as a tiny module-level script or a
  `python -c` payload; CPython docs' `AttributeError` caveat about `__main__`-local
  targets applies to the mp route.
- **Handshake, never sleep.** Deterministic ordering comes from a pipe: the child prints
  a token (`READY`) on stdout after it holds the lock and the parent blocks on
  `readline()` with a timeout. Poll for a *state predicate* with a deadline
  (`while time.monotonic() < deadline`) rather than a fixed `sleep`. Avoid
  `multiprocessing.Manager().Lock()` for this — it coordinates the *test*, but the thing
  under test is the file lock, so a manager lock would mask ordering bugs.
- **Reap in a fixture `finally`.** Because pytest-timeout's signal method does not kill
  children, every child needs `try/finally: proc.kill(); proc.wait(timeout=…)`, and
  `proc.communicate(timeout=…)` rather than an unbounded `wait()`. Budget the whole test
  well under the 120s ceiling (a lock test should be < 5s).
- **Criterion 4 (kill mid-hold) is the cleanest of the four**: `subprocess.Popen` the
  child, wait for `READY`, `proc.kill()` (SIGKILL), `proc.wait()`, then assert the parent
  acquires. This is a *pure kernel-behaviour* assertion — "the kernel releases the
  process's advisory lock the moment it dies, file present or not" (portalocker) — so it
  is inherently non-flaky and needs no TTL, no `clean_stale_lock`, and no pid check. It
  also proves recovery does not depend on the unlink path at all.
- **Non-vacuity for criterion 2** (the caller's explicit requirement that the fixture
  first show the OLD path acquiring): implement the old path *in the test file* — an
  inline `unlink → reopen → flock` helper — assert it **does** double-acquire, then assert
  the production `acquire` refuses. My scratch probe is exactly that control and it
  works. This satisfies `feedback_mutation_test_guards_and_fixtures` ("a guard that can't
  fail doesn't count") and `feedback_guards_stop_one_seam_short`: the control must call
  the *production* `acquire`, not a re-implementation of the fix. A weaker alternative —
  monkeypatching the predicate — is not equivalent, because the defect lives in the
  acquire *path*, not only the predicate.

---

## Consensus vs debate

**Consensus (strong, unanimous across sources):** never unlink a lockfile that others
coordinate on by path; bind mutual exclusion to the kernel lock, not the filename;
process liveness is established by lock contention, not by reading a pid.

**Debate:** whether a lock should ever be *breakable* on a timer. flufl.lock says yes but
only as a **renewable lease** with `refresh()`; Kleppmann says a timer alone is never
sufficient for correctness and demands fencing; filelock/portalocker/trbs-pid sidestep it
entirely by having no TTL. The resolution for a **single-host, kernel-locked** resource is
that the debate does not apply: the distributed-systems TTL exists because there is no
shared kernel to ask. pyfinagent has one. Importing a lease design here adds the failure
mode without adding any capability.

---

## Pitfalls (from literature + this codebase)

- Tuning `_LOCK_TTL_SEC` to 1.5×7200 "fixes" the reported symptom and leaves defects 2
  and 3 fully live. Do not stop there.
- Removing the `age` term but keeping `unlink`-on-contention still yields split-brain
  the moment `pid_alive` is wrong (recycled pid, or a pidfile written by a process that
  died before flushing).
- An unbounded retry loop on inode mismatch is a hang. Portalocker bounds it by the
  acquire timeout; bound it by attempts (3–5) and fail closed to `CycleLockError`.
- Deriving the TTL from `settings.paper_cycle_max_seconds` **at import time** re-freezes
  it: `settings_api.py` lets the operator change the budget at runtime
  (`ge=300, le=21600`). Derive per call.
- Changing `is_stale` silently changes Slack watchdog wording (`scheduler.py:132-138`);
  disclose it as a behaviour change even though it is an improvement.

---

## Recommendation

**1. The predicate — liveness only; age is never sufficient.**

```
is_stale  =  (not pid_alive)        # necessary, not sufficient
```

and — this is the important half — **`is_stale` must stop being the authority for
destructive action.** The flock is the authority. Concretely: `inspect_lock()` may keep
reporting `age_sec` and a *separate* `age_exceeds_budget` field for the watchdog/digest
(it is genuinely useful signal that a cycle is overrunning), but the only thing that may
gate an `unlink` is "no process holds the flock". Source: portalocker ("the kernel
releases the process's advisory lock the moment it dies, file present or not"),
trbs/pid (no TTL term at all), flufl.lock (a TTL is only safe as a refreshed lease, which
this pidfile is not — it is written once and never touched again).

**2. The acquire path — delete the stale-reacquire branch entirely.**

Replace `cycle_lock.py:119-132` with: on `BlockingIOError`, close the fd and raise
`CycleLockError`. Full stop. No inspect, no clean, no reopen. A live flock holder means a
live cycle — by kernel guarantee — and a dead holder cannot be holding the flock, so the
branch's premise is empty. This is filelock's and trbs/pid's design. It also makes the
`acquire` path *shorter* than today's, which is the right direction for a P0 safety fix.

**3. Verify-after-lock, as defence in depth.**

```
for attempt in range(_ACQUIRE_MAX_ATTEMPTS):      # 5
    fd = os.open(path, O_RDWR|O_CREAT, 0o644)
    try: flock(fd, LOCK_EX|LOCK_NB)
    except BlockingIOError: close(fd); raise CycleLockError(...)
    st_fd = os.fstat(fd)
    try: st_path = os.stat(path)
    except FileNotFoundError: close(fd); continue      # swapped under us
    if (st_fd.st_dev, st_fd.st_ino) == (st_path.st_dev, st_path.st_ino):
        break                                          # we hold the CURRENT file
    close(fd); continue
else:
    raise CycleLockError("lockfile kept being replaced under us")
```

Source: portalocker 4.0.0 / issue #115 — "acquire now re-verifies, after locking, that its
handle still names the current path, and retries within the timeout if a race is
detected." Proven working on this host (§probe, D/E rows). Bound the loop; never spin.

**4. Stop unlinking on release.**

`cycle_lock.py:151-159`: keep `LOCK_UN` + `close`, drop the `unlink` at `:153`. If the
pidfile's contents matter for observability, truncate-and-rewrite a
`{"state":"released", ...}` payload instead of deleting. Source: filelock's in-source
invariant ("Unlinking a locked file on Unix splits waiters across inodes and breaks mutual
exclusion for processes that coordinate via the same path") and Flohr. **This is the fix
for the third defect I proved above, and it is not in the step's stated scope — flag it to
the operator rather than silently widening or silently omitting it.** The leftover file is,
in portalocker's words, "inert litter."

Keep exactly one unlink site — `clean_stale_lock` at startup — and gate it on the strong
test rather than `is_stale`: *acquire the flock first; only a process that has just won
the flock may remove or rewrite the file.* That is race-free by construction, because the
winner is by definition the only one there.

**5. What criterion 3's "TTL derived from the cycle budget" should bind to.**

- **Bind to `settings.paper_cycle_max_seconds`** — `backend/config/settings.py:33`,
  value `7200.0`, the same field `autonomous_loop.py:439` uses for the `asyncio.timeout`.
  That is the only honest source of truth: it makes the lock and the cycle read the
  *same* number by construction, which is precisely the drift that produced this bug.
- **Read it lazily, inside the function**, e.g.
  `float(getattr(get_settings(), "paper_cycle_max_seconds", 7200.0)) * _TTL_MULTIPLE`,
  never a module-level constant. `settings_api.py:171` allows `300 ≤ x ≤ 21600` at
  runtime via `PUT /api/settings`, so an import-time snapshot would reintroduce the
  same class of staleness. Note `autonomous_loop.py:439`'s fallback literal is a stale
  `1800.0`; do not copy it — use `7200.0` or, better, no fallback.
- **And say plainly what the derived TTL is *for*.** After recommendations 1–4 it must
  **not** gate any unlink or any acquire decision. Its legitimate consumers are
  observability: the Slack watchdog line (`scheduler.py:120-141`) and any
  overrun alerting, which want "this cycle has exceeded its budget" — a true and useful
  statement that is simply not the same statement as "this lock is stale." If 85.5 keeps a
  TTL that can still condemn a lock, it will have renamed the bug rather than fixed it.

**Suggested guard against regression:** assert in a test that
`cycle_lock`'s TTL *tracks* `settings.paper_cycle_max_seconds` (monkeypatch the setting to
an odd value and assert the derived TTL moves), rather than asserting a literal — that is
the mutation-resistant form of the criterion, and it is what replaces
`test_phase_38_6_ttl_constant_is_90_minutes`.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **10**
- [x] 10+ unique URLs total — **22**
- [x] Recency scan (last 2 years) performed + reported — one material finding (portalocker 4.0.0 / #115)
- [x] Full pages/sources read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module (1 acquire site, 1 startup site, 1 Slack consumer, 4 test files, settings + settings_api)
- [x] Contradictions / consensus noted (flufl.lock + Kleppmann vs the no-TTL libraries; resolved on the single-host axis)
- [x] Claims cited per-claim
- [ ] Tool-call and word budget for `moderate` exceeded — disclosed at the top

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 12,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 1,
    "dry": false
  },
  "summary": "The defect class is lock split-brain via lockfile inode swap; portalocker 4.0.0 (issue #115) names it and ships the exact remedy the caller hypothesised -- after flock succeeds, re-verify (st_dev, st_ino) of fstat(fd) against stat(path) and retry in a bounded loop. The stronger, unanimous remedy is to never unlink a lockfile others coordinate on by path (filelock states this as an in-source invariant; Flohr's 'Never Delete Your PID File' is the canonical treatment). Liveness is strictly stronger than age: the kernel releases an advisory lock the instant the holder dies, file present or not, so BlockingIOError already means a live holder and no age test improves on it; filelock/portalocker/trbs-pid carry no TTL at all, and flufl.lock's TTL is a refreshed lease, which cycle_lock's write-once pidfile is not. Recommendation: is_stale = (not pid_alive) only, delete the stale-reacquire branch outright, add bounded verify-after-lock, stop unlinking on release, and derive any TTL lazily from settings.paper_cycle_max_seconds (settings.py:33, 7200.0, runtime-mutable) for OBSERVABILITY only. Two findings beyond the step premise: (a) the finally block unlinks before LOCK_UN (cycle_lock.py:153 vs :157), which I proved produces a double-acquire on the NORMAL release path with no TTL involvement; (b) the verification command -k 'cycle_lock or 85_5' does not collect the 8-test 38.6 regression suite and matches one false-friend parametrize id -- and two existing 38.6 tests pin the superseded behaviour and must be updated.",
  "brief_path": "handoff/current/research_brief_85.5.md",
  "gate_passed": true
}
```
