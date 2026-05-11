---
step: phase-23.1.22
cycle_date: 2026-04-30
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_22.py'
covers: [phase-23.1.20, phase-23.1.21, phase-23.1.22]
---

# Experiment Results — phase-23.1.22 (consolidates 23.1.20 + 23.1.21)

## Summary — root cause via SIGUSR1 dump

User reported "pause/resume crashes the backend, 30s timeout"
multiple times across yesterday and today. Initial hypotheses
chased downstream symptoms:
- **23.1.20** thought BQ slowness was the cause (added 5s
  timeout + 503 + Retry-After to resume_trading + kill-switch
  GET).
- **23.1.21** caught a separate 19-hour silent hang via
  forensics (root cause: ThreadPoolExecutor shutdown(wait=True)
  blocking event loop on stuck Anthropic call). Shipped daemon
  thread + faulthandler + external watchdog + ProcessType=Interactive.

**The actual root cause was found TODAY when the user clicked
resume at 18:42:54.** I caught the hang fresh and sent SIGUSR1
to the running backend (the faulthandler I'd just shipped in
phase-23.1.21). The thread dump showed:

```
Current thread:
  File ".../kill_switch.py", line 95 in snapshot   ← wants self._lock
  File ".../kill_switch.py", line 116 in resume    ← already holds self._lock
```

`threading.Lock()` is **NOT reentrant**. `resume()` acquires
the lock, then calls `self.snapshot()` which tries to acquire
the SAME lock — instant deadlock that froze the entire asyncio
event loop. Same bug in `pause()`.

This is THE actual bug. Phases 20 and 21 are valuable defenses
in depth, but neither would have helped — the hang was a Python
deadlock entirely inside in-memory state.

## Three coordinated cycles (all landed)

### phase-23.1.20 — timeout hardening (defense in depth)
- `resume_trading` BQ call wrapped in `asyncio.timeout(5)`;
  returns 503 + `Retry-After: 5` on TimeoutError
- `get_kill_switch_state` same hardening; degrades gracefully
  with portfolio=None on timeout
- `bq.get_paper_portfolio` enforces 30s `result(timeout=30)`
  (CLAUDE.md "BQ timeout: 30s" rule)

### phase-23.1.21 — diagnostic + auto-recovery + secondary root cause
- **`_spawn_real_agent`**: replaced `ThreadPoolExecutor(max_workers=1)`
  with `threading.Thread(daemon=True)` + `join(timeout=60)`.
  ThreadPoolExecutor's `__exit__` calls `shutdown(wait=True)`
  which blocks the asyncio caller forever if the worker is
  stuck on a non-cancellable HTTP read.
- **faulthandler SIGUSR1**: `backend/main.py` lifespan registers
  `faulthandler.register(SIGUSR1, all_threads=True)`. **This is
  what caught today's deadlock.**
- **External watchdog**: `scripts/launchd/backend_watchdog.sh`
  + `.plist`. Pings /api/health every 60s. On 3 consecutive
  failures: SIGUSR1 (capture stack), then `launchctl kickstart -k`.
- **App Nap exemption**: `ProcessType=Interactive` +
  `LegacyTimers=true` in backend plist.

### phase-23.1.22 — THE actual deadlock fix
- New `_snapshot_locked()` helper that doesn't re-acquire the
  lock. `pause()` and `resume()` call it directly. Public
  `snapshot()` still acquires the lock for external callers.

## Files modified

- `backend/services/kill_switch.py` (+11 lines: `_snapshot_locked`
  helper, `pause`/`resume` use it instead of `snapshot()`)
- `backend/services/ticket_queue_processor.py` (daemon-thread
  pattern replacing ThreadPoolExecutor)
- `backend/api/paper_trading.py` (asyncio.timeout(5) wraps
  + JSONResponse import)
- `backend/db/bigquery_client.py` (result(timeout=30))
- `backend/main.py` (faulthandler register)
- `~/Library/LaunchAgents/com.pyfinagent.backend.plist`
  (ProcessType=Interactive, LegacyTimers=true)

## Files added

- `tests/services/test_kill_switch_no_deadlock.py` (4 tests)
- `tests/services/test_spawn_agent_no_block.py` (3 tests)
- `tests/api/test_pause_resume_timeout.py` (3 tests)
- `tests/verify_phase_23_1_22.py` (immutable verification —
  consolidates the three cycles)
- `scripts/launchd/backend_watchdog.sh` + `.plist` (installed
  via `launchctl load -w`)
- `handoff/current/phase-23.1.{20,21,22}-{external-research,internal-codebase-audit}.md`

## Live verification (post-fix backend)

```
$ python -c "from backend.api.paper_trading import pause_trading, resume_trading, KillSwitchActionRequest
import asyncio, time
t0 = time.monotonic(); print(asyncio.run(pause_trading(...)))   # 0.00s
t0 = time.monotonic(); print(asyncio.run(resume_trading(...)))  # 1.71s (BQ breach check)
t0 = time.monotonic(); print(asyncio.run(pause_trading(...)))   # 0.00s"
```

Pre-fix: hung indefinitely. Post-fix: 0-2 seconds.

```
$ kill -USR1 <pid>
# Stack dump of all 17 threads written to backend.log within 200ms.
# This is the diagnostic that caught the deadlock today.
```

## Test results

```
$ pytest tests/services/test_kill_switch_no_deadlock.py tests/services/test_spawn_agent_no_block.py tests/api/test_pause_resume_timeout.py -q
..........                                                                [100%]
10 passed in 15.42s
```

## Backwards compatibility

- `_snapshot_locked` is a private helper; `snapshot()` public
  API unchanged.
- daemon-thread pattern preserves return shape on success.
- asyncio.timeout(5) is well above normal BQ latency.
- faulthandler registration is purely additive.
- Watchdog is a separate process; backend works without it.

## Honest disclosures

1. **Cascade of three phases for one root cause**. Phase-23.1.20
   chased BQ-timeout (wrong tree). Phase-23.1.21 caught the
   ThreadPoolExecutor blocker (real, but a SECOND bug —
   different code path, different symptom). Phase-23.1.22
   nailed the deadlock. The phase-23.1.21 faulthandler was
   the diagnostic that closed the case.

2. **The 23.1.20 + 23.1.21 fixes are still load-bearing.**
   They harden against future BQ slowness, future stuck
   subagent calls, App Nap suspension, and provide post-mortem
   dump capability. The user-visible "pause hangs" is the
   23.1.22 deadlock; the others guard against different
   failure modes.

3. **No data integrity risk.** All fixes are operational —
   no trade or position state was affected.

## Phase 2 (deferred)

- Audit ALL `with self._lock:` blocks for re-entrant call
  patterns. Currently only `pause`/`resume` were buggy, but
  the same shape could exist in `KillSwitchState.update_*`
  paths or other services.
- Consider switching `KillSwitchState._lock` from
  `threading.Lock()` to `threading.RLock()` as a defensive
  default. RLock has slightly higher overhead but is
  reentrant.
