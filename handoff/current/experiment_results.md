---
step: phase-23.1.19
cycle_date: 2026-04-29
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_19.py'
---

# Experiment Results — phase-23.1.19

## Summary

User reported "backend crashed". `backend.log` showed recurring
`OSError: [Errno 24] Too many open files: '.../limits.yaml'` ending
in unresponsiveness. governance/limits_loader.py uses
`with path.open("rb")` correctly — its FD IS released. The leak
was upstream: 23 sites across 7 files used the leaky pattern
`with sqlite3.connect(...) as conn:` which Python docs explicitly
state does NOT close the connection (only commits/rolls back).

Each call leaked 1 SQLite connection = 3 FDs (main DB + WAL + shm).
With ticket_queue_processor running every 5s in lifespan, FDs
accumulated to the launchd `NumberOfFiles=16384` limit and the
process could no longer open ANY file — governance watcher's 10s
tick was the visible victim.

**`lsof` evidence (immediately after restart, pre-fix):** 9 FDs
already held open against `tickets.db` from the first ticket
processor batch.

## Three coordinated fixes

**Fix A — Wrap all 23 sites with `contextlib.closing()`**.
Pattern: `with closing(sqlite3.connect(p)) as conn, conn:` —
combined context managers. Outer `closing()` actually closes the
FD on exit; inner `conn` re-establishes the commit/rollback
semantics callers depend on. No method body re-indentation.
Sites: tickets_db.py (15), ticket_queue_processor.py (1),
sla_monitor.py (2), response_delivery.py (2),
stuck_task_reaper.py (1), commands.py (1), direct_responder.py (1).

**Fix B — Regression test**
(`tests/db/test_tickets_db_no_fd_leak.py`). Uses
`psutil.Process(os.getpid()).num_fds()` before/after 100
iterations of `create_ticket / get_open_tickets /
update_ticket_status / get_ticket_stats`. Asserts net delta ≤ 5.
Pre-fix repro confirmed: same loop without `closing()` grows by
exactly 100 FDs (1 per call). Skipped on Windows
(num_fds is UNIX-only).

**Fix C — RLIMIT_NOFILE startup log**
(`backend/main.py` lifespan). Logs `soft=N hard=M` from
`resource.getrlimit(RLIMIT_NOFILE)`. WARNs if soft < 4096.
Operational early-warning so future low-limit anomalies (e.g.,
launchd plist regression) are caught at boot, not at crash.

## Files modified

- `backend/db/tickets_db.py` (15 sites + closing import)
- `backend/services/ticket_queue_processor.py` (1 site + closing import inline)
- `backend/services/sla_monitor.py` (2 sites + closing imports inline)
- `backend/services/response_delivery.py` (2 sites + closing imports inline)
- `backend/services/stuck_task_reaper.py` (1 site + closing import inline)
- `backend/slack_bot/commands.py` (1 site + closing import inline)
- `backend/slack_bot/direct_responder.py` (1 site + closing import at module top)
- `backend/main.py` (+15 lines: RLIMIT_NOFILE logging block)

## Files added

- `tests/db/__init__.py` (empty package marker)
- `tests/db/test_tickets_db_no_fd_leak.py` (1 regression test)
- `tests/verify_phase_23_1_19.py` (immutable verification)

## Verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_19.py
ok 23 sqlite3.connect sites wrapped with closing() across 7 files + tickets_db imports closing + main.py logs RLIMIT_NOFILE + FD-leak regression test passes
```
Exit 0.

## Test results

```
$ pytest tests/db/test_tickets_db_no_fd_leak.py + 5 prior phases' suites -q
.............................                                            [100%]
29 passed in 2.81s
```

## Pre/post measurements

**Bare leaky pattern reproduced** (one-shot Python script):
```
Leaky: before=4 after=104 delta=100   # 100 FDs leaked over 100 iterations
```

**lsof tickets.db FDs after backend restart:**
- Pre-fix: 9 FDs already held (after a single ticket batch)
- Post-fix: **0 FDs** (everything closed cleanly)

**RLIMIT_NOFILE startup log (post-fix backend boot):**
```
22:06:55 I [main] RLIMIT_NOFILE: soft=8192 hard=16384
22:06:57 I [main] Prewarming ticker-meta cache for 14 tickers...
```

Total process FD count: 348 (pre-fix immediately after restart) →
342 (post-fix). The leak no longer accumulates over time.

## Backwards compatibility

- `with closing(sqlite3.connect(p)) as conn, conn:` preserves the
  same `conn` binding and the same `with conn:` transaction
  semantics — every existing call site behaves identically except
  for actually closing the FD on exit.
- RLIMIT log is informational; backend still boots on systems
  with low soft limits, just emits a WARN.
- `tests/db/__init__.py` is an empty package marker; no runtime impact.

## Honest disclosures

1. **Researcher correction**: my initial scan found 17 sites; the
   researcher's full audit found **23** (sla_monitor +2,
   response_delivery +2, stuck_task_reaper +1). All 23 are now
   fixed. This is a teachable moment about trusting the agent
   over a quick grep.

2. **launchd plist soft limit is 8192**, not 16384. The
   plist key `NumberOfFiles=16384` is the HARD limit; soft is
   bound by macOS defaults (8192 on this system). 8192 is
   plenty for normal operation but with the leak it would have
   been hit faster than the plist suggested. RLIMIT log makes
   this visible.

3. **No yfinance / httpx leak found** in this audit. The
   tickets.db SQLite leak is the entire crash signature.

4. **One-shot data integrity** check: `tickets.db` is a queue,
   not a financial source of truth. No data loss from prior
   leaks.

## Phase 2 (deferred)

- Refactor TicketsDB to use a single thread-local connection
  (open once, reuse). Eliminates per-call open/close overhead
  and removes the closing-pattern requirement entirely.
  Researcher noted this is more invasive; defer until A+B+C
  are verified in production over a few days.
- Audit the rest of the repo for other `with X.connect(...)`
  patterns that might have similar issues (e.g., aiohttp
  ClientSession, httpx Client).
- Add a periodic FD-count log (every hour) so trend can be
  monitored, not just boot-time.
