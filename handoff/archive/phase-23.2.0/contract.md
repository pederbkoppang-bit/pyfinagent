---
step: phase-23.1.21
title: Backend silent-hang root-cause fix (daemon thread + faulthandler + external watchdog)
cycle_date: 2026-04-30
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_21.py'
research_brief: handoff/current/phase-23.1.21-external-research.md (also see phase-23.1.21-internal-codebase-audit.md)
---

# Contract — phase-23.1.21

## Hypothesis

User flagged "backend is down" via UI red-dot. Forensic state:
PID 40904 ran 20h12m alive but silent — last log line at
23:13:35 yesterday during normal operation. Process state `S`
(sleeping), 17 threads, 449 FDs (well under limit), no error,
no crash, no exit. launchd `KeepAlive=true` did NOT respawn
because it only fires on EXIT, not hang. Even APScheduler's
5-second `process_batch` jobs stopped logging — the entire
asyncio event loop was blocked.

**Root cause** (per researcher's audit, confidence HIGH):
`backend/services/ticket_queue_processor.py::_spawn_real_agent`
creates a `ThreadPoolExecutor(max_workers=1)` inside a `with`
block, calls `future.result(timeout=60)`, and the executor's
`__exit__` calls `shutdown(wait=True)`. If the underlying
Anthropic HTTP call hangs at the TCP level (no RST), the
worker thread never terminates and the `with` block never
exits, blocking `loop.run_in_executor(...)`, blocking
`process_single_ticket`, blocking `process_queue_batch`,
blocking the entire event loop. No new requests served. No
new logs. TCP accept loop dead.

If we (1) replace the `ThreadPoolExecutor` with a
`threading.Thread(daemon=True)` so the asyncio caller is
released even when the worker thread is stuck, (2) register
`faulthandler` on `SIGUSR1` so future hangs can dump thread
stacks before kicking, (3) add an external launchd watchdog
that curls `/api/health` every 60s and `kickstart -k`s on 3
failures, and (4) add `ProcessType=Interactive` to the plist
to defensively exempt from App Nap, then this class of
silent hang is structurally closed AND auto-recovered AND
post-mortem-debuggable.

## Research-gate summary

- External brief: `handoff/current/phase-23.1.21-external-research.md`
  — 7 sources read in full (launchd.plist man page, Python
  faulthandler docs, APScheduler user guide, APScheduler pool
  docs, Apple App Nap docs, uvicorn.org, OneUptime blog).
  Recency scan 2024-2026. `gate_passed: true`.
- Internal audit: `handoff/current/phase-23.1.21-internal-codebase-audit.md`
  — 11 files inspected. Threading-lock inventory, App Nap
  analysis, APScheduler thread-pool analysis, phase-23.1.19
  closing-pattern verification, launchd KeepAlive semantics,
  heartbeat file analysis. **Top hypothesis identified the
  exact code path with file:line anchors.**

## Plan steps

1. **Fix Root-Cause** — `backend/services/ticket_queue_processor.py::_spawn_real_agent`
   (~line 231): replace `with ThreadPoolExecutor(max_workers=1) as pool:`
   pattern with `threading.Thread(target=..., daemon=True)`. After
   `thread.join(timeout=60)`, if `thread.is_alive()`, log a
   warning and return failure (the daemon thread will be cleaned
   up at process exit; the asyncio caller is released).

2. **Fix C — faulthandler SIGUSR1** in `backend/main.py` lifespan:
   ```python
   import faulthandler, signal
   faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
   ```
   Operators (or the watchdog) send SIGUSR1 first to capture
   thread state in stderr, then SIGKILL via kickstart.

3. **Fix A — External watchdog launchd job**
   (`scripts/launchd/com.pyfinagent.backend-watchdog.plist` +
   `scripts/launchd/backend_watchdog.sh`). Runs every 60s.
   curls `/api/health` with 5s timeout. On 3 consecutive
   failures: `kill -USR1 $PID` (capture stack), wait 2s,
   `launchctl kickstart -k gui/<uid>/com.pyfinagent.backend`.
   Counter file in `~/Library/Caches/`.

4. **Fix D — App Nap exemption** in
   `~/Library/LaunchAgents/com.pyfinagent.backend.plist`:
   `ProcessType: Interactive` + `LegacyTimers: true`. Belt-and-
   suspenders defensive add.

5. **Fix E — Document** in `CLAUDE.md` Critical Rules: manual
   recovery command + watchdog architecture + faulthandler
   diagnostic.

6. **Tests**:
   - `tests/services/test_spawn_agent_no_block.py` —
     mock the agent call to hang forever; assert the asyncio
     caller is released within 65s with a clear failure status
     (not a 30s+ hang).
   - `tests/test_watchdog_script.py` — exercise the
     `backend_watchdog.sh` logic via shell-script execution
     against a stub /health that returns 503.

7. **Immutable verification** (`tests/verify_phase_23_1_21.py`):
   asserts daemon-thread pattern in _spawn_real_agent,
   faulthandler block in main.py, watchdog plist + script
   exist, ProcessType in launchd plist.

## Immutable verification command

```bash
source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_21.py
```

Must exit 0 with one ok-line.

## Acceptance criteria

- `pytest tests/services/test_spawn_agent_no_block.py -q` passes.
- `python tests/verify_phase_23_1_21.py` exits 0.
- Backend `kill -USR1 <pid>` writes a stack dump to backend.log.
- Watchdog plist installed via `launchctl load` and visible
  in `launchctl list`.

## Backwards compatibility

- Daemon-thread replacement preserves the same return shape on
  success; only the timeout path is now non-blocking.
- faulthandler registration is purely additive (a signal
  handler) — zero impact on normal flow.
- Watchdog is a separate process that can be installed but is
  optional — backend works without it.
- ProcessType=Interactive is a hint to launchd; not a behavior
  change for the process itself.

## References

- `handoff/current/phase-23.1.21-external-research.md`
- `handoff/current/phase-23.1.21-internal-codebase-audit.md`
- `backend/services/ticket_queue_processor.py:231` (the hang site)
- `backend/main.py:109+` (lifespan startup)
- `~/Library/LaunchAgents/com.pyfinagent.backend.plist`
