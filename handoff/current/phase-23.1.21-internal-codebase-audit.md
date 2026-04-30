# Phase-23.1.21 Internal Codebase Audit — Backend Silent Hang

**Date:** 2026-04-29
**Investigator:** Researcher agent
**Context:** Backend PID 40904 ran 20h12m without logging or responding to TCP connections. Process was alive (S state, 17 threads, 449 FDs). Last log line was a normal APScheduler job entry at 23:13:35. `launchctl kickstart -k` restored it.

---

## 1. Thread and Lock Inventory

### All threading.Lock instances

| File | Line | Variable | Lock Type | Purpose |
|------|------|----------|-----------|---------|
| `backend/tools/alt_data.py` | 30 | `_CACHE_LOCK` | `threading.Lock()` | Alt-data in-memory cache |
| `backend/agents/cost_tracker.py` | 105 | `_lock` (field) | `threading.Lock()` | Per-agent cost accum |
| `backend/agents/_genai_client.py` | 29 | `_client_lock` | `threading.Lock()` | Gemini client singleton init |
| `backend/governance/limits_loader.py` | 50 | `_init_lock` | `threading.Lock()` | One-time governance load |
| `backend/api/job_status_api.py` | 77 | `_lock` | `threading.Lock()` | Job status map |
| `backend/services/api_cache.py` | 31 | `self._lock` | `threading.Lock()` | In-memory TTL cache |
| `backend/services/cycle_health.py` | 72 | `self._lock` | `threading.Lock()` | JSONL write serialization |
| `backend/services/kill_switch.py` | 46 | `self._lock` | `threading.Lock()` | Kill-switch state |
| `backend/services/perf_tracker.py` | 34 | `self._lock` | `threading.Lock()` | Latency ring buffer |
| `backend/services/live_prices.py` | 40 | `self._lock` | `threading.Lock()` | Live price cache |
| `backend/services/observability/alerting.py` | 53 | `self._lock` | `threading.Lock()` | Alert deduplication |
| `backend/services/observability/api_call_log.py` | 59, 200 | `_lock`, `_llm_lock` | `threading.Lock()` | API call log, LLM log |

### asyncio.Lock instances

| File | Line | Variable | Note |
|------|------|----------|------|
| `backend/agents/mas_events.py` | 97 | `self._lock` | Conditional: `asyncio.Lock() if asyncio.get_event_loop().is_running() else None` |

**Red flag:** `mas_events.py:97` calls `asyncio.get_event_loop()` to decide which lock type to use at construction time. If this object is constructed from a thread that has no running event loop (e.g., during early import before lifespan starts), it falls back to `None` — meaning the lock is disabled for that instance. If constructed inside the event loop, it uses an asyncio.Lock. This is a correctness issue but not a deadlock path per se.

### Cross-lock dependency analysis

No static lock-ordering violations were found. Each lock is narrowly scoped:

- `api_cache._lock` is never held while calling into `kill_switch`, `perf_tracker`, or `cycle_health`.
- `kill_switch._lock` is held only during `pause()`, `resume()`, `snapshot()`, `update_sod_nav()`, `update_peak()`. None of these acquire any other lock.
- `perf_tracker._lock` is held only during `record()`, `summarize()`, `export_tsv()`, `clear()`. No nested locking.
- `governance._init_lock` is held only during the one-time init path. No nested locking.
- `cycle_health._lock` is held only during the JSONL write in `record_cycle_end()`. No nested locking.

**Verdict on deadlock via lock cycle:** Low probability. The locks are all leaf-level — no code acquires more than one of these locks simultaneously, so there is no opportunity for a classic lock-ordering cycle.

---

## 2. macOS Process Suspension / App Nap Analysis

### What the plist actually has

`~/Library/LaunchAgents/com.pyfinagent.backend.plist` (read in full):

```
ProgramArguments: [caffeinate, -i, -s, uvicorn, ...]
KeepAlive: true
ProcessType: (absent — defaults to "Standard")
LegacyTimers: (absent)
EnableTransactions: (absent)
```

**No `ProcessType` key is present.** That means launchd treats this as a `Standard` job. On macOS, `Standard` jobs are subject to App Nap when the system is idle.

**The caffeinate wrapper:** `-i` prevents the system from idle-sleeping; `-s` prevents sleep when connected to AC power. However, caffeinate is a child process of launchd — the parent of uvicorn is caffeinate, not launchd directly. App Nap acts on the process itself (its `QOS_CLASS`), not the system sleep state. A process marked as background-class by the OS can have its CPU budget throttled to near-zero even when the system is not sleeping.

**The distinction matters:**
- `caffeinate -i -s`: tells the kernel not to sleep the SYSTEM.
- App Nap: throttles the PROCESS (timer coalescing + CPU QOS downgrade) when the OS decides it is idle (no user interaction, no active windows, no outstanding NSProcessInfo assertions).

A launchd agent with no `ProcessType` key defaults to Standard. Apple's documentation states that without a ProcessType, the job receives "default OS resource restrictions," which in practice means App Nap can apply if the system decides the process is not directly serving the user.

**Is App Nap the confirmed cause?** Not confirmed — the 19-hour gap is consistent with App Nap onset at a low-activity window (overnight), but it is also consistent with asyncio event loop blockage. Both hypotheses remain open.

**How to confirm:** Run `sudo spindump 40904` during a future hang before kickstarting. The report will show whether threads are sleeping in `QOS_CLASS_BACKGROUND` or blocked on a syscall.

---

## 3. APScheduler Thread-Pool Analysis

### How process_batch is wired

`backend/main.py:179-196`: At lifespan startup, an `AsyncIOScheduler` is created and `process_batch` (an `async def`) is added as an interval job every 5 seconds. This is NOT a ThreadPool job — it runs as a coroutine on the asyncio event loop.

**Key fact:** `AsyncIOScheduler` dispatches `async def` jobs by scheduling them as coroutines on the event loop. If the event loop is blocked, the scheduler's internal wakeup call also blocks. There is no separate thread pool in this path.

The `ticket_queue_processor.py` exposes `process_queue_batch()` which `process_batch` calls. Looking at that function:

- `process_single_ticket()` (line 286) calls `await asyncio.sleep(10)` on EVERY first attempt (line 361) and up to 240s on retries (line 357).
- Inside `process_single_ticket`, `loop.run_in_executor(None, self.spawn_agent_session, ...)` (line 364) dispatches to the default `ThreadPoolExecutor`. The default pool has `min(32, os.cpu_count() + 4)` workers in Python 3.8+, which is usually 8-36 on a modern Mac.
- `spawn_agent_session` calls `_spawn_real_agent`, which spawns a `ThreadPoolExecutor(max_workers=1)` (line 231) per call and blocks in `future.result(timeout=60)`.

**Thread exhaustion path:** If multiple tickets are queued, `max_concurrent=1` limits to 1 ticket at a time. However, if a single ticket's `_spawn_real_agent` call hangs past the 60s timeout AND the `ThreadPoolExecutor(max_workers=1)` context manager blocks waiting for the thread to terminate (which it does — Python's `with ThreadPoolExecutor()` calls `shutdown(wait=True)` on `__exit__`), then the run_in_executor thread holding the work is never released until the 60s future resolves.

**The 60s Anthropic call + executor chain:** `future.result(timeout=60)` raises `TimeoutError` after 60s but the underlying `call_anthropic` thread in the pool continues running until the HTTP connection closes or errors. Since the pool is created with `max_workers=1` inside a `with` block, Python's `ThreadPoolExecutor.__exit__` calls `shutdown(wait=True)` — meaning the coroutine awaiting `run_in_executor` is blocked until that thread actually terminates. If the Anthropic SDK's HTTP connection hangs at the OS TCP level (no TCP RST, no timeout from the SDK), that thread never terminates. This would block the single run_in_executor thread in the asyncio thread pool for the duration, preventing event loop progress for any task awaiting that executor.

**Is this the hang cause?** This is the most likely candidate for blocking the event loop if there was a stuck ticket in an IN_PROGRESS state. The `process_batch` coroutine itself blocks on `gather(*tasks)` which blocks on `process_single_ticket` which blocks on `run_in_executor`. The scheduler's next 5s trigger never fires because the event loop is executing the previous batch's `gather`. APScheduler's `AsyncIOScheduler` will coalesce missed fires and run only once when the loop becomes available — so if the loop was blocked for 19 hours, APScheduler would have been silent for the entire duration.

**This matches the forensic evidence exactly:** "even APScheduler process_batch 5s jobs stopped logging" — correct, because they couldn't fire while the event loop was blocked in the gather.

---

## 4. Phase-23.1.19 SQLite FD Fix Analysis

The pattern used throughout (e.g., `backend/db/tickets_db.py:57`, `backend/services/sla_monitor.py:56`):

```python
with closing(sqlite3.connect(self.db_path)) as conn, conn:
    ...
```

The double context manager semantics:
- `closing(conn).__exit__` calls `conn.close()` — guaranteed to run.
- `conn.__exit__` (sqlite3.Connection as context manager) calls `conn.commit()` on success or `conn.rollback()` on exception.

**Order of operations on exception:** Python evaluates multiple `with` targets left-to-right. On `__exit__`, they are called in reverse order: `conn.__exit__` (rollback) first, then `closing(conn).__exit__` (close) second. If `conn.__exit__` raises (e.g., during rollback on a corrupted journal), `closing(conn).__exit__` is still guaranteed to execute because Python's compound-with calls each `__exit__` independently (PEP 343). The connection WILL close.

**FK violation scenario:** If a write fails with `IntegrityError`, `conn.__exit__` calls `rollback()`. SQLite `rollback()` on a shared-cache or WAL database can block momentarily but will not deadlock in a single-process setup. The connection closes afterward. No deadlock risk here.

**Verdict:** The phase-23.1.19 fix is correct. `with closing(conn) as conn, conn:` is safe and ensures FD release even on exception.

---

## 5. launchd KeepAlive Semantics

From the plist and Apple docs: `KeepAlive: true` respawns the process when it EXITS. It does NOT detect hung-but-alive processes. There is no `WatchdogTimeout` key in launchd.plist(5) for user-land LaunchAgents. The kernel's watchdog (watchdogd) only monitors WindowServer and a handful of system daemons — it is not configurable for third-party processes.

**Confirmed via man page read:** No `WatchdogTimeout` exists in the launchd plist spec for user-defined jobs.

---

## 6. Heartbeat File Analysis

`handoff/.cycle_heartbeat.json` is written by `CycleHealthLog._write_heartbeat()` at `backend/services/cycle_health.py:135`. This is called by `record_cycle_start()` and `record_cycle_end()`. These methods are called from the autonomous trading loop (`autonomous_loop.py`), NOT from the queue processor or the APScheduler process_batch job.

The heartbeat is therefore a cycle-level dead-man's switch for the paper trading loop, not a process-level liveness signal. It does not update every 30s unless the trading cycle runs every 30s (it runs daily). It is not the trigger for the launchd restart — it is just data for the frontend freshness strip.

**Is anything watching the heartbeat to kick the backend?** No. The `compute_freshness()` function reads it for the UI dashboard only. No external watchdog reads it to restart the process.

---

## 7. Most Likely Hang Scenario (Ranked)

1. **PRIMARY (HIGH CONFIDENCE): Stuck ticket in the queue processor blocking the asyncio event loop.**
   - A ticket was in IN_PROGRESS state.
   - `process_single_ticket` awaited `loop.run_in_executor(None, spawn_agent_session, ...)`.
   - Inside `spawn_agent_session`, `_spawn_real_agent` created a `ThreadPoolExecutor(max_workers=1)` and called `future.result(timeout=60)`.
   - The Anthropic HTTP connection did not time out at the TCP level (OS-level connection lingered without a RST), causing the worker thread to hang indefinitely.
   - `ThreadPoolExecutor.__exit__` (called via `with` block) called `shutdown(wait=True)`, blocking the run_in_executor thread.
   - The `asyncio.gather()` in `process_queue_batch` blocked.
   - The APScheduler `process_batch` coroutine never returned.
   - The event loop became permanently blocked. No new requests served. No new log lines. TCP accept loop never ran.

2. **SECONDARY (MODERATE CONFIDENCE): App Nap CPU throttling.**
   - No `ProcessType` key means the process can be subject to App Nap.
   - Overnight, with no user activity, macOS could have downgraded the process to background QOS.
   - Under extreme throttling, the process appears alive (state S, threads exist) but makes no forward progress.
   - This would also explain why caffeinate didn't help — it kept the system awake but not the process's QOS tier.

3. **TERTIARY (LOW CONFIDENCE): Lock contention in a non-async context.**
   - All threading.Lock instances are narrowly scoped. No cross-lock cycles found.

---

## 8. Internal Files Inspected

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/main.py` | 406 | FastAPI entry, lifespan, scheduler wiring | Read (L1-350) |
| `backend/services/kill_switch.py` | 177 | Kill-switch state + threading.Lock | Read in full |
| `backend/services/api_cache.py` | 140 | In-memory TTL cache + threading.Lock | Read in full |
| `backend/services/perf_tracker.py` | 148 | Latency tracker + threading.Lock | Read in full |
| `backend/services/cycle_health.py` | 228 | Heartbeat + cycle JSONL | Read in full |
| `backend/services/ticket_queue_processor.py` | 560 | Ticket queue, agent spawn, 60s Anthropic call | Read (L1-500) |
| `backend/governance/limits_loader.py` | 156 | Immutable limits watcher + os._exit | Read in full |
| `backend/agents/mas_events.py` | ~210 | asyncio.Lock conditional init | Read (L90-110) |
| `~/Library/LaunchAgents/com.pyfinagent.backend.plist` | 40 | launchd job definition | Read in full |
| `backend/db/tickets_db.py` | ~470 | SQLite ticket store | Scanned (closing pattern) |
| `backend/services/sla_monitor.py` | ~80 | SLA monitor | Scanned (closing pattern) |
