# Internal Codebase Audit — Phase 23.1.19
# SQLite FD Leak: Full Site Inventory

Generated: 2026-04-29

---

## 1. Root Cause Summary

`with sqlite3.connect(path) as conn:` invokes the connection object's context manager,
which commits or rolls back the open transaction but **does not close the connection**.
The Python docs (3.12+) state this explicitly:

> "The context manager neither implicitly opens a new transaction nor closes the
> connection. If you need a closing context manager, consider using contextlib.closing()."

Each call site therefore leaks one open connection object. CPython will eventually
garbage-collect it, but the GC is not guaranteed to run promptly, and in practice
each open connection keeps:
- 1 FD to the main `tickets.db` file
- 1 FD to `tickets.db-wal` (WAL file, created on first write)
- 1 FD to `tickets.db-shm` (shared-memory index for WAL)

This matches the lsof observation of 9 FDs (3 per leaked connection object) after
only a few calls. With `process_queue_batch` running every 5 s and each batch
invoking multiple TicketsDB methods, FDs accumulate steadily until the process
hits the macOS launchd soft limit (default 256 in older sessions, 10240 in newer
ones, or 16384 when raised via the NumberOfFiles plist).

---

## 2. Full Site Inventory (23 sites, not 17)

The initial count of 17 missed 6 additional sites in three service files.

### 2.1 backend/db/tickets_db.py — 15 sites

| Line | Method | Description |
|------|--------|-------------|
| 56  | `_init_database` | Schema + index creation at init |
| 156 | `create_ticket` | INSERT + ticket_number increment |
| 183 | `is_duplicate_envelope` | SELECT by envelope_id |
| 192 | `mark_duplicate` | UPDATE + SELECT by envelope_id |
| 210 | `get_ticket` | SELECT by id |
| 223 | `get_ticket_by_number` | SELECT by ticket_number |
| 236 | `get_open_tickets` | SELECT OPEN tickets |
| 306 | `update_ticket_status` | UPDATE status + timestamps |
| 319 | `acknowledge_ticket` | UPDATE acknowledged_at |
| 331 | `get_ticket_stats` | 3 SELECT aggregates |
| 368 | `get_sla_breaches` | SELECT elapsed > SLA |
| 389 | `cleanup_old_tickets` | UPDATE RESOLVED -> CLOSED |
| 404 | `get_ticket_queue_position` | SELECT COUNT subquery |
| 421 | `update_queue_positions` | SELECT + UPDATE in loop |
| 461 | `clear_queue` | DROP + reinit (calls _init_database again) |

### 2.2 backend/services/ticket_queue_processor.py — 2 sites

| Line | Method | Description |
|------|--------|-------------|
| 40  | `_increment_retries` | UPDATE retries inline; bypasses TicketsDB layer |
| — | (no other direct calls) | All other SQLite access goes through TicketsDB methods |

Note: `_increment_retries` is a raw `sqlite3.connect` call that duplicates a DB
operation that properly belongs in `TicketsDB`. This is both a leak site and a
layering violation.

### 2.3 backend/slack_bot/commands.py — 1 site

| Line | Method | Description |
|------|--------|-------------|
| 245 | `handle_any_message` ("clear queue" branch) | DELETE + INSERT directly; bypasses `TicketsDB.clear_queue()` |

Another layering violation: this raw call duplicates `clear_queue()` logic.

### 2.4 backend/slack_bot/direct_responder.py — 1 site

| Line | Method | Description |
|------|--------|-------------|
| 194 | `_build_ticket_status` | SELECT status/count + recent tickets |

This file uses `str(db_path)` on a `Path` object, which is correct but the
connection is not closed.

### 2.5 backend/services/sla_monitor.py — 2 sites (previously uncounted)

| Line | Method | Description |
|------|--------|-------------|
| 53  | `check_sla_breaches` (approx.) | SELECT overdue tickets |
| 109 | (second method) | UPDATE or follow-up SELECT |

`sla_monitor.py` runs every 300 s (per `app.py` line 64). Lower cadence than the
queue processor but still leaking.

### 2.6 backend/services/response_delivery.py — 2 sites (previously uncounted)

| Line | Method | Description |
|------|--------|-------------|
| 234 | `deliver_ticket_response` (approx.) | SELECT ticket details |
| 270 | (second method) | UPDATE delivery status |

Called from `process_single_ticket` on every successfully processed ticket.

### 2.7 backend/services/stuck_task_reaper.py — 1 site (previously uncounted)

| Line | Method | Description |
|------|--------|-------------|
| 48  | `reap_stuck_tasks` (approx.) | SELECT + UPDATE stuck/hung tickets |

Runs every 60 s (per `app.py` line 68).

---

## 3. Call Cadence Analysis

Source: `backend/main.py:159-183` (queue processor) and `backend/slack_bot/app.py`

| Caller | Interval | Sites called per tick | Approx FD rate |
|--------|----------|----------------------|----------------|
| `process_queue_batch` via APScheduler in `main.py` | every 5 s | `update_queue_positions` (1) + `get_open_tickets` (1) + per-ticket: `update_ticket_status` x2, `_increment_retries` x1 | ~5–10 FD/min worst case |
| SLA monitor (`sla_monitor.py`) | every 300 s | 2 sites | ~24 FD/hr |
| Stuck-task reaper | every 60 s | 1 site | ~60 FD/hr |
| Slack `handle_any_message` | on every message | `is_duplicate_envelope` (via ingestion), `create_ticket`, `acknowledge_ticket` | demand-driven |

The queue processor at 5 s intervals is the dominant driver. Over 8 hours the
process opens approximately 2,400–4,800 connections without closing any, saturating
the FD table.

The `process_queue_batch` runs inside `backend/main.py` (the uvicorn process), so
the FD leak is in the same process as the limits_loader watcher. When the process
FD table exhausts, `_watcher_loop` fails to `path.open("rb")` every 10 s, and
subsequently the health endpoint, uvicorn access logs, and any other file-opening
operation all fail. This explains the cascading 500s.

---

## 4. Existing Tests (tests/test_tickets_db.py)

File: `/Users/ford/.openclaw/workspace/pyfinagent/tests/test_tickets_db.py`

The existing test is a standalone `test_phase_1_schema()` function (not pytest).
It exercises: `create_ticket`, `is_duplicate_envelope`, `update_ticket_status`,
`get_open_tickets`, `acknowledge_ticket`, `get_ticket_stats`. It uses a temp DB
at `/tmp/test_tickets.db` and cleans up after itself.

**Impact of Fix A:** The fix changes `with sqlite3.connect(path)` to
`with closing(sqlite3.connect(path))`. This changes how connections are closed but
not what queries execute or what values are returned. The existing test should pass
unchanged after the fix.

**What the test does NOT check:** FD counts, connection lifecycle, or concurrent
access. A new `tests/db/test_tickets_db_no_fd_leak.py` is warranted (Fix B below).

---

## 5. limits_loader.py — Confirmed NOT the Source

`backend/governance/limits_loader.py:57`:
```python
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(64 * 1024), b""):
        h.update(chunk)
```

`Path.open()` returns a `pathlib`-backed `io.BufferedReader`. Python's `io` types
fully implement the context manager protocol including `__exit__ -> close()`.
The FD is released correctly on every call.

`limits_loader.py` is the **victim** (cannot open the file when FD table is full),
not the source.

---

## 6. No Other SQLite Callers in backend/ (Other Than Listed)

grep results show 23 total sites across 7 files. No other `sqlite3.connect` calls
exist in the backend. The full set is:
- `backend/db/tickets_db.py` (15)
- `backend/services/ticket_queue_processor.py` (1 — `_increment_retries`)
- `backend/services/sla_monitor.py` (2)
- `backend/services/response_delivery.py` (2)
- `backend/services/stuck_task_reaper.py` (1)
- `backend/slack_bot/commands.py` (1)
- `backend/slack_bot/direct_responder.py` (1)

---

## 7. Fix Surface Map

### Fix A — contextlib.closing() wrap at every site

Pattern for tickets_db.py methods:
```python
# Before (leaks FD):
with sqlite3.connect(self.db_path) as conn:
    ...

# After (closes on exit):
from contextlib import closing
with closing(sqlite3.connect(self.db_path)) as conn:
    with conn:          # transaction context
        ...
```

Note: the inner `with conn:` is optional for read-only queries but required
wherever `conn.commit()` is currently called explicitly. Replacing the outer
`with sqlite3.connect` with `closing()` removes the automatic commit/rollback;
the inner `with conn:` restores it.

**Complete file:line list requiring Fix A:**

```
backend/db/tickets_db.py:56
backend/db/tickets_db.py:156
backend/db/tickets_db.py:183
backend/db/tickets_db.py:192
backend/db/tickets_db.py:210
backend/db/tickets_db.py:223
backend/db/tickets_db.py:236
backend/db/tickets_db.py:306
backend/db/tickets_db.py:319
backend/db/tickets_db.py:331
backend/db/tickets_db.py:368
backend/db/tickets_db.py:389
backend/db/tickets_db.py:404
backend/db/tickets_db.py:421
backend/db/tickets_db.py:461
backend/services/ticket_queue_processor.py:40
backend/services/sla_monitor.py:53
backend/services/sla_monitor.py:109
backend/services/response_delivery.py:234
backend/services/response_delivery.py:270
backend/services/stuck_task_reaper.py:48
backend/slack_bot/commands.py:245
backend/slack_bot/direct_responder.py:194
```

23 sites total.

### Fix B — Regression test (new file)

`tests/db/test_tickets_db_no_fd_leak.py`

Key pattern:
```python
import psutil, os
proc = psutil.Process(os.getpid())
fds_before = proc.num_fds()

db = TicketsDB("/tmp/test_fd_leak.db")
for _ in range(100):
    db.create_ticket(...)
    db.get_open_tickets()
    db.update_ticket_status(...)
    db.get_ticket_stats()

fds_after = proc.num_fds()
delta = fds_after - fds_before
assert delta <= 5, f"FD leak: {delta} FDs leaked over 100 iterations"
```

The bound of 5 accounts for interpreter-internal FD churn (random.urandom,
log file handles, etc.) while clearly failing on the pre-fix leak of ~300 FDs.

### Fix C — Startup RLIMIT log in backend/main.py lifespan

```python
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
logging.info(f"RLIMIT_NOFILE: soft={soft} hard={hard}")
if soft < 4096:
    logging.warning(f"RLIMIT_NOFILE soft={soft} is dangerously low; consider ulimit -n 65536")
```

Add this near the top of the `@asynccontextmanager async def lifespan()` block,
before any service starts. If soft is 256 (old macOS terminal sessions) the
warning will fire on every start.

### Fix D — Thread-local connection (deferred)

Refactoring `TicketsDB` to hold a thread-local `sqlite3.Connection` eliminates
per-call open/close overhead entirely. Implementation requires:
- `threading.local()` storage on the `TicketsDB` instance
- A `_get_conn()` helper that checks `self._local.conn` and creates if absent
- An explicit `close_all()` method for test teardown

This is the most robust long-term solution but a more invasive change. Defer until
after Fix A + B + C are verified in production.

---

## 8. Recommendation

Apply **Fix A + Fix B + Fix C** in a single PR. Order of application:

1. Fix C first (startup log — zero risk, immediate observability)
2. Fix A (closing() wrap — mechanical diff, 23 sites, no logic change)
3. Fix B (regression test — confirms Fix A held)

Defer Fix D. The per-call open/close pattern introduced by Fix A is slightly less
efficient than a persistent connection (1–2 ms overhead per call) but is correct
and safe. For a 5 s batch interval this overhead is negligible.

The two layering violations (`_increment_retries` raw SQL in the processor,
`clear_queue` raw SQL in commands.py) should be cleaned up as part of Fix A —
route them through `TicketsDB` methods to eliminate duplicate logic. This is not
strictly required for the FD fix but reduces the blast radius of any future
schema change.
