---
step: phase-23.1.19
title: backend FD leak fix (sqlite3 closing()) + regression guard + RLIMIT log
cycle_date: 2026-04-29
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_19.py'
research_brief: handoff/current/phase-23.1.19-external-research.md (also see phase-23.1.19-internal-codebase-audit.md)
---

# Contract — phase-23.1.19

## Hypothesis

Backend crashed with `OSError: [Errno 24] Too many open files` on
`limits.yaml`. The governance watcher uses `with path.open("rb")`
correctly — its FDs ARE released. The leak is upstream.

`grep -c "with sqlite3.connect"` initially identified 17 sites;
researcher's full audit found **23 sites across 7 files**:

- `backend/db/tickets_db.py`: 15 sites
- `backend/services/ticket_queue_processor.py`: 1 site
- `backend/services/sla_monitor.py`: 2 sites
- `backend/services/response_delivery.py`: 2 sites
- `backend/services/stuck_task_reaper.py`: 1 site
- `backend/slack_bot/commands.py`: 1 site
- `backend/slack_bot/direct_responder.py`: 1 site

**Root cause:** Python sqlite3 docs explicitly state
`with sqlite3.connect(...) as conn:` only commits/rolls back the
transaction, **does NOT close the connection**. Each call leaks 1
SQLite connection = 3 FDs (main db + WAL + shm). With
`ticket_queue_processor` running every 5s in lifespan, FDs
accumulate to the OS soft limit and the process can no longer
open ANY file (governance watcher victim).

If we (A) wrap every site with `contextlib.closing(...)` so the
connection actually closes, (B) add a regression test that asserts
no FD growth across 100 method calls, and (C) log RLIMIT_NOFILE at
boot with a WARNING when soft limit is dangerously low, then this
class of bug is closed and any future regression is caught both
locally (test) and operationally (boot log).

## Research-gate summary

- External brief: `handoff/current/phase-23.1.19-external-research.md`
  — 6 sources read in full (Python sqlite3 official docs, Python.org
  discussion thread, alexwlchan.net 2024 TIL, blog.rtwilson worked
  example, Python resource module docs, psutil docs). 15 URLs
  collected. Recency scan 2024-2026 confirms no Python 3.14 change
  to sqlite3 context manager semantics. `gate_passed: true`.
- Internal audit: `handoff/current/phase-23.1.19-internal-codebase-audit.md`
  — 7 files inspected with all 23 file:line anchors and concrete
  patches.

Researcher recommends **A + B + C**. Defers D (thread-local
connection refactor) — more invasive; A is sufficient.

## Plan steps

1. **Fix A — Wrap all 23 sites with `contextlib.closing()`**.
   Pattern:
   ```python
   from contextlib import closing
   with closing(sqlite3.connect(path)) as conn:
       with conn:        # restore commit/rollback semantics
           ...
   ```
   For pure-read methods, the inner `with conn:` is optional but
   kept for consistency with write methods.

2. **Fix B — Regression test**
   `tests/db/test_tickets_db_no_fd_leak.py`. Uses
   `psutil.Process(os.getpid()).num_fds()` before / after 100
   iterations of common TicketsDB methods. Asserts net delta ≤ 5.
   Skipped on Windows (psutil.num_fds is UNIX-only).

3. **Fix C — Startup RLIMIT_NOFILE log** in `backend/main.py`
   lifespan entry. Logs `(soft, hard)` from
   `resource.getrlimit(resource.RLIMIT_NOFILE)`. WARNs when
   soft < 4096.

4. **Immutable verification**
   `tests/verify_phase_23_1_19.py` — greps the source for the
   `closing(` wrap at every previously-leaky site, asserts no
   bare `with sqlite3.connect` pattern remains in the 7 files,
   confirms the new test exists and passes, confirms
   `RLIMIT_NOFILE` is logged from `main.py`.

## Immutable verification command

```bash
source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_19.py
```

Must exit 0 with one ok-line.

## Acceptance criteria

- `pytest tests/db/test_tickets_db_no_fd_leak.py -q` passes.
- `python tests/verify_phase_23_1_19.py` exits 0.
- `lsof -p <backend pid>` after backend restart + 5 minutes of
  ticket-processor activity shows ≤ 5 FDs to tickets.db (was 9
  immediately after restart, growing).
- backend.log shows `RLIMIT_NOFILE: soft=N hard=M` at startup.
- No new `OSError: [Errno 24]` lines in backend.log.

## Backwards compatibility

- `closing()` wraps the connection — no API change for callers.
- Inner `with conn:` preserves commit/rollback semantics that
  callers rely on.
- Test is sandboxed via `tmp_path` and skipped on Windows.
- RLIMIT log is informational; backend always boots even on
  low-limit systems (just logs WARN).

## References

- `handoff/current/phase-23.1.19-external-research.md`
- `handoff/current/phase-23.1.19-internal-codebase-audit.md`
- Python sqlite3 docs:
  https://docs.python.org/3/library/sqlite3.html (canonical
  source for the bug)
- 23 fix sites listed in the internal audit by file:line
