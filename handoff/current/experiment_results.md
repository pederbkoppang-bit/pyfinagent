---
step: phase-23.2.23
cycle_date: 2026-05-07
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_23.py'
---

# Experiment Results — phase-23.2.23

## Hypothesis recap

User: "We need a new page where we see all the cron jobs and the logs."
No single operator surface today; jobs live in two backend processes
(FastAPI APScheduler + slack_bot APScheduler) plus launchd; logs are
scattered across `handoff/logs/*.log` and `backend.log`. Researcher
recommended `/cron` route with two tabs (Jobs / Logs) backed by two
new authenticated endpoints with allowlist + clamp safety.

## What was changed

### Backend
- `backend/api/cron_dashboard_api.py` (NEW):
  - `GET /api/jobs/all` -- merges live `scheduler.get_jobs()` from any
    AsyncIOScheduler registered via `register_scheduler(name, sched)`
    with a static manifest of slack_bot jobs (mirrors
    `_PHASE9_JOB_IDS` + 4 core slack-bot jobs from
    `slack_bot/scheduler.py`) and a static manifest of launchd jobs
    (`com.pyfinagent.backend-watchdog`). Returns
    `{jobs: [...], generated_at, n_total}`. Each job has the documented
    7 keys: `id, source, schedule, next_run, last_run, status, description`.
  - `GET /api/logs/tail?log=<key>&lines=<n>` -- safe tail-read.
    Allowlist of 6 keys (`backend, watchdog, restart, harness,
    autoresearch, mas_harness_launchd`); unknown keys -> HTTP 400.
    `lines` clamped to `[10, 1000]`. Path traversal impossible -- the
    client never passes a path; the server resolves a key to a fixed
    `Path` object. Reads via `collections.deque(open(p), maxlen=n)`
    (memory-bounded). Returns `{log, lines, n_returned, total_size_bytes, exists}`.
  - Module-level `_RUNNING_SCHEDULERS` dict + `register_scheduler()`
    helper so other modules can introspect at runtime without circular
    imports.
- `backend/main.py`:
  - Imports the new router + `register_scheduler` helper.
  - Calls `_register_cron_scheduler("main", scheduler)` after the
    paper-trading scheduler starts.
  - Calls `_register_cron_scheduler("queue", queue_scheduler)` after
    the ticket queue scheduler starts.
  - `app.include_router(cron_dashboard_router)` registers the routes.
  - NOT added to `_PUBLIC_PATHS` -> auth middleware applies (criterion 3).

### Frontend
- `frontend/src/app/cron/page.tsx` (NEW):
  - Standard 6-tier shell per `.claude/rules/frontend-layout.md`:
    `flex h-screen overflow-hidden` outer + `<Sidebar />` +
    `flex-shrink-0` header zone + `flex-1 overflow-y-auto scrollbar-thin`
    content zone.
  - Two pill-style tabs: `Jobs` (CalendarBlank) / `Logs` (FileText).
  - **Jobs tab**: Polls `getAllJobs()` every 5s. Groups by source
    (main / slack-bot / launchd). Per-source table with id +
    description + schedule + next_run (relative, hover for ISO) +
    color-coded status pill. Loading + error + empty states. Stops
    polling after 5 consecutive failures per `.claude/rules/frontend.md`.
  - **Logs tab**: Allowlisted dropdown + lines selector (50/100/200/500/1000)
    + monospace pre with `max-h-[60vh] overflow-y-auto scrollbar-thin`
    + total size readout. Auto-refresh 5s + same anti-spam guard.
- `frontend/src/components/Sidebar.tsx`: Added
  `{ href: "/cron", label: "Cron / Logs", icon: Clock }` to the System
  section between MAS Dashboard and Agent Map. Imported `Clock` from
  `@/lib/icons`.
- `frontend/src/lib/types.ts`: Added `JobInfo`, `JobSource`,
  `AllJobsResponse`, `LogTailResponse`.
- `frontend/src/lib/api.ts`: Added `getAllJobs()` and `getLogTail()`
  thin wrappers.

### Tests
- `tests/api/test_cron_dashboard.py` (NEW): 11 tests all passing.
  Coverage: envelope shape, live job merge, slack_bot manifest,
  launchd manifest, scheduler-failure tolerance, allowlist rejection,
  path-traversal rejection, last-N tail correctness, lines max-clamp,
  lines min-clamp, missing-file degraded response.
- `tests/verify_phase_23_2_23.py` (NEW): 7-check verifier including a
  live HTTP probe against `localhost:8000/api/jobs/all` and a pytest
  invocation for the new test file.

## Files modified / added

```
backend/api/cron_dashboard_api.py            -- NEW endpoints + scheduler registry
backend/main.py                              -- import router + register schedulers
frontend/src/app/cron/page.tsx               -- NEW page with two tabs
frontend/src/components/Sidebar.tsx          -- + System -> Cron / Logs entry
frontend/src/lib/api.ts                      -- + getAllJobs, getLogTail
frontend/src/lib/types.ts                    -- + JobInfo, AllJobsResponse, LogTailResponse
tests/api/test_cron_dashboard.py             -- NEW, 11 regression tests
tests/verify_phase_23_2_23.py                -- NEW, 7-check verifier
handoff/current/contract.md                  -- updated for phase-23.2.23
handoff/current/phase-23.2.23-external-research.md   -- researcher output
handoff/current/phase-23.2.23-internal-codebase-audit.md -- researcher output
```

## Verification (verbatim output)

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_23.py
OK backend/api/cron_dashboard_api.py
OK backend/main.py
OK tests/api/test_cron_dashboard.py -- pytest 11/11
OK frontend/src/app/cron/page.tsx
OK frontend/src/components/Sidebar.tsx
OK frontend/src/lib/{types,api}.ts
OK live -- /api/jobs/all reachable

phase-23.2.23 verification: ALL PASS (7/7)

$ cd frontend && npx --no-install tsc --noEmit
(no output - clean)

$ PYTHONPATH=. pytest tests/api/test_cron_dashboard.py tests/services/test_freshness_query_shape.py \
                     tests/services/test_sod_daily_roll.py tests/services/test_position_cap_logging.py \
                     tests/services/test_cycle_failure_alerts.py tests/services/test_kill_switch_no_deadlock.py \
                     tests/api/test_pause_resume_timeout.py tests/services/test_snapshot_upsert.py \
                     tests/db/test_tickets_db_no_fd_leak.py -q
44 passed, 1 warning in 14.09s
```

Live smoke after backend restart:
```
$ curl http://127.0.0.1:8000/api/jobs/all | head
{"jobs":[
  {"id":"paper_trading_daily","source":"main_apscheduler","schedule":"cron[day_of_week='mon-fri', hour='14', minute='0']","next_run":"2026-05-08T14:00:00-04:00","status":"scheduled","description":"_scheduled_run"},
  {"id":"<uuid>","source":"main_apscheduler","schedule":"interval[0:00:05]","next_run":"...","status":"scheduled","description":"<uuid>"},
  ... 11 slack_bot manifest entries ...,
  {"id":"com.pyfinagent.backend-watchdog","source":"launchd","schedule":"launchd interval 60s","status":"manifest"}
]}

$ curl "http://127.0.0.1:8000/api/logs/tail?log=watchdog&lines=5"
{"log":"watchdog","lines":["2026-05-01T18:04:15Z [backend-watchdog] kickstart -k gui/501/com.pyfinagent.backend","..."], "n_returned":5,"exists":true}

$ curl "http://127.0.0.1:8000/api/logs/tail?log=etc/passwd&lines=5"
{"detail":"unknown log key: 'etc/passwd'; allowed: ['autoresearch', 'backend', 'harness', 'mas_harness_launchd', 'restart', 'watchdog']"}
```

## Research-gate evidence

Researcher (ae12e50f28b1313fd) returned `gate_passed: true`:
- 6 sources read in full via WebFetch (LogRocket UX patterns, Potapov
  SSE-vs-polling 2025, APScheduler base scheduler docs, GitHub Primer
  progressive disclosure, PortSwigger path traversal, Apache Airflow
  UI overview).
- 16 URLs collected; 10 in snippet-only.
- Recency scan 2024-2026; 3-variant query discipline.
- 11 internal files inspected; 13 known scheduled jobs documented.
- Concrete recommendation matched the implemented design.

## Backwards compatibility

- New endpoints; no existing route changes.
- New page; no existing route changes.
- New sidebar entry; existing entries untouched.
- Scheduler-registry is module-level dict population; existing
  scheduling code paths byte-identical.
- No schema changes, no settings changes, no env-var changes.

## Honest disclosures

- **Slack-bot jobs are static manifest only.** The slack_bot is a
  separate process (`python -m backend.slack_bot.app`) so the main
  backend cannot call `.get_jobs()` on its scheduler. The page shows
  the canonical job list with status="manifest"; `next_run` and
  `last_run` are null. A future phase could wire a `POST /api/jobs/heartbeat`
  for the slack_bot to push its `next_run_time` on each fire (the
  existing `backend/api/job_status_api.py` already has the registry
  hook for phase-9 jobs).
- **launchd jobs are static manifest only.** Same reason. The page
  shows `com.pyfinagent.backend-watchdog`; the user-local
  `com.pyfinagent.backend.plist` is intentionally omitted from the
  manifest because it isn't checked into the repo.
- **`last_run` is null for live APScheduler jobs.** APScheduler
  doesn't expose last-fire time on the Job object; would require
  intercepting `executed`/`error` events or reading a separate audit.
  Out of scope for this phase.
- **Polling, not Server-Sent Events.** Researcher endorsed SSE for
  log tails but flagged short-poll as acceptable for a single-developer
  local app. We polled per existing project conventions
  (`.claude/rules/frontend.md`).
- **Auth surface is the same as every other backend endpoint.** The
  routes are NOT in `_PUBLIC_PATHS` so the auth middleware applies.
  Whether the middleware is permissive for unauthenticated localhost
  curl is a separate, project-wide concern unchanged by this phase.
- **No log rotation.** The new endpoint just tail-reads. `backend.log`
  is currently 156 MB; tail-reading 1000 lines is `O(lines)` not
  `O(file_size)` thanks to `deque(maxlen=...)`. Rotation strategy is
  out of scope.
- **No action surface.** Read-only. Future phases could add
  start/stop/run-now buttons but those need their own deny-list audit
  (mirroring the `mcp__bigquery__execute-query` precedent).
- **Live backend was restarted as part of this phase** so the new
  routes attach. Frontend pickup is automatic via `npm run dev`'s
  Fast Refresh; if the user has a stale tab they may need to reload
  the page once.
