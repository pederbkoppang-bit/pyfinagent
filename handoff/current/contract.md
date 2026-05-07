---
step: phase-23.2.23
title: Cron / Logs operator dashboard page + supporting backend endpoints
cycle_date: 2026-05-07
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_2_23.py'
research_brief: handoff/current/phase-23.2.23-external-research.md (also see phase-23.2.23-internal-codebase-audit.md)
---

# Contract — phase-23.2.23

## Hypothesis

User: "We need a new page where we see all the cron jobs and the logs."
Today the user has no single place to see what's scheduled and what's
recently logged — APScheduler jobs live in two processes (FastAPI +
slack_bot), launchd jobs live in plists, logs live in
`handoff/logs/*.log` + `backend.log`. Researcher recommendation: a new
`/cron` route with two tabs (Jobs / Logs) backed by two new authenticated
read-only endpoints (`/api/jobs/all` and `/api/logs/tail`).

## Research-gate summary

Researcher (ae12e50f28b1313fd) returned `gate_passed: true`:
- 6 sources read in full via WebFetch (LogRocket UI patterns,
  Potapov SSE-vs-polling 2025, APScheduler base scheduler docs,
  GitHub Primer progressive disclosure, PortSwigger path traversal,
  Apache Airflow UI overview)
- 16 URLs collected; 10 in snippet-only
- Recency scan 2024-2026; GitHub Actions 1000-line backscroll +
  Potapov 2025 SSE recommendation are the new findings; canonical
  short-poll guidance is acceptable for a single-developer local app
- 11 internal files inspected with file:line anchors
- 13 known scheduled jobs documented (2 main + 11 slack_bot/launchd)
- Concrete recommendation: route, sidebar entry, tab layout, endpoint
  shapes, allowlist constraints

## Immutable success criteria (verbatim — DO NOT EDIT)

1. New backend endpoint `GET /api/jobs/all` returns a JSON envelope
   `{jobs: [{id, source, schedule, next_run, last_run, status,
   description}], generated_at}`. `source` is one of
   `{main_apscheduler, slack_bot, launchd}`. Slack-bot and launchd
   entries come from a curated static manifest if cross-process
   introspection is not available; main_apscheduler entries come
   from live `scheduler.get_jobs()` introspection.
2. New backend endpoint `GET /api/logs/tail?log=<key>&lines=<n>`
   returns the last N lines of an allowlisted log file as
   `{log: <key>, lines: ["...", ...], n_returned, total_size_bytes}`.
   The `log` parameter is REJECTED unless it matches the allowlist
   `{backend, watchdog, restart, harness, autoresearch,
   mas_harness_launchd}`. `lines` is clamped to `[10, 1000]`.
   Path traversal MUST be impossible — the endpoint never accepts
   nor echoes a raw path.
3. Both endpoints are protected by the existing auth middleware
   (i.e., NOT added to `_PUBLIC_PATHS`).
4. New page `frontend/src/app/cron/page.tsx` follows the standard
   6-tier shell (`flex h-screen overflow-hidden` outer, `<Sidebar />`,
   `flex-shrink-0` header zone, `flex-1 overflow-y-auto scrollbar-thin`
   content zone) per `.claude/rules/frontend-layout.md`. Two tabs:
   `Jobs` and `Logs`. Phosphor icons only (no emoji).
5. Sidebar entry added: `{ href: "/cron", label: "Cron / Logs", icon:
   Clock }` in the System section of `frontend/src/components/Sidebar.tsx`
   (between MAS Dashboard and Agent Map). `Clock` exported from
   `frontend/src/lib/icons.ts`.
6. Jobs tab renders a sortable table with columns id / source /
   schedule / next_run / last_run / status. Status color-coded:
   emerald=ok, rose=failed, amber=in_progress, slate=never_run.
   No emoji. Empty state for "no jobs reported" with a Phosphor
   `IconWarning` and a link back to `/agents`.
7. Logs tab renders a dropdown (the 6 allowlisted keys), a lines
   selector (50/100/200/500/1000), a monospace pre block with the
   tail, and a "Refresh" button. Auto-refresh polls every 5s when
   the page is focused; counts consecutive failures and stops after
   5 per `.claude/rules/frontend.md`. Loading + error + empty states
   per the same rule.
8. Backend tests:
   - `tests/api/test_jobs_all.py` — at least 3 tests: shape contract,
     auth required, every entry has the documented keys.
   - `tests/api/test_logs_tail.py` — at least 4 tests: allowlist
     enforced (rejects an arbitrary path), lines clamp enforced
     (10/1000), happy-path tail returns the last N lines, auth
     required.
9. `python tests/verify_phase_23_2_23.py` exits 0 (asserts file
   existence, sidebar entry, allowlist constants, page shell
   structure, no-emoji, AST parse).
10. `cd frontend && npx --no-install tsc --noEmit` exits 0 (no new
    type errors).
11. The new page renders against the live backend (localhost:8000
    healthy) and shows at least the `paper_trading_daily` and
    `process_batch` jobs from `scheduler.get_jobs()`. Verified via
    a smoke check using `curl /api/jobs/all` after backend restart.

## Plan steps

1. **Track the running APScheduler instances** in a module-level
   registry so `/api/jobs/all` can call `.get_jobs()`. Edit
   `backend/main.py` lifespan: store `scheduler` and `queue_scheduler`
   into a module-level dict (e.g., `_running_schedulers["main"] =
   scheduler; _running_schedulers["queue"] = queue_scheduler`). No
   behavior change to existing scheduling.
2. **New backend file** `backend/api/cron_dashboard_api.py`:
   - `GET /api/jobs/all` — merges (a) live introspection of the two
     APSchedulers via `scheduler.get_jobs()`, (b) static manifest of
     slack_bot's 11 jobs (mirroring `_PHASE9_JOB_IDS` +
     morning/evening_digest/watchdog_health_check/prompt_leak_redteam
     from `slack_bot/scheduler.py`), (c) static launchd manifest
     (`com.pyfinagent.backend-watchdog.plist`).
   - `GET /api/logs/tail?log=<key>&lines=<n>`:
     - `_LOG_PATHS` allowlist dict mapping key → resolved Path.
     - Reject unknown keys with HTTP 400.
     - Clamp `lines` to `[10, 1000]`.
     - Stream-tail using `collections.deque(open(p, encoding='utf-8',
       errors='replace'), maxlen=lines)`.
     - Returns `{log, lines: [...], n_returned, total_size_bytes}`.
   - Register the new router in `backend/main.py`.
3. **Tests**:
   - `tests/api/test_jobs_all.py` — uses `TestClient` against a
     stubbed scheduler; asserts shape and key presence; asserts auth
     required (401 without token).
   - `tests/api/test_logs_tail.py` — uses `tmp_path` to create fake
     allowlisted log files with predictable lines; asserts shape,
     allowlist rejection (400 on `log=etc/passwd` and unknown key),
     line clamp (lines=5 -> 10, lines=10000 -> 1000), auth required.
4. **Frontend**:
   - `frontend/src/app/cron/page.tsx` — 6-tier shell, two tabs, polled
     fetch via `apiGet`. No new component primitives — reuse existing
     button / card / scrollbar-thin classes from prior pages.
   - `frontend/src/components/Sidebar.tsx` — add the System entry.
   - `frontend/src/lib/icons.ts` — re-export `Clock` directly.
   - `frontend/src/lib/types.ts` — add `JobInfo` and `LogTailResponse`
     types.
   - `frontend/src/lib/api.ts` — add `getAllJobs` and `getLogTail`
     thin wrappers.
5. **Verifier** `tests/verify_phase_23_2_23.py`:
   - AST-parse modified .py files
   - Assert allowlist dict + `_LOG_PATHS` keys present in
     `cron_dashboard_api.py`.
   - Assert both endpoints registered in `backend/main.py`.
   - Assert sidebar entry references `/cron`.
   - Assert the new page file exists with the required shell + tabs.
   - Grep no-emoji on new frontend files.
   - Run the new pytests.
6. **Live verification** (after backend restart):
   - `curl http://127.0.0.1:8000/api/jobs/all -H "Authorization: ..."`
     returns a non-empty job list including `paper_trading_daily`.
   - `curl /api/logs/tail?log=watchdog&lines=20` returns 20 lines.
7. **Append `harness_log.md`** AFTER Q/A PASS.

## Out of scope

- **Slack-bot live introspection.** The slack_bot is a separate
  process; cross-process IPC for live `get_jobs()` is overkill for a
  single-developer local app. Static manifest with last-fired
  timestamp from the existing job_status_api heartbeat registry is
  sufficient. A future phase could wire a `POST /api/jobs/heartbeat`
  for the slack_bot to push its `next_run_time` on each fire.
- **Server-Sent Events for log streaming.** Researcher endorsed SSE
  but flagged short-poll as acceptable for a local app. Polling at 5s
  with stop-after-5-failures matches existing conventions.
- **Log rotation / archival.** Existing logs are append-only; rotation
  is out of scope.
- **Job-trigger (start/stop/run-now) actions.** Read-only first; any
  action surface should be a separate phase with its own deny-list
  audit (matches the `mcp__bigquery__execute-query` precedent).
- **Cost / rate-limit metrics.** Already covered by
  `/api/observability/cost-budget` and the existing dashboard tile.

## Backwards compatibility

- New endpoints; no existing route changes.
- New page; no existing route changes.
- New sidebar entry; existing entries untouched.
- Scheduler-registry is module-level dict population; existing code
  paths are byte-identical.
- No schema changes, no settings changes.

## References

- Researcher: `handoff/current/phase-23.2.23-external-research.md`,
  `handoff/current/phase-23.2.23-internal-codebase-audit.md`
- `backend/main.py:163-220` (scheduler init sites)
- `backend/api/paper_trading.py:895-934` (paper_trading_daily registration)
- `backend/slack_bot/scheduler.py:31-90, 336-382` (slack-bot jobs +
  phase-9 registry)
- `backend/api/job_status_api.py` (existing heartbeat registry, reused)
- `frontend/src/app/agents/page.tsx` (precedent for system-internals page)
- `.claude/rules/frontend-layout.md` (6-tier shell mandatory)
- `.claude/rules/frontend.md` (polling stop-after-5 rule, no emoji)
- PortSwigger path-traversal allowlist guidance
- Apache Airflow tabbed-UI precedent
