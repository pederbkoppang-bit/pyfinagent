# Phase-23.2.23 Internal Codebase Audit
## Cron / Logs Page — Job + Log Inventory

**Tier:** moderate  
**Date:** 2026-05-05  
**Scope:** Enumerate every scheduled job, every operator-actionable log file, existing endpoints to reuse, auth pattern, and frontend anatomy compliance.

---

## 1. Scheduled Jobs Inventory

### 1A. APScheduler Jobs in `backend/api/paper_trading.py` (main process)

| Job ID | Cron Expression | Function | File:Line | Notes |
|--------|----------------|----------|-----------|-------|
| `paper_trading_daily` | `cron hour=<settings.paper_trading_hour> minute=0 day_of_week=mon-fri TZ=America/New_York` | `_scheduled_run` → calls `run_daily_cycle` | `paper_trading.py:914` (`_add_scheduler_job`) | Only registered if `settings.paper_trading_enabled`. Job ID defined at `paper_trading.py:35` as `_scheduler_job_id = "paper_trading_daily"`. The scheduler itself is created in `main.py:167` (lifespan) and passed into `init_scheduler` at `paper_trading.py:901`. |

### 1B. APScheduler Jobs in `backend/main.py` lifespan (main process)

| Job ID | Cron/Interval | Function | File:Line | Notes |
|--------|--------------|----------|-----------|-------|
| (inline `process_batch`) | `interval seconds=5` | `process_batch` → calls `get_queue_processor().process_queue_batch(batch_size=10)` | `main.py:210` | Ticket queue processor. Registered on a dedicated `queue_scheduler` instance. No explicit job id set. |

The `queue_scheduler` is a separate `AsyncIOScheduler` instance from the paper trading one; both live in the same FastAPI process.

### 1C. APScheduler Jobs in `backend/slack_bot/scheduler.py` (separate process)

These run in the Slack bot process (`python -m backend.slack_bot.app`), not in the FastAPI process. They cannot be introspected via `scheduler.get_jobs()` from the backend.

| Job ID | Cron/Interval Expression | Function | File:Line | Notes |
|--------|------------------------|----------|-----------|-------|
| `morning_digest` | `cron hour=<settings.morning_digest_hour> minute=0 TZ=America/New_York` | `_send_morning_digest` | `scheduler.py:35` | Posts portfolio + reports digest to Slack channel. |
| `evening_digest` | `cron hour=<settings.evening_digest_hour> minute=0 TZ=America/New_York` | `_send_evening_digest` | `scheduler.py:46` | Posts EOD portfolio + trades digest to Slack channel. |
| `watchdog_health_check` | `interval minutes=<settings.watchdog_interval_minutes>` | `_watchdog_health_check` | `scheduler.py:58` | Probes `/api/health`; posts to Slack on failure only. |
| `prompt_leak_redteam` | `cron hour=3 minute=15 TZ=America/New_York` | `_nightly_prompt_leak_redteam` | `scheduler.py:71` | Nightly red-team audit; appends to `handoff/prompt_leak_redteam_audit.jsonl`. |

### 1D. Phase-9 Jobs in `backend/slack_bot/scheduler.py::register_phase9_jobs` (separate process)

Registered via `register_phase9_jobs(scheduler)` on the same Slack-bot scheduler. Module-level mapping at `scheduler.py:361`:

| Job ID | Cron Expression | Module | File:Line |
|--------|----------------|--------|-----------|
| `daily_price_refresh` | `cron hour=1` | `backend.slack_bot.jobs.daily_price_refresh` | `scheduler.py:362` |
| `weekly_fred_refresh` | `cron day_of_week=sun hour=2` | `backend.slack_bot.jobs.weekly_fred_refresh` | `scheduler.py:363` |
| `nightly_mda_retrain` | `cron hour=3` | `backend.slack_bot.jobs.nightly_mda_retrain` | `scheduler.py:364` |
| `hourly_signal_warmup` | `cron minute=5` (every hour) | `backend.slack_bot.jobs.hourly_signal_warmup` | `scheduler.py:365` |
| `nightly_outcome_rebuild` | `cron hour=4` | `backend.slack_bot.jobs.nightly_outcome_rebuild` | `scheduler.py:366` |
| `weekly_data_integrity` | `cron day_of_week=mon hour=5` | `backend.slack_bot.jobs.weekly_data_integrity` | `scheduler.py:367` |
| `cost_budget_watcher` | `cron hour=6` | `backend.slack_bot.jobs.cost_budget_watcher` | `scheduler.py:368` |

These 7 job IDs mirror `_JOB_NAMES` in `backend/api/job_status_api.py:52-60` (the canonical heartbeat registry).

### 1E. launchd Plists

| Label | Interval | Script | Log Output | Repo Location |
|-------|----------|--------|-----------|---------------|
| `com.pyfinagent.backend-watchdog` | every 60 seconds (`StartInterval=60`) | `scripts/launchd/backend_watchdog.sh` | `handoff/logs/backend-watchdog.log` | `scripts/launchd/com.pyfinagent.backend-watchdog.plist` (line 7, 9, 16) |
| `com.pyfinagent.backend` | on-load (launchd managed) | uvicorn backend | n/a (user-local plist, not in repo) | `~/Library/LaunchAgents/com.pyfinagent.backend.plist` — user-local, not tracked in git |

Other launchd plists referenced by the log files:
- `ablation.launchd.log` implies a plist for the ablation experiment runner (not currently found in `scripts/launchd/`; likely removed or never committed).
- `autoresearch.launchd.log` implies a plist for autoresearch (same — not in repo).
- `mas-harness.launchd.log` implies a plist for the MAS harness runner (same).

These three are referenced by their `.launchd.log` files but the plists appear to be user-local and not committed. They should be documented as "historically active launchd agents" on the Cron page, sourced from log-file presence.

---

## 2. Log Files Inventory

All under `/Users/ford/.openclaw/workspace/pyfinagent/handoff/logs/` unless noted.

| File | Size | Last Modified | Writer | Purpose | Operator-actionable? | Rotation? |
|------|------|--------------|--------|---------|----------------------|-----------|
| `backend-watchdog.log` | 1.5 KB | 2026-05-05 (active) | `scripts/launchd/backend_watchdog.sh` via launchd `StandardOutPath` | Watchdog health-check results; backend restart events | YES — shows if backend is being auto-restarted | No rotation configured (launchd appends); kept small by infrequent writes |
| `backend-restart.log` | 1.9 KB | 2026-04-24 | Watchdog script or manual restart | Backend restart timestamps + exit codes | YES — postmortem for unexpected restarts | Manual |
| `autoresearch.log` | 4.7 KB | 2026-04-19 | Autoresearch harness script | Autoresearch optimization cycle output | YES — shows last autoresearch run results | Manual |
| `mas-harness.log` | 2.8 MB | 2026-04-19 | MAS harness (`run_harness.py`) | Full harness cycle output: PLAN/GENERATE/EVALUATE phases | YES — primary harness diagnostic | No rotation; grows unboundedly |
| `ablation.log` | 2.2 KB | 2026-04-19 | Feature ablation script | Feature ablation experiment output | Moderate | Manual |
| `seed_stability_output.log` | 1.1 MB | 2026-04-16 | Seed stability test script | Backtest seed stability test output | Low (historical) | Manual |
| `backend.log` (repo root) | 156 MB | 2026-05-07 (active) | uvicorn/FastAPI via launchd `StandardErrorPath` | Main backend process stderr — all `logger.*` calls from `main.py`, API handlers, services | YES — primary backend diagnostic | No rotation; **156 MB and growing** — tail-only is essential |
| `monthly_approval_state.json` | 365 B | 2026-04-24 | Monthly approval API | HITL monthly approval state | Low (structured JSON, not a log) | n/a |
| `row_count_snapshot.json` | 2 B | 2026-04-27 | Data integrity job | BQ row count snapshot | Low (structured JSON) | n/a |
| `ablation.launchd.log` | 59 B | 2026-04-17 | launchd stderr capture | launchd's own stderr for ablation plist | NO — launchd noise, rarely useful | n/a |
| `autoresearch.launchd.log` | 204 B | 2026-04-17 | launchd stderr capture | launchd's own stderr for autoresearch plist | NO | n/a |
| `mas-harness.launchd.log` | 0 B | 2026-04-16 | launchd stderr capture | Empty (no launchd-level errors) | NO | n/a |

**Operator-actionable subset (expose on Cron/Logs page):**
1. `backend.log` — primary backend diagnostic (tail last N lines only; file is 156 MB)
2. `backend-watchdog.log` — watchdog / restart events
3. `backend-restart.log` — restart timeline
4. `mas-harness.log` — harness cycle output
5. `autoresearch.log` — autoresearch results

**Skip on the page:** `*.launchd.log` (launchd noise), `*.json` files (structured state, not logs), `ablation.log` and `seed_stability_output.log` (historical one-offs).

---

## 3. Existing Endpoints to Reuse

### `GET /api/jobs/status` (`backend/api/job_status_api.py:105`)
- Returns `{ jobs: [{ name, last_run_at, last_duration_s, status, last_error }] }` for the 7 phase-9 Slack-bot jobs.
- Status values: `never_run | ok | failed | in_progress | skipped_idempotent`.
- In-memory registry pre-seeded. Cross-process delivery via `POST /heartbeat` (not yet wired from Slack bot).
- Reuse: YES — use as the primary job-status source. Will show `never_run` for all 7 until the heartbeat wiring is implemented. The Cron page should render this honestly with a "heartbeat not yet wired" note.
- Path is already in `_PUBLIC_PATHS` at `main.py:286` — no auth required (consistent with other status tiles).

### `GET /api/paper-trading/status` (`backend/api/paper_trading.py`)
- Returns scheduler active state + last cycle result. Can be used to display `paper_trading_daily` job status.

### `GET /api/observability/freshness` (`backend/api/observability_api.py:25`)
- Returns per-source last_tick_age for signal freshness. Not a job-status endpoint per se, but could be used as a secondary health indicator on the Jobs tab.

### No existing log-tail endpoint
- There is no `GET /api/logs/tail` or any log streaming endpoint. This must be built new.

---

## 4. Auth + Safety Patterns

### Auth
- `frontend/src/lib/api.ts` sends a Bearer token extracted from the `authjs.session-token` cookie (lines 54-58).
- All backend routes except `_PUBLIC_PATHS` go through `auth_and_security_middleware` (`main.py:301`) which calls `get_current_user`.
- A new `/api/logs/tail` endpoint should be authenticated (not added to `_PUBLIC_PATHS`).

### Log tail safety — path traversal prevention
- The endpoint MUST NOT accept a file path from the client. Instead, expose a curated allowlist of named log keys mapped server-side to absolute paths. (Source: PortSwigger Web Security Academy — "compare user input with a whitelist of permitted values.")
- Example allowlist structure:
  ```python
  _LOG_ALLOWLIST = {
      "backend": Path("/Users/ford/.openclaw/workspace/pyfinagent/backend.log"),
      "watchdog": Path(".../handoff/logs/backend-watchdog.log"),
      "harness": Path(".../handoff/logs/mas-harness.log"),
      "autoresearch": Path(".../handoff/logs/autoresearch.log"),
      "restart": Path(".../handoff/logs/backend-restart.log"),
  }
  ```
- The API should accept `?log=backend&lines=200` — `log` is matched against the allowlist key only; any unrecognized key returns 400.
- Rate limiting: given local-only deployment, a simple `100 requests/minute` per IP via a lightweight counter is sufficient. The existing `api_cache.py` TTL cache can double as a rate-limiter for the tail endpoint.

### Polling pattern
- The existing pattern throughout the frontend (e.g., `agents/page.tsx`, `paper-trading` page) is `setInterval` polling every 5-10 seconds via `apiGet`. No SSE is used anywhere in the codebase currently.
- For log tails: polling `GET /api/logs/tail?log=backend&lines=100` every 5 seconds is consistent with the existing pattern and sufficient for a local operator dashboard. SSE would be cleaner but introduces complexity not seen elsewhere in the codebase.
- Polling failure limit: `frontend/src/app/(conventions in rules/frontend.md)` requires stopping after 5 consecutive failures with an error message.

---

## 5. Frontend Page Anatomy

### Existing "System" section in Sidebar (`frontend/src/components/Sidebar.tsx:53-59`)
```
System
  - MAS Dashboard  (/agents)   icon: Robot
  - Agent Map      (/agent-map) icon: Graph
```
New entry to add: `Cron / Logs` at `/cron` with `Clock` Phosphor icon.

`Clock` is already exported from `frontend/src/lib/icons.ts` (line 75 as `BiasRecency = Clock`). To use it for the nav entry, add a direct `Clock` re-export to `icons.ts`.

### Analogous precedent: `frontend/src/app/agents/page.tsx`
- Uses `Sidebar` + `<main>` shell with two-zone layout (fixed header + scrollable content).
- Has a tab bar with 4 tabs: Live Stream, Run History, Agent Map, OpenClaw.
- Each tab is a distinct operational view. The Cron/Logs page should follow the same pattern.
- `agents/page.tsx` already demonstrates polling for live events, history tables, and external-process data (OpenClaw).

### Proposed page route
- `frontend/src/app/cron/page.tsx`

### Proposed tab structure
- **Jobs** tab: table of all known jobs (both main-process and Slack-bot), their schedule, last run, status, next run.
- **Logs** tab: log file selector (dropdown of 5 operator-actionable logs) + tail display (last N lines, auto-refresh every 5s).

This "two-tab" layout is consistent with `agents/page.tsx` (4 tabs) and avoids the complexity of a side-by-side layout which would require breakpoint-aware CSS and doesn't benefit the operator (they either need job status or log detail at a given moment, not both simultaneously).

### No-emoji enforcement
- No emoji anywhere in the page. Use Phosphor icons: `Clock` (nav), `CheckCircle` / `XCircle` / `MinusCircle` (status dots), `File` (log selector), `ArrowClockwise` (refresh).
- Pre-flight grep: `grep -n "emoji\|🟢\|🔴\|⚠\|✅\|❌" frontend/src/app/cron/page.tsx` before commit.

---

## 6. Backend API Endpoint Shapes Proposed

### `GET /api/jobs/all` (new — to be built in GENERATE)
Returns all known jobs across both processes in a unified schema:

```json
{
  "jobs": [
    {
      "id": "paper_trading_daily",
      "name": "Paper Trading Daily Cycle",
      "process": "backend",
      "schedule": "cron mon-fri hour=<X>:00 ET",
      "next_run_time": "2026-05-06T14:00:00+00:00",
      "status": "ok | never_run | failed | in_progress",
      "last_run_at": "2026-05-05T14:00:00+00:00",
      "last_duration_s": 42.1,
      "last_error": null
    }
  ]
}
```

This merges:
- Live APScheduler introspection (main-process jobs: `paper_trading_daily`, `process_batch`) via `scheduler.get_jobs()` — fields: `job.id`, `job.name`, `str(job.trigger)`, `job.next_run_time`.
- Static heartbeat registry from `job_status_api._registry` (Slack-bot jobs + phase-9 jobs).
- Static launchd metadata (watchdog plist: every 60s; last run inferred from log timestamp).

### `GET /api/logs/tail` (new — to be built in GENERATE)
```
GET /api/logs/tail?log=backend&lines=100
```
Response:
```json
{
  "log": "backend",
  "path_label": "backend.log",
  "lines": ["2026-05-07 19:52:01 I [main] ...", "..."],
  "total_bytes": 163577856,
  "truncated": true
}
```
Security: `log` param validated against `_LOG_ALLOWLIST`; path never echoed in error messages; file read with `tail -n <lines>` equivalent via `deque(open(...), maxlen=lines)`.

---

## 7. Internal Files Inspected

| File | Lines Read | Role | Status |
|------|-----------|------|--------|
| `backend/main.py` | 1-400 | FastAPI lifespan, scheduler startup, auth middleware, router registration | Active |
| `backend/slack_bot/scheduler.py` | 1-383 | Slack-bot APScheduler: 4 core jobs + 7 phase-9 jobs via `register_phase9_jobs` | Active |
| `backend/api/paper_trading.py` | 1-80, 895-934 | Paper trading scheduler registration, `init_scheduler`, `_add_scheduler_job` | Active |
| `backend/api/job_status_api.py` | 1-153 | Phase-9 job heartbeat registry, `GET /api/jobs/status` | Active |
| `backend/api/observability_api.py` | 1-79 | Observability: freshness + latency endpoints | Active |
| `scripts/launchd/com.pyfinagent.backend-watchdog.plist` | 1-25 | Watchdog launchd plist: 60s interval, log path | Active |
| `frontend/src/components/Sidebar.tsx` | 1-60 | Nav sections — System has 2 entries, new Cron entry needed | Active |
| `frontend/src/app/agents/page.tsx` | 1-80 | Analogous system-internals page with 4-tab pattern | Active |
| `frontend/src/lib/api.ts` | 1-60 | Auth token extraction pattern (cookie-based Bearer) | Active |
| `frontend/src/lib/icons.ts` | 1-120 | Icon re-exports; `Clock` available as `BiasRecency`, needs direct re-export | Active |
| `handoff/logs/` directory | ls -lh | All 12 files: sizes, dates, writers | Active |
