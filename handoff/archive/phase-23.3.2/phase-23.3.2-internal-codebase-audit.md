# Phase-23.3.2 Internal Codebase Audit
# Slack-Bot APScheduler Jobs — Registration, Liveness, and Cross-Process Gap

Accessed: 2026-05-07

---

## 1. The 4 Core Jobs — Registration Sites

All four jobs are registered in `backend/slack_bot/scheduler.py` inside `start_scheduler(app: AsyncApp)`.

| Job ID | Lines | Trigger type | Expression | Function called |
|---|---|---|---|---|
| `morning_digest` | 35-44 | cron | `hour=settings.morning_digest_hour, minute=0, tz=America/New_York` | `_send_morning_digest(app)` |
| `evening_digest` | 47-56 | cron | `hour=settings.evening_digest_hour, minute=0, tz=America/New_York` | `_send_evening_digest(app)` |
| `watchdog_health_check` | 59-66 | interval | `minutes=settings.watchdog_interval_minutes` | `_watchdog_health_check(app)` |
| `prompt_leak_redteam` | 71-79 | cron | `hour=3, minute=15, tz=America/New_York` | `_nightly_prompt_leak_redteam(app)` |

All four use `replace_existing=True` so reloads do not raise `ConflictingIdError`.
(`backend/slack_bot/scheduler.py:35-79`)

The scheduler is an `AsyncIOScheduler` (APScheduler async) and is only started if `settings.slack_channel_id` is truthy (line 28-30). If `SLACK_CHANNEL_ID` is unset, `start_scheduler` returns early and NONE of the four jobs are registered. The caller gets no error — only a `logger.warning`.

---

## 2. `start_scheduler` is Called on Slack-Bot Startup

`backend/slack_bot/app.py:56` calls `start_scheduler(app)` unconditionally inside `main()`. The Bolt `AsyncSocketModeHandler` is started AFTER the scheduler, meaning if the scheduler fails or is skipped (no `SLACK_CHANNEL_ID`), the socket connection still opens. The four jobs live ONLY in the slack-bot OS process (`python -m backend.slack_bot.app`, PID 16385 as reported).

The main FastAPI backend (`backend/main.py`) does NOT call `start_scheduler`. It registers its own schedulers (`main` and `queue`) via `_register_cron_scheduler` at lines 175 and 228. The slack-bot scheduler is entirely separate.

---

## 3. Logging Configuration for the Slack Bot

`backend/slack_bot/app.py:46` calls `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")` inside `main()`. There is no dedicated log file configured anywhere in the slack-bot modules.

When run as `python -m backend.slack_bot.app` (confirmed by `pgrep`), logs go to stdout/stderr of that process. Since there is no launchd plist for the slack-bot (no `com.pyfinagent.slackbot.plist` found in `~/Library/Logs/`), its output is not captured to a log file by the OS. It is a manual foreground process.

The log allowlist in `backend/api/cron_dashboard_api.py:102-110` does NOT include a `slackbot` key. There is no way to tail slack-bot logs from the `/cron` dashboard's Logs tab.

If the process was started in a terminal session with output redirection, the log is wherever that terminal wrote it. There is no authoritative on-disk log file for the slack-bot process in the current configuration.

---

## 4. What `/api/jobs/status` Returns — The 7 Phase-9 Jobs Only

`backend/api/job_status_api.py` defines a static `_JOB_NAMES` tuple at lines 52-60 containing EXACTLY the 7 phase-9 job IDs:
```
daily_price_refresh, weekly_fred_refresh, nightly_mda_retrain,
hourly_signal_warmup, nightly_outcome_rebuild, weekly_data_integrity, cost_budget_watcher
```

`morning_digest`, `evening_digest`, `watchdog_health_check`, and `prompt_leak_redteam` are NOT in this registry. The `GET /api/jobs/status` endpoint returns exactly 7 rows, always, pre-seeded with `status="never_run"` until a `POST /api/jobs/heartbeat` delivers an event.

---

## 5. Live Curl Confirmation — `/api/jobs/status`

```json
{
  "jobs": [
    {"name": "daily_price_refresh",     "status": "never_run", "last_run_at": null, ...},
    {"name": "weekly_fred_refresh",     "status": "never_run", "last_run_at": null, ...},
    {"name": "nightly_mda_retrain",     "status": "never_run", "last_run_at": null, ...},
    {"name": "hourly_signal_warmup",    "status": "never_run", "last_run_at": null, ...},
    {"name": "nightly_outcome_rebuild", "status": "never_run", "last_run_at": null, ...},
    {"name": "weekly_data_integrity",   "status": "never_run", "last_run_at": null, ...},
    {"name": "cost_budget_watcher",     "status": "never_run", "last_run_at": null, ...}
  ]
}
```

Confirmed: `morning_digest`, `evening_digest`, `watchdog_health_check`, and `prompt_leak_redteam` are absent. All 7 phase-9 jobs show `never_run` because the `POST /api/jobs/heartbeat` wiring from the slack-bot is not yet active (noted explicitly in `job_status_api.py:13-14`).

---

## 6. No HTTP Server Inside `backend/slack_bot/` That Exposes Scheduler State

Grep across all `backend/slack_bot/` `.py` files finds no FastAPI/Flask/aiohttp app definition, no `/healthz` route, no Bolt `http_handler`, and no `get_jobs()` call exposed to an HTTP caller. The slack-bot runs Socket Mode (outbound WebSocket only). There is no inbound HTTP server in the slack-bot process.

The only health-like behavior is:
- `backend/slack_bot/direct_responder.py:73-102` — the bot responds to Slack messages containing "status" / "health" by calling `http://localhost:8000/api/health` (the MAIN backend's health endpoint, not the slack-bot's own state).
- `backend/slack_bot/app_home.py:56-57` — the App Home tab checks `http://localhost:8000/api/health` via a synchronous `httpx.get` call.

Neither of these exposes scheduler state or proves the APScheduler is running.

---

## 7. `/api/health` and `/api/observability/freshness` — Do Not Reflect Slack-Bot Liveness

`/api/health` is the main FastAPI backend's own health check. It reflects backend process health only, not slack-bot process liveness. No cross-process probe is made.

`/api/observability/freshness` (`backend/api/observability_api.py`, alias to `compute_freshness`) reflects signal data freshness from BigQuery. It has no knowledge of the slack-bot process.

---

## 8. The Heartbeat Wiring Gap

`backend/api/job_status_api.py:13-14` documents the gap explicitly:

> "A hidden `POST /heartbeat` endpoint lets the Slack-bot (or any harness integration) deliver event dicts to this registry. Until that wiring lands, `GET /status` returns `status="never_run"` for every job"

The `POST /api/jobs/heartbeat` endpoint exists at `job_status_api.py:135-143` and is fully functional. But no code in `backend/slack_bot/` ever calls it. The `heartbeat()` context manager in `job_runtime.py:67-114` uses `sink=None`, which defaults to `logger.info` (line 83). The phase-9 jobs (daily_price_refresh, etc.) use `heartbeat()` from `job_runtime.py` but wire no HTTP sink. So all 11 jobs (4 core + 7 phase-9) are effectively dark to the main backend's registry.

---

## 9. `/api/jobs/all` — The Cron Tab Endpoint

`backend/api/cron_dashboard_api.py:162-186` implements `GET /api/jobs/all`, which:
1. Calls `.get_jobs()` on all schedulers registered in `_RUNNING_SCHEDULERS` — currently only the main APScheduler and the queue scheduler (both inside the FastAPI process).
2. Appends static manifest entries for all 11 slack-bot jobs (`_SLACK_BOT_JOBS`, lines 62-85) and 1 launchd job.

The four core jobs appear in `_SLACK_BOT_JOBS` at lines 63-69 with `status="manifest"` (set by `_static_to_dict`, line 155) and `next_run=None`. This is exactly what the /cron Jobs tab shows. The status badge "manifest" is rendered in the frontend (`frontend/src/app/cron/page.tsx:64`) as a grey pill.

The comment at `cron_dashboard_api.py:59-60` explicitly states: "The MAIN backend cannot call .get_jobs() on its scheduler. Mirror the canonical job list... last_run + status remain 'unknown' -- a future phase can wire a heartbeat POST."

---

## Summary Table

| Claim | Evidence |
|---|---|
| All 4 jobs registered in `start_scheduler` | `scheduler.py:35-79` |
| `start_scheduler` called on bot startup | `app.py:56` |
| No dedicated log file for slack-bot | `app.py:46` + no launchd log |
| `/api/jobs/status` covers 7 phase-9 jobs only, not 4 core | `job_status_api.py:52-60`, live curl |
| `/api/jobs/heartbeat` exists but not wired | `job_status_api.py:135-143`, `job_runtime.py:83` |
| No inbound HTTP server in slack-bot process | grep of `backend/slack_bot/` |
| `/api/health` and `/api/observability/freshness` blind to slack-bot | `main.py:188-193` |
| `/api/jobs/all` shows 4 core jobs as `status="manifest"` | `cron_dashboard_api.py:62-69, 155` |
