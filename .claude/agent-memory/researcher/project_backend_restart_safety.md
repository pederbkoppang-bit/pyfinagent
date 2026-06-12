---
name: backend-restart-safety
description: Backend restart can NEVER double-fire the daily paper-trading cron (MemoryJobStore + forward-only next_run_time); watchdog/kickstart topology; researcher sandbox cannot read backend/.env
metadata:
  type: project
---

Restarting the backend (launchctl kickstart -k gui/$UID/com.pyfinagent.backend) can NEVER re-fire the `paper_trading_daily` job for a fire time that already passed — code-level certainty, established phase-61.1 (2026-06-12).

**Why:** three independent reasons. (1) `main.py:264-272` builds `AsyncIOScheduler()` with NO jobstores arg = MemoryJobStore; no schedule state survives restart, and `_real_add_job` (installed apscheduler 3.11.2, `schedulers/base.py:1066-1068`) computes `next_run_time = trigger.get_next_fire_time(None, now)` — CronTrigger with `previous_fire_time=None` (`triggers/cron/__init__.py:205-222`) searches strictly forward from `now`, so a freshly added job has zero past run times and the misfire/coalesce machinery never engages. (2) `misfire_grace_time=3600` + `coalesce=True` (paper_trading.py:1299-1322) only matter within a live process (event-loop stall) or persistent stores. (3) Exactly three `run_daily_cycle` call sites exist (paper_trading.py:1031 dry-run, :1279 manual run-now, :1329 cron callback) — no run-on-startup path.

**How to apply:** when a deploy/restart decision needs the "will it double-trade?" answer, the answer is no — restart any time EXCEPT mid-cycle (kickstart -k SIGKILLs in-flight cycles, bypassing finally blocks per backend_watchdog.sh:56-58; check `/api/paper-trading/status` first). Only constraint for flag pickup: restart BEFORE the next 18:00 UTC (14:00 ET, PAPER_TRADING_HOUR=14) cycle. Post-restart proof: /status `next_run` shows the next future fire.

Topology facts: backend plist runs `caffeinate -i -s <venv>/uvicorn` single-process (no --workers/--reload), KeepAlive=true, ThrottleInterval=5; the watchdog (`com.pyfinagent.backend-watchdog`, every 60s) restarts via the SAME kickstart -k label after 3 failed health checks — launchd enforces one instance per label, so no double-process topology exists. `get_settings()` is lru_cache per-process (settings.py:539-541); restart is the only deterministic flag pickup; no eager module-level Settings() snapshot exists in backend/. Slack bot is a separate process that never executes the trading path — flag flips don't require touching it.

Sandbox fact: the researcher agent is PERMISSION-DENIED on `backend/.env` (both Bash and Read) — .env audits must be delegated to Main with an exact grep command. Related: [[cron-scheduler-control-topology]], [[slack-bot-supervision-topology]].
