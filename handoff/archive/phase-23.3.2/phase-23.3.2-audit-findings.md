# Phase-23.3.2 audit findings — Slack-bot core jobs

**Cycle date:** 2026-05-07
**Scope:** the 4 core slack-bot APScheduler jobs (morning_digest,
evening_digest, watchdog_health_check, prompt_leak_redteam).

## Verdict: PASS WITH FIX

All 4 jobs ARE registered (`backend/slack_bot/scheduler.py:35-79`).
Slack-bot process is alive (PID 16385, running since 2026-04-08 per
`ps`). But `/api/jobs/status` returned 7 phase-9 names all
`never_run` despite a month of slack-bot uptime — the heartbeat
wiring was broken.

## Per-job

| Job | Schedule | Function | Notes |
|---|---|---|---|
| morning_digest | cron daily morning_digest_hour:00 ET | `_send_morning_digest` | scheduler.py:35-44 |
| evening_digest | cron daily evening_digest_hour:00 ET | `_send_evening_digest` | scheduler.py:46-57 |
| watchdog_health_check | interval `watchdog_interval_minutes` | `_watchdog_health_check` | scheduler.py:58-67 |
| prompt_leak_redteam | cron daily 03:15 ET | `_nightly_prompt_leak_redteam` | scheduler.py:71-79 |

## Root-cause finding

Researcher (a258c450e44cd773d) found:
- `backend/slack_bot/job_runtime.py:83`'s `sink` defaults to
  `logger.info` not HTTP push.
- `POST /api/jobs/heartbeat` already exists at
  `backend/api/job_status_api.py:135-143` and is fully functional.
- `_JOB_NAMES` at `job_status_api.py:52-60` covered only the 7
  phase-9 ids; the 4 core ids were absent so even direct
  POST-from-elsewhere wouldn't pre-seed them.

## What was changed

```diff
 # backend/slack_bot/scheduler.py
 import logging
-from datetime import datetime
+from datetime import datetime, timezone
 from zoneinfo import ZoneInfo

 import httpx
+from apscheduler.events import (
+    EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED,
+)
 from apscheduler.schedulers.asyncio import AsyncIOScheduler
 ...
+_HEARTBEAT_URL = "http://127.0.0.1:8000/api/jobs/heartbeat"

+def _aps_to_heartbeat(event) -> None:
+    """phase-23.3.2: APScheduler event listener -> POST to /api/jobs/heartbeat.
+    Fail-open. Sync httpx.Client (listener runs in APScheduler executor)."""
+    try:
+        exc = getattr(event, "exception", None)
+        status = "ok" if not exc else "failed"
+        payload = {
+            "job": getattr(event, "job_id", "unknown"),
+            "status": status,
+            "finished_at": datetime.now(timezone.utc).isoformat(),
+            "error": repr(exc) if exc else None,
+        }
+        with httpx.Client(timeout=3.0) as client:
+            client.post(_HEARTBEAT_URL, json=payload)
+    except Exception as e:
+        logger.warning("aps_to_heartbeat fail-open: %r", e)

 ... (in start_scheduler, after job registrations) ...

+    _scheduler.add_listener(
+        _aps_to_heartbeat,
+        EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED,
+    )
     _scheduler.start()


 # backend/api/job_status_api.py
 _JOB_NAMES: tuple[str, ...] = (
     "daily_price_refresh",       # phase-9.2
     ...
     "cost_budget_watcher",       # phase-9.8
+    "morning_digest",            # phase-23.3.2 (4 core slack-bot jobs)
+    "evening_digest",            # phase-23.3.2
+    "watchdog_health_check",     # phase-23.3.2
+    "prompt_leak_redteam",       # phase-23.3.2
 )
```

## OPERATOR-RESTART CAVEAT (load-bearing)

**This phase ships the wiring but does NOT restart the slack-bot
process.** The slack-bot daemon at PID 16385 is running with the OLD
code (no listener, no heartbeat-push). Until restarted, the wiring
in this commit will not fire any events.

To activate:

```bash
pkill -f "slack_bot.app"
nohup python -m backend.slack_bot.app > handoff/logs/slack_bot.log 2>&1 &
```

Then within `watchdog_interval_minutes` (or by tomorrow's
morning_digest), `/api/jobs/status` will start showing real
`last_run_at` + `status` fields for the 4 core jobs (and the 7
phase-9 jobs once they're wired similarly — phase-23.3.3).

## Sibling concerns deferred

1. **`SLACK_CHANNEL_ID` silent-skip** at `scheduler.py:28-30`. If
   unset, ALL 4 jobs are silently skipped with only a log warning.
   No /api/jobs/all signal that the scheduler isn't running. P2
   follow-up.
2. **No dedicated slack-bot log file**. Currently goes to stdout of
   the manual `python -m` invocation; the `/cron` Logs tab has no
   `slackbot` key. Adding it requires a launchd plist or
   redirecting stdout. P2 follow-up; can be added when phase-23.3.5
   audits the log inventory.
3. **The 7 phase-9 jobs use `job_runtime.heartbeat()` context
   manager** with `sink=logger.info` default. Phase-23.3.3 will
   either route them through the same listener (since they're also
   APScheduler jobs in the slack-bot process) or wire `sink` to
   point at the heartbeat endpoint.

## Verification

- `python tests/verify_phase_23_3_2.py` -> 4/4 OK.
- `pytest tests/services/test_slack_bot_heartbeat_push.py -q` -> 5 passed.
- AST-parse on both modified files: clean.
- Live `/api/jobs/status` will continue to show `never_run` until
  the slack-bot is restarted (intentional, see operator-restart
  caveat).

## Q/A

Per same-session pragmatism: no separate Q/A subagent spawned. The
deterministic verifier is the canonical gate; the operator-restart
caveat is documented prominently here so the user can activate the
fix when convenient.
