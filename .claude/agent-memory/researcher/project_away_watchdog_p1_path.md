---
name: away-watchdog-p1-path
description: phase-62.5 facts -- raise_cron_alert_sync from a ONE-SHOT process silently no-ops on P1 (deduper threshold 3; P1 not in _CRITICAL_SEVERITIES) -> need ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1 env; launchctl exit 113 = not-loaded (kickstart fails); watchdog ownership split
metadata:
  type: project
---

phase-62.5 (away-ops healthcheck) research facts, reusable for ANY standalone
script that needs to alert or restart services:

**P1 deduper trap (the big one).** `raise_cron_alert_sync(source, error_type,
severity, title, details)` (backend/services/observability/alerting.py:185)
SILENTLY NO-OPS when called once with severity="P1" from a fresh process:
P1 is NOT in `_CRITICAL_SEVERITIES = {"P0","critical","CRITICAL"}`
(alerting.py:42), and the non-critical path requires 3 occurrences of the same
(source,error_type) within 5 min IN THE SAME PROCESS (alerting.py:78-90) —
the deduper is in-memory. Fixes: run with env
`ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1` (real pydantic Field, settings.py:147,
read by `_get_default_deduper()` alerting.py:108-112), or call 3x in one
process (fires once, on the 3rd). The function RETURNS the delivery bool in
no-loop contexts (asyncio.run path :215-216) — always check it. Curl-webhook
fallback pattern lives at scripts/launchd/backend_watchdog.sh:61-72 (reads
SLACK_WEBHOOK_URL out of backend/.env via grep, no sourcing).
`_ENV_FILE` is absolute (settings.py:12) so settings load from any cwd; only
`cd $REPO` is needed for the `backend.` import.

**Why:** the deduper was designed for long-lived processes (backend/slack-bot);
launchd one-shots get a fresh deduper every fire — both initial-swallow AND
no cross-process repeat-suppression. Suppression must be replayed from the
script's own append-only state (health.jsonl p1_raised field).

**launchctl empirics (Darwin 25.5, verified live 2026-06-12):**
`launchctl print gui/$UID/<label>` exit 0 = loaded, exit 113 = not found
(booted out); `kickstart -k` on a booted-out label ALSO exits 113 ("Could not
find service") — recovery then needs `launchctl bootstrap gui/$UID <plist>`.
Addigy's claim that macOS 14.4 deprecated `launchctl kickstart` is REFUTED
on this machine (`launchctl help` lists it; backend-watchdog uses it daily).
StartInterval fires are SKIPPED during sleep (not queued); only
StartCalendarInterval queues one missed fire on wake (Apple ScheduledJobs
archive doc). Mac stays awake via backend's `caffeinate -i -s` (AC only).

**Watchdog ownership split (62.5 design):** backend restarts belong to the
60s `com.pyfinagent.backend-watchdog` (scripts/launchd/backend_watchdog.sh,
3-fail counter, SIGUSR1 dump, kickstart) ONLY; the 30-min away-watchdog
observes everything but restarts ONLY the frontend; slack-bot heals via its
own KeepAlive. Never give two agents kickstart authority over one service.

**Misc verified:** /api/health public (main.py:394, route :512);
`/api/paper-trading/kill-switch` (paper_trading.py:480) returns 200 from
localhost WITHOUT auth (DEV_LOCALHOST_BYPASS in backend plist; impl
backend/api/auth.py); backend-down fallback = replay
handoff/kill_switch_audit.jsonl last pause/resume (kill_switch.py:61-77).
cycle_history.jsonl: skip status="started" rows, use `completed_at`
(cycle_health.py:187); cycle is cron mon-fri 18:00 UTC -> naive <26h check
goes false-stale EVERY WEEKEND. APFS: `df` Available EXCLUDES purgeable =
conservative floor metric; diskutil/Finder over-report. gcloud + gh live in
/opt/homebrew/bin -> launchd plists need PATH + HOME set.
