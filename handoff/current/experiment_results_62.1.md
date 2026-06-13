# Experiment Results -- phase-62.1 (GENERATE, RE-EXECUTION)

**Step:** 62.1 -- Slack bot under launchd + restart on current code. **Date:** 2026-06-13
(AM away session). **State:** criteria 1 + 2 COMPLETE (in-window); criterion 3 PENDING the
14:00 CEST scheduled morning digest from the restarted process (PM-owned per commit
875e25d4). **Expected Q/A verdict:** CONDITIONAL on criterion 3. **No status flip this AM.**

## What changed (and what did NOT)

NO files edited. NO plist change. The launchd cutover was already done (Jun-12). The ONLY
action: a code-only **restart in place** of the already-launchd-managed bot via
`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot` (rail-9 authorized restart
path; research-confirmed canonical verb for loading freshly-committed code when the plist
is unchanged).

## Scope-honesty note (the 8-second "staleness")

The pre-restart bot (PID 38084, lstart `Fri Jun 12 13:14:13`) was 8s OLDER than commit
1be98e83 (`13:14:21`), so criterion 2 (lstart > commit-time) failed *as literally
worded*. BUT the changed files' mtimes are BEFORE that lstart:
- `backend/slack_bot/operator_tokens.py`  mtime `2026-06-12 13:13:28 +0200`
- `backend/services/observability/alerting.py` mtime `2026-06-12 13:12:16 +0200`
So the pre-restart bot had ALREADY loaded the current code (files saved 13:12-13:13, bot
started 13:14:13); the 8s gap was a pure commit-timestamp artifact (`git commit` ran 8s
after the restart). The restart this AM was done to satisfy criterion 2's immutable
literal wording AND as clean pre-departure hygiene (confirm the bot restarts + reconnects
cleanly before the 3-week unattended window). It was NOT correcting a real code-staleness.

## Pre-restart crash-loop guard (research-prescribed)

    $ .venv/bin/python -c "from backend.slack_bot.app import create_app; print('IMPORT_OK', callable(create_app))"
    IMPORT_OK create_app callable= True
    smoke_exit=0

Import clean -> KeepAlive will not crash-loop on a bad import.

## Restart + IMMUTABLE verification.command (verbatim)

    $ launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot   # exit 0
    $ launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | grep -E 'state|pid' \
        && ps -o lstart= -p $(... awk '/pid =/{print $3}') \
        && git log -1 --format=%ci -- backend/slack_bot/
        state = running
        pid = 83982
            state = active
            state = active
    Sat Jun 13 07:46:26 2026
    2026-06-12 13:14:21 +0200

=> NEW lstart (Sat Jun 13 07:46:26 2026) POSTDATES the newest slack_bot commit
(2026-06-12 13:14:21 +0200) by ~18.5h, AND postdates both changed-file mtimes. Current
committed slack_bot code is unambiguously loaded.

## Criterion 1 -- launchd agent + old PIDs dead (verbatim)

    $ pgrep -fl backend.slack_bot.app          -> 83982 ... -m backend.slack_bot.app   (count=1)
    $ kill -0 26147   -> "no such process"     (old MANUAL bot dead)
    $ kill -0 38084   -> "no such process"     (prior launchd PID, replaced by 83982)
    $ PlistBuddy -c 'Print :KeepAlive' <plist> -> true
    plist EnvironmentVariables mirror the backend plist shape (PATH = backend's venv-first
    PATH verbatim; PYTHONUNBUFFERED=1). KeepAlive=true, RunAtLoad=true, ThrottleInterval=5.

## Healthy Socket Mode reconnect (fresh log, post-restart)

    2026-06-13 07:46:26  apscheduler.scheduler: Scheduler started
    2026-06-13 07:46:26  backend.slack_bot.scheduler: Scheduler started: morning digest at
                         8:00, evening digest at 17:00, watchdog every 15 min
    2026-06-13 07:46:26  Ticket queue processor / SLA monitoring / Stuck-Task Reaper started
    2026-06-13 07:46:27  slack_bolt.AsyncApp: A new session (s_277925562) has been established
    2026-06-13 07:46:27  slack_bolt.AsyncApp: Bolt app is running!

Reconnect < 1s; exactly one Socket Mode session; no ERROR/Traceback after the restart.

## Criterion 3 -- digest from the NEW process: PENDING until MON 2026-06-15 (weekend-gated)

The post-restart scheduler (PID 83982) is armed for the 8:00 ET morning + 17:00 ET evening
digests. **CORRECTION (Q/A Cycle-64 catch):** digests are weekend-gated --
`_send_morning_digest`/`_send_evening_digest` skip on non-trading days via
`_is_us_trading_day_now()` (scheduler.py:543, phase-51.3). Today (Sat 2026-06-13) + Sun
2026-06-14 are non-trading, so NO qualifying digest fires this weekend (the bot logs a
`skipped: ... not a US trading day` line). **Earliest qualifying digest from the launchd
bot = Monday 2026-06-15, 14:00 CEST** (or 23:00 CEST). A one-shot send_away_digest.py is a
SCRIPT process, not the launchd bot, so it cannot satisfy "from the NEW process" -- not
triggered off-schedule. The Monday closing session (AM or PM) captures the real
`digest sent` log line + Slack permalink into live_check_62.1.md and runs the closing Q/A.
Architected design: commit 875e25d4 ("PM-session ... owns 62.1/62.2 evidence closure").

## Files

No repo code/config edited. Restart-only. Artifacts: contract_62.1.md (re-execution),
research_brief_62.1.md (revalidated), this file, live_check_62.1.md (updated),
harness_log.md Cycle 64 append.
