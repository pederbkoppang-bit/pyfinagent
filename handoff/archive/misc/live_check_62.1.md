# live_check -- phase-62.1: Slack bot under launchd + restart on current code

Date: **2026-06-13** (AM away session, re-execution). Supersedes the 2026-06-12 record
below the divider. Status: **criteria 1 + 2 COMPLETE; criterion 3 PENDING** the 14:00 CEST
scheduled morning digest from the restarted process (PM-owned per commit 875e25d4).

## A. Criterion 1 -- launchd agent + old process dead (verbatim)

    $ pgrep -fl backend.slack_bot.app
    83982 .../Python -m backend.slack_bot.app          (pgrep count = 1)
    $ kill -0 26147   ->  "kill 26147 failed: no such process"   (old MANUAL bot dead)
    $ kill -0 38084   ->  "kill 38084 failed: no such process"   (prior launchd PID, replaced)
    $ PlistBuddy -c 'Print :KeepAlive' ~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist
    true

Plist `~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist`: KeepAlive=true,
RunAtLoad=true, ThrottleInterval=5, ProcessType=Interactive, LegacyTimers=true,
EnvironmentVariables PATH = the backend plist's venv-first PATH verbatim + PYTHONUNBUFFERED=1
(mirrors com.pyfinagent.backend.plist's environment shape), ProgramArguments=[.venv/bin/
python, -m, backend.slack_bot.app], WorkingDirectory=repo, logs to handoff/logs/slack_bot.log.

## B. Criterion 2 -- verification command output (verbatim)

    $ launchctl kickstart -k gui/$(id -u)/com.pyfinagent.slack-bot     # exit 0
    $ launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | grep -E 'state|pid'
        state = running
        pid = 83982
            state = active
            state = active
    $ ps -o lstart= -p 83982
    Sat Jun 13 07:46:26 2026
    $ git log -1 --format=%ci -- backend/slack_bot/
    2026-06-12 13:14:21 +0200

=> bot process start (**Sat Jun 13 07:46:26 2026**) POSTDATES the newest slack_bot commit
(**2026-06-12 13:14:21 +0200**, 1be98e83) by ~18.5h. Sharper corroboration (commit-time !=
file-edit-time): lstart also postdates both changed-file mtimes -- operator_tokens.py
`2026-06-12 13:13:28`, alerting.py `2026-06-12 13:12:16`. Current committed code loaded.
Exactly ONE process (pgrep count = 1). Pre-restart import smoke-test passed (crash-loop
guard). Healthy Socket Mode reconnect (fresh log):

    2026-06-13 07:46:26  Scheduler started: morning digest at 8:00, evening digest at 17:00, watchdog every 15 min
    2026-06-13 07:46:27  slack_bolt.AsyncApp: A new session (s_277925562) has been established
    2026-06-13 07:46:27  slack_bolt.AsyncApp: Bolt app is running!

## C. Criterion 3 -- digest from the NEW process: PENDING until MON 2026-06-15 (weekend-gated)

The post-restart process (PID 83982) is armed for the 8:00 ET morning digest + 17:00 ET
evening digest. **CORRECTION (Q/A catch, Cycle 64):** digests are weekend-gated --
`_send_morning_digest`/`_send_evening_digest` skip on non-trading days via
`_is_us_trading_day_now()` (scheduler.py:543, phase-51.3), logging a
`morning_digest skipped: ... is not a US trading day` line instead of sending. Today
(Sat 2026-06-13) and Sun 2026-06-14 are non-trading days, so NO qualifying digest fires
this weekend. **Earliest qualifying digest from the launchd bot = Monday 2026-06-15,
08:00 ET = 14:00 CEST** (or 23:00 CEST evening). A one-shot send_away_digest.py would be a
SCRIPT process, not the launchd bot, so it cannot satisfy "from the NEW process" -- not
triggered off-schedule.

**Monday closing-session action (2026-06-15):** (1) capture the real `Morning digest
sent` (or `Evening digest sent`) log line from handoff/logs/slack_bot.log -- NOT the
`skipped: ... not a US trading day` line -- emitted by the launchd bot; verify the
emitting process's lstart still postdates commit 1be98e83; (2) paste the Slack message
permalink here; (3) run the closing Q/A and flip 62.1 to done.

## Honesty note

The pre-restart bot already held the current code (file mtimes predate its lstart); the
8s commit-timestamp gap was an artifact. The restart satisfies criterion 2's immutable
literal wording AND is clean pre-departure hygiene (proven: the bot restarts + reconnects
in <1s). It was not correcting real code-staleness.

---

## (SUPERSEDED) 2026-06-12 record

The original cutover record (PID 2585 lstart 2026-06-12 10:22:44, crontab race removed,
manual PID 26147 killed) is archived in git history at the Jun-12 commits. Criteria 1-2
were re-established above on the current process after subsequent commits (through
1be98e83) and an intervening KeepAlive restart cycle moved the running PID to 38084 then
83982. The crontab-monitor removal + dual-supervisor closure from Jun-12 remain in effect.
