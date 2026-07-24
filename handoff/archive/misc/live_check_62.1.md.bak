# live_check -- phase-62.1: Slack bot under launchd + restart on current code

Date: 2026-06-12. Status: criteria 1-2 COMPLETE; criterion 3 (digest from NEW process)
due at the 08:00 ET / 14:00 Oslo morning digest today.

## A. Criterion 1 -- launchd agent + old process dead

Plist: ~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist -- KeepAlive=true,
RunAtLoad=true, ThrottleInterval=5, ProcessType=Interactive + LegacyTimers=true (research:
the manual bot ran niced SN, worse for APScheduler timers), WorkingDirectory=repo
(load-bearing for -m), backend plist's PATH verbatim (imsg + bare-python subprocess deps),
NO caffeinate (backend's system-wide -s assertion covers the machine), logs to
handoff/logs/slack_bot.log (gitignored; fresh file = unambiguous NEW-process attribution).

Old manual process: PID 26147 (started 2026-06-11 05:03:44 per research -- predating the
phase-60.1/60.4 commits) killed; verification verbatim:

    manual bot dead
    kill-0-exit=1 (1 = process gone)

DUAL-SUPERVISOR RACE CLOSED (research finding): the */5 crontab line invoking
scripts/slack_bot_monitor.sh (non-atomic pgrep-then-nohup) was REMOVED from the crontab
(verified: grep count 0), AND the script itself rewritten launchd-first (delegates to
launchctl kickstart when the agent exists; legacy nohup only if the plist is absent) --
two independent layers against duplicate bot instances. Note: sandboxed crontab installs
stall under macOS TCC; the installs completed via background retries (exit 0) and the
final state is verified above.

## B. Criterion 2 -- verification command output (verbatim)

    $ launchctl print gui/$(id -u)/com.pyfinagent.slack-bot | grep -E 'state|pid'
        state = running
        pid = 2585
    $ ps -o lstart= -p 2585
    fre. 12 jun. 10.22.44 2026
    $ git log -1 --format=%ci -- backend/slack_bot/
    2026-06-11 16:30:22 +0200

=> bot process start (2026-06-12 10:22:44 +0200) POSTDATES the newest slack_bot commit
(2026-06-11 16:30:22 +0200): phase-60.4 bot code is loaded. Exactly ONE process
(pgrep count = 1). Startup log verbatim (handoff/logs/slack_bot.log):

    2026-06-12 10:22:44,959 INFO ... Ticket queue processor started
    2026-06-12 10:22:44,960 INFO ... SLA monitoring started
    2026-06-12 10:22:44,961 INFO ... Stuck-Task Reaper started (15-minute timeout)
    2026-06-12 10:22:45,546 INFO slack_bolt.AsyncApp: A new session (s_277980774) has been established
    2026-06-12 10:22:45,546 INFO slack_bolt.AsyncApp: Bolt app is running!

Expected benign artifact (research-predicted): one price-refresh catch-up fire ~20s after
bootstrap (idempotent by day).

## C. Criterion 3 -- digest from the NEW process: PENDING until 14:00 Oslo

Evidence shape: the morning-digest send line in handoff/logs/slack_bot.log (new file --
only the new process writes here) + the Slack message timestamp/permalink.

## Hygiene finding (queued, NOT fixed here)

The remaining */2 crontab line exports a PLAINTEXT Slack bot token inline. Same class as
the FRED key (operator deferred). Added to the return-day ask list; rotating/moving it
into the script's env loading is out of 62.1 scope.
