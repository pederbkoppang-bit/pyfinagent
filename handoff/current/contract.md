# Contract -- phase-62.1: Slack bot under launchd + restart on current code

Date: 2026-06-12. Goal: goal-away-ops. Window: NOW (research GO, clean 08:10-08:55 UTC).

## Research-gate summary

Brief: handoff/current/research_brief.md (gate_passed: true, 5 sources in full -- launchd.info,
ss64 launchctl, Slack Socket Mode docs, APScheduler base docs, ss64 caffeinate; recency scan).
PREMISE CORRECTIONS: (1) stale bot runs code-as-of 2026-06-11 05:03:44 (not 06-05) -- still
predates phase-60.1/60.4 commits, restart justified. (2) Socket Mode is LOAD-BALANCED, not
broadcast -- the dual-instance threat is the duplicated AsyncIOScheduler + queue-processor/
SLA-monitor/reaper (app.py:60-68), not event double-delivery. (3) CRITICAL: a user crontab
monitor (scripts/slack_bot_monitor.sh, */5, non-atomic pgrep-then-nohup) would race launchd
into a second instance -- REMOVE THAT CRONTAB LINE FIRST (keep the */2 slack_mention_checker
line). Plist: backend template MINUS caffeinate (backend's -s assertion is system-wide);
ProgramArguments=[.venv/bin/python, -m, backend.slack_bot.app]; WorkingDirectory=repo
(load-bearing for -m); backend's PATH verbatim (imsg + bare python subprocess deps);
KeepAlive=true; RunAtLoad=true; ThrottleInterval=5; ProcessType=Interactive + LegacyTimers
=true (current bot runs niced SN -- worse for APScheduler timers); logs to handoff/logs/
slack_bot.log (gitignored; fresh file = unambiguous NEW-process attribution). No secrets in
plist (settings.py:12 loads backend/.env by absolute path, CWD-independent). SIGTERM = hard
kill, risk LOW (in-memory jobstore, idempotency-keyed jobs, price-refresh catch-up will
fire once benignly post-bootstrap); NO signal handler in this step (scope + APScheduler
#567). Digest hours are settings DEFAULTS 8/17 ET -- confirm live via settings load (the
.env file itself is deny-ruled).

## Immutable success criteria (verbatim from masterplan 62.1)

1. "com.pyfinagent.slack-bot launchd agent exists with KeepAlive=true, mirroring
   com.pyfinagent.backend.plist's environment shape; the old manual PID is dead (kill -0
   fails)"
2. "ps lstart of the launchd-managed bot process is LATER than the newest git commit
   touching backend/slack_bot/ (verbatim paste of both)"
3. "a morning or evening digest observed in Slack from the NEW process (permalink or
   screenshot path in live_check_62.1.md)"

verification.command (verbatim): launchctl print gui/$(id -u)/com.pyfinagent.slack-bot |
grep -E 'state|pid' && ps -o lstart= -p $(launchctl print gui/$(id -u)/com.pyfinagent.
slack-bot | awk '/pid =/{print $3}') && git log -1 --format=%ci -- backend/slack_bot/

Criterion-3 timing: first digest from the new process = morning digest 08:00 ET (14:00
Oslo) today. Q/A is spawned ONCE after that evidence lands (complete-evidence single
spawn, not an early CONDITIONAL round).

## Plan (cutover sequence, research-prescribed)

1. Confirm digest hours via settings load (no digest in the next ~30 min).
2. Remove the slack_bot_monitor.sh crontab line (keep slack_mention_checker); verify.
3. Write ~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist per the research spec.
4. Import smoke test: .venv/bin/python -c "from backend.slack_bot.app import create_app"
   (KeepAlive + boot-crash = 5s restart loop otherwise).
5. Kill the manual bot PID; confirm pgrep empty.
6. launchctl bootstrap gui/$(id -u) <plist>; verify exactly ONE process + startup lines in
   handoff/logs/slack_bot.log.
7. At/after 14:00 Oslo: confirm the morning digest arrived (operator channel) -- permalink
   or the bot log send-line into live_check_62.1.md; expect one benign price-refresh
   catch-up fire post-bootstrap (documented, not a defect).
8. experiment_results.md -> fresh Q/A -> harness_log -> flip.

## Out of scope

Signal handlers in app.py; token handler (62.2); digest content changes (62.8); any
phase-9 job logic.

## References

research_brief.md (62.1); launchd.info; ss64 launchctl + caffeinate; docs.slack.dev Socket
Mode; APScheduler base scheduler docs.
