---
name: slack-bot-supervision-topology
description: Slack bot is now LAUNCHD-supervised (com.pyfinagent.slack-bot, KeepAlive, since ~2026-06-12); crontab monitor line REMOVED; old cron-only topology is historical; safe one-shot digest-send path still valid
metadata:
  type: project
---

**UPDATED 2026-06-12 (phase-62.5 audit): the cron→launchd migration HAPPENED.**
The slack_bot (`python -m backend.slack_bot.app`) is now supervised by
**launchd**: `~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist` (mtime
2026-06-12 10:21) with `KeepAlive=true`, `RunAtLoad=true`, `ThrottleInterval=5`,
venv python, logs to `handoff/logs/slack_bot.log`. Verified: `launchctl list`
shows `com.pyfinagent.slack-bot` with a live PID; `crontab -l` no longer
contains `slack_bot_monitor.sh` (only `slack_mention_checker.sh */2` remains,
which does NOT start the bot). `scripts/slack_bot_monitor.sh` itself was
rewritten (lines 4-9, 25-39): if the launchd plist is present it uses
`launchctl kickstart` (idempotent, launchd serializes instance management);
the legacy `nohup` spawn survives ONLY as a fallback when the plist is absent.

**Historical (pre-2026-06-12) topology — superseded:** user-crontab monitor
`*/5` greping ps + `nohup python -m backend.slack_bot.app &` (PPID 1, looked
orphaned but was cron-managed). The phase-54.2 double-instance warning (cron
+ launchd simultaneously = each process runs its own AsyncIOScheduler at
scheduler.py:196 → digests + 11 crons DOUBLE-FIRE) remains TRUE as a hazard
class — it is why the monitor script now defers to launchd instead of
nohup-spawning. If you ever see double digests, check for a stray non-launchd
bot process (`ps aux | grep slack_bot.app` should show exactly one, parented
to launchd).

Mac still does not system-sleep: backend plist wraps uvicorn in
`caffeinate -i -s` (AC power only).

**Send ONE digest safely (unchanged, still valid):** there is NO in-process
trigger endpoint (`/api/jobs/{id}/trigger` is `paper_trading_daily`-only,
cron_dashboard_api.py:517). Use a standalone one-shot: bare
`AsyncWebClient(bot_token)` + existing `format_morning_digest()` +
`chat_postMessage(channel=slack_channel_id)` — a Web-API POST that opens ZERO
Socket Mode connections and never touches the running bot. Digest is
$0/template-only (formatters.py imports only math+datetime; no LLM).

See [[project_cron_scheduler_control_topology]] for the APScheduler control
surface, [[project_slack_digest_calendar_guard]] for the trading-day guard,
and [[project_away_watchdog_p1_path]] for how standalone scripts must raise
P1s (deduper trap).
