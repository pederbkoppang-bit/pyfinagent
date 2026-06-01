---
name: slack-bot-supervision-topology
description: How the slack_bot is actually launched/supervised (cron monitor, NOT launchd) + the double-instance risk if you add a launchd plist; the safe one-shot digest-send path
metadata:
  type: project
---

The slack_bot (`python -m backend.slack_bot.app`) is supervised by a **user-crontab
process monitor, NOT launchd** — phase-54.2 finding (2026-06-01). phase-54.1
incorrectly concluded the bot was "unsupervised" because it checked only
`launchctl list` and missed `crontab -l`.

**Actual topology:**
- `scripts/slack_bot_monitor.sh` runs `*/5 * * * *` (user crontab; added 2026-04-01,
  stable). Greps `ps aux | grep "python.*backend.slack_bot.app"`; if absent, `cd`s to
  repo + `source .venv/bin/activate` + `nohup python -m backend.slack_bot.app &`. The
  `nohup &` detaches → child reparents to PID 1 → that is WHY the bot shows PPID 1 /
  no launchd plist (looks orphaned but is cron-managed).
- On restart it fires an iMessage to the operator (+4794810537) via `imsg`
  (`/opt/homebrew/bin/imsg`, installed). Independent of Slack + Mac console.
- Separate `*/2 * * * *` `slack_mention_checker.sh` does NOT start the bot.
- Mac will NOT system-sleep: backend runs under `caffeinate -i -s` →
  `pmset -g` shows `sleep prevented by caffeinate`.

**Why: this matters / How to apply:** when asked to make the bot "resilient", do NOT
add a launchd KeepAlive `RunAtLoad` plist alongside the cron — it creates a SECOND
instance. Slack allows up to 10 Socket Mode connections per app token and routes each
inbound payload to ONE random connection (slash-commands NOT duplicated), BUT each
process runs its own `AsyncIOScheduler` (scheduler.py:196) → morning/evening digests
+ all 11 phase-9/core crons DOUBLE-FIRE. The two supervisors also mask each other's
death. Safe cron→launchd migration (if ever): comment the cron line →
`pkill -f backend.slack_bot.app` → confirm `ps|wc -l`=0 → `launchctl bootstrap
gui/$(id -u) <plist>` (KeepAlive{SuccessfulExit:false}+RunAtLoad+ThrottleInterval:5,
don't crash <10s per Apple); rollback `launchctl bootout gui/$(id -u) <plist>`.

**Send ONE digest safely (no 2nd instance):** there is NO in-process trigger endpoint
(`/api/jobs/{id}/trigger` is `paper_trading_daily`-only, cron_dashboard_api.py:517).
Use a standalone one-shot: bare `AsyncWebClient(bot_token)` + existing
`format_morning_digest()` + `chat_postMessage(channel=slack_channel_id)` — a Web-API
POST that opens ZERO Socket Mode connections and never touches the running bot.

Digest is **$0/template-only** (formatters.py imports only math+datetime; no LLM) →
NOT operator-gated. See [[project_cron_scheduler_control_topology]] for the
APScheduler control surface and [[project_slack_digest_calendar_guard]] for the
trading-day guard on the digests.
