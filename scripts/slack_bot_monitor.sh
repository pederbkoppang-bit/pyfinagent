#!/bin/bash
# Slack Bot Process Monitor
# phase-62.1 (goal-away-ops): launchd-first. com.pyfinagent.slack-bot owns the
# bot (KeepAlive=true). This cron monitor no longer nohup-spawns when the
# launchd agent exists -- the old check-then-start was non-atomic and could
# race launchd into a SECOND bot instance (duplicate digests + duplicate
# phase-9 schedulers). When the agent exists, restart is delegated to
# launchctl kickstart (idempotent; launchd serializes instance management).
# The legacy nohup path remains ONLY as a fallback if the plist is absent.
# Cron line: */5 * * * * /path/to/slack_bot_monitor.sh

set -e

SLACK_BOT_NAME="backend.slack_bot.app"
SLACK_BOT_LOG="/Users/ford/.openclaw/workspace/pyfinagent/backend_slack.log"
PYFINAGENT_PATH="/Users/ford/.openclaw/workspace/pyfinagent"
LAUNCHD_LABEL="com.pyfinagent.slack-bot"
LAUNCHD_TARGET="gui/$(id -u)/$LAUNCHD_LABEL"

is_slack_bot_running() {
    ps aux | grep -E "python.*$SLACK_BOT_NAME" | grep -v grep > /dev/null 2>&1
}

# launchd-managed path (phase-62.1+)
if launchctl print "$LAUNCHD_TARGET" > /dev/null 2>&1; then
    if ! is_slack_bot_running; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Bot down despite launchd KeepAlive; kickstarting $LAUNCHD_LABEL" >> "$SLACK_BOT_LOG"
        launchctl kickstart "$LAUNCHD_TARGET" >> "$SLACK_BOT_LOG" 2>&1 || true
        imsg send --to "+4794810537" --text "SLACK BOT was down despite launchd KeepAlive; kickstart issued $(date '+%Y-%m-%d %H:%M:%S')" 2>/dev/null || true
    fi
    exit 0
fi

# Legacy fallback (launchd agent absent)
start_slack_bot() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Slack bot (legacy nohup path; launchd agent absent)..." >> "$SLACK_BOT_LOG"
    cd "$PYFINAGENT_PATH"
    source .venv/bin/activate
    nohup python -m backend.slack_bot.app >> "$SLACK_BOT_LOG" 2>&1 &
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Slack bot started (PID: $!)" >> "$SLACK_BOT_LOG"
}

if ! is_slack_bot_running; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Slack bot not running. Restarting..." >> "$SLACK_BOT_LOG"
    start_slack_bot
    imsg send --to "+4794810537" --text "SLACK BOT CRASHED AND RESTARTED (legacy path) $(date '+%Y-%m-%d %H:%M:%S')" 2>/dev/null || true
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Slack bot is running." >> "$SLACK_BOT_LOG"
fi
