#!/bin/bash
# Slack Bot Process Monitor
# Ensures slack_bot.app stays running and restarts if it crashes
# Install in cron: */5 * * * * /path/to/slack_bot_monitor.sh

set -e

SLACK_BOT_NAME="backend.slack_bot.app"
SLACK_BOT_LOG="/Users/ford/.openclaw/workspace/pyfinagent/backend_slack.log"
PYFINAGENT_PATH="/Users/ford/.openclaw/workspace/pyfinagent"

# Function to check if process is running
is_slack_bot_running() {
    ps aux | grep -E "python.*$SLACK_BOT_NAME" | grep -v grep > /dev/null 2>&1
}

# Function to start slack bot
start_slack_bot() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Slack bot..." >> "$SLACK_BOT_LOG"
    cd "$PYFINAGENT_PATH"
    source .venv/bin/activate
    nohup python -m backend.slack_bot.app >> "$SLACK_BOT_LOG" 2>&1 &
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Slack bot started (PID: $!)" >> "$SLACK_BOT_LOG"
}

# Main logic
if ! is_slack_bot_running; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Slack bot not running. Restarting..." >> "$SLACK_BOT_LOG"
    start_slack_bot
    
    # Send alert to iMessage
    imsg send --to "+4794810537" --text "⚠️ SLACK BOT CRASHED & RESTARTED
    
Time: $(date '+%Y-%m-%d %H:%M:%S GMT+2')
Action: Process monitor detected bot down, restarted automatically
Status: Now running again
Log: $SLACK_BOT_LOG"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Slack bot is running." >> "$SLACK_BOT_LOG"
fi
