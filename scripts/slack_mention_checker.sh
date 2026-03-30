#!/bin/bash
# Phase 2.11: Slack Mention Checker
# Cron job that polls #ford-approvals for mentions and routes to Ford session
# Runs every 2-3 minutes as a sibling to Gateway Watchdog

set -e

WORKSPACE_DIR="${HOME}/.openclaw/workspace"
CHANNEL_ID="C0ANTGNNK8D"
LOCK_FILE="/tmp/slack_mention_checker.lock"
STATE_FILE="${WORKSPACE_DIR}/pyfinagent/handoff/.slack_mention_state"

# Prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    AGE=$(($(date +%s) - $(stat -f%m "$LOCK_FILE" 2>/dev/null || echo 0)))
    if [ $AGE -lt 120 ]; then
        exit 0
    fi
fi
touch "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

# Load last processed timestamp
LAST_TS=$(cat "$STATE_FILE" 2>/dev/null || echo "0")

# Load Slack bot token from config if not set
if [ -z "$SLACK_BOT_TOKEN" ]; then
    SLACK_BOT_TOKEN=$(cat ~/.openclaw/openclaw.json 2>/dev/null | jq -r '.channels.slack.botToken' 2>/dev/null || echo "")
fi

# Query Slack for new messages (requires SLACK_BOT_TOKEN env var)
if [ -z "$SLACK_BOT_TOKEN" ]; then
    echo "[slack_mention_checker] SLACK_BOT_TOKEN not found" >> /tmp/slack_mention_checker.log
    exit 1
fi

# Use curl + jq to fetch recent messages from #ford-approvals
RESPONSE=$(curl -s -X POST https://slack.com/api/conversations.history \
    -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"channel\": \"$CHANNEL_ID\", \"oldest\": \"$LAST_TS\", \"limit\": 10}")

# Parse response
ERROR=$(echo "$RESPONSE" | jq -r '.error' 2>/dev/null)
if [ "$ERROR" != "null" ] && [ ! -z "$ERROR" ]; then
    echo "[slack_mention_checker] Error: $ERROR" >> /tmp/slack_mention_checker.log
    exit 1
fi

# Process messages in chronological order (oldest first)
MESSAGES=$(echo "$RESPONSE" | jq -r '.messages | reverse | .[] | @base64')

for MSG in $MESSAGES; do
    MSG_JSON=$(echo "$MSG" | base64 -d)
    TS=$(echo "$MSG_JSON" | jq -r '.ts')
    TEXT=$(echo "$MSG_JSON" | jq -r '.text')
    USER=$(echo "$MSG_JSON" | jq -r '.user')
    
    # Update last processed timestamp
    echo "$TS" > "$STATE_FILE"
    
    # Check if message mentions Ford
    if echo "$TEXT" | grep -qE '<@U0A0CTMGF5J>|<@B[0-9A-Z]+>|@Ford'; then
        # Route to Ford's main session via OpenClaw
        # This sends a message to the Ford agent to respond
        MENTION_MSG="📬 *New Slack mention from <@$USER>*\n\`\`\`\n$TEXT\n\`\`\`\n\n*Action:* Please respond to this message in #ford-approvals"
        
        # Use OpenClaw CLI to send to Ford session (if available)
        if command -v openclaw &> /dev/null; then
            openclaw message send \
                --session "agent:main:slack:channel:c0antgnnk8d" \
                --message "$MENTION_MSG" \
                2>/dev/null || echo "[slack_mention_checker] Failed to route mention" >> /tmp/slack_mention_checker.log
        fi
        
        # Log that we processed this mention
        echo "[slack_mention_checker] Processed mention at $TS from user $USER" >> /tmp/slack_mention_checker.log
    fi
done

exit 0
