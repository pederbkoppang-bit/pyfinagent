#!/usr/bin/env bash
# phase-23.1.21: external liveness watchdog for the pyfinagent backend.
#
# launchd KeepAlive=true respawns the backend on EXIT but cannot detect a
# hang (process alive, accept loop dead). This script is run by a separate
# launchd agent every 60 seconds. On 3 consecutive /api/health failures it:
#   1. sends SIGUSR1 to dump all thread stacks (faulthandler) to backend.log
#   2. sleeps 2s so the dump lands on disk
#   3. runs `launchctl kickstart -k` to forcibly restart the backend
#
# On any successful health check the consecutive-failure counter resets to 0.

set -u

HEALTH_URL="http://127.0.0.1:8000/api/health"
SERVICE_LABEL="com.pyfinagent.backend"
COUNTER_FILE="${HOME}/Library/Caches/com.pyfinagent.backend.watchdog.fails"
LOG_TAG="[backend-watchdog]"
TIMEOUT=5
FAILURE_THRESHOLD=3

mkdir -p "$(dirname "$COUNTER_FILE")"

# Read counter (default 0)
fails=0
if [ -f "$COUNTER_FILE" ]; then
    fails=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)
fi

# Health check
if curl -sS -m "$TIMEOUT" -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null | grep -q "^200$"; then
    if [ "$fails" -gt 0 ]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $LOG_TAG health OK; resetting fails (was $fails)"
    fi
    echo 0 > "$COUNTER_FILE"
    exit 0
fi

# Failure path
fails=$((fails + 1))
echo "$fails" > "$COUNTER_FILE"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $LOG_TAG health FAIL ($fails / $FAILURE_THRESHOLD)"

if [ "$fails" -lt "$FAILURE_THRESHOLD" ]; then
    exit 0
fi

# Threshold breached -> diagnose then restart
PID=$(pgrep -f "uvicorn backend.main" | head -1)
if [ -n "$PID" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $LOG_TAG sending SIGUSR1 to PID $PID for stack dump"
    kill -USR1 "$PID" 2>/dev/null || true
    sleep 2
fi

UID_NUM=$(id -u)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $LOG_TAG kickstart -k gui/$UID_NUM/$SERVICE_LABEL"
launchctl kickstart -k "gui/$UID_NUM/$SERVICE_LABEL"

# Reset counter so we don't restart in a loop while the service warms back up.
echo 0 > "$COUNTER_FILE"
