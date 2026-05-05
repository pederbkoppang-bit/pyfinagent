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

# phase-23.2.18: post a Slack alert BEFORE the kickstart -k SIGKILL so the
# operator is told the backend is being force-restarted. SIGKILL bypasses
# Python finally blocks; the in-process notifier cannot fire from inside
# the doomed process. Read the webhook URL out of backend/.env without
# sourcing the file (avoid leaking other secrets into this script's env).
ENV_FILE="$(cd "$(dirname "$0")/../.." && pwd)/backend/.env"
WEBHOOK_URL=""
if [ -f "$ENV_FILE" ]; then
    WEBHOOK_URL=$(grep -E '^SLACK_WEBHOOK_URL=' "$ENV_FILE" | head -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
fi
if [ -n "$WEBHOOK_URL" ]; then
    HOSTNAME_S=$(hostname -s 2>/dev/null || hostname)
    PAYLOAD=$(printf '{"text":"[P1] pyfinagent backend kickstart -k on %s. Watchdog: 3 consecutive /api/health failures. PID=%s. SIGUSR1 stack dump captured to backend.log. Backend is being force-restarted now."}' "$HOSTNAME_S" "${PID:-?}")
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $LOG_TAG posting Slack alert before kickstart"
    curl -sS -m 5 -X POST -H 'Content-Type: application/json' -d "$PAYLOAD" "$WEBHOOK_URL" >/dev/null 2>&1 || \
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $LOG_TAG Slack alert curl failed (continuing to kickstart)"
fi

UID_NUM=$(id -u)
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $LOG_TAG kickstart -k gui/$UID_NUM/$SERVICE_LABEL"
launchctl kickstart -k "gui/$UID_NUM/$SERVICE_LABEL"

# Reset counter so we don't restart in a loop while the service warms back up.
echo 0 > "$COUNTER_FILE"
