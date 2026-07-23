#!/bin/bash
# Start backend and frontend services for pyfinAgent
# Called by Gateway Watchdog cron and can be run manually
#
# phase-75.11 (sre-ops-02/sre-ops-09): PRIMARY path is
# `launchctl kickstart -k` -- the supervised-restart primitive already
# proven at scripts/away_ops/healthcheck.sh:172 ("-k" kills the running
# service then restarts it under launchd supervision). The prior design
# force-killed uvicorn with SIGKILL and redirected a freshly nohup'd
# process into a truncating single-arrow redirect onto backend.log; that
# raced launchd's own KeepAlive respawn into an unsupervised zombie AND
# truncated a log that launchd itself holds an O_APPEND file descriptor on.
#
# The direct-launch path is preserved ONLY for a machine where the launchd
# agents are not yet installed -- gate it behind LEGACY_DIRECT=1 explicitly.
# It uses a SCOPED, SIGTERM-first stop (never -9, never a bare process-name
# match) so it cannot collide with an unrelated process.

set -e

PROJ_DIR="/Users/ford/.openclaw/workspace/pyfinagent"
VENV="$PROJ_DIR/.venv/bin/activate"
UID_N=$(id -u)

if [ "${LEGACY_DIRECT:-0}" != "1" ]; then
    echo "Restarting backend + frontend via launchctl kickstart (primary path)"
    launchctl kickstart -k "gui/$UID_N/com.pyfinagent.backend"
    launchctl kickstart -k "gui/$UID_N/com.pyfinagent.frontend"
    sleep 3
else
    echo "LEGACY_DIRECT=1 -- using the unsupervised direct-launch path"
    echo "(scoped SIGTERM-then-wait, never -9; prefer the launchctl path above)"

    # Scoped stop: match ONLY 'uvicorn backend.main', never a bare 'uvicorn'
    # or 'next dev' name match (those hit unrelated processes historically).
    # SIGTERM first, wait for a graceful exit, THEN just report -- never -9.
    if pgrep -f 'uvicorn backend.main' >/dev/null 2>&1; then
        echo "Stopping existing uvicorn backend.main"
        pkill -f 'uvicorn backend.main' 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            pgrep -f 'uvicorn backend.main' >/dev/null 2>&1 || break
            sleep 1
        done
        if pgrep -f 'uvicorn backend.main' >/dev/null 2>&1; then
            echo "uvicorn backend.main did not exit after SIGTERM+5s -- leaving it; investigate manually"
        fi
    fi

    # Start backend
    cd "$PROJ_DIR"
    source "$VENV"
    nohup python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 >> backend.log 2>&1 &
    echo "Backend started (PID: $!)"

    # Start frontend
    cd "$PROJ_DIR/frontend"
    nohup npx next start -p 3000 >> ../frontend.log 2>&1 &
    echo "Frontend started (PID: $!)"

    sleep 3
fi

# Verify (both paths)
if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
    echo "Backend (8000) healthy"
else
    echo "Backend (8000) not responding"
fi

if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "Frontend (3000) healthy"
else
    echo "Frontend (3000) not responding"
fi
