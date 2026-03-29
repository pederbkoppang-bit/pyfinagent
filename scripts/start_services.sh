#!/bin/bash
# Start backend and frontend services for pyfinAgent
# Called by Gateway Watchdog cron and can be run manually

set -e

PROJ_DIR="/Users/ford/.openclaw/workspace/pyfinagent"
VENV="$PROJ_DIR/.venv/bin/activate"

# Kill any existing instances
pkill -9 uvicorn 2>/dev/null || true
pkill -9 "next dev" 2>/dev/null || true
sleep 2

# Start backend
cd "$PROJ_DIR"
source "$VENV"
nohup python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
echo "Backend started (PID: $!)"

# Start frontend  
cd "$PROJ_DIR/frontend"
nohup npx next start -p 3000 > ../frontend.log 2>&1 &
echo "Frontend started (PID: $!)"

sleep 3

# Verify
if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
    echo "✅ Backend (8000) healthy"
else
    echo "❌ Backend (8000) not responding"
fi

if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend (3000) healthy"
else
    echo "❌ Frontend (3000) not responding"
fi
