#!/usr/bin/env bash
# Launchd-invoked wrapper for the nightly autoresearch memo.
# Loads backend/.env (for ANTHROPIC_API_KEY), activates the venv,
# runs run_memo.py, logs to handoff/autoresearch.log.

set -euo pipefail

REPO="/Users/ford/.openclaw/workspace/pyfinagent"
LOG="$REPO/handoff/autoresearch.log"

cd "$REPO"

# Source backend/.env into the environment (POSIX-compatible).
# Only picks up lines of the form KEY=value, ignores comments and blanks.
if [ -f "$REPO/backend/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$REPO/backend/.env"
    set +a
fi

# Activate venv
# shellcheck disable=SC1091
. "$REPO/.venv/bin/activate"

echo "[$(date -Iseconds)] START nightly autoresearch" >> "$LOG"
if python "$REPO/scripts/autoresearch/run_memo.py" >> "$LOG" 2>&1; then
    echo "[$(date -Iseconds)] END nightly autoresearch OK" >> "$LOG"
else
    rc=$?
    echo "[$(date -Iseconds)] END nightly autoresearch FAIL rc=$rc" >> "$LOG"
    exit "$rc"
fi
echo "---" >> "$LOG"
