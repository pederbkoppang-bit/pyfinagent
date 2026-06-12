#!/usr/bin/env bash
# Launchd-invoked wrapper for the nightly autoresearch memo.
# Loads backend/.env (for ANTHROPIC_API_KEY), activates the venv,
# runs run_memo.py, logs to handoff/autoresearch.log.

set -euo pipefail

REPO="/Users/ford/.openclaw/workspace/pyfinagent"
LOG="$REPO/handoff/autoresearch.log"

cd "$REPO"

# Source backend/.env into the environment (POSIX-compatible).
# phase-62.6 fix: the file is DOTENV format, not shell -- a wrapped comment
# line with an unbalanced quote (introduced by an operator paste 2026-06-12)
# made a raw `. .env` die with "unexpected EOF". Source a SANITIZED stream
# instead: only KEY=value lines, comments/garbage dropped, shell quote
# semantics preserved for well-formed lines.
if [ -f "$REPO/backend/.env" ]; then
    _envtmp=$(mktemp)
    grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$REPO/backend/.env" > "$_envtmp" 2>/dev/null || true
    set -a
    # shellcheck disable=SC1090
    . "$_envtmp"
    set +a
    rm -f "$_envtmp"
fi

# Activate venv
# shellcheck disable=SC1091
. "$REPO/.venv/bin/activate"

echo "[$(date -Iseconds)] START nightly autoresearch" >> "$LOG"
# phase-62.6 (goal-away-ops): --preflight-only keeps the nightly at $0 during
# the away window (deps now installed; a full run would spend on Anthropic
# nightly). Remove the flag on the operator token 'AUTORESEARCH SPEND: RESUME'.
if python "$REPO/scripts/autoresearch/run_memo.py" --preflight-only >> "$LOG" 2>&1; then
    echo "[$(date -Iseconds)] END nightly autoresearch OK" >> "$LOG"
else
    rc=$?
    echo "[$(date -Iseconds)] END nightly autoresearch FAIL rc=$rc" >> "$LOG"
    exit "$rc"
fi
echo "---" >> "$LOG"
