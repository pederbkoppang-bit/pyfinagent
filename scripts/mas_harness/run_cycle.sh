#!/usr/bin/env bash
# scripts/mas_harness/run_cycle.sh
#
# Launchd-invoked wrapper that runs one MAS harness cycle against the
# Go-Live checklist. Picks the next tractable unchecked item, drives a
# research -> plan -> generate -> evaluate -> log -> push cycle via
# `claude -p`, and exits. On schedule, launchd fires us again.
#
# Lockfile prevents concurrent cycles. Prompt lives next door in
# cycle_prompt.md.

set -euo pipefail

REPO="/Users/ford/.openclaw/workspace/pyfinagent"
PROMPT="$REPO/scripts/mas_harness/cycle_prompt.md"
LOCKFILE="$REPO/handoff/.mas-harness.lock"
LOGFILE="$REPO/handoff/mas-harness.log"
CLAUDE_BIN="/Users/ford/.local/bin/claude"

cd "$REPO"

# ── Lockfile (prevents overlapping cycles) ──────────────────────
if [ -f "$LOCKFILE" ]; then
    lock_pid=$(cat "$LOCKFILE" 2>/dev/null || echo "")
    if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
        echo "[$(date -Iseconds)] SKIP previous cycle still running (pid=$lock_pid)" >> "$LOGFILE"
        exit 0
    fi
    # Stale lock
    rm -f "$LOCKFILE"
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

# ── Sync main before cycle ──────────────────────────────────────
# Runtime log churn (backend.log, backend_slack.log, etc.) can leave
# the tracked tree dirty between cycles. Auto-stash tracked-only
# changes so rebase can proceed; pop after. Never touches untracked.
git checkout main >> "$LOGFILE" 2>&1 || true
AUTOSTASH=0
if ! git diff --quiet --ignore-submodules -- || ! git diff --cached --quiet --ignore-submodules --; then
    if git stash push -m "mas-harness-autostash-$(date +%s)" >> "$LOGFILE" 2>&1; then
        AUTOSTASH=1
    fi
fi
git pull --rebase origin main >> "$LOGFILE" 2>&1 || {
    [ "$AUTOSTASH" = "1" ] && git stash pop >> "$LOGFILE" 2>&1 || true
    echo "[$(date -Iseconds)] ABORT pull failed" >> "$LOGFILE"
    exit 1
}
[ "$AUTOSTASH" = "1" ] && git stash pop >> "$LOGFILE" 2>&1 || true

# ── Run the cycle ───────────────────────────────────────────────
CYCLE_START=$(date -Iseconds)
echo "[$CYCLE_START] START cycle" >> "$LOGFILE"

# `claude -p <prompt>` runs non-interactively, prints the final assistant
# message, exits. We pipe the prompt file in via stdin rather than as an
# arg to avoid command-line length limits.
CYCLE_OUTPUT=$(
    "$CLAUDE_BIN" \
        -p \
        --dangerously-skip-permissions \
        --model claude-opus-4-6 \
        < "$PROMPT" \
        2>&1
) || {
    CYCLE_EXIT=$?
    echo "[$(date -Iseconds)] FAIL claude exit=$CYCLE_EXIT" >> "$LOGFILE"
    echo "$CYCLE_OUTPUT" | tail -20 >> "$LOGFILE"
    exit "$CYCLE_EXIT"
}

echo "[$(date -Iseconds)] END cycle" >> "$LOGFILE"
# Append the last 5 lines of claude's output so the log shows the
# MAS_HARNESS_CYCLE_* sentinel line the prompt instructs Claude to emit.
echo "$CYCLE_OUTPUT" | tail -5 >> "$LOGFILE"
echo "---" >> "$LOGFILE"

exit 0
