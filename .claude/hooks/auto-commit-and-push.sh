#!/usr/bin/env bash
# Post-write hook: auto-commit-and-push when a masterplan step status
# flips to "done" in .claude/masterplan.json. Wired as the 4th hook in
# the existing PostToolUse + Write/Edit(.claude/masterplan.json) chain.
#
# Sequence on a step-done flip:
#   1. git add -A
#   2. git commit -m "<phase-id>: <step name>"
#   3. invoke post-commit-changelog.sh manually (subprocess git from a hook
#      does NOT re-trigger PostToolUse(Bash), so the changelog hook must
#      be called directly — see phase-23.6.4 design)
#   4. git push origin main (logs failure, never fails the masterplan Write)
#
# Loop prevention: the changelog hook's own "chore: auto-changelog ..."
# auto-commit is NOT a Write to masterplan.json, so this hook does not
# re-fire on its own commits.

set -euo pipefail

PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

MASTERPLAN="$PROJECT_ROOT/.claude/masterplan.json"
LOG_DIR="$PROJECT_ROOT/handoff/logs"
LOG_FILE="$LOG_DIR/auto-push.log"
CHANGELOG_HOOK="$PROJECT_ROOT/.claude/hooks/post-commit-changelog.sh"

mkdir -p "$LOG_DIR"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" >> "$LOG_FILE"; }

if [ ! -f "$MASTERPLAN" ]; then
    exit 0
fi

cd "$PROJECT_ROOT"

# --- Detect status-flip-to-done via Python (jq isn't guaranteed on macOS) ---
FLIPPED_STEP=$(python3 - "$MASTERPLAN" << 'PYEOF' 2>/dev/null || true
import json
import subprocess
import sys

masterplan_path = sys.argv[1]

def load_done_ids(blob: str) -> dict:
    """Walk a masterplan JSON blob and return {id: name} for all status=done."""
    out: dict[str, str] = {}
    if not blob.strip():
        return out
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return out

    def walk(node):
        if isinstance(node, dict):
            if node.get("status") == "done" and "id" in node:
                out[str(node["id"])] = str(node.get("name", node["id"]))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return out


with open(masterplan_path, encoding="utf-8") as f:
    curr_blob = f.read()
try:
    prev_blob = subprocess.run(
        ["git", "show", "HEAD:.claude/masterplan.json"],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout
except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
    prev_blob = ""

prev_done = load_done_ids(prev_blob)
curr_done = load_done_ids(curr_blob)
newly_done = {k: v for k, v in curr_done.items() if k not in prev_done}

if not newly_done:
    sys.exit(0)


def step_sort_key(sid: str):
    """Sort by dotted-number id so '23.6.10' > '23.6.9'."""
    parts = []
    for part in sid.split("."):
        try:
            parts.append((0, int(part)))
        except ValueError:
            parts.append((1, part))
    return parts


top_id = sorted(newly_done.keys(), key=step_sort_key)[-1]
print(top_id)
print(newly_done[top_id])
PYEOF
)

if [ -z "$FLIPPED_STEP" ]; then
    # No new status=done detected -- masterplan.json edit was metadata
    # only (e.g. adding a step, changing priority). Silent exit.
    exit 0
fi

STEP_ID=$(echo "$FLIPPED_STEP" | sed -n '1p')
STEP_NAME=$(echo "$FLIPPED_STEP" | sed -n '2p')

if [ -z "$STEP_ID" ]; then
    exit 0
fi

# --- Build commit subject ---
# Prefer "phase-<id>: <name>" if the id looks like a phase-style number,
# else just "<id>: <name>".
case "$STEP_ID" in
    [0-9]*)  SUBJECT="phase-${STEP_ID}: ${STEP_NAME}" ;;
    *)       SUBJECT="${STEP_ID}: ${STEP_NAME}" ;;
esac

# Truncate to keep the subject line readable (git's soft 72-char convention).
if [ "${#SUBJECT}" -gt 100 ]; then
    SUBJECT="${SUBJECT:0:97}..."
fi

# --- Stage everything ---
# Broad capture; the pre-commit pre-tool-use-danger guard + gitignore for
# .env files cover safety. git add -A also picks up untracked archive
# snapshots from archive-handoff.sh (which runs ahead of us in the chain).
# Retry once: when the PostToolUse hook chain runs all 4 hooks back-to-back,
# the sibling masterplan-memory-sync.sh + archive-handoff.sh occasionally
# leave .git/index.lock or a transient FS state that makes the first
# `git add -A` race. A single 1s retry resolves it empirically (observed
# 2026-05-11 during phase-23.7.0 live integration test).
add_stderr=$(git add -A 2>&1)
add_rc=$?
if [ "$add_rc" -ne 0 ]; then
    log "git add -A attempt 1 failed (rc=$add_rc): $add_stderr -- retrying after 1s"
    sleep 1
    add_stderr=$(git add -A 2>&1)
    add_rc=$?
    if [ "$add_rc" -ne 0 ]; then
        log "git add -A attempt 2 failed (rc=$add_rc): $add_stderr -- giving up"
        exit 0
    fi
fi

# Nothing to commit? Maybe sibling hooks already committed. Silent exit.
if git diff --cached --quiet 2>/dev/null; then
    log "step $STEP_ID flipped to done but nothing staged (sibling hook already committed?)"
    exit 0
fi

# --- Commit ---
if ! git commit -m "$SUBJECT" >> "$LOG_FILE" 2>&1; then
    log "git commit failed for step $STEP_ID (subject: $SUBJECT)"
    exit 0
fi

COMMIT_HASH=$(git log -1 --format="%h" 2>/dev/null || echo "unknown")
log "step $STEP_ID committed as $COMMIT_HASH ($SUBJECT)"

# --- Invoke the changelog hook directly ---
# Subprocess git from inside a hook does NOT trigger PostToolUse(Bash),
# so we must call the changelog hook ourselves. The changelog hook will
# then run its own `git commit -m "chore: auto-changelog ..."` which (a)
# does not re-trigger PostToolUse(Bash) for the same reason and (b) is
# skip-listed by the auto-commit-and-push detection (chore commit doesn't
# write masterplan.json).
if [ -x "$CHANGELOG_HOOK" ] || [ -f "$CHANGELOG_HOOK" ]; then
    bash "$CHANGELOG_HOOK" >> "$LOG_FILE" 2>&1 || log "changelog hook returned non-zero"
fi

# --- Push ---
if git push origin main >> "$LOG_FILE" 2>&1; then
    log "step $STEP_ID pushed to origin/main"
else
    log "WARN: git push failed for step $STEP_ID -- commits remain local; retry manually"
fi

exit 0
