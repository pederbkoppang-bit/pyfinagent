#!/usr/bin/env bash
# archive-handoff.sh — Move handoff/current/* into handoff/archive/phase-<id>/
# when a step in .claude/masterplan.json transitions to status=done.
# Triggered by PostToolUse on Write(.claude/masterplan.json).
# Idempotent: skips steps whose archive dir already exists, and gracefully
# no-ops on the first run (no HEAD masterplan to diff against).

set -euo pipefail

REPO="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -z "$REPO" ] && exit 0

MASTERPLAN="$REPO/.claude/masterplan.json"
CURRENT_DIR="$REPO/handoff/current"
ARCHIVE_ROOT="$REPO/handoff/archive"

[ -f "$MASTERPLAN" ] || exit 0
[ -d "$CURRENT_DIR" ] || exit 0

# Gather "just-completed" step ids by diffing HEAD vs working tree.
# Prints one step id per line. Empty output = nothing to archive.
NEWLY_DONE=$(python3 - "$REPO" "$MASTERPLAN" << 'PYEOF'
import json, subprocess, sys
from pathlib import Path

repo, mp_path = sys.argv[1], sys.argv[2]

def load(s):
    try:
        return json.loads(s)
    except Exception:
        return {"phases": []}

def step_statuses(doc):
    out = {}
    for p in doc.get("phases", []):
        for s in p.get("steps", []):
            sid = s.get("id")
            if sid:
                out[sid] = s.get("status")
    return out

with open(mp_path) as f:
    after = json.load(f)

try:
    before_raw = subprocess.check_output(
        ["git", "-C", repo, "show", "HEAD:.claude/masterplan.json"],
        stderr=subprocess.DEVNULL,
    ).decode()
    before = json.loads(before_raw)
except Exception:
    before = {"phases": []}

before_s = step_statuses(before)
after_s = step_statuses(after)

for sid, status in after_s.items():
    if status == "done" and before_s.get(sid) != "done":
        print(sid)
PYEOF
)

[ -z "$NEWLY_DONE" ] && exit 0

# Archive each newly-done step. Exit codes swallowed so a partial failure
# does not block the tool call that triggered us.
archive_step() {
    local sid="$1"
    local target="$ARCHIVE_ROOT/phase-$sid"

    # Idempotency: if a dir for this id already exists, append a numeric
    # suffix so we never clobber prior evidence.
    if [ -d "$target" ]; then
        local n=2
        while [ -d "${target}-v${n}" ]; do n=$((n + 1)); done
        target="${target}-v${n}"
    fi

    mkdir -p "$target"

    local moved=0
    for f in contract.md experiment_results.md evaluator_critique.md research.md; do
        if [ -f "$CURRENT_DIR/$f" ]; then
            # Use git mv when possible (preserves history); fall back to mv.
            if git -C "$REPO" mv "handoff/current/$f" "handoff/archive/$(basename "$target")/$f" 2>/dev/null; then
                moved=$((moved + 1))
            elif mv "$CURRENT_DIR/$f" "$target/$f" 2>/dev/null; then
                moved=$((moved + 1))
            fi
        fi
    done

    echo "[archive-handoff] step $sid -> $(basename "$target") ($moved files)" >&2
}

while IFS= read -r sid; do
    [ -z "$sid" ] && continue
    archive_step "$sid" || true
done <<< "$NEWLY_DONE"

exit 0
