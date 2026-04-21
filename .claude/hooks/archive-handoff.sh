#!/usr/bin/env bash
# archive-handoff.sh — Move handoff/current/* into handoff/archive/phase-<id>/
# when a step in .claude/masterplan.json transitions to status=done.
# Triggered by PostToolUse on Write(.claude/masterplan.json).
# Idempotent: skips steps whose archive dir already exists, and gracefully
# no-ops on the first run (no HEAD masterplan to diff against).
#
# REMEDIATION GUARD (2026-04-20): exit early if the remediation flag
# file exists. Bug: when HEAD masterplan isn't committed, every `done`
# step looks newly-done vs HEAD, so this hook churns the archive dir on
# every masterplan write. `.claude/archive-handoff.disabled` bypasses
# the hook until the operator removes it.
if [ -f "${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/archive-handoff.disabled" ]; then
    exit 0
fi

set -euo pipefail

REPO="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$REPO" ]; then
  REPO="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$REPO" ]; then
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
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
    # phase-4.16.2 fix: masterplan step ids are inconsistent --
    # some are bare `4.14.1`, others already prefixed `phase-6.1`.
    # Strip any leading `phase-` so we do not produce `phase-phase-6.1/`.
    local short_sid="${sid#phase-}"
    local target="$ARCHIVE_ROOT/phase-$short_sid"

    # Idempotency: if a dir for this id already exists, append a numeric
    # suffix so we never clobber prior evidence.
    if [ -d "$target" ]; then
        local n=2
        while [ -d "${target}-v${n}" ]; do n=$((n + 1)); done
        target="${target}-v${n}"
    fi

    mkdir -p "$target"

    # Rolling phase-level files: COPY (not move) so downstream verifiers can
    # keep reading them between step transitions. The per-step snapshot goes
    # to the archive; the live file keeps serving cross-verification.
    # phase-4.16.2 fix: add research_brief.md (actual file name since
    # phase-4.9; `research.md` was the old name and never matched).
    local copied=0
    for f in contract.md experiment_results.md evaluator_critique.md research.md research_brief.md; do
        if [ -f "$CURRENT_DIR/$f" ]; then
            if cp "$CURRENT_DIR/$f" "$target/$f" 2>/dev/null; then
                copied=$((copied + 1))
            fi
        fi
    done

    # Step-specific files: MOVE (these are the per-substep contracts like
    # handoff/current/4.5.9-contract.md, which do belong only to one step).
    # phase-4.16.2 fix: match BOTH `<sid>-*.md` AND `phase-<sid>-*.md`
    # (the `phase-` prefix became the convention from ~phase-4.14 onward
    # and the old single-glob left 150 files stranded).
    local moved=0
    for f in "$CURRENT_DIR/${sid}-"*.md "$CURRENT_DIR/phase-${sid}-"*.md; do
        if [ -f "$f" ]; then
            local base="$(basename "$f")"
            if git -C "$REPO" mv "handoff/current/$base" "handoff/archive/$(basename "$target")/$base" 2>/dev/null; then
                moved=$((moved + 1))
            elif mv "$f" "$target/$base" 2>/dev/null; then
                moved=$((moved + 1))
            fi
        fi
    done

    echo "[archive-handoff] step $sid -> $(basename "$target") (copied=$copied moved=$moved)" >&2
}

while IFS= read -r sid; do
    [ -z "$sid" ] && continue
    archive_step "$sid" || true
done <<< "$NEWLY_DONE"

exit 0
