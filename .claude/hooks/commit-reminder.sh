#!/usr/bin/env bash
# commit-reminder.sh -- When a masterplan step flips to done, stderr-nudge
# the operator to commit if the working tree is dirty.
#
# Non-blocking: always exits 0, only writes to stderr. Purpose is to get the
# user into the habit of committing per-step so post-commit-changelog.sh
# captures the work in CHANGELOG.md rather than requiring a manual summary
# at the end of a long session.
#
# Triggered by PostToolUse on Write(.claude/masterplan.json).

set -euo pipefail

# Resolve project root with the same 3-step fallback used by the other hooks.
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
[ -z "$PROJECT_ROOT" ] && exit 0

MASTERPLAN="$PROJECT_ROOT/.claude/masterplan.json"
[ -f "$MASTERPLAN" ] || exit 0

# Find steps that just transitioned to done (same diff logic as archive-handoff.sh).
NEWLY_DONE=$(python3 - "$PROJECT_ROOT" "$MASTERPLAN" << 'PYEOF' 2>/dev/null || true
import json, subprocess, sys
repo, mp_path = sys.argv[1], sys.argv[2]

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

# Nothing just flipped to done -> silent exit.
[ -z "$NEWLY_DONE" ] && exit 0

# Any uncommitted work in the tree?
DIRTY="$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | head -1 || true)"
[ -z "$DIRTY" ] && exit 0

# One line per step so the nudge is informative.
STEP_LIST="$(echo "$NEWLY_DONE" | paste -sd, -)"
CHANGED_COUNT="$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

{
  echo "[commit-reminder] Step(s) $STEP_LIST marked done with $CHANGED_COUNT uncommitted change(s)."
  echo "[commit-reminder] Commit now so post-commit-changelog.sh captures the work."
  echo "[commit-reminder]   git add -A && git commit -m 'Phase $STEP_LIST: <summary>'"
} >&2

exit 0
