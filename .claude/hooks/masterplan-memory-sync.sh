#!/usr/bin/env bash
# masterplan-memory-sync.sh — Syncs masterplan state to memory layers
# Triggered by PostToolUse on Write(.claude/masterplan.json)
# Does NOT commit (avoids infinite loop)

set -euo pipefail

PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
MASTERPLAN="$PROJECT_ROOT/.claude/masterplan.json"
MEMORY_DIR="$HOME/.claude/projects/-Users-ford/memory"
EPISODIC_DIR="$HOME/.openclaw/workspace/memory"

if [ ! -f "$MASTERPLAN" ]; then exit 0; fi

# 1. Update Claude Code auto-memory topic file (masterplan_state.md)
python3 - "$MASTERPLAN" "$MEMORY_DIR/masterplan_state.md" << 'PYEOF'
import json, sys

mp_path, out_path = sys.argv[1], sys.argv[2]
with open(mp_path) as f:
    mp = json.load(f)

lines = [
    "---",
    "name: Masterplan State",
    "description: Current phase/step status from masterplan.json — read this to know where the project stands",
    "type: project",
    "---", "",
    f"Last synced: {mp.get('updated_at', 'unknown')}", "",
]

# phase-4.16.2 follow-up fix: phases and steps may be missing `status`
# or `name` in older masterplan buckets -- `.get(..., default)` keeps
# the hook non-blocking so PostToolUse does not log a Python traceback
# on every masterplan.json write.
for phase in mp.get("phases", []):
    p_status = phase.get("status")
    if not p_status:
        # Derive phase status from its steps so the summary is useful.
        step_statuses = [s.get("status", "pending") for s in phase.get("steps", [])]
        if step_statuses and all(s == "done" for s in step_statuses):
            p_status = "done"
        elif any(s == "in-progress" for s in step_statuses):
            p_status = "in-progress"
        elif any(s == "blocked" for s in step_statuses):
            p_status = "blocked"
        else:
            p_status = "pending"
    icon = {"done": "[x]", "in-progress": "[~]", "pending": "[ ]", "blocked": "[!]"}.get(p_status, "[ ]")
    gate = ""
    g = phase.get("gate")
    if isinstance(g, dict) and not g.get("approved", True):
        gate = f" — GATE: {g.get('reason','<no reason>')}"
    elif isinstance(g, str):
        gate = f" — GATE: {g}"
    phase_id = phase.get("id", "?")
    phase_name = phase.get("name", "<unnamed>")
    lines.append(f"{icon} **{phase_id}**: {phase_name} ({p_status}){gate}")

    for step in phase.get("steps", []):
        s_status = step.get("status", "pending")
        s_icon = {"done": "[x]", "in-progress": "[~]", "pending": "[ ]", "blocked": "[!]"}.get(s_status, "[ ]")
        s_id = step.get("id", "?")
        s_name = step.get("name", "<unnamed>")
        lines.append(f"  {s_icon} {s_id}: {s_name}")

with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
PYEOF

# 2. Append to OpenClaw episodic memory (today's daily log)
TODAY=$(date +%Y-%m-%d)
DAILY_LOG="$EPISODIC_DIR/$TODAY.md"

# Only append if the daily log dir exists
if [ -d "$EPISODIC_DIR" ]; then
    CHANGES=$(python3 -c "
import json
with open('$MASTERPLAN') as f:
    mp = json.load(f)
active = []
for p in mp.get('phases', []):
    for s in p.get('steps', []):
        st = s.get('status', 'pending')
        if st in ('in-progress', 'done'):
            icon = '[x]' if st == 'done' else '[~]'
            sid = s.get('id', '?')
            sname = s.get('name', '<unnamed>')
            active.append(f'  {icon} {sid}: {sname}')
if active:
    print('\n'.join(active[-5:]))
" 2>/dev/null || echo "")

    if [ -n "$CHANGES" ]; then
        echo "" >> "$DAILY_LOG"
        echo "### $(date +%H:%M) — Masterplan Updated" >> "$DAILY_LOG"
        echo "$CHANGES" >> "$DAILY_LOG"
    fi
fi
