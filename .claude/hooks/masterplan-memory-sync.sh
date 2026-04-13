#!/usr/bin/env bash
# masterplan-memory-sync.sh — Syncs masterplan state to memory layers
# Triggered by PostToolUse on Write(.claude/masterplan.json)
# Does NOT commit (avoids infinite loop)

set -euo pipefail

MASTERPLAN="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}/.claude/masterplan.json"
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

for phase in mp.get("phases", []):
    icon = {"done": "[x]", "in-progress": "[~]", "pending": "[ ]", "blocked": "[!]"}.get(phase["status"], "[ ]")
    gate = ""
    if phase.get("gate") and not phase["gate"].get("approved", True):
        gate = f" — GATE: {phase['gate']['reason']}"
    lines.append(f"{icon} **{phase['id']}**: {phase['name']} ({phase['status']}){gate}")

    for step in phase.get("steps", []):
        s_icon = {"done": "[x]", "in-progress": "[~]", "pending": "[ ]"}.get(step["status"], "[ ]")
        lines.append(f"  {s_icon} {step['id']}: {step['name']}")

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
        if s['status'] in ('in-progress', 'done'):
            icon = '[x]' if s['status'] == 'done' else '[~]'
            active.append(f'  {icon} {s[\"id\"]}: {s[\"name\"]}')
if active:
    print('\n'.join(active[-5:]))
" 2>/dev/null || echo "")

    if [ -n "$CHANGES" ]; then
        echo "" >> "$DAILY_LOG"
        echo "### $(date +%H:%M) — Masterplan Updated" >> "$DAILY_LOG"
        echo "$CHANGES" >> "$DAILY_LOG"
    fi
fi
