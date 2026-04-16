#!/usr/bin/env bash
# TeammateIdle hook: keep teammates working until their assigned phase passes
# Exit 2 = block (keep working), stderr = feedback message
# Exit 0 = allow idle

set -euo pipefail

# Resolve project root with a safe fallback chain:
#   1) $CLAUDE_PROJECT_DIR (set by Claude Code)
#   2) git rev-parse (only if cwd is inside a repo)
#   3) the script's own dirname/../.. (hooks live in <root>/.claude/hooks/)
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
MASTERPLAN="$PROJECT_ROOT/.claude/masterplan.json"

if [ ! -f "$MASTERPLAN" ]; then
  exit 0  # no masterplan, allow idle
fi

# Check if any step is still in-progress
IN_PROGRESS=$(python3 -c "
import json, sys
with open('$MASTERPLAN') as f:
    mp = json.load(f)
pending = []
for phase in mp['phases']:
    for step in phase.get('steps', []):
        if step['status'] == 'in-progress':
            pending.append(f\"  [{step['id']}] {step['name']}\")
if pending:
    print('\n'.join(pending))
    sys.exit(0)
else:
    sys.exit(1)
" 2>/dev/null) || true

if [ -n "$IN_PROGRESS" ]; then
  echo "Steps still in-progress — keep working:" >&2
  echo "$IN_PROGRESS" >&2
  exit 2  # block idle, feed back what needs doing
fi

exit 0  # allow idle
