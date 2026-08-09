#!/usr/bin/env bash
# Stop hook: refuse to stop while a masterplan step is still in-progress.
#
# WHY THIS EXISTS AS A SCRIPT (2026-08-09):
# The previous implementation was a `type: agent` hook whose prompt said
# "check .claude/masterplan.json" -- a RELATIVE path. The hook agent does
# not run with the project as its cwd, so that resolved to
# ~/.claude/masterplan.json, which does not exist. The agent found no
# file, therefore no in-progress steps, therefore returned ok:true every
# single time. The gate had never blocked a stop in its lifetime.
# Proven live: that hook agent messaged this session asking a peer to read
# "/Users/ford/.claude/masterplan.json".
#
# The failure mode is the point: a missing file read as "nothing to do"
# is indistinguishable from a genuine pass. This script makes the blind
# case LOUD (systemMessage) while still failing open, so the gate can
# never again pass silently for the wrong reason.
#
# Status vocabulary: the writer (scripts/generate_masterplan.py) emits
# "in-progress" with a HYPHEN. The underscore form is also accepted
# because .claude/skills/masterplan/SKILL.md treats both as open.
#
# Exit 0 + no output      = allow stop (nothing in-progress)
# Exit 0 + decision:block = refuse stop, reason fed back to Claude
# Exit 0 + systemMessage  = allow stop, but tell the operator the gate was blind

set -uo pipefail

payload=""
if [ ! -t 0 ]; then
    payload=$(cat 2>/dev/null || true)
fi

# Loop prevention: if this hook already blocked once this turn, allow the
# stop through rather than trapping the session in a block->retry cycle.
if [ -n "$payload" ]; then
    active=$(printf '%s' "$payload" | jq -r '.stop_hook_active // false' 2>/dev/null || echo false)
    if [ "$active" = "true" ]; then
        exit 0
    fi
fi

# Resolve project root -- same proven chain as teammate-idle-check.sh:
#   1) $CLAUDE_PROJECT_DIR   2) git toplevel   3) this script's ../..
PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
MASTERPLAN="$PROJECT_ROOT/.claude/masterplan.json"

# FAIL OPEN, BUT LOUDLY. This is the exact case the old gate swallowed.
if [ ! -f "$MASTERPLAN" ]; then
    jq -nc --arg p "$MASTERPLAN" \
        '{systemMessage: ("Stop gate BLIND: masterplan not found at " + $p + " -- allowing stop, but the in-progress check did NOT run.")}' \
        2>/dev/null || echo '{"systemMessage":"Stop gate BLIND: masterplan not found -- allowing stop, check did NOT run."}'
    exit 0
fi

# Count in-progress steps. Any parse failure is reported, never swallowed.
OPEN=$(python3 -c '
import json, sys
OPEN_STATUSES = {"in-progress", "in_progress"}
try:
    with open(sys.argv[1]) as f:
        mp = json.load(f)
except Exception as e:
    print("PARSE_ERROR:%s" % e)
    raise SystemExit(0)
rows = []
for phase in mp.get("phases", []):
    for step in phase.get("steps", []):
        if step.get("status") in OPEN_STATUSES:
            rows.append("[%s] %s" % (step.get("id", "?"), step.get("name", "")[:100]))
print("\n".join(rows))
' "$MASTERPLAN" 2>/dev/null) || OPEN="PARSE_ERROR:python3 invocation failed"

case "$OPEN" in
    PARSE_ERROR:*)
        jq -nc --arg e "${OPEN#PARSE_ERROR:}" \
            '{systemMessage: ("Stop gate BLIND: could not parse masterplan (" + $e + ") -- allowing stop, check did NOT run.")}' \
            2>/dev/null || echo '{"systemMessage":"Stop gate BLIND: masterplan parse failed -- allowing stop."}'
        exit 0
        ;;
esac

if [ -n "$OPEN" ]; then
    jq -nc --arg steps "$OPEN" \
        '{decision:"block", reason:("Masterplan steps are still in-progress -- finish or reclassify them before stopping:\n" + $steps)}'
    exit 0
fi

exit 0
