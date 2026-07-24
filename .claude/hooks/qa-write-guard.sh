#!/usr/bin/env bash
# phase-75.20.1: qa-write-guard -- PreToolUse hook (matcher: Write|Edit).
#
# WHY: qa.md grants a read-only-on-file-contents evaluator, but its
# `memory: project` frontmatter makes the upstream loader auto-enable
# Read/Write/Edit ("... so the subagent can manage its memory files",
# sub-agents doc). The tools allowlist therefore CANNOT exclude
# Write/Edit without killing Q/A memory curation. This hook restores the
# enforcement: it DENIES Write/Edit for the acting `qa` subagent unless
# the target is inside .claude/agent-memory/qa/ (the memory dir the
# injection exists to serve).
#
# Identity source: hooks doc -- PreToolUse common input carries
# `agent_type` ("Agent name ... for custom subagents, this is the
# frontmatter name") when the hook fires inside a subagent call. Main's
# own tool calls carry no agent_type -> always allowed.
#
# KNOWN GAP (permissions doc): Write/Edit hooks do not intercept Bash
# subprocess writes; the Main-side post-verdict git-status cleanliness
# rule (per-step-protocol.md section 4) is the covering control.
#
# Exit 2 = block (PreToolUse convention). Exit 0 = allow.
# FAIL-OPEN by design: missing fields, malformed JSON, or an internal
# error must never brick the session -- only an explicit qa-outside-
# memory match blocks.

payload=""
if [ ! -t 0 ]; then
    payload=$(cat 2>/dev/null || true)
fi

GUARD_LOG="${CLAUDE_PROJECT_DIR:-$(pwd)}/handoff/logs/qa_write_guard.log"
mkdir -p "$(dirname "$GUARD_LOG")" 2>/dev/null || true

decision=$(printf '%s' "$payload" | python3 -c '
import json, os, sys, datetime

# Any failure in this block must end in "allow" (fail-open).
try:
    raw = sys.stdin.read()
    d = json.loads(raw) if raw.strip() else {}
except Exception:
    print("allow malformed-payload")
    sys.exit(0)

agent_type = d.get("agent_type") or ""
tool_name = d.get("tool_name") or ""
tool_input = d.get("tool_input") or {}
file_path = ""
if isinstance(tool_input, dict):
    file_path = tool_input.get("file_path") or ""

# Always-on shape log: doubles as the empirical confirmation of which
# fields the installed Claude Code actually populates (log-only leg).
try:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    line = json.dumps({"ts": ts, "agent_type": agent_type,
                       "tool_name": tool_name, "file_path": file_path})
    print("LOG " + line, file=sys.stderr)
except Exception:
    pass

MEMORY_DIR = ".claude/agent-memory/qa/"

if agent_type == "qa" and tool_name in ("Write", "Edit"):
    # normpath collapses ../ traversal so the segment check cannot be
    # escaped by a path that merely CONTAINS the memory dir substring.
    norm = os.path.normpath(file_path.replace("\\", "/"))
    if MEMORY_DIR.rstrip("/") + "/" not in norm + "/":
        print("deny qa-write-outside-memory")
        sys.exit(0)
print("allow ok")
' 2>>"$GUARD_LOG")

# Nothing decided (python missing/crashed) -> fail open.
case "$decision" in
    deny*)
        echo "qa-write-guard: BLOCKED -- the qa evaluator is read-only on file contents" >&2
        echo "(Write/Edit allowed only under .claude/agent-memory/qa/; see" >&2
        echo "per-step-protocol.md section 4 and phase-75.20.1)" >&2
        exit 2
        ;;
    *)
        exit 0
        ;;
esac
