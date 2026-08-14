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
# THIS IS A CONVENTION CHECK, NOT A SECURITY BOUNDARY (phase-86.64).
# Stated plainly because the difference decides what may be relied on it.
# Two independent reasons, both MEASURED by driving this hook:
#   1. CHANNEL. The settings.json matcher is Write|Edit, so a write issued
#      through Bash is never routed here at all. Measured: an identical qa
#      identity writing an identical target path returns exit 2 via
#      tool_name=Write and exit 0 via tool_name=Bash, and the Bash write
#      added ZERO lines to this hook log.
#   2. IDENTITY. agent_type is chosen by the SPAWNER (RFC 9700 s4.15), so a
#      caller that wants past this simply does not name itself qa-anything.
# It therefore raises the cost of an ACCIDENTAL breach and cannot stop a
# deliberate one. The covering control for the deliberate case is the
# Main-side post-verdict git-status cleanliness rule
# (per-step-protocol.md section 4). Unguarded tool names are enumerated in
# handoff/current/experiment_results_86.64.md criterion 2.
#
# Exit 2 = block (PreToolUse convention). Exit 0 = allow.
#
# FAIL DIRECTION IS MIXED, AND THE PREVIOUS COMMENT HERE WAS WRONG.
# It claimed "missing fields, malformed JSON, or an internal error must
# never brick the session -- only an explicit qa-outside-memory match
# blocks." The second half is false and was falsified by measurement:
#   * HOOK-ITSELF-BROKE  -> FAIL-OPEN, as claimed. Malformed JSON, an empty
#     payload, and python3 absent from PATH all exit 0.
#   * PATH INDETERMINATE on a qa Write/Edit -> FAIL-CLOSED, exit 2. A
#     missing tool_input key, a null tool_input, an empty dict, an empty
#     file_path and a non-dict tool_input ALL deny, because normpath("")
#     is "." and "." is not inside the memory dir. That is NOT an "explicit
#     qa-outside-memory match", and it is the safe direction for a control
#     whose whole job is to refuse when it cannot read the target -- so the
#     BEHAVIOUR is kept and the DESCRIPTION is corrected.
# A non-qa identity is unaffected by all of the above and always allows.

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
# phase-86.33 P0 -- MEASURE BEFORE REDESIGNING. The hooks doc documents
# agent_id as a second PreToolUse identity field; this guard has never read
# or logged it, so every identity conclusion in 86.31 and 86.33 rests on
# agent_type ALONE. agent_type is chosen by the SPAWNER (RFC 9700 s4.15: a
# server SHOULD NOT let clients influence their client_id), so a field the
# spawner does not choose may discriminate cases agent_type cannot.
# LOG-ONLY: recorded, and used in NO decision below. Changing policy on an
# unmeasured field would be the same mistake one level down.
agent_id = d.get("agent_id") or ""
tool_name = d.get("tool_name") or ""
tool_input = d.get("tool_input") or {}
file_path = ""
if isinstance(tool_input, dict):
    file_path = tool_input.get("file_path") or ""

# phase-86.33 criterion 2 -- MEASURE WHAT THE PLATFORM ACTUALLY SENDS.
# This guard reads four fields and logged only those, so the log could never
# answer whether the payload carries any OTHER identity field. Criterion 2
# requires that be measured by driving the real hook rather than inferred from
# documentation, so the top-level KEY SET is now recorded.
# KEYS ONLY, never values: a value could carry file content or prompt text, and
# this log is committed to the repo. Sorted for stable diffing.
# LOG-ONLY, exactly like the agent_id leg above: it feeds NO decision below.
# NOTE FOR ANY FUTURE EDITOR -- this whole python body lives inside a bash
# single-quoted block. One apostrophe anywhere in it terminates that block and
# the guard degrades to allow-everything, silently. That is criterion 6 of this
# step and it is not hypothetical. Write comments without apostrophes.
try:
    payload_keys = sorted(d.keys()) if isinstance(d, dict) else []
except Exception:
    payload_keys = []

# Always-on shape log: doubles as the empirical confirmation of which
# fields the installed Claude Code actually populates (log-only leg).
try:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    line = json.dumps({"ts": ts, "agent_type": agent_type,
                       "agent_id": agent_id,
                       "tool_name": tool_name, "file_path": file_path,
                       "payload_keys": payload_keys})
    print("LOG " + line, file=sys.stderr)
except Exception:
    pass

MEMORY_DIR = ".claude/agent-memory/qa/"


def is_qa_role(name):
    """True for the qa ROLE, not just the literal type name.

    phase-86.31 cycle 2. The predicate used to be `agent_type == "qa"`. But
    `agent_type` carries the SPAWN NAME when Main names an Agent() call, so
    every cycle-numbered Q/A walked straight past the guard. MEASURED from
    the log this hook writes (handoff/logs/qa_write_guard.log, 3110 records):
    27 DISTINCT qa-* identities -- qa-80-2-c2, qa-85-5-c3, qa-36-12-cycle6,
    ... -- issuing 113 Write/Edit events, 69 of them OUTSIDE the memory dir
    and therefore ALLOWED. Two of those are exactly the no-self-eval breach
    the guard exists to prevent: `qa-80-2-c2` EDITED
    frontend/src/lib/api.ts, and `qa-80-2` WROTE
    handoff/current/evaluator_critique.md -- the artifact Main is
    contractually the verbatim scribe for. Found by the phase-86.31 cycle-1
    Q/A; the checker written alongside it could not see the class because
    it only ever drove this hook with the literal "qa".

    DISCLOSED RESIDUAL, not closed: `workflow-subagent` (80 events) and
    `general-purpose` (22) are NOT matched here and remain unguarded. They
    are deliberately left alone because they are indistinguishable from
    LEGITIMATE writers -- the researcher rail launches under such names and
    write-first is mandatory for it, so matching them would break the other
    Layer-3 rail. The post-verdict `git status` cleanliness rule owned by Main
    (per-step-protocol.md section 4) is the named covering control, and the
    gap is queued as its own masterplan step.
    """
    n = (name or "").strip().lower()
    return n == "qa" or n.startswith("qa-") or n.startswith("qa_")


if is_qa_role(agent_type) and tool_name in ("Write", "Edit"):
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
