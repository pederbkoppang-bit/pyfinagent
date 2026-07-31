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
#
# Fail-open: 2026-05-16 fix. trap exit 0 honors the design comment
# ("never blocking the masterplan Write that triggered it") even when
# `set -e` would otherwise kill the script silently mid-flow. Observed
# symptom before the fix: hook logs `INVOKED ... pid=N` (+ optional
# `INFO: live_check artifact present`) but produces no commit/push log
# lines, and the harness surfaces "PostToolUse:Edit hook error --
# Failed with non-blocking status code: No stderr output". The
# underlying root cause (where exactly set -e fires) is still being
# diagnosed; trap + set -uo pipefail makes the harness-visible
# behavior consistent with the design intent of fail-open.

trap 'exit 0' EXIT
set -uo pipefail

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

# phase-23.8.4 observability: every invocation produces a log entry so the
# next cycle that mis-fires can distinguish "hook never dispatched" (no
# INVOKED line) from "hook dispatched but newly_done was empty" (INVOKED
# line followed by silent exit at line ~114). Cheap; the existing
# newly_done detection still gates the actual commit/push work.
log "INVOKED auto-commit-and-push pid=$$"

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

# --- live_check gate (phase-23.8.1 / audit R-1) ---
# If the newly-done step has a non-empty verification.live_check field,
# require handoff/current/live_check_<step_id>.md to exist before pushing.
# Helper is a standalone Python module so the gate logic is unit-testable
# (see tests/verify_phase_23_8_1.py). Fail-open to "proceed" on any
# helper error -- consistent with the rest of this hook's discipline of
# never blocking the masterplan Write that triggered it.
GATE_HELPER="$PROJECT_ROOT/.claude/hooks/lib/live_check_gate.py"
if [ -f "$GATE_HELPER" ]; then
    GATE_DECISION=$(python3 "$GATE_HELPER" "$MASTERPLAN" "$STEP_ID" "$PROJECT_ROOT/handoff/current" 2>/dev/null || echo "proceed")
    case "$GATE_DECISION" in
        skip)
            log "WARN: live_check field set for $STEP_ID but handoff/current/live_check_${STEP_ID}.md is missing -- auto-push skipped. Create the file with verbatim live-system evidence and re-trigger by re-editing the masterplan, OR run \`git push origin main\` manually if appropriate."
            exit 0
            ;;
        passed)
            log "INFO: live_check artifact present for $STEP_ID -- gate satisfied"
            ;;
        proceed|*)
            # No live_check field set, or unrecognized output -> proceed as today.
            ;;
    esac
fi

# --- harness_log gate (phase-38.4 / OPEN-13) ---
# Mirrors live_check_gate above. Default-OFF: only fires when
# HARNESS_LOG_GATE_ENABLED=true (operator opts in once they're satisfied
# the doctrine doesn't false-positive). When enabled, requires
# `handoff/harness_log.md` to contain a `phase=<step_id>` token before
# the auto-push. Closes the failure mode where Main flips status=done
# without first appending the cycle block (phase-34 cycle 9 retro).
# Fail-open on any helper error -- consistent with hook discipline.
HARNESS_LOG_GATE_HELPER="$PROJECT_ROOT/.claude/hooks/lib/harness_log_gate.py"
HARNESS_LOG_FILE="$PROJECT_ROOT/handoff/harness_log.md"
if [ -f "$HARNESS_LOG_GATE_HELPER" ]; then
    HL_DECISION=$(python3 "$HARNESS_LOG_GATE_HELPER" "$HARNESS_LOG_FILE" "$STEP_ID" 2>/dev/null || echo "proceed")
    case "$HL_DECISION" in
        skip)
            log "WARN: harness_log gate ENABLED + handoff/harness_log.md missing 'phase=$STEP_ID' token -- auto-push skipped. Append the cycle block (see cycle format) and re-trigger by re-editing the masterplan, OR run \`git push origin main\` manually if appropriate."
            exit 0
            ;;
        passed)
            log "INFO: harness_log gate satisfied for $STEP_ID"
            ;;
        warn)
            # phase-81.0: the missing middle state. Audible, but does NOT hold
            # the push -- see harness_log_gate.py's docstring for why a gate
            # that could only ever HOLD was a gate nobody could afford to turn
            # on. Reaching this arm at all means someone set
            # HARNESS_LOG_GATE_ENABLED=true; the gate ships disabled.
            log "WARN: harness_log gate -- handoff/harness_log.md has no 'phase=$STEP_ID' token in its tail. The step is closing WITHOUT a cycle-block append (log-is-last doctrine). Push NOT held (warn mode); set HARNESS_LOG_GATE_MODE=block to make this blocking."
            printf '{"systemMessage":"harness_log gate: step %s is closing with no cycle-block append in handoff/harness_log.md (warn mode -- push proceeded)."}\n' "$STEP_ID"
            ;;
        proceed|*)
            # Gate disabled OR step-id missing -> proceed as today.
            ;;
    esac
fi

# --- verdict gate (phase-71.3) ---
# Mirrors live_check_gate above. Reads the machine-readable Q/A verdict
# (handoff/current/evaluator_critique.json, persisted by Main) so the
# status-flip gate reads the VERDICT, not prose. FAIL-OPEN: only HOLDS the
# push when the JSON is present, matches THIS step_id, and carries an
# explicit non-PASS verdict (CONDITIONAL/FAIL or ok false). Missing /
# unreadable / stale / PASS -> proceed. Never blocks the masterplan Write.
VERDICT_GATE_HELPER="$PROJECT_ROOT/.claude/hooks/lib/verdict_gate.py"
# phase-81.0: resolve the PER-STEP filename first, rolling name as fallback.
# Measured 2026-07-31: handoff/current/ holds six evaluator_critique_<id>.json
# files written 2026-07-26 -- i.e. AFTER the 2026-07-24 housekeeping sweep that
# moved the rolling evaluator_critique.json to handoff/archive/misc/. Main's
# write convention drifted rolling -> per-step and this hook was never updated,
# so it had been reading a path nothing writes. Restoring only the rolling name
# would have fixed exactly one cycle and gone dark again.
# phase-81.2: the helper now owns resolution and searches an ordered chain
# (current per-step -> current rolling -> archive/phase-<sid>/ -> archive/misc/),
# so a housekeeping sweep can no longer disarm this gate by relocating its input.
# Line 1 of the output is the decision token; line 2 is the source that answered.
if [ -f "$VERDICT_GATE_HELPER" ]; then
    VG_RAW=$(python3 "$VERDICT_GATE_HELPER" --resolve "$STEP_ID" "$PROJECT_ROOT/handoff" 2>/dev/null || printf 'proceed\nnone\n')
    VG_DECISION=$(printf '%s\n' "$VG_RAW" | sed -n '1p')
    VG_SOURCE=$(printf '%s\n' "$VG_RAW" | sed -n '2p')
    [ -n "$VG_DECISION" ] || VG_DECISION="proceed"
    [ -n "$VG_SOURCE" ] || VG_SOURCE="none"
    # A verdict answered from the archive is a DIFFERENT situation from one
    # answered out of handoff/current/ -- it means this step's evidence has
    # already been swept. Surface it rather than letting both read as a pass.
    case "$VG_SOURCE" in
        archive:*)
            log "NOTE: verdict gate for $STEP_ID resolved from '$VG_SOURCE' -- the step's verdict JSON is no longer in handoff/current/. The gate still fired (that is phase-81.2 working), but the artifact has been archived away from where it was written."
            # phase-81.2 cycle 2 (Q/A observation a): auto-push.log is gitignored
            # (.gitignore:76) and PostToolUse stdout is not shown in transcript, so a
            # log-only NOTE is silent to the operator -- exactly the failure class this
            # phase exists to kill. The no_input arm already emits a systemMessage; this
            # arm needs one too, or the operator-visible channel goes quiet precisely
            # when a step gates on a swept verdict.
            printf '{"systemMessage":"verdict gate: step %s gated on a verdict resolved from %s -- its evidence is no longer in handoff/current/. The gate fired correctly; the artifact has been archived away from where it was written."}\n' "$STEP_ID" "$VG_SOURCE"
            ;;
    esac
    case "$VG_DECISION" in
        hold)
            log "WARN: verdict gate -- evaluator_critique.json for $STEP_ID is not a clean PASS (verdict!=PASS or ok=false) -- auto-push held. Resolve the Q/A blockers + re-run a fresh Q/A to PASS, then re-trigger by re-editing the masterplan, OR run \`git push origin main\` manually if appropriate."
            exit 0
            ;;
        passed)
            log "INFO: verdict gate satisfied for $STEP_ID (evaluator_critique.json verdict=PASS)"
            ;;
        no_input)
            # phase-81.0: FAIL-OPEN (we continue), but no longer SILENT. Until
            # 81.0 this case was folded into `proceed`, which is how the gate
            # sat dark for 13 consecutive step closes after 2026-07-20 with no
            # trace. Note the log file itself is gitignored (.gitignore:76) and
            # PostToolUse stdout is not shown in the transcript, so the
            # operator-visible signal is the systemMessage below -- without it
            # this warning would be as silent as the bug it reports.
            log "WARN: verdict gate had NO INPUT for $STEP_ID -- neither handoff/current/evaluator_critique_${STEP_ID}.json nor the rolling evaluator_critique.json exists. The step closed WITHOUT a machine-readable Q/A verdict; nothing verified this push."
            printf '{"systemMessage":"verdict gate: NO INPUT for step %s -- closed without a machine-readable Q/A verdict (fail-open, push proceeded). Write handoff/current/evaluator_critique_%s.json to arm it."}\n' "$STEP_ID" "$STEP_ID"
            ;;
        proceed|*)
            # File exists but cannot gate (unreadable / stale / mismatched
            # step-id / no verdict field) -> proceed as today (fail-open).
            ;;
    esac
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
