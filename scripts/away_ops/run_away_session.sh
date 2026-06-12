#!/usr/bin/env bash
# run_away_session.sh {am|pm} -- goal-away-ops phase-62.3.
# Launchd-driven wrapper for the two daily unattended Claude Code sessions.
# Evolves scripts/mas_harness/run_cycle.sh with away-specific policy:
#   - dirty tree -> recovery prompt (run_cycle's abort would deadlock 3 weeks)
#   - git pull failure -> offline mode (work local; push retried later)
#   - sentinel pre-flight failure OR absence -> digest-only prompt (fail-closed
#     to report-only, never silent)
#   - HALT-DEV operator token -> AM exits, PM degrades to digest-only
#   - every failure path logs to handoff/away_ops/session.log and exits 0 so
#     the launchd calendar never stalls
# Lockfile hardened vs run_cycle.sh: noclobber atomic create + PID liveness +
# process-name check (check-then-create is not atomic; PIDs get recycled).
# AWAY_SESSION_DRY_RUN=1 exercises lock/log/prompt-selection paths with an
# echo substitute for claude (used by the 62.3 acceptance proof and 62.7).

REPO="/Users/ford/.openclaw/workspace/pyfinagent"
CLAUDE_BIN="/Users/ford/.local/bin/claude"
GTIMEOUT="/opt/homebrew/bin/gtimeout"
LOCK="$REPO/handoff/.away-session.lock"
OPS="$REPO/handoff/away_ops"
SLOG="$OPS/session.log"
TOKENS="$REPO/handoff/operator_tokens.jsonl"

SESSION="${1:-am}"
case "$SESSION" in
    am) CAP=14400; MAX_TURNS=250 ;;
    pm) CAP=7200;  MAX_TURNS=120 ;;
    *)  echo "usage: $0 {am|pm}" >&2; exit 0 ;;
esac

mkdir -p "$OPS" 2>/dev/null
cd "$REPO" || exit 0

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
slog() { printf '[%s] [%s] %s\n' "$(ts)" "$SESSION" "$1" >> "$SLOG" 2>/dev/null; }

# ── Lock: atomic create via noclobber; stale reap via PID + name check ──
acquire_lock() {
    local attempt
    for attempt in 1 2; do
        if ( set -o noclobber; printf '%s\n' "$$" > "$LOCK" ) 2>/dev/null; then
            return 0
        fi
        local holder
        holder=$(head -1 "$LOCK" 2>/dev/null)
        if [ -n "$holder" ] && ps -p "$holder" -o command= 2>/dev/null | grep -q "run_away_session"; then
            return 1  # live session holds it
        fi
        slog "stale lock (pid=${holder:-empty}) reaped"
        rm -f "$LOCK"
    done
    return 1
}

if ! acquire_lock; then
    slog "SKIP -- another session holds the lock"
    exit 0
fi
trap 'rm -f "$LOCK"' EXIT

slog "START session cap=${CAP}s max_turns=${MAX_TURNS} dry_run=${AWAY_SESSION_DRY_RUN:-0}"

# ── HALT-DEV / RESUME-DEV (last one wins) ───────────────────────────────
PROMPT_KIND="$SESSION"
if [ -f "$TOKENS" ]; then
    halt_line=$(grep -n '"key": "HALT-DEV"' "$TOKENS" 2>/dev/null | tail -1 | cut -d: -f1)
    resume_line=$(grep -n '"key": "RESUME-DEV"' "$TOKENS" 2>/dev/null | tail -1 | cut -d: -f1)
    if [ -n "$halt_line" ] && [ "${resume_line:-0}" -lt "$halt_line" ]; then
        if [ "$SESSION" = "am" ]; then
            slog "HALT-DEV active (token line $halt_line) -- AM session exits"
            slog "END session result=HALTED"
            exit 0
        fi
        slog "HALT-DEV active -- PM degrades to digest-only"
        PROMPT_KIND="digest_only"
    fi
fi

if [ "${AWAY_SESSION_DRY_RUN:-0}" != "1" ]; then
    # ── Sentinel pre-flight (62.4): absent or failing => digest-only ─────
    if [ "$PROMPT_KIND" != "digest_only" ]; then
        if [ ! -x "$REPO/scripts/away_ops/sentinel.sh" ]; then
            slog "sentinel missing -- fail-closed to digest-only"
            PROMPT_KIND="digest_only"
        elif ! bash "$REPO/scripts/away_ops/sentinel.sh" >> "$SLOG" 2>&1; then
            slog "sentinel FAILED -- downgrading to digest-only"
            PROMPT_KIND="digest_only"
        fi
    fi
    # ── Dirty tree => recovery (a crashed prior session left WIP) ────────
    if [ "$PROMPT_KIND" = "am" ] || [ "$PROMPT_KIND" = "pm" ]; then
        if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
            slog "dirty tree detected -- recovery prompt selected"
            PROMPT_KIND="recovery"
        fi
    fi
    # ── Sync (offline-tolerant) ──────────────────────────────────────────
    if ! git pull --rebase origin main >> "$SLOG" 2>&1; then
        git rebase --abort >> "$SLOG" 2>&1
        slog "git pull failed -- OFFLINE MODE (work local; push retried by hooks/next session)"
    fi
fi

PROMPT_FILE="$REPO/scripts/away_ops/prompt_${PROMPT_KIND}.md"
if [ ! -f "$PROMPT_FILE" ]; then
    slog "prompt file missing: $PROMPT_FILE -- END session result=NO_PROMPT"
    exit 0
fi
slog "prompt=$PROMPT_KIND file=$(basename "$PROMPT_FILE")"

OUT_JSON="$OPS/session_${SESSION}_$(date -u +%Y%m%dT%H%M%SZ).json"

if [ "${AWAY_SESSION_DRY_RUN:-0}" = "1" ]; then
    sleep "${AWAY_SESSION_DRY_RUN_SLEEP:-5}"
    printf '{"result":"dry-run-ok","total_cost_usd":0}\n' > "$OUT_JSON"
    rc=0
else
    "$GTIMEOUT" -k 60 "$CAP" "$CLAUDE_BIN" -p \
        --dangerously-skip-permissions \
        --model claude-opus-4-8 \
        --max-turns "$MAX_TURNS" \
        --output-format json \
        < "$PROMPT_FILE" > "$OUT_JSON" 2>> "$SLOG"
    rc=$?
fi

case "$rc" in
    0)   slog "claude exited 0" ;;
    124) slog "TIMEOUT -- gtimeout TERM at ${CAP}s (WIP checkpoint expected per prompt)" ;;
    137) slog "TIMEOUT-KILL -- survived TERM, KILLed after grace" ;;
    *)   slog "claude exited rc=$rc (crash or limit)" ;;
esac

# ── Cost + limit surfacing (June-15 Agent SDK credit; digest reads these) ─
if [ -s "$OUT_JSON" ]; then
    cost=$(python3 -c "import json,sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get('total_cost_usd', 'n/a'))
except Exception:
    print('unparseable')" "$OUT_JSON" 2>/dev/null)
    slog "COST total_cost_usd=$cost out=$(basename "$OUT_JSON")"
    if grep -qiE 'usage limit|session limit|credit.*(exhaust|limit)|out of credit' "$OUT_JSON" 2>/dev/null; then
        slog "LIMIT_HIT -- Agent SDK credit or session limit reached; session output truncated"
    fi
fi

slog "END session result=rc${rc}"
exit 0
