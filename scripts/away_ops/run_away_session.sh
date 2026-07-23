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
# phase-66.4: env-overridable for auth drills (stub binary emitting the 401
# envelope); production plists set nothing and get the real path.
CLAUDE_BIN="${AWAY_SESSION_CLAUDE_BIN:-/Users/ford/.local/bin/claude}"
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
    # 62.4 refinement (preflight-test discovery): handoff/audit/*.jsonl are
    # appended by hooks AFTER every commit and handoff/away_ops/ holds the
    # wrapper's own logs -- both are perpetually dirty by design and would
    # route EVERY session into recovery. Real WIP = anything else dirty.
    if [ "$PROMPT_KIND" = "am" ] || [ "$PROMPT_KIND" = "pm" ]; then
        dirty=$(git status --porcelain 2>/dev/null | grep -vE '^.. (handoff/audit/|handoff/away_ops/|handoff/logs/)')
        if [ -n "$dirty" ]; then
            slog "dirty tree detected (non-evidence paths) -- recovery prompt selected"
            PROMPT_KIND="recovery"
        fi
    fi
    # ── Sync (offline-tolerant; skipped in preflight-test mode) ──────────
    if [ "${AWAY_SESSION_TEST_PREFLIGHT:-0}" != "1" ]; then
        # phase-75.11 (sre-ops-07): gtimeout-wrap the pull so a half-open
        # GitHub connection cannot hold the away-session lock forever --
        # purely additive: the existing `if ! ...` below already routes ANY
        # nonzero rc (including gtimeout's 124-on-timeout) to the offline
        # branch, so no new branch is needed.
        if ! "$GTIMEOUT" -k 10 120 git pull --rebase origin main >> "$SLOG" 2>&1; then
            git rebase --abort >> "$SLOG" 2>&1
            slog "git pull failed -- OFFLINE MODE (work local; push retried by hooks/next session)"
        fi
    fi
fi

PROMPT_FILE="$REPO/scripts/away_ops/prompt_${PROMPT_KIND}.md"
if [ ! -f "$PROMPT_FILE" ]; then
    slog "prompt file missing: $PROMPT_FILE -- END session result=NO_PROMPT"
    exit 0
fi
slog "prompt=$PROMPT_KIND file=$(basename "$PROMPT_FILE")"

# phase-62.4: preflight-test mode -- exercises the REAL pre-flight chain
# (HALT-DEV, sentinel, dirty-tree, prompt selection) with no git/claude side
# effects, so the sentinel->digest-only wiring is testable (criterion 3).
if [ "${AWAY_SESSION_TEST_PREFLIGHT:-0}" = "1" ]; then
    slog "PREFLIGHT-TEST prompt=$PROMPT_KIND -- exiting before sync/claude"
    echo "PREFLIGHT_PROMPT=$PROMPT_KIND"
    exit 0
fi

OUT_JSON="$OPS/session_${SESSION}_$(date -u +%Y%m%dT%H%M%SZ).json"

# ── phase-66.4: AUTH-DEAD latch -- page-once-and-skip on credential death ─
# The 2026-06-20..07-06 mode: 34 consecutive sessions burned their slot on a
# dead credential, each logged as generic "crash or limit". With the latch
# open, a session spends ONE 20s-capped probe instead of a full launch; on
# probe success it clears the latch and proceeds (recovery is automatic).
AUTH_STATE="$OPS/auth_page_state.json"
if [ "${AWAY_SESSION_DRY_RUN:-0}" != "1" ] && [ -f "$AUTH_STATE" ] \
    && python3 -c 'import json,sys; sys.exit(0 if json.load(open("'"$AUTH_STATE"'")).get("incident_open") else 1)' 2>/dev/null; then
    slog "AUTH-DEAD latch active -- probing credential before launch"
    printf 'ping' | "$GTIMEOUT" -k 5 20 "$CLAUDE_BIN" -p \
        --model claude-opus-4-8 --max-turns 1 --output-format json \
        > "$OPS/auth_probe_last.json" 2>>"$SLOG"
    probe_rc=$?
    if [ "$probe_rc" -eq 0 ] && ! grep -q '"api_error_status": *401' "$OPS/auth_probe_last.json" 2>/dev/null; then
        python3 -c 'import json,datetime; json.dump({"incident_open": False, "cleared_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cleared_by": "wrapper_probe"}, open("'"$AUTH_STATE"'", "w"))' 2>>"$SLOG"
        slog "AUTH-RECOVERED -- probe ok, latch cleared, proceeding with session"
    else
        slog "AUTH-DEAD latch active + probe still failing (rc=$probe_rc) -- session skipped (no full launch, no page; paged once at incident open)"
        slog "END session result=auth-dead-skip"
        exit 0
    fi
fi

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

# ── phase-66.4: 401-in-session = credential death, not a generic crash ───
# Detector per research_brief_66.4.md: rc!=0 AND api_error_status 401 in the
# session JSON (never key on subtype -- 401 sessions carry subtype "success").
# Page exactly once per incident via the bot-token path; the latch makes
# every subsequent slot a cheap probe-and-skip (block above).
if [ "$rc" -ne 0 ] && grep -q '"api_error_status": *401' "$OUT_JSON" 2>/dev/null; then
    slog "AUTH-DEAD -- 401 in session JSON (credential expired/corrupted)"
    latch_open=$(python3 -c 'import json; print(str(json.load(open("'"$AUTH_STATE"'")).get("incident_open", False)).lower())' 2>/dev/null || echo false)
    if [ "$latch_open" != "true" ]; then
        BOT_TOKEN=$(grep -m1 '^SLACK_BOT_TOKEN=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        CHANNEL=$(grep -m1 '^SLACK_CHANNEL_ID=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        [ -z "$CHANNEL" ] && CHANNEL="C0ANTGNNK8D"
        sent=false
        if [ -n "$BOT_TOKEN" ]; then
            resp=$(curl -s -m 10 -X POST https://slack.com/api/chat.postMessage \
                -H "Authorization: Bearer $BOT_TOKEN" \
                -H 'Content-type: application/json; charset=utf-8' \
                --data "{\"channel\":\"$CHANNEL\",\"text\":\"P1 AWAY: Claude credential DEAD (401) -- session $(basename "$OUT_JSON") died at launch. Scheduled sessions AND the cc_rail trading rail will fail until re-login on the host (claude /login, or claude setup-token for a 1-year credential). Paged ONCE; further sessions probe-and-skip. See docs/runbooks/credential-expiry-monitoring.md\"}" 2>/dev/null)
            printf '%s' "$resp" | grep -q '"ok":true' && sent=true
        fi
        python3 -c 'import json,datetime; json.dump({"incident_open": True, "opened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "detail": "401 in '"$(basename "$OUT_JSON")"'", "paged": '"$([ "$sent" = "true" ] && echo True || echo False)"'}, open("'"$AUTH_STATE"'", "w"))' 2>>"$SLOG"
        slog "AUTH-DEAD paged (delivered=$sent); latch OPEN -- subsequent sessions probe-and-skip"
    else
        slog "AUTH-DEAD latch already open -- no page (once per incident)"
    fi
fi

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
