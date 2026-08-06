#!/usr/bin/env bash
# Launchd-invoked wrapper for the nightly autoresearch memo.
# Loads backend/.env (for ANTHROPIC_API_KEY), activates the venv,
# runs run_memo.py, logs to handoff/autoresearch.log.

set -euo pipefail

# phase-76.9.2: AUTORESEARCH_REPO override is a TEST SEAM (fixture repos in
# backend/tests/test_phase_76_9_2_max_bridge.py); production launchd runs
# never set it and get the hardcoded default, byte-identical to before.
REPO="${AUTORESEARCH_REPO:-/Users/ford/.openclaw/workspace/pyfinagent}"
LOG="$REPO/handoff/autoresearch.log"

cd "$REPO"

# Source backend/.env into the environment (POSIX-compatible).
# phase-62.6 fix: the file is DOTENV format, not shell -- a wrapped comment
# line with an unbalanced quote (introduced by an operator paste 2026-06-12)
# made a raw `. .env` die with "unexpected EOF". Source a SANITIZED stream
# instead: only KEY=value lines, comments/garbage dropped, shell quote
# semantics preserved for well-formed lines.
if [ -f "$REPO/backend/.env" ]; then
    _envtmp=$(mktemp)
    grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$REPO/backend/.env" > "$_envtmp" 2>/dev/null || true
    set -a
    # shellcheck disable=SC1090
    . "$_envtmp"
    set +a
    rm -f "$_envtmp"
fi

# Activate venv
# shellcheck disable=SC1091
. "$REPO/.venv/bin/activate"

echo "[$(date -Iseconds)] START nightly autoresearch" >> "$LOG"
# phase-62.6 kept this at $0 (--preflight-only) through the away window.
# 2026-07-07: operator RESUMED real nightly spend (in-session AskUserQuestion,
# the AUTORESEARCH SPEND: RESUME decision) -- flag removed; the sentinel's
# token-derived metered figure (66.3) makes the spend honestly visible.
# Scheduled-run evidence: tonight's cron is the proof (39.1 doctrine).
FAIL_STATE="$REPO/handoff/away_ops/autoresearch_fail_state.json"
PAGE_AFTER_N="${SRE_OPS_AUTORESEARCH_PAGE_AFTER:-3}"
mkdir -p "$(dirname "$FAIL_STATE")" 2>/dev/null

# phase-76.9.2: fail-state increment + page-after-N, factored so BOTH failure
# branches (max-rail preflight below, run_memo failure) share the verbatim
# 75.11 seam. Body moved unchanged from the former run-failure else-branch.
_record_fail_and_page() {
    _rc="$1"; _ctx="$2"
    prev_fails=0
    if [ -f "$FAIL_STATE" ]; then
        prev_fails=$(python3 -c 'import json; print(int(json.load(open("'"$FAIL_STATE"'")).get("consecutive_fails", 0)))' 2>/dev/null || echo 0)
    fi
    new_fails=$((prev_fails + 1))
    python3 -c 'import json; json.dump({"consecutive_fails": '"$new_fails"'}, open("'"$FAIL_STATE"'", "w"))' 2>>"$LOG" || true

    if [ "$new_fails" -ge "$PAGE_AFTER_N" ]; then
        BOT_TOKEN=$(grep -m1 '^SLACK_BOT_TOKEN=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        CHANNEL=$(grep -m1 '^SLACK_CHANNEL_ID=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        [ -z "$CHANNEL" ] && CHANNEL="C0ANTGNNK8D"
        if [ -n "$BOT_TOKEN" ]; then
            curl -s -m 10 -X POST https://slack.com/api/chat.postMessage \
                -H "Authorization: Bearer $BOT_TOKEN" \
                -H 'Content-type: application/json; charset=utf-8' \
                --data "{\"channel\":\"$CHANNEL\",\"text\":\"P1 AUTORESEARCH: $new_fails consecutive nightly autoresearch failures ($_ctx rc=$_rc). See $LOG.\"}" >/dev/null 2>&1 || true
        fi
    fi
}

# ── phase-76.9.2: Anthropic Max-rail routing. When ON, every LLM call in
# run_memo routes bridge (127.0.0.1:18797) -> claude-code-proxy (:18796) ->
# `claude -p` on the Max plan at $0 metered. LOUD FAIL doctrine: if the bridge
# is down we page + exit non-zero -- NEVER silently fall through to the metered
# direct API (that silent fallback is the exact spend this flag exists to kill).
#
# phase-82.11 (2026-08-06): DEFAULT FLIPPED 0 -> 1. This supersedes the
# phase-76.9.2 default-OFF choice deliberately, on the operator's standing
# instruction, recorded verbatim in handoff/current/contract_82.11.md:
#   "82.11 metered rail: $0-metered stands -> move autoresearch OFF it or
#    disable it. Do NOT buy credits."
# The metered direct API had exhausted its credit balance and failed the
# nightly run on 12 consecutive dates (2026-07-26..2026-08-06, measured from
# handoff/autoresearch/). The bridge was measured live the same day:
# GET 127.0.0.1:18797/health -> {"ok":true,"proxy":"claude-code-cli"} and a
# real POST /v1/messages round-tripped. The default lives HERE rather than in
# backend/.env because .env is gitignored (.gitignore:5), so an .env flip would
# be unauditable.
# OPERATOR REVERT: add `AUTORESEARCH_USE_MAX_RAIL=0` to backend/.env -- the env
# var still wins over this default, and no restart is needed (this script
# re-sources .env every invocation, :22-30).
if [ "${AUTORESEARCH_USE_MAX_RAIL:-1}" = "1" ]; then
    MAX_RAIL_URL="${AUTORESEARCH_MAX_RAIL_URL:-http://127.0.0.1:18797}"
    if curl -sf -m 10 "$MAX_RAIL_URL/health" >/dev/null 2>&1; then
        export ANTHROPIC_API_URL="$MAX_RAIL_URL"
        export ANTHROPIC_BASE_URL="$MAX_RAIL_URL"
        # Dummy key: run_memo requires non-empty; the rail ignores auth; any
        # leakage to api.anthropic.com fails 401 = provable $0 metered.
        export ANTHROPIC_API_KEY="max-rail-dummy-key"
        echo "[$(date -Iseconds)] max-rail ON -- routing via $MAX_RAIL_URL (dummy key, \$0 metered)" >> "$LOG"
    else
        echo "[$(date -Iseconds)] END nightly autoresearch FAIL rc=78 (max-rail preflight: bridge $MAX_RAIL_URL down; NOT falling back to metered API)" >> "$LOG"
        _record_fail_and_page 78 "max-rail preflight"
        exit 78
    fi
fi

if python "$REPO/scripts/autoresearch/run_memo.py" >> "$LOG" 2>&1; then
    echo "[$(date -Iseconds)] END nightly autoresearch OK" >> "$LOG"
    python3 -c 'import json; json.dump({"consecutive_fails": 0}, open("'"$FAIL_STATE"'", "w"))' 2>>"$LOG" || true
else
    rc=$?
    echo "[$(date -Iseconds)] END nightly autoresearch FAIL rc=$rc" >> "$LOG"

    # phase-75.11 (sre-ops-04) paging seam, factored into
    # _record_fail_and_page above (phase-76.9.2) -- body unchanged.
    _record_fail_and_page "$rc" "run_memo"
    exit "$rc"
fi
echo "---" >> "$LOG"
