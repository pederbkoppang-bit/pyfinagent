#!/usr/bin/env bash
# Launchd-invoked wrapper for the nightly autoresearch memo.
# Loads backend/.env (for ANTHROPIC_API_KEY), activates the venv,
# runs run_memo.py, logs to handoff/autoresearch.log.

set -euo pipefail

REPO="/Users/ford/.openclaw/workspace/pyfinagent"
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

if python "$REPO/scripts/autoresearch/run_memo.py" >> "$LOG" 2>&1; then
    echo "[$(date -Iseconds)] END nightly autoresearch OK" >> "$LOG"
    python3 -c 'import json; json.dump({"consecutive_fails": 0}, open("'"$FAIL_STATE"'", "w"))' 2>>"$LOG" || true
else
    rc=$?
    echo "[$(date -Iseconds)] END nightly autoresearch FAIL rc=$rc" >> "$LOG"

    # ── phase-75.11 (sre-ops-04): paging seam -- page after N consecutive
    # failures (bot-token pattern reused verbatim from healthcheck.sh's P1
    # fallback). This job already logged FAIL rc above; only the page was
    # missing.
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
                --data "{\"channel\":\"$CHANNEL\",\"text\":\"P1 AUTORESEARCH: $new_fails consecutive nightly autoresearch failures (rc=$rc). See $LOG.\"}" >/dev/null 2>&1 || true
        fi
    fi
    exit "$rc"
fi
echo "---" >> "$LOG"
