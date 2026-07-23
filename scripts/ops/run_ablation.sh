#!/usr/bin/env bash
# scripts/ops/run_ablation.sh -- phase-75.11 (sre-ops-04).
#
# Launchd-invoked wrapper for the nightly ablation study. Replaces the LIVE
# com.pyfinagent.ablation.plist's raw `. backend/.env` ProgramArguments
# sourcing, which crash-failed ~37 nights on backend/.env:81's unbalanced
# quote (its StandardOut was 0 bytes -- the job died before logging a
# single line). The sourcing block below is copied VERBATIM from
# scripts/autoresearch/run_nightly.sh:19-27 (the phase-62.6 sanitized-grep
# fix for the exact same failure mode) rather than re-implemented.
set -euo pipefail

REPO="${SRE_OPS_REPO:-/Users/ford/.openclaw/workspace/pyfinagent}"
LOG="$REPO/handoff/logs/ablation.log"
FAIL_STATE="$REPO/handoff/away_ops/ablation_fail_state.json"
PAGE_AFTER_N="${SRE_OPS_ABLATION_PAGE_AFTER:-3}"

mkdir -p "$(dirname "$LOG")" "$(dirname "$FAIL_STATE")" 2>/dev/null
cd "$REPO"

# --- BEGIN verbatim copy of scripts/autoresearch/run_nightly.sh:19-27 ---
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
# --- END verbatim copy ---

# shellcheck disable=SC1091
. "$REPO/.venv/bin/activate"

echo "[$(date -Iseconds)] START ablation" >> "$LOG"
if python "$REPO/scripts/ablation/run_ablation.py" --next-untested >> "$LOG" 2>&1; then
    echo "[$(date -Iseconds)] END ablation OK" >> "$LOG"
    python3 -c 'import json; json.dump({"consecutive_fails": 0}, open("'"$FAIL_STATE"'", "w"))' 2>>"$LOG" || true
else
    rc=$?
    echo "[$(date -Iseconds)] END ablation FAIL rc=$rc" >> "$LOG"

    # ── Paging seam: page after N consecutive failures (bot-token pattern
    # reused verbatim from healthcheck.sh's P1 fallback) ────────────────
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
                --data "{\"channel\":\"$CHANNEL\",\"text\":\"P1 ABLATION: $new_fails consecutive nightly ablation failures (rc=$rc). See $LOG.\"}" >/dev/null 2>&1 || true
        fi
    fi
    exit "$rc"
fi
echo "---" >> "$LOG"
