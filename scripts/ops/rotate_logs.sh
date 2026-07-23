#!/usr/bin/env bash
# scripts/ops/rotate_logs.sh -- phase-75.11 (sre-ops-01).
#
# The ONLY prior rotation authority lived inside the away-watchdog
# (scripts/away_ops/healthcheck.sh:246-255) and covered backend.log alone;
# frontend.log, handoff/logs/slack_bot.log, and handoff/logs/auto-push.log
# had NO rotation authority at all. Measured 2026-07-24: backend.log = 112MB
# (2.2x its 50MB cap), frontend.log = 15MB, slack_bot.log = 5.0MB,
# auto-push.log = 692KB; the away-watchdog itself had been dead ~17 days
# (health.jsonl frozen since 2026-07-06) with nobody paged.
#
# This script is a repo-shipped artifact. It is NOT bootstrapped as a
# machine launchd agent by this step -- that is operator token
# OPS-ROTATE-BOOTSTRAP (see handoff/current/ops_rotate_runbook_75.11.md).
#
# cp+truncate (never mv/rename) is REQUIRED for all four logs: three are
# launchd StandardOutPath/StandardErrorPath targets (backend.log,
# frontend.log, and indirectly the others via their own writers) where the
# owning process holds an O_APPEND file descriptor. POSIX write() re-derives
# EOF on every call, so `cp` then `: > file` leaves the SAME open FD writing
# fresh from offset 0 with no restart required -- lsof-verified on this
# machine (healthcheck.sh's cp+truncate produced .gz archives Jun 12 + Jul 6
# with zero backend restarts). A rename+SIGHUP model (newsyslog) was
# evaluated and REJECTED in the phase-75.11 research: launchd (not the
# child) owns the FD for launchd-captured logs, so there is no process to
# signal into reopening -- a renamed file just keeps receiving writes via
# the stale inode (the classic "logging stops after rotation" failure).
set -uo pipefail
# NOT `-e`: a single log's rotation failing (e.g. permissions) must not
# abort the watchdog-liveness alarm check below.

REPO="${SRE_OPS_REPO:-/Users/ford/.openclaw/workspace/pyfinagent}"
OPS="$REPO/handoff/away_ops"
HEALTH="$OPS/health.jsonl"
LOGDIR="$REPO/handoff/logs"
ERRLOG="$OPS/rotate_logs_err.log"
mkdir -p "$LOGDIR" "$OPS" 2>/dev/null

SIZE_CAP_BYTES="${SRE_OPS_SIZE_CAP_BYTES:-52428800}"   # 50MB -- matches the healthcheck.sh precedent
KEEP_ARCHIVES="${SRE_OPS_KEEP_ARCHIVES:-10}"            # gzip archives kept per log name

ts_compact() { date -u +%Y%m%dT%H%M%SZ; }

# ── cp+truncate rotation (model: scripts/away_ops/healthcheck.sh:246-255) ──
rotate_one() {
    local src="$1" name="$2"
    [ -f "$src" ] || return 0
    local size
    size=$(stat -f%z "$src" 2>/dev/null || echo 0)
    [ "$size" -gt "$SIZE_CAP_BYTES" ] || return 0
    local rts archive
    rts=$(ts_compact)
    archive="$LOGDIR/${name}.${rts}"
    if cp "$src" "$archive" 2>>"$ERRLOG"; then
        : > "$src"
        gzip "$archive" 2>>"$ERRLOG" || true
        echo "rotated $name (${size} bytes -> ${archive}.gz)"
    fi
}

# The four named logs at their REAL paths (backend.log + frontend.log at
# repo root via launchd StandardOutPath; the other two already live under
# handoff/logs/).
rotate_one "$REPO/backend.log"     "backend.log"
rotate_one "$REPO/frontend.log"    "frontend.log"
rotate_one "$LOGDIR/slack_bot.log" "slack_bot.log"
rotate_one "$LOGDIR/auto-push.log" "auto-push.log"

# ── Archive retention: keep only the newest KEEP_ARCHIVES .gz per log name.
prune_archives() {
    local name="$1"
    # shellcheck disable=SC2012
    ls -t "$LOGDIR/${name}".*.gz 2>/dev/null | tail -n "+$((KEEP_ARCHIVES + 1))" | while IFS= read -r old; do
        [ -n "$old" ] && rm -f "$old" 2>>"$ERRLOG"
    done
}
prune_archives "backend.log"
prune_archives "frontend.log"
prune_archives "slack_bot.log"
prune_archives "auto-push.log"

# ── Watchdog-liveness alarm seam (sre-ops-01 addendum) ──────────────────
# handoff/away_ops/health.jsonl is written ONLY by the away-watchdog
# (com.pyfinagent.away-watchdog / healthcheck.sh). If that agent stops
# firing, nothing currently notices -- measured: it was dead for 17 days
# (2026-07-06 .. 2026-07-24) before this step. This rotation agent runs on
# its own independent StartInterval, so it doubles as the away-watchdog's
# liveness monitor: page once per incident (bot-token pattern reused
# VERBATIM from healthcheck.sh's P1 fallback -- same creds, same Slack Web
# API call, same python-free grep extraction), and clear the latch once
# health.jsonl is fresh again so the next death re-pages.
ALARM_STATE="$OPS/logrotate_alarm_state.json"
STALE_THRESHOLD_S="${SRE_OPS_STALE_THRESHOLD_S:-7200}"   # 2h, per contract

page_bot_token() {
    # Reused verbatim pattern from scripts/away_ops/healthcheck.sh's P1
    # fallback: python-free credential extraction + curl Web API POST.
    local text="$1"
    local bot_token channel resp
    bot_token=$(grep -m1 '^SLACK_BOT_TOKEN=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
    channel=$(grep -m1 '^SLACK_CHANNEL_ID=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
    [ -z "$channel" ] && channel="C0ANTGNNK8D"
    [ -z "$bot_token" ] && return 1
    resp=$(curl -s -m 10 -X POST https://slack.com/api/chat.postMessage \
        -H "Authorization: Bearer $bot_token" \
        -H 'Content-type: application/json; charset=utf-8' \
        --data "{\"channel\":\"$channel\",\"text\":\"$text\"}" 2>/dev/null)
    printf '%s' "$resp" | grep -q '"ok":true'
}

now_s=$(date +%s 2>/dev/null || echo 0)
if [ -f "$HEALTH" ]; then
    health_mtime=$(stat -f%m "$HEALTH" 2>/dev/null || echo 0)
    age_s=$((now_s - health_mtime))
else
    age_s=999999999   # absent entirely == maximally stale
fi

incident_open=false
if [ -f "$ALARM_STATE" ]; then
    incident_open=$(python3 -c 'import json; print(str(json.load(open("'"$ALARM_STATE"'")).get("incident_open", False)).lower())' 2>/dev/null || echo false)
fi

if [ "$age_s" -gt "$STALE_THRESHOLD_S" ]; then
    if [ "$incident_open" != "true" ]; then
        sent=false
        if page_bot_token "P1 LOGROTATE-WATCHDOG: handoff/away_ops/health.jsonl is $((age_s / 3600))h stale -- the away-watchdog (com.pyfinagent.away-watchdog) appears dead. Check: launchctl print gui/\$UID/com.pyfinagent.away-watchdog ; recover with: launchctl kickstart -k gui/\$UID/com.pyfinagent.away-watchdog"; then
            sent=true
        fi
        python3 -c 'import json,datetime; json.dump({"incident_open": True, "opened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "age_s": '"$age_s"', "paged": '"$([ "$sent" = "true" ] && echo True || echo False)"'}, open("'"$ALARM_STATE"'", "w"))' 2>>"$ERRLOG"
        echo "ALARM: health.jsonl stale ${age_s}s, paged=$sent"
    fi
elif [ "$incident_open" = "true" ]; then
    python3 -c 'import json,datetime; json.dump({"incident_open": False, "cleared_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}, open("'"$ALARM_STATE"'", "w"))' 2>>"$ERRLOG"
    echo "ALARM cleared: health.jsonl fresh again"
fi

exit 0
