#!/usr/bin/env bash
# healthcheck.sh -- goal-away-ops phase-62.5.
# Run by com.pyfinagent.away-watchdog (every 30 min) and by PM sessions.
# Observes everything; restarts ONLY the frontend (backend recovery is owned
# by the 60s com.pyfinagent.backend-watchdog -- two kickstart authorities on
# one service is a double-restart race; the slack-bot self-heals via its
# KeepAlive plist). Appends ONE JSON line per run to handoff/away_ops/
# health.jsonl. Pages P1 only after 2 consecutive failed FRONTEND restarts,
# via raise_cron_alert_sync with ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1 (the
# in-memory deduper dies with each one-shot process -- research catch) and a
# raw-webhook curl fallback; re-page suppression via health.jsonl replay.

REPO="/Users/ford/.openclaw/workspace/pyfinagent"
OPS="$REPO/handoff/away_ops"
HEALTH="$OPS/health.jsonl"
UID_N=$(id -u)
mkdir -p "$OPS" 2>/dev/null
cd "$REPO" || exit 0

ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

svc_state() {  # 0=loaded/running info printed; "running pid", "loaded", "absent"
    local label="$1"
    if out=$(launchctl print "gui/$UID_N/$label" 2>/dev/null); then
        local pid
        pid=$(printf '%s' "$out" | awk '/pid = /{print $3; exit}')
        if [ -n "$pid" ]; then echo "running:$pid"; else echo "loaded"; fi
    else
        echo "absent"
    fi
}

backend_state=$(svc_state com.pyfinagent.backend)
frontend_state=$(svc_state com.pyfinagent.frontend)
slackbot_state=$(svc_state com.pyfinagent.slack-bot)

api_health=$(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8000/api/health 2>/dev/null || echo 000)
frontend_http=$(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:3000/login 2>/dev/null || echo 000)

# Kill-switch: endpoint first; backend-down fallback = audit-log replay (never
# false-P1 on unknown -- backend-down is its own field).
ks="unknown"
ks_json=$(curl -s -m 10 http://localhost:8000/api/paper-trading/kill-switch 2>/dev/null)
if [ -n "$ks_json" ]; then
    ks=$(printf '%s' "$ks_json" | python3 -c 'import json,sys
try: print(str(json.load(sys.stdin).get("paused", "unknown")).lower())
except Exception: print("unknown")' 2>/dev/null || echo unknown)
fi
if [ "$ks" = "unknown" ] && [ -f "$REPO/handoff/kill_switch_audit.jsonl" ]; then
    ks=$(python3 -c '
import json
last = "unknown"
for line in open("'"$REPO"'/handoff/kill_switch_audit.jsonl", encoding="utf-8"):
    try: e = json.loads(line)
    except Exception: continue
    if e.get("event") == "pause": last = "true"
    elif e.get("event") == "resume": last = "false"
print(last)' 2>/dev/null || echo unknown)
fi

# Cycle freshness: honest recording, NEVER paged on (weekend false-stale).
read -r cycle_age_h cycle_fresh <<< "$(python3 -c '
import json, datetime
age_h, fresh = -1, "unknown"
try:
    last = None
    for line in open("'"$REPO"'/handoff/cycle_history.jsonl", encoding="utf-8"):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("status") == "started": continue
        t = e.get("completed_at") or e.get("ts") or e.get("started_at")
        if t: last = t
    if last:
        dt = datetime.datetime.fromisoformat(str(last).replace("Z", "+00:00"))
        age_h = round((datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds() / 3600, 1)
        fresh = "true" if age_h < 26 else "false"
except Exception:
    pass
print(age_h, fresh)' 2>/dev/null || echo '-1 unknown')"

disk_free_gb=$(df -g "$REPO" 2>/dev/null | awk 'NR==2{print $4}' || echo -1)
adc_ok=$(gcloud auth application-default print-access-token >/dev/null 2>&1 && echo true || echo false)
gh_ok=$(gh auth status >/dev/null 2>&1 && echo true || echo false)

# ── Frontend-only restart authority ──────────────────────────────────────
restarts=0; restart_failed=false; restart_note=""
if [ "$frontend_http" != "200" ] || [ "${frontend_state%%:*}" != "running" ]; then
    before="$frontend_state/http$frontend_http"
    if launchctl kickstart -k "gui/$UID_N/com.pyfinagent.frontend" 2>>"$OPS/healthcheck_err.log"; then
        restart_note="kickstart"
    else
        # booted-out label -> kickstart fails 113 -> bootstrap fallback
        if launchctl bootstrap "gui/$UID_N" "$HOME/Library/LaunchAgents/com.pyfinagent.frontend.plist" 2>>"$OPS/healthcheck_err.log"; then
            restart_note="kickstart-113-then-bootstrap"
        else
            restart_note="kickstart-and-bootstrap-FAILED"
            restart_failed=true
        fi
    fi
    restarts=1
    sleep 15
    frontend_state=$(svc_state com.pyfinagent.frontend)
    frontend_http=$(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:3000/login 2>/dev/null || echo 000)
    [ "$frontend_http" != "200" ] && restart_failed=true
    restart_note="$restart_note before=$before after=$frontend_state/http$frontend_http"
fi

# ── P1 after 2 consecutive failed restarts (health.jsonl replay dedupe) ──
# 62.5 cycle-2 (Q/A BLOCK find): settings.slack_webhook_url is EMPTY on this
# machine, so raise_cron_alert_sync returns False BEFORE sending (alerting.py
# :150-156) -- and the original curl fallback read a NONEXISTENT settings
# attr. Fallback rewired to the BOT-TOKEN chat.postMessage path (the same
# credential that delivers the daily digests -- live-proven), with creds
# grepped python-free from backend/.env (no shared venv failure domain).
# HEALTHCHECK_TEST_P1=1 forces this branch for delivery drills (no real
# restart failures needed); test runs do not set p1_raised in the JSON line.
p1_raised=false
p1_branch=false
if [ "${HEALTHCHECK_TEST_P1:-0}" = "1" ]; then
    p1_branch=true
elif [ "$restart_failed" = "true" ]; then
    prev=$(tail -1 "$HEALTH" 2>/dev/null)
    prev_failed=$(printf '%s' "$prev" | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin); print("true" if d.get("restart_failed") else "false", "true" if d.get("p1_raised") else "false")
except Exception: print("false false")' 2>/dev/null || echo "false false")
    read -r was_failed was_paged <<< "$prev_failed"
    [ "$was_failed" = "true" ] && [ "$was_paged" != "true" ] && p1_branch=true
fi
if [ "$p1_branch" = "true" ]; then
    sent=$(cd "$REPO" && ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1 .venv/bin/python -c "
from backend.services.observability.alerting import raise_cron_alert_sync
ok = raise_cron_alert_sync(source='away_watchdog', error_type='frontend_restart_failed',
    severity='P1', title='AWAY: frontend restart failed twice',
    details='healthcheck could not recover com.pyfinagent.frontend ($restart_note)')
print('true' if ok else 'false')" 2>>"$OPS/healthcheck_err.log" || echo false)
    if [ "$sent" != "true" ]; then
        # bot-token fallback: python-free cred extraction + curl Web API.
        BOT_TOKEN=$(grep -m1 '^SLACK_BOT_TOKEN=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        CHANNEL=$(grep -m1 '^SLACK_CHANNEL_ID=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        [ -z "$CHANNEL" ] && CHANNEL="C0ANTGNNK8D"
        if [ -n "$BOT_TOKEN" ]; then
            resp=$(curl -s -m 10 -X POST https://slack.com/api/chat.postMessage \
                -H "Authorization: Bearer $BOT_TOKEN" \
                -H 'Content-type: application/json; charset=utf-8' \
                --data "{\"channel\":\"$CHANNEL\",\"text\":\"P1 AWAY-WATCHDOG: frontend restart failed twice ($ts). $restart_note. Reply HALT-DEV to stop dev sessions; trading kill-switch is independent.\"}" 2>/dev/null)
            printf '%s' "$resp" | grep -q '"ok":true' && sent=true
        fi
    fi
    if [ "${HEALTHCHECK_TEST_P1:-0}" = "1" ]; then
        slog_p1="P1-TEST delivery=$sent"
        printf '[%s] %s\n' "$ts" "$slog_p1" >> "$OPS/healthcheck_err.log" 2>/dev/null
        echo "P1_TEST_DELIVERY=$sent"
    else
        p1_raised="$sent"
    fi
fi

ok=true
[ "${backend_state%%:*}" != "running" ] && ok=false
[ "$api_health" != "200" ] && ok=false
[ "$frontend_http" != "200" ] && ok=false
[ "${slackbot_state%%:*}" != "running" ] && ok=false
[ "$adc_ok" != "true" ] && ok=false
[ "${disk_free_gb:-0}" -lt 20 ] 2>/dev/null && ok=false

printf '{"ts":"%s","ok":%s,"backend":"%s","frontend":"%s","slack_bot":"%s","api_health":%s,"frontend_http":%s,"kill_switch_paused":"%s","cycle_age_h":%s,"cycle_fresh_26h":"%s","disk_free_gb":%s,"adc_ok":%s,"gh_ok":%s,"restarts_performed":%s,"restart_failed":%s,"restart_note":"%s","p1_raised":%s}\n' \
    "$ts" "$ok" "$backend_state" "$frontend_state" "$slackbot_state" \
    "${api_health:-0}" "${frontend_http:-0}" "$ks" "${cycle_age_h:--1}" "$cycle_fresh" \
    "${disk_free_gb:--1}" "$adc_ok" "$gh_ok" "$restarts" "$restart_failed" "$restart_note" "$p1_raised" \
    >> "$HEALTH" 2>/dev/null

[ "$ok" = "true" ] && exit 0 || exit 1
