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

# ── phase-66.4: Claude credential probe (auth vs network) ────────────────
# Two layers: (a) `claude auth status` = LOCAL credential presence (cannot
# catch an expired-but-present refresh token -- the actual 2026-06/07 mode);
# (b) the newest away session JSON scanned for api_error_status 401 -- the
# API-delivered auth signal. Network errors (ECONNRESET) carry no 401 and do
# not open an auth incident. Page once per INCIDENT via a latch file (NOT
# the tail-1 dedupe above, which re-pages every other 30-min watchdog run);
# a 401 session older than the latch's cleared_at never re-opens the
# incident (prevents re-page after recovery before the next session runs).
AUTH_STATE="$OPS/auth_page_state.json"
CLAUDE_BIN_HC="$HOME/.local/bin/claude"; [ -x "$CLAUDE_BIN_HC" ] || CLAUDE_BIN_HC="claude"
auth_status_ok=$("$CLAUDE_BIN_HC" auth status >/dev/null 2>&1 && echo true || echo false)
read -r auth_ok auth_detail <<< "$(python3 - "$OPS" "$AUTH_STATE" "$auth_status_ok" <<'PYEOF' 2>/dev/null || echo "unknown probe_error"
import datetime, glob, json, os, sys
ops, state_path, status_ok = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
ok, detail = True, "ok"
if not status_ok:
    ok, detail = False, "auth_status_rc_nonzero"
cleared_at = None
try:
    st = json.load(open(state_path))
    if st.get("cleared_at"):
        cleared_at = datetime.datetime.fromisoformat(st["cleared_at"])
except Exception:
    pass
sessions = sorted(glob.glob(os.path.join(ops, "session_*.json")), key=os.path.getmtime)
if sessions:
    newest = sessions[-1]
    try:
        body = open(newest, encoding="utf-8", errors="replace").read()
    except Exception:
        body = ""
    if '"api_error_status": 401' in body or '"api_error_status":401' in body:
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(newest), datetime.timezone.utc)
        if cleared_at is None or mt > cleared_at:
            ok, detail = False, "401_in_" + os.path.basename(newest)
print("true" if ok else "false", detail)
PYEOF
)"
auth_p1=false
if [ "${HEALTHCHECK_TEST_AUTH_P1:-0}" = "1" ] || [ "$auth_ok" = "false" ]; then
    latch_open=$(python3 -c 'import json; print(str(json.load(open("'"$AUTH_STATE"'")).get("incident_open", False)).lower())' 2>/dev/null || echo false)
    if [ "${HEALTHCHECK_TEST_AUTH_P1:-0}" = "1" ] || [ "$latch_open" != "true" ]; then
        BOT_TOKEN=$(grep -m1 '^SLACK_BOT_TOKEN=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        CHANNEL=$(grep -m1 '^SLACK_CHANNEL_ID=' "$REPO/backend/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' | tr -d "'")
        [ -z "$CHANNEL" ] && CHANNEL="C0ANTGNNK8D"
        auth_sent=false
        if [ -n "$BOT_TOKEN" ]; then
            if [ "${HEALTHCHECK_TEST_AUTH_P1:-0}" = "1" ]; then
                auth_msg="P1 AWAY-WATCHDOG [DRILL 66.4]: forced auth-probe page -- delivery test only, credential is NOT actually dead."
            else
                auth_msg="P1 AWAY-WATCHDOG: Claude credential auth failure detected ($auth_detail, $ts). Scheduled sessions AND the cc_rail trading rail will fail until re-login on the host (claude /login, or claude setup-token for a 1-year credential). Paged ONCE per incident. Runbook: docs/runbooks/credential-expiry-monitoring.md"
            fi
            resp=$(curl -s -m 10 -X POST https://slack.com/api/chat.postMessage \
                -H "Authorization: Bearer $BOT_TOKEN" \
                -H 'Content-type: application/json; charset=utf-8' \
                --data "{\"channel\":\"$CHANNEL\",\"text\":\"$auth_msg\"}" 2>/dev/null)
            printf '%s' "$resp" | grep -q '"ok":true' && auth_sent=true
        fi
        if [ "${HEALTHCHECK_TEST_AUTH_P1:-0}" = "1" ]; then
            # drill isolation (62.5 doctrine): real delivery, NO latch write,
            # no auth_p1 in the JSON line.
            printf '[%s] AUTH-P1-TEST delivery=%s\n' "$ts" "$auth_sent" >> "$OPS/healthcheck_err.log" 2>/dev/null
            echo "AUTH_P1_TEST_DELIVERY=$auth_sent"
        else
            auth_p1="$auth_sent"
            python3 -c 'import json,datetime; json.dump({"incident_open": True, "opened_at": "'"$ts"'", "detail": "'"$auth_detail"'", "paged": '"$([ "$auth_sent" = "true" ] && echo True || echo False)"', "paged_by": "healthcheck"}, open("'"$AUTH_STATE"'", "w"))' 2>>"$OPS/healthcheck_err.log"
        fi
    fi
elif [ -f "$AUTH_STATE" ]; then
    # Healthy observation closes an open incident so the NEXT death re-pages.
    python3 -c '
import json, datetime
p = "'"$AUTH_STATE"'"
try:
    st = json.load(open(p))
except Exception:
    st = {}
if st.get("incident_open"):
    json.dump({"incident_open": False, "cleared_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cleared_by": "healthcheck_healthy"}, open(p, "w"))
' 2>>"$OPS/healthcheck_err.log"
fi

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

# ── backend.log rotation (62.6): cp+truncate is restart-free because the
# launchd FDs carry O_APPEND (lsof-verified; POSIX write re-derives EOF).
# Archives are gitignored + compressed + never deleted (they hold the
# pre-redaction FRED key; forensics pending the deferred rotation).
log_rotated=false
BLOG="$REPO/backend.log"
if [ -f "$BLOG" ] && [ "$(stat -f%z "$BLOG" 2>/dev/null || echo 0)" -gt 52428800 ]; then
    rts=$(date -u +%Y%m%dT%H%M%SZ)
    if cp "$BLOG" "$REPO/handoff/logs/backend.log.$rts" 2>>"$OPS/healthcheck_err.log"; then
        : > "$BLOG"
        gzip "$REPO/handoff/logs/backend.log.$rts" 2>>"$OPS/healthcheck_err.log" || true
        log_rotated=true
    fi
fi

ok=true
[ "${backend_state%%:*}" != "running" ] && ok=false
[ "$api_health" != "200" ] && ok=false
[ "$frontend_http" != "200" ] && ok=false
[ "${slackbot_state%%:*}" != "running" ] && ok=false
[ "$adc_ok" != "true" ] && ok=false
[ "$auth_ok" = "false" ] && ok=false  # phase-66.4: credential death is a health failure ("unknown" stays fail-open)
[ "${disk_free_gb:-0}" -lt 20 ] 2>/dev/null && ok=false

printf '{"ts":"%s","ok":%s,"backend":"%s","frontend":"%s","slack_bot":"%s","api_health":%s,"frontend_http":%s,"kill_switch_paused":"%s","cycle_age_h":%s,"cycle_fresh_26h":"%s","disk_free_gb":%s,"adc_ok":%s,"gh_ok":%s,"auth_ok":"%s","auth_detail":"%s","auth_p1":%s,"restarts_performed":%s,"restart_failed":%s,"restart_note":"%s","p1_raised":%s,"log_rotated":%s}\n' \
    "$ts" "$ok" "$backend_state" "$frontend_state" "$slackbot_state" \
    "${api_health:-0}" "${frontend_http:-0}" "$ks" "${cycle_age_h:--1}" "$cycle_fresh" \
    "${disk_free_gb:--1}" "$adc_ok" "$gh_ok" "$auth_ok" "$auth_detail" "$auth_p1" "$restarts" "$restart_failed" "$restart_note" "$p1_raised" "$log_rotated" \
    >> "$HEALTH" 2>/dev/null

[ "$ok" = "true" ] && exit 0 || exit 1
