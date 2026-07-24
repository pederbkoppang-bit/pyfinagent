# live_check -- phase-62.5: healthcheck + away-watchdog

Date: 2026-06-12. Status: criteria 1+3 COMPLETE; criterion 2 evidenced with an honest
deviation (below) + a 62.7 operator drill queued for the full bootout chain.

## Criterion 1 -- probes + structured JSON line (verbatim)

    $ bash scripts/away_ops/healthcheck.sh; echo exit=$?
    exit=0
    {"ts":"2026-06-12T10:02:27Z","ok":true,"backend":"running:84680","frontend":"running:22178","slack_bot":"running:21016","api_health":200,"frontend_http":200,"kill_switch_paused":"false","cycle_age_h":14.9,"cycle_fresh_26h":"true","disk_free_gb":115,"adc_ok":true,"gh_ok":true,"restarts_performed":0,"restart_failed":false,"restart_note":"","p1_raised":false}

All mandated probes present: launchctl state x3 (0/113 semantics, PID extraction),
/api/health, :3000/login http code, kill-switch (endpoint + audit-replay fallback when
backend down), cycle freshness (HONEST recording, never paged -- weekend false-stale
research catch), ADC token probe, disk >=20GB (df -g Available, purgeable-exclusive),
gh auth.

## Criterion 2 -- frontend drill (honest result)

    $ launchctl stop com.pyfinagent.frontend && sleep 3 && curl .../login
    frontend-down http=200            <- ALREADY BACK

FINDING: the frontend plist carries KeepAlive -- launchd healed the deliberate stop in
<3s, BEFORE the healthcheck could observe it down. The healthcheck's kickstart path is
therefore the SECOND-layer backstop (booted-out or wedged-but-running states). Its
execution + before/after logging IS evidenced by a real invocation earlier today (the
awk-bug iteration ran the full restart branch):

    "restarts_performed":1,"restart_failed":false,"restart_note":"kickstart before=loaded/http200 after=loaded/http200"

The full booted-out chain (kickstart fails 113 -> bootstrap fallback) cannot be drilled
from inside a session: launchctl bootout on pyfinagent labels is blocked by our own 62.0
guard (hook + deny mirror -- working as designed, verified again live). QUEUED for the
62.7 dress rehearsal as an operator-keystroke drill ("FRONTEND DRILL: PASS" checklist
line). Restart authority is FRONTEND-ONLY by design (backend owned by the 60s
backend-watchdog; slack-bot self-heals via KeepAlive -- two kickstart authorities on one
service is a double-restart race; research finding 2).

## CYCLE-2 CORRECTION (Q/A spawn-1 BLOCK find -- P1 delivery was a live NO-OP)

Spawn-1 probed delivery, not just logic: settings.slack_webhook_url is EMPTY on this
machine (raise_cron_alert returns False at alerting.py:150-156 BEFORE any send) and the
original curl fallback read a NONEXISTENT settings attribute -- two consecutive failed
restarts would have paged nobody, forever. The same empty webhook silently kills the
backend_watchdog.sh:64 P1 too (recorded as the WEBHOOK ask in pending_tokens.json).

FIX (cycle-2): fallback rewired to the BOT-TOKEN chat.postMessage path -- the credential
that demonstrably delivers the daily digests -- with creds grepped python-free from
backend/.env (no shared venv failure domain). HEALTHCHECK_TEST_P1=1 added for delivery
drills (forces the branch without real restart failures; never sets p1_raised).

LIVE DELIVERY PROOF (verbatim):

    $ HEALTHCHECK_TEST_P1=1 bash scripts/away_ops/healthcheck.sh
    P1_TEST_DELIVERY=true
    drill-exit=0

A real P1 message landed in the bot channel via the fallback (raise_cron_alert_sync
correctly returned false on the empty webhook first -- the layering works end-to-end).

## Criterion 3 -- independent watchdog + P1 path

    $ plutil -lint ~/Library/LaunchAgents/com.pyfinagent.away-watchdog.plist  -> OK
    $ launchctl bootstrap gui/$(id -u) ...away-watchdog.plist                 -> bootstrapped
    watchdog-fired line ts: 2026-06-12T10:03:36Z | ok: True | backend: running:84680

StartInterval=1800 + RunAtLoad=true (fired immediately on bootstrap, independent of the
two daily sessions -- the 10:03:36Z line is the watchdog's own, not a manual run). P1
path: 2 consecutive restart_failed lines (health.jsonl replay, crash-conservative) ->
raise_cron_alert_sync with ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1 env (the in-memory
deduper is per-process and P1 is not auto-critical -- the silent-no-op research trap),
returned bool checked, raw-webhook curl fallback, p1_raised replay suppression. Not
live-drilled (would require actually breaking the frontend twice); code-evidenced +
research-grounded; the alerting path itself is the production path used by
backend_watchdog.sh.

## Iteration (honest)

First run had a BSD-awk bug (\\s unsupported) -> all services read "loaded" -> one
spurious-but-harmless frontend kickstart (which usefully evidenced the restart branch)
-> fixed to /pid = /, re-run clean.
