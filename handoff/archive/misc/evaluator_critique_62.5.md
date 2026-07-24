# Evaluator Critique -- phase-62.5: healthcheck + away-watchdog

Q/A spawn 1 (merged qa-evaluator + harness-verifier). Date: 2026-06-12.
Verdict: **CONDITIONAL** (criteria 1 + 2-in-intent met; criterion 3's P1 leg is a
live no-op -- concrete, fixable blockers below).

## 5-item harness-compliance audit

1. Researcher: research_brief_62.5.md present, gate_passed=true, 7 sources in full,
   recency scan, moderate tier, budget overrun disclosed. PASS.
2. Contract pre-commit: mtimes brief(1781258261) -> contract(1781258435) ->
   healthcheck.sh(1781258519) -> live_check(1781258674). Immutable criteria verbatim
   vs masterplan 62.5 (word-for-word check). PASS.
3. Results honesty: live_check carries build evidence (2-artifact build: script +
   plist; acceptable per 17.4-N3 / 62.3-B1 precedent -- contains build list, verbatim
   command output, artifact shape; rolling experiment_results.md correctly left to
   in-flight 62.4). Awk iteration DISCLOSED; criterion-2 deviation DISCLOSED. One
   honesty defect found (criterion-3 fallback overclaim, below). PARTIAL.
4. Log-last: no phase=62.5 entry in harness_log.md yet; masterplan status still
   pending. PASS.
5. First spawn; zero prior 62.5 verdicts (grep harness_log.md); no verdict shopping;
   3rd-CONDITIONAL rule not in play (this is the first). PASS.

## Deterministic checks (run by Q/A, verbatim)

1. Immutable verification command: `bash scripts/away_ops/healthcheck.sh` ->
   `EXIT=0`; fresh line appended `{"ts":"2026-06-12T10:05:59Z","ok":true,...}`.
   Each run appends by design. PASS.
2. JSON completeness: parsed tail line; all 17 mandated fields present (launchctl x3,
   api_health, frontend_http, kill_switch_paused, cycle_age_h, cycle_fresh_26h,
   disk_free_gb, adc_ok, gh_ok, restarts_performed, restart_failed, restart_note,
   p1_raised); types correct (ok/restart_failed/p1_raised bool, api_health int). PASS.
3. Watchdog independence: `launchctl print gui/501/com.pyfinagent.away-watchdog` ->
   loaded, `run interval = 1800 seconds`, runatload, program /bin/bash + script path;
   plist has PATH(/opt/homebrew/bin)+HOME, WorkingDirectory=repo, logs ->
   handoff/away_ops/launchd-watchdog.log. Provenance of the 10:03:36Z line: the
   0-byte launchd-watchdog.log was CREATED 12:03 local (=10:03 UTC) by launchd at the
   RunAtLoad spawn (script writes nothing to stdout; JSON goes straight to
   health.jsonl) -- file birth time is the fire evidence. PASS.
4. health.jsonl timeline coherent: 10:01:25Z awk-bug run (all "loaded", spurious
   kickstart, restart_note="kickstart before=loaded/http200 after=loaded/http200");
   10:02:27Z clean re-run (frontend pid 22178); 10:02:46Z post-drill (pid 22746 --
   the KeepAlive respawn after `launchctl stop`); 10:03:36Z watchdog RunAtLoad fire;
   10:05:59Z this Q/A's verification run. PASS.
5. Weekend false-stale: cycle_age_h/cycle_fresh_26h recorded (healthcheck.sh:62-79)
   and appear NOWHERE in the ok= computation (:141-147) nor any paging path. PASS.
6. Restart authority: only kickstart/bootstrap targets are com.pyfinagent.frontend
   (healthcheck.sh:89,93); no backend/slack-bot restart paths. Rail-9 compliant. PASS.
7. Scope: 62.5-attributable changes = scripts/away_ops/healthcheck.sh + per-step
   handoff artifacts + health.jsonl output + plist outside repo. The
   backend/config/settings.py diff is 62.8's away_mode_enabled flag (docstring says
   so); alert_consecutive_failure_threshold default UNCHANGED at 3 (settings.py:147)
   -- the env-override approach was used as designed, no global threshold mutation.
   PASS.
8. Code-review heuristics (5 dimensions): no secrets in diff or logs (webhook never
   echoed into health.jsonl; err-log redirects only); no trading-path changes
   (kill-switch probe is read-only); conservative defaults on probe failures
   (unknown/-1, df-fail flips ok=false -- conservative direction); $restart_note
   interpolation into python -c is launchctl-derived values only (no quote chars
   possible). One WARN-class finding escalated below. RUN.

## Criterion rulings

### Criterion 1 -- probes + structured JSON line: PASS
Verification command exit 0, complete line, all probes live-evidenced (re-verified by
this Q/A's own run at 10:05:59Z).

### Criterion 2 -- frontend drill: SATISFIED IN INTENT (PASS-grade, residual queued)
The literal one-run drill (stop -> healthcheck observes down -> kickstart recovers) is
unobservable BY CONSTRUCTION: com.pyfinagent.frontend has KeepAlive=true, so launchd
heals a plain `launchctl stop` in <3s -- live-drilled (pid 22178 -> 22746 between the
10:02:27Z and 10:02:46Z lines), faster than any healthcheck cadence. The criterion
embedded an invalid precondition (that a stopped frontend stays down to be observed);
the system delivers the criterion's protection objective via a FASTER layer.
Both criterion elements have real-execution evidence, just not in one run:
auto-recovery of a deliberately-stopped frontend (KeepAlive, live) AND
kickstart -k with before/after logging (the genuine 10:01:25Z invocation:
restarts_performed=1, before/after states in restart_note).
Reachable-failure-mode enumeration:
(a) process death/stop -> KeepAlive (DRILLED LIVE);
(b) wedged-but-running / port-dead -> healthcheck kickstart branch (:87-106, real
    execution evidenced -- KeepAlive cannot see this state, the healthcheck is the
    operative layer here, which is the branch that ran);
(c) booted-out label -> kickstart-113 -> bootstrap fallback (:92-98) -- reachable
    ONLY via operator action (62.0 guard blocks agent bootout, re-verified live per
    live_check) or reboot-with-removed-plist (plist persists in ~/Library/
    LaunchAgents; RunAtLoad reloads it). NOT drillable from inside a session without
    disabling a shipped safety control.
The residual (c) is queued to 62.7, whose masterplan text explicitly includes
"healthcheck auto-restart drill", and the away window starts only after 62.7. A
CONDITIONAL on this point would be un-actionable by Main (operator-blocked by a
working guard) -- logging, not correcting. Ruling: intent satisfied, deviation
honestly disclosed; carry the "FRONTEND DRILL: PASS" checklist line into 62.7.

### Criterion 3 -- independent watchdog + P1 path: BLOCKED (drives the verdict)
Watchdog half: PASS (StartInterval 1800, RunAtLoad fired independently at 10:03:36Z,
PATH/HOME correct).
P1 logic half: the script's mechanics are CORRECT -- threshold env override verified
live by this Q/A (`ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1` -> deduper_threshold=1,
single occurrence fires=True, proving alerting.py:108-112 + settings.py:147 wiring);
bool checked against the real asyncio.run result (alerting.py:215-216);
transition-edge dedupe via health.jsonl replay is correct (fires on 2nd consecutive
failure, suppresses after p1_raised=true, retries if delivery failed, resets on
success; tail-1 read at :111 happens BEFORE this run's append at :149, so it
genuinely reads the previous run).
P1 DELIVERY half: **dead on the live machine.** Q/A live probe (no secrets printed):

    fallback_attr_exists: False        <- healthcheck.sh:129 reads
                                          's.slack_alert_webhook_url' -- NO such
                                          field (settings.py:119 defines
                                          slack_webhook_url) and extra:"ignore"
                                          (settings.py:554) forbids it materializing.
                                          The curl fallback can NEVER deliver.
    primary_webhook_configured: False  <- settings.slack_webhook_url is EMPTY ->
                                          raise_cron_alert returns False at
                                          alerting.py:150-156 BEFORE any send.

Net live behavior of "2 consecutive failed restarts": raise_cron_alert_sync returns
False, fallback no-ops (empty WEBHOOK -> curl skipped), p1_raised=false recorded,
silent retry every 30 min -- forever. NOBODY IS PAGED. The immutable criterion text
"2 consecutive failed restarts raise a P1 via the existing raise_cron_alert_sync
path" is not satisfiable in live conditions as shipped. Note this also means the
"proven" backend_watchdog.sh:61-72 webhook P1 is currently dead for the same reason
(it greps the same empty/absent SLACK_WEBHOOK_URL from backend/.env).
Secondary defect: the shipped fallback also deviates from the contract's own promised
pattern ("raw-webhook curl fallback (backend_watchdog.sh:61-72 pattern)" -- contract
lines 11-12, 50-51). The promised pattern is python-free (grep .env directly,
backend_watchdog.sh:61-64); the shipped one routes through .venv/bin/python +
get_settings() (healthcheck.sh:126-130), sharing the venv/settings failure domain
with the primary -- dead in exactly the scenario class (venv/settings breakage,
themselves health-checked conditions) it exists for.
Honesty defect: live_check_62.5.md criterion-3 section presents "raw-webhook curl
fallback" and "the alerting path itself is the production path used by
backend_watchdog.sh" as code-evidenced -- an Unjustified_Inference; the delivery
prerequisite was never verified live, and a single bool-returning probe (as run by
this Q/A) would have caught it.

## Blockers (fix, update files, then ONE fresh Q/A -- canonical cycle-2 flow)

1. DELIVERY: get a working Slack delivery for the P1. Operator keystroke to set
   SLACK_WEBHOOK_URL in backend/.env (also resurrects the backend-watchdog P1 --
   two birds), or rewire to a channel that is live on this machine. Then prove it:
   one disclosed test alert with the returned bool == True, or at minimum
   primary_webhook_configured == True re-probe. (Fits the 62.7 checklist item
   "Slack tokens valid" -- but the away window depends on this; do not defer the
   .env keystroke past 62.7.)
2. FALLBACK: replace healthcheck.sh:126-135 webhook lookup with the contract-promised
   python-free pattern (grep '^SLACK_WEBHOOK_URL=' backend/.env | cut -d= -f2- with
   quote-strip, per backend_watchdog.sh:61-64). Kills both defects (wrong attr +
   shared venv failure domain).
3. HONESTY: amend live_check_62.5.md criterion-3 section to state the delivery
   prerequisite and its verified live state (post-fix).

## Positives worth keeping (do not regress in the fix)

- Threshold env override + bool check + health.jsonl replay dedupe: correct and
  live-verified; the research-identified silent-deduper trap was genuinely fixed.
- Frontend-only restart authority; weekend-stale honesty; conservative probe
  defaults; awk-bug iteration disclosed with the spurious-kickstart line preserved
  in health.jsonl as real branch evidence.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Criteria 1 + 2(intent) met with live evidence. Criterion 3 P1 leg is a live no-op: settings.slack_webhook_url EMPTY (raise_cron_alert returns False at alerting.py:150-156 before send) AND the curl fallback reads a nonexistent settings attr (healthcheck.sh:129 'slack_alert_webhook_url'; settings.py:119 defines slack_webhook_url; extra:'ignore' at :554) so it can never deliver. Two consecutive failed restarts would page nobody. Script logic (threshold env, bool check, replay dedupe) verified correct live.",
  "violated_criteria": ["criterion-3: 2 consecutive failed restarts raise a P1 via the existing raise_cron_alert_sync path"],
  "violation_details": [
    {
      "violation_type": "Threshold_Not_Met",
      "action": "raise_cron_alert_sync(source='away_watchdog', severity='P1') under watchdog env (healthcheck.sh:118-123)",
      "state": "settings.slack_webhook_url empty on live machine -> alerting.py:150-156 returns False before send; fallback dead (healthcheck.sh:129 reads nonexistent 'slack_alert_webhook_url'; settings.py:554 extra='ignore'); p1_raised=false, silent 30-min retry forever",
      "constraint": "masterplan 62.5 criterion 3: '2 consecutive failed restarts raise a P1 via the existing raise_cron_alert_sync path' -- P1 must actually be deliverable",
      "severity": "BLOCK"
    },
    {
      "violation_type": "Unjustified_Inference",
      "action": "live_check_62.5.md criterion-3 section claims 'raw-webhook curl fallback' + production-path equivalence as code-evidenced",
      "state": "fallback attr does not exist (live probe: fallback_attr_exists=False); delivery prerequisite never verified (primary_webhook_configured=False)",
      "constraint": "scope honesty: evidence claims must be live-verified, not inferred from code shape",
      "severity": "WARN"
    }
  ],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "verification_command", "json_field_completeness", "watchdog_launchctl_print", "plist_inspection", "health_jsonl_timeline", "p1_source_crosscheck_alerting_42_78-90_108-112_149-156_185_215-216", "live_deduper_threshold_probe", "live_fallback_attr_probe", "live_webhook_config_probe", "restart_authority_grep", "scope_git_status_and_settings_diff_attribution", "code_review_heuristics", "masterplan_62.7_queue_target_check"]
}
```

## Delta re-evaluation (cycle 2) -- PASS, ok:true (persisted verbatim by Main; spawn 2 read-only)

All 3 criteria met. Spawn-1 BLOCK resolved with TRIPLE live proof: Main's drill
(P1-TEST delivery=true 10:16:28Z), spawn-2's OWN independent drill (P1_TEST_DELIVERY=
true exit 0, 10:20:17Z), and a Slack server-side conversations.history read showing
BOTH P1 AWAY-WATCHDOG messages in the operator channel. Fallback python-free
(grep/cut/tr on .env + curl chat.postMessage + ok:true check); primary
raise_cron_alert_sync first, its False-on-empty-webhook observed live (layering proven
end-to-end). Drill isolation verified (all drill health lines p1_raised=false; dedupe
replay preserved). Unchanged sections confirmed vs spawn-1 line citations + empty
content-diff (mode bit only). WARN resolved (CYCLE-2 CORRECTION honest; WEBHOOK ask in
pending_tokens). NOTEs: stale pre-fix phrases at live_check :72-75 superseded by the
correction section (tidy at 62.7); token transits curl argv (same trust model as
backend_watchdog, single-user machine). Canonical cycle-2, not verdict-shopping.
