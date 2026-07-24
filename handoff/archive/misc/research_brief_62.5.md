# Research Brief — phase-62.5 (goal-away-ops): away-ops healthcheck.sh + 30-min watchdog plist

Tier: moderate (caller said "simple-to-moderate"). Date: 2026-06-12.
Tool-budget note: caller mandated a 6-item internal audit + >=5 full external reads; actual usage ran over the moderate 18-call budget (~32 calls). Disclosed per protocol.

## Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/ScheduledJobs.html | 2026-06-12 | official doc (Apple) | WebFetch full | StartCalendarInterval: "if the computer is asleep when the job should have run, your job will run when the computer wakes up." StartInterval & all others: "skipped when the computer is turned off or asleep; they will not run until the next designated time occurs." Powered off: both skip. |
| https://docs.cloud.google.com/sdk/gcloud/reference/auth/application-default/print-access-token | 2026-06-12 | official doc (Google) | WebFetch full | "generates and prints an access token for the current Application Default Credential (ADC)"; "default access token lifetime is 3600 seconds"; --lifetime up to 43200s is impersonation-only. Each call mints a fresh token — a 0 exit proves the refresh-token path works end-to-end. |
| https://docs.cloud.google.com/docs/authentication/application-default-credentials | 2026-06-12 | official doc (Google) | WebFetch full (after 301 redirect) | macOS ADC path confirmed: `$HOME/.config/gcloud/application_default_credentials.json`; tokens minted from local ADC carry `cloud-platform` scope. Page is silent on refresh-token expiry — treat `print-access-token` exit code as the authoritative liveness probe. |
| https://eclecticlight.co/2023/04/27/where-does-macos-get-its-volume-free-space-figures-from/ | 2026-06-12 | authoritative blog (Howard Oakley) | WebFetch full | "df shouldn't give you any allowance for purging, and only unconditionally free space." Finder/Disk Utility get purgeable-inclusive figures from the CacheDelete subsystem (`deleted` daemon). For scripts, df is the conservative figure; API-grade is `NSURLVolumeAvailableCapacityForImportantUsageKey`. |
| https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-06-12 | book (Google SRE) | WebFetch full | "Every page should be actionable." "Every page response should require intelligence." "I can only react with a sense of urgency a few times a day before I become fatigued." Symptoms over causes. Grounds the P1-only-on-restart-failure-transition design. |
| https://aws.amazon.com/builders-library/implementing-health-checks/ | 2026-06-12 | official doc (AWS Builders' Library) | WebFetch full | Liveness vs dependency checks; "built cautious rate limiting and control feedback loops so that the automation stops and engages humans when thresholds are crossed" — exactly the 2-consecutive-failed-restarts → P1 pattern; fail-open when the whole fleet looks unhealthy. |
| https://addigy.com/blog/macos-14-4-and-the-addigy-mdm-watchdog/ | 2026-06-12 | vendor blog | WebFetch full | [ADVERSARIAL-ish] "In the release of macOS 14.4, Apple has deprecated support for several commands, including launchctl kickstart" (advises `launchctl kill`). CONTRADICTED locally: Darwin 25.5.0 `launchctl help` still lists `kickstart  Forces an existing service to start.`, and the active backend-watchdog + operator npm-install workflow use it in production daily. Treat as a future-removal watch item, not a blocker. |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://alvinalexander.com/mac-os-x/launchd-plist-examples-startinterval-startcalendarinterval/ | blog | Apple archive doc is the canonical source for the same semantics |
| https://developer.apple.com/forums/thread/52369 | forum | anecdotal; Apple doc covers it |
| https://blog.darnell.io/automation-on-macos-with-launchctl/ | blog | duplicate of Apple semantics |
| https://github.com/DomT4/homebrew-autoupdate/issues/59 | community | real-world StartInterval→Calendar migration motivation only |
| https://blog.salrashid.dev/articles/2022/extend_gcp_token_lifetime/ | blog | impersonation-only lifetime extension; not our path |
| https://docs.cloud.google.com/sdk/gcloud/reference/auth/print-access-token | official doc | sibling command (non-ADC variant) |
| https://mjtsai.com/blog/2025/03/10/purgeable-disk-space/ | blog roundup | 2025 recency hit; eclecticlight is the primary |
| https://eclecticlight.co/2023/04/15/what-is-purgeable-disk-space/ + /2023/04/17/the-finder-confuses... | blog | same author, same conclusion as the read-in-full piece |
| https://daisydiskapp.com/guide/4/en/PurgeableSpace/ | vendor doc | purgeable counted as "available" in Finder — corroborates |
| https://upstat.io/blog/health-check-implementation-guide | vendor blog | 30-60s intervals, alert-fatigue caveats — corroborates AWS/SRE |
| https://betterstack.com/community/guides/monitoring/health-checks/ | vendor guide | generic; AWS piece is the authority |
| https://oneuptime.com/blog/post/2026-01-30-docker-health-check-best-practices/view | blog | 2026 recency hit; Docker-specific |
| https://incident.io/blog/escalation-policy-anti-patterns | vendor blog | escalation anti-patterns; corroborates SRE chapter |
| https://rootly.com/sre/eliminate-alert-fatigue-smart-incident-management (+3 sibling rootly pages) | vendor | 2026 State of Production Reliability: 83% of on-call engineers ignore/dismiss alerts at least occasionally |
| https://addigy.com/blog/addigy-new-mdm-watchdog-agent-how-to-resolve-mdm-issues-with-macos/ | vendor blog | Mar-2026 MDM watchdog release note |
| https://docs.level.io/en/articles/9926456-level-watchdog-task | vendor doc | scheduled-watchdog-as-backup-to-service-manager pattern (matches our coexist design) |
| ~6 further hits (apple discussions x3, fig.io, github htslib #803, sentry watchdog-terminations, etc.) | community | low weight |

Search-query discipline: year-less canonical ("launchd StartInterval vs StartCalendarInterval sleep missed jobs"; "gcloud ... print-access-token token lifetime refresh"; "health check script design unattended systems"; "alert fatigue SRE paging philosophy"), last-2-year ("APFS purgeable space df ... 2024"), current-year frontier ("macOS launchd watchdog health check agent 2026"). ~45 unique URLs collected.

## Recency scan (2024-2026)
Performed. Findings: (1) Addigy 2024 (macOS 14.4) kickstart-deprecation claim — evaluated against canonical behavior and REFUTED on this machine (Darwin 25.5.0, see read-in-full table); design still gains a `bootstrap` fallback. (2) 2026 State of Production Reliability Report (via rootly): 83% of on-call engineers ignore alerts at least occasionally — reinforces page-on-transition-only. (3) mjtsai 2025 purgeable-space roundup — no change vs eclecticlight 2023 canon. (4) No 2024-2026 source supersedes Apple's StartInterval/StartCalendarInterval semantics (archive doc remains canonical). Nothing else in the window changes the plan.

## Key findings (external)
1. **StartInterval fires are SKIPPED during sleep, not queued** — "All other launchd jobs are skipped when the computer is turned off or asleep" (Apple, ScheduledJobs). StartCalendarInterval queues exactly one missed fire for wake. The criterion mandates StartInterval(1800); acceptable because the Mac is held awake by the backend's `caffeinate -i -s` wrapper (com.pyfinagent.backend.plist ProgramArguments) — note `-s` only holds on AC power. Add `RunAtLoad=true` so a reboot probes immediately.
2. **A single `print-access-token` call is a complete ADC liveness probe** (Google): it mints a fresh ~3600s token from the refresh token in `~/.config/gcloud/application_default_credentials.json` (present, mode 600, dated Apr 24). Exit 0 = refresh path + network OK; nonzero (e.g. invalid_grant) = reauth needed.
3. **df Available is the right metric for a floor** (eclecticlight): it excludes purgeable space — conservative direction (never false-passes the 20GB floor). Finder/Disk Utility/diskutil "available" include purgeable via CacheDelete and over-report. Live: `df -g /Users/ford/.openclaw/workspace/pyfinagent` → Available 115 (1G-blocks, floored to integer GB — fine for a 20GB threshold).
4. **Page on transition, record everything else** (Google SRE): "Every page should be actionable"; a fresh-process watchdog that re-pages every 30 min on a steady-state failure manufactures alert fatigue. The P1 must fire on ENTERING the 2-consecutive-failed-restarts state, then be suppressed until a success intervenes.
5. **Stop-and-engage-humans is the canonical remediation limiter** (AWS): rate-limit automation, engage humans past a threshold. The "2 consecutive failed restarts → P1" criterion is exactly this pattern.
6. **kickstart deprecation claim (Addigy, macOS 14.4)** — refuted locally; keep `kickstart -k` (criterion-pinned, in production use here) but log its exit code and fall back to `bootstrap` when the service is not loaded (see A1/A5 below).

## Internal code inventory
| File | Lines | Role | Status |
|---|---|---|---|
| `~/Library/LaunchAgents/com.pyfinagent.backend-watchdog.plist` | all | 60s StartInterval agent → backend_watchdog.sh | LOADED (launchctl list: status 0) |
| `scripts/launchd/backend_watchdog.sh` | 1-79 | backend hang detector: 3-fail counter file, SIGUSR1 dump, webhook P1, `kickstart -k` backend | ACTIVE, proven (phase-23.1.21/23.2.18) |
| `~/Library/LaunchAgents/com.pyfinagent.frontend.plist` | all | next dev :3000, `KeepAlive=true`, RunAtLoad | RUNNING (pid 79156) |
| `~/Library/LaunchAgents/com.pyfinagent.backend.plist` | all | uvicorn under `caffeinate -i -s`, KeepAlive=true | RUNNING (pid 84680) |
| `~/Library/LaunchAgents/com.pyfinagent.slack-bot.plist` | all | bot under launchd, KeepAlive=true (plist dated 2026-06-12) | RUNNING (pid 8551) |
| `scripts/slack_bot_monitor.sh` | 4-9, 25-39 | legacy cron monitor — REMOVED from crontab (verified `crontab -l`: only slack_mention_checker remains); script now kickstarts the launchd target if plist present | decommissioned from cron |
| `backend/services/observability/alerting.py` | 42, 51-90, 100-116, 119-182, 185-219 | AlertDeduper + raise_cron_alert(_sync) | the P1 path; see A2 gotcha |
| `backend/config/settings.py` | 12, 147-149, 545 | `_ENV_FILE` absolute; `alert_consecutive_failure_threshold=3` Field; env_file model_config | env-var overridable |
| `backend/tools/slack.py` | 12 | `send_notification(webhook_url, message, metadata, alert_type)` async webhook helper | used by raise_cron_alert |
| `backend/main.py` | 393-409, 512-547 | `_PUBLIC_PATHS` incl. `/api/health`; health route + response shape | LIVE 200 |
| `backend/api/paper_trading.py` | 34, 480-518, 778-780 | router prefix `/api/paper-trading`; GET `/kill-switch` response shape; `_KILL_SWITCH_AUDIT_PATH` | LIVE 200 from localhost, no auth |
| `backend/api/auth.py` | (grep hit) | DEV_LOCALHOST_BYPASS localhost auth bypass (set in backend plist env) | behavior live-verified |
| `backend/services/kill_switch.py` | 40-77, 109, 159, 191, 210, 217 | audit replay: `pause`→paused, `resume`→unpaused; other events (`sod_snapshot`, `peak_update`) ignored for state | the fallback idiom to copy |
| `backend/services/cycle_health.py` | 15-16, 36, 149-218 (esp. 187, 205-206), 264-290 | cycle_history.jsonl writer + staleness verdict; "skip 'started' rows -- they have completed_at=null" | the 26h-check idiom to copy |
| `handoff/cycle_history.jsonl` | tail | `{"cycle_id","started_at","completed_at","duration_ms","status",...}` ISO8601 `+00:00`; status `started`/`completed`; pairs of rows per cycle | last completed 2026-06-11T19:10:40Z |
| `handoff/kill_switch_audit.jsonl` | tail | events seen: `resume`, `peak_update`, `sod_snapshot` (+ `pause` per writer) | last resume 2026-06-11 |
| `scripts/away_ops/run_away_session.sh` | 1-60 | away-ops conventions: noclobber lockfile, `slog` ts format, exit-0 discipline | idioms to reuse |
| `~/Library/LaunchAgents/com.pyfinagent.away-session-{am,pm}.plist` | all | away-ops plist conventions: PATH incl. /opt/homebrew/bin, logs under handoff/away_ops/ | template for new plist |

### A1. Existing backend-watchdog — ACTIVE; decision: COEXIST with disjoint ownership
The backend-watchdog is loaded and live (`launchctl list` shows `com.pyfinagent.backend-watchdog` status 0; plist `StartInterval=60`, runs `scripts/launchd/backend_watchdog.sh`). It owns: backend hang detection (3 consecutive `/api/health` failures, counter file `~/Library/Caches/com.pyfinagent.backend.watchdog.fails`, backend_watchdog.sh:17-46), SIGUSR1 stack dump (:49-54), direct-webhook P1 (:61-72), `kickstart -k` of `com.pyfinagent.backend` (:74-76).

**Decision: COEXIST.** The away-watchdog (30-min) OBSERVES everything and RESTARTS ONLY THE FRONTEND. It must NOT restart the backend: the 60s watchdog already recovers a hung/dead backend within ~3 min (10x faster than the 30-min cadence could), and two agents kickstarting the same service is exactly the double-restart race the caller fears. If the away-watchdog finds the backend down, that fact is itself diagnostic ("the 60s watchdog has been failing for >=30 min") — record it in health.jsonl; the backend-watchdog's own P1 (:68) has already paged. Not "subsume": merging would either degrade backend MTTR to 30 min or run gcloud/gh/disk/BQ-adjacent checks every 60s for no benefit, and it would churn proven recovery machinery (memory: biggest risk = regression into a working engine). Not "extend": backend_watchdog.sh's single-purpose 80-line shape is its reliability feature. Disjointness rule for GENERATE: backend → backend-watchdog only; frontend → away-watchdog only; slack-bot → launchd KeepAlive only (observed, never restarted by either script).

Note: the slack-bot is now launchd-supervised (plist dated 2026-06-12, KeepAlive=true); the old crontab monitor is GONE from `crontab -l` (only `slack_mention_checker.sh */2` remains, and `slack_bot_monitor.sh` itself was rewritten to kickstart the launchd target). No supervision conflict for the bot either.

### A2. raise_cron_alert_sync — signature + the one-shot-process TRAP
Signature (alerting.py:185-191): `raise_cron_alert_sync(source: str, error_type: str, severity: str, title: str, details: dict | str) -> bool`. Module-level imports are stdlib-only (alerting.py:31-38); `get_settings()` + `send_notification` are deferred inside the call (:144-146), so a standalone import is cheap. `_ENV_FILE` is ABSOLUTE (settings.py:12), so settings load from any cwd; run via `cd $REPO && .venv/bin/python -c "..."` so the `backend.` package resolves. No bot token needed — it posts via `settings.slack_webhook_url` webhook (alerting.py:148-156). In `python -c` there is no running loop → `asyncio.run` path → returns the REAL delivery bool (:215-216). No HTTP route exposes this; the localhost POST-to-backend relay alternative is strictly worse (it dies when the backend dies — the main thing being monitored).

**TRAP (load-bearing):** P1 is NOT in `_CRITICAL_SEVERITIES = frozenset({"P0", "critical", "CRITICAL"})` (alerting.py:42). Non-critical alerts need `consecutive_threshold=3` occurrences of the same (source, error_type) within a 5-min window IN THE SAME PROCESS before firing (:78-90). A fresh one-shot process calling `raise_cron_alert_sync(severity="P1")` ONCE accumulates occurrences=1 < 3 → `should_fire` returns False → **silent no-op** (and `asyncio.run` still returns False, so it is detectable). Two clean fixes:
- **(preferred)** run the one-shot with env `ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1` — `alert_consecutive_failure_threshold` is a real pydantic Field (settings.py:147) read by `_get_default_deduper()` (alerting.py:108-112); env var overrides env-file. Scope is naturally contained to the one-shot process.
- (alternative) call it 3x in one process (fires exactly once, on the 3rd call). Works but couples to deduper internals.
Check the returned bool; if False (or the python invocation itself fails — e.g. broken venv, which is among the things being health-checked), fall back to the raw-webhook curl pattern of backend_watchdog.sh:61-72, logging which path delivered. Cross-process re-page suppression does NOT come from the deduper (its memory dies with the process) — it must come from health.jsonl replay (A4).

### A3. cycle_history.jsonl parse + kill-switch fallback when backend is down
Line shape (writer: cycle_health.py:264-290 + record_cycle_end): `{"cycle_id", "started_at", "completed_at", "duration_ms", "status", "n_trades", "error_count", ...}`; timestamps ISO8601 with `+00:00`; each cycle writes a `status:"started"` row (completed_at=null) then a `status:"completed"` row. For the <26h check: scan from the END for the last row with `completed_at != null` (skip "started" rows — same rule as cycle_health.py:187), parse `completed_at`, compare to now-UTC. Last completed: 2026-06-11T19:10:40Z; cycle fires daily at 18:00 UTC.

Kill-switch: primary = `GET http://127.0.0.1:8000/api/paper-trading/kill-switch` (paper_trading.py:480; prefix :34) → `{"paused": bool, "pause_reason", ...}` — live-verified 200 WITHOUT auth from localhost (DEV_LOCALHOST_BYPASS=1 in the backend plist env; bypass implemented in backend/api/auth.py). When the backend is down the endpoint dies with it → fallback: replay `handoff/kill_switch_audit.jsonl` exactly like `KillSwitchState._load_from_audit` (kill_switch.py:61-77): last `pause`/`resume` event wins; ignore `peak_update`/`sod_snapshot`; per-line try/except on parse (malformed/partial tail lines must not crash the check). Record e.g. `"kill_switch": "paused=false (source=audit_fallback)"` and DO NOT raise any kill-switch P1 when the cause is backend-down — backend-down is its own separately-recorded finding (criterion explicitly requires no false-P1 here; SRE: page the symptom once).

### A4. "2 consecutive failed restarts" counter — replay health.jsonl, no second state file
Recommend deriving the counter from health.jsonl itself: each run appends one line including `frontend_restart_attempted: bool`, `frontend_restart_ok: bool` (post-restart re-probe of :3000), and `p1_raised: bool`. On the next run, read the trailing lines: consecutive runs with `restart_attempted && !restart_ok` = the counter. P1 fires iff this run is the >=2nd consecutive failure AND no `p1_raised:true` since the last success (transition-edge paging; re-page naturally if a success intervenes and it breaks again). Crash-safety: append-only JSONL with a single O_APPEND write is the project's proven audit idiom (kill_switch_audit, cycle_history); a crash mid-run loses at most the current line and the counter degrades CONSERVATIVELY (undercounts by one run → pages one cycle later, never spuriously). A separate counter file (the backend_watchdog.sh:17/41 `echo >` pattern) has a truncate-then-crash window that silently resets to 0 and adds a second source of truth — rejected. Parser must skip unparseable lines (kill_switch.py:66-70 idiom). Keep restart attempts going after the P1 (kickstart is idempotent and cheap; self-heals transients) but keep paging suppressed until a success resets the state — this is the AWS "automation stops [escalating] and engages humans" shape.

### A5. /api/health + frontend probe — both live-verified
`GET /api/health` exists (main.py:512), is auth-exempt (`_PUBLIC_PATHS`, main.py:394), returns `{"status":"ok","service":"pyfinagent-backend","version":...,"mcp_servers":{...},"limits_digest":...}` (main.py:541-547). Live: 200. Check `.status == "ok"`, 5s timeout (`curl -m 5`, same as backend_watchdog.sh:19,31). Frontend: `GET http://localhost:3000/login` → 200 live-verified auth-free (NextAuth sign-in page); `/` returns 302 (redirect to login). Probe `/login`, expect 200; treat connect-refused/timeout/5xx as down. launchctl state probe: `launchctl print gui/$(id -u)/<label>` → exit 0 = loaded (live-verified), exit 113 = not loaded/booted out (live-verified on a fake label; `kickstart` on a missing label also exits 113 with `Could not find service`). Recovery flow per the criterion: log before-state (`launchctl print` summary + :3000 code) → `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` → wait/re-probe → log after-state. If `print` showed exit 113 (deliberately booted out rather than stopped/hung), kickstart WILL fail — fall back to `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.frontend.plist`, still logging the kickstart attempt first (criterion names kickstart; the bootstrap fallback covers the case kickstart cannot by design). Note: frontend has `KeepAlive=true`, so a *killed* frontend self-heals via launchd; the watchdog's restart path matters for hung-but-alive, port-dead, and booted-out states.

### A6. Disk-free on APFS — use df Available (purgeable-EXCLUSIVE) at the repo path
`df -g /Users/ford/.openclaw/workspace/pyfinagent | awk 'NR==2{print $4}'` → 115 today (disk3s5 Data volume; / and Data share the container pool so both show the same Available). df's Available excludes purgeable space (eclecticlight) = under-reports = SAFE direction for a 20GB floor. Do NOT use `diskutil info` "available" or Finder figures (purgeable-inclusive via CacheDelete → over-report → can false-pass the floor). `df -g` floors to whole GB — irrelevant at a 20GB threshold. `df -k` if sub-GB precision ever matters.

## Consensus vs debate (external)
Consensus: page-on-actionable-only / page-on-transition (SRE, AWS, incident.io, rootly all align); df-excludes-purgeable (eclecticlight, daisydisk, mjtsai, Apple discussions all align); StartInterval sleep-skip vs StartCalendarInterval wake-queue (Apple + every secondary source). Debate: Addigy's "kickstart deprecated in 14.4" vs observed reality on Darwin 25.5 (kickstart present in `launchctl help` and in daily production use here) — resolved in favor of local evidence, with a bootstrap fallback as the hedge; no other source corroborates removal.

## Pitfalls (from literature + audit)
1. **Silent P1 swallow** — single fresh-process P1 call is a no-op (alerting.py:42 + :85-86). Use `ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1` env override; check the returned bool; curl-webhook fallback.
2. **Weekend false-stale** — the paper cycle is cron mon-fri (18:00 UTC); a naive `<26h` check goes red from ~Saturday 20:00 UTC until Monday's cycle, ~2 days of false alarm per week. The criterion mandates *verifying/recording* <26h, and mandates P1 only for failed restarts — so: record `cycle_age_h` + `cycle_fresh_26h` honestly, and gate any future *paging* on weekday logic (cycle_health.py:149-218 `is_weekday_et` is the in-repo precedent). Do not page on this field in 62.5.
3. **StartInterval sleep-skip** — missed fires are dropped, not queued (Apple). Mitigated by the backend's `caffeinate -i -s` (AC power only). Add `RunAtLoad=true` for reboot coverage.
4. **launchd PATH** — gcloud and gh live at `/opt/homebrew/bin`; launchd jobs get a minimal PATH. Plist must set PATH incl. `/opt/homebrew/bin` (copy the away-session plist's EnvironmentVariables, which also sets HOME — gcloud needs HOME to find ADC).
5. **kickstart vs booted-out** — `kickstart -k` exits 113 if the service was booted out; needs the bootstrap fallback (A5). Before/after logging makes the failure mode auditable either way.
6. **Re-page fatigue** — fresh process each fire = no in-memory dedup; suppression must be replayed from health.jsonl (A4). 83% of on-call engineers admit ignoring alerts (2026 reliability report via rootly) — one crisp P1 beats 16/day.
7. **Partial-line JSONL reads** — tail line may be mid-write; per-line try/except (kill_switch.py:66-70 idiom).
8. **Don't echo secrets** — the healthcheck must not print the webhook URL or tokens into health.jsonl/logs (backend_watchdog.sh:60-64 reads the webhook without sourcing .env — copy that).
9. **gcloud latency** — print-access-token does a network round-trip (~1-2s, can be slower); give it its own timeout (e.g. `gtimeout 30` — /opt/homebrew/bin/gtimeout is already used by run_away_session.sh) so a wedged gcloud doesn't hang the whole healthcheck.

## Application to pyfinagent (build map for GENERATE)
- `scripts/away_ops/healthcheck.sh`: checks in order — (1) `launchctl print gui/$UID/{backend,frontend,slack-bot}` exit codes; (2) `curl -m 5 127.0.0.1:8000/api/health` + jq `.status`; (3) `curl -m 5 -o /dev/null -w %{http_code} localhost:3000/login`; (4) kill-switch via `/api/paper-trading/kill-switch` (paper_trading.py:480) with kill_switch_audit.jsonl replay fallback (kill_switch.py:61-77); (5) last `completed_at` in cycle_history.jsonl < 26h (skip started-rows, cycle_health.py:187); (6) `gcloud auth application-default print-access-token >/dev/null` under gtimeout; (7) `df -g $REPO` Available >= 20; (8) `gh auth status` exit code. Append ONE JSON line to `handoff/away_ops/health.jsonl` with all fields + restart/before/after + p1 bookkeeping. Frontend-only recovery: kickstart -k, re-probe, bootstrap fallback on 113. P1 on 2nd consecutive failed restart via `cd $REPO && ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1 .venv/bin/python -c "from backend.services.observability.alerting import raise_cron_alert_sync; ..."` (source=`away_watchdog`, error_type=`frontend_restart_failed`, severity=`P1`), bool-checked, curl-webhook fallback. Reuse run_away_session.sh idioms: `ts()`/`slog`-style logging, exit-0 discipline so launchd never throttles the agent.
- `com.pyfinagent.away-watchdog.plist`: Label com.pyfinagent.away-watchdog; ProgramArguments /bin/bash + script path; `StartInterval=1800`; `RunAtLoad=true`; NO KeepAlive (periodic job, not daemon); ProcessType Background; EnvironmentVariables PATH=/opt/homebrew/bin:... + HOME=/Users/ford; WorkingDirectory=$REPO; Std{Out,Err}Path → handoff/logs/away-watchdog.log (process log; distinct from the structured health.jsonl). Mirror backend-watchdog.plist + away-session-am.plist shapes.

## Risks & gotchas + GO/NO-GO
**GO.** Every integration point exists and was verified live today: /api/health 200, :3000/login 200, kill-switch endpoint 200 unauthenticated from localhost, launchctl exit codes (0/113) empirically pinned, gh auth 0, df 115GB, ADC file present, alert function signature + deduper semantics read in source. No blockers. The four things GENERATE must not get wrong: the P1 deduper trap (pitfall 1), the weekend-stale honesty (pitfall 2), frontend-only restart ownership (A1), and PATH/HOME in the plist (pitfall 4).

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7)
- [x] 10+ unique URLs total (~45)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not snippets) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] Internal exploration covered every relevant module (watchdog, plists x5, alerting, kill_switch, cycle_health, paper_trading, main, settings, slack tools, away-ops scripts, crontab)
- [x] Contradictions / consensus noted (kickstart deprecation claim resolved against local evidence)
- [x] All claims cited per-claim

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 38,
  "urls_collected": 45,
  "recency_scan_performed": true,
  "internal_files_inspected": 18,
  "report_md": "handoff/current/research_brief_62.5.md",
  "gate_passed": true
}
```
