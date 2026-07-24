# Research Brief — phase-66.4 Credential-expiry resilience

Status: internal audit COMPLETE; external research in progress
Tier: moderate (caller said SIMPLE-to-MODERATE)
Date: 2026-07-07 (session started 2026-07-06)

## Question
Outside-the-process auth resilience: (1) daily auth probe in
`scripts/away_ops/healthcheck.sh` distinguishing auth failure from network
error, ONE deduped P1 via the proven bot-token path; (2)
`run_away_session.sh` page-once-and-skip on 401; (3) pre-expiry warning
>=24h if the credential surface exposes expiry, else documented infeasibility.

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| scripts/away_ops/healthcheck.sh | 189 | 30-min watchdog probe + frontend restart + P1 | live (62.5/62.6) |
| scripts/away_ops/run_away_session.sh | 167 | 2x-daily claude -p wrapper | live (62.3/62.4) |
| ~/Library/LaunchAgents/com.pyfinagent.away-watchdog.plist | — | StartInterval 1800 + RunAtLoad; PATH incl. ~/.local/bin | live |
| handoff/away_ops/session_am_20260621T053010Z.json | 1 | canonical 401 failure artifact | evidence |
| handoff/away_ops/session.log | — | wrapper log; 06-21 shows rc=1 on 401 | evidence |
| .gitignore | :24,:72-73 | `*.log`, `handoff/logs/`, `handoff/*.log` | trap map |
| backend/services/observability/alerting.py | :46,:75-80 | python P1 path (separate from shell path) | 66.1-repaired |

### 1. healthcheck.sh end-to-end (file:line)
- Probes: launchd svc_state x3 (:33-35), api_health :37, frontend_http :38,
  kill-switch endpoint + audit-replay fallback :42-59, cycle freshness :62-79
  (recorded, never paged — weekend false-stale), disk :81, **adc_ok :82,
  gh_ok :83 — the natural slot for a new `claude_auth` probe** (same
  one-liner boolean shape).
- JSON line :182-186 now has **18 fields** (17 from 62.5 + `log_rotated`
  from 62.6): ts, ok, backend, frontend, slack_bot, api_health,
  frontend_http, kill_switch_paused, cycle_age_h, cycle_fresh_26h,
  disk_free_gb, adc_ok, gh_ok, restarts_performed, restart_failed,
  restart_note, p1_raised, log_rotated. New fields append here.
- P1 trigger :117-129: only fires for restart_failed; dedupe = `tail -1`
  health.jsonl replay (prev.restart_failed && !prev.p1_raised).
  **Defect to not copy: tail-1 replay re-pages every OTHER run during a
  sustained outage** (run N pages, N+1 suppressed, N+2 sees prev
  p1_raised=false → pages again) ≈ 1 page/hour, not once-per-incident.
  A daily auth P1 needs a different dedupe: scan health.jsonl backwards
  for last auth_p1 within 24h, or a latch/state file.
- P1 delivery :130-148: python `raise_cron_alert_sync` with
  `ALERT_CONSECUTIVE_FAILURE_THRESHOLD=1` env (:131), then **bot-token
  fallback :139-148**: grep `SLACK_BOT_TOKEN`/`SLACK_CHANNEL_ID` from
  backend/.env (python-free), curl chat.postMessage, default channel
  C0ANTGNNK8D, success test `grep -q '"ok":true'` :147.
- `HEALTHCHECK_TEST_P1=1` drill :119, :150-156: forces the branch, logs
  `P1-TEST delivery=` to healthcheck_err.log, echoes `P1_TEST_DELIVERY=`,
  and **does NOT set p1_raised in the JSON line** (drill isolation — test
  runs never poison the dedupe replay). An auth-probe drill mode should
  mirror this.
- "Daily" semantics: plist StartInterval 1800 → 48 runs/day (+RunAtLoad on
  boot). So "daily probe" = probe every 30 min, page-once-per-24h dedupe;
  OR gate the probe itself on a last-probe timestamp. StartInterval fires
  are skipped during sleep (memory: 62.5), Mac held awake on AC.
- Exit: `ok` aggregation :174-180 (auth failure should flip ok=false);
  exit 1 on not-ok :188 (launchd ignores rc).

### 2. run_away_session.sh + the 401 signature
- claude invocation :135-141 (`gtimeout -k 60 $CAP claude -p
  --dangerously-skip-permissions --model claude-opus-4-8 --max-turns N
  --output-format json < prompt > OUT_JSON`), rc case :144-149 — rc=1 on
  auth failure logs only `"claude exited rc=1 (crash or limit)"`
  (session.log 2026-06-21T05:34:38Z, verbatim) then **exits 0; nothing
  pages, next slot fires in 12h → the 34-session silent burn**.
- OUT_JSON path :128: `handoff/away_ops/session_${SESSION}_<ts>.json`.
- **Canonical 401 artifact** (session_am_20260621T053010Z.json:1):
  `"type":"result","subtype":"success","is_error":true,
  "api_error_status":401,"num_turns":1,"duration_api_ms":0,
  "result":"Failed to authenticate. API Error: 401 Invalid authentication
  credentials","total_cost_usd":0`. Note `subtype:"success"` — do NOT key
  on subtype. **Cheapest reliable detector: rc!=0 AND
  `grep -q '"api_error_status":401' "$OUT_JSON"`** (grep-able, python-free);
  result-string grep `Failed to authenticate` is the fallback for older
  shapes. duration_ms=265138 → a dead-auth session still burns ~4.4 min.
- Page-once state file: dirty-tree filter :97 excludes `handoff/audit/`,
  `handoff/away_ops/`, `handoff/logs/` → a state file under
  **handoff/away_ops/** (e.g. `auth_page_state.json`) will NOT trip the
  recovery prompt. git check-ignore: `handoff/away_ops/*.json[l]` is
  NOT gitignored (will appear untracked; away sweeps may commit it — fine,
  it is not secret); only `*.log` under it is ignored (.gitignore:24).
  **Do not name a state file `*.log`** (17.4 lesson) if it must be
  committed; conversely naming it `.log` keeps it out of git entirely.
- Existing wrapper spots to hook: rc case :144-149 (detect), cost block
  :152-163 (already parses OUT_JSON with python3 — a 401 parse can ride
  the same block), HALT-DEV precedent :66-78 for skip semantics.

### 3. Two P1 paths confirmed separate
- Shell path (healthcheck.sh :130-148): python raise_cron_alert_sync
  attempt + python-free bot-token curl fallback; dedupe = health.jsonl
  replay (its OWN state, not the python deduper).
- Python path (backend/services/observability/alerting.py): P1 in
  `_CRITICAL_SEVERITIES` :46 bypasses threshold AND repeat-window :75-80
  since 66.1 → **every P1 call pages; exactly-once must be caller-side**
  (66.1 memory). The in-memory deduper dies with each one-shot process
  anyway (62.5 memory) — so for healthcheck the ONLY real dedupe is the
  shell-side state replay. Drill isolation verified in 62.5 cycle
  ("drill isolation + dedupe purity verified") and re-confirmed at
  :150-156 (test runs write no p1_raised).

### 4. Credential surface on THIS machine (read-only, redacted)
- `claude --version` = 2.1.201.
- `claude auth status` (rc=0) prints JSON by default; `--json` accepted,
  same shape. Fields: loggedIn, authMethod ("claude.ai"), apiProvider
  ("firstParty"), email, orgId, orgName, subscriptionType ("max").
  **NO expiry field in auth status output.**
- `~/.claude/.credentials.json` **ABSENT** — storage is the macOS login
  keychain, service `Claude Code-credentials`, acct `ford` (created
  2026-04-16, mdat 2026-07-06T21:16:04Z = yesterday's restore).
- Keychain payload keys (values REDACTED): `claudeAiOauth.accessToken`
  (len 108), `.refreshToken` (len 108), **`.expiresAt` = 1783401364079 =
  2026-07-07T05:16:04Z** — exactly mdat+8h → `expiresAt` is the
  **ACCESS-token expiry (8h TTL, auto-refreshed)**, NOT the refresh-token
  lifetime (which remains unexposed/unpublished). `.scopes`
  = [user:file_upload, user:inference, user:mcp_servers, user:profile,
  user:sessions:claude_code], `.subscriptionType` = max,
  `.rateLimitTier` = default_claude_max_20x.
- **Criterion-3 verdict:** a ">=24h before expiry" warning on the REFRESH
  token is INFEASIBLE (no surface exposes it). Feasible proxies:
  (a) `expiresAt` staleness — if `now - expiresAt > ~16h` despite sessions
  having run, refresh is dead (during normal ops the 12h session cadence
  refreshes it; 8h TTL + 14.5h max gap between sessions means grace must
  be >~16h to avoid weekend/idle false positives); (b) keychain `mdat`
  age as last-successful-refresh timestamp. Both are lagging
  (post-failure) indicators, not pre-expiry warnings — document as such.
- `claude auth status` reported loggedIn:true while holding a possibly
  already-expired access token → treat it as a LOCAL credential-presence
  check, not proof of API life. The 17-day outage ran with a credential
  PRESENT but invalid (claude-code#61912 corruption signature, prior
  brief). **Probe design consequence: auth status alone catches
  logged-OUT; catching expired-but-present needs the newest
  session_*.json api_error_status:401 scan (zero-cost) or a metered live
  ping.** Distinguishing network error: curl to slack/api or the existing
  api_health probe failing SIMULTANEOUSLY indicates network, not auth.
- `claude setup-token --help` exists: "Set up a long-lived authentication
  token (requires Claude subscription)" — the 1-year recovery path
  (prior brief: higher precedence slot).
- Keychain read via `security find-generic-password -w` worked WITHOUT a
  prompt from this session; launchd context should match (same user login
  keychain) but a locked keychain can fail it — probe must fail-open
  (unknown, not page).

### 5. gitignore / dirty-tree trap summary
- `.gitignore:24` `*.log` (anywhere), `:72` `handoff/logs/`, `:73`
  `handoff/*.log`; `handoff/away_ops/session.log` confirmed ignored via
  :24. `health.jsonl` / `session_*.json` / any new `*.json` under
  away_ops: NOT ignored, but wrapper-excluded from dirty-tree (:97).
- Placement rule for 66.4 state: `handoff/away_ops/auth_page_state.json`
  (or a field appended to health.jsonl lines) — both dirty-tree-safe;
  jsonl-field wins for auditability (same replay idiom as p1_raised).

## External research

### Search queries run (3-variant discipline)
- Current-year: "claude code setup-token long-lived token auth status CLI 2026"
- Year-less canonical: "dead man's switch cron job monitoring heartbeat missed check-in"
- Last-2-year: "alert fatigue deduplication page once per incident state machine shell monitoring 2025"

### Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://code.claude.com/docs/en/authentication | 2026-07-07 | official doc | WebFetch full | macOS = encrypted Keychain; Linux = ~/.claude/.credentials.json 0600. Auth precedence (6 slots): cloud provider > ANTHROPIC_AUTH_TOKEN > ANTHROPIC_API_KEY > apiKeyHelper > **CLAUDE_CODE_OAUTH_TOKEN (setup-token)** > subscription OAuth from /login. `claude setup-token` = one-year OAuth token, "prints a token to the terminal. It does not save the token anywhere"; inference-scoped; requires subscription. `--bare` does NOT read CLAUDE_CODE_OAUTH_TOKEN. apiKeyHelper re-called "after 5 minutes or on HTTP 401". |
| https://code.claude.com/docs/en/cli-reference | 2026-07-07 | official doc | WebFetch full | **"claude auth status — Show authentication status as JSON. Use --text for human-readable output. Exits with code 0 if logged in, 1 if not."** That is the entire documented exit contract; result-JSON fields (is_error, api_error_status) are NOT documented — the session artifact is the authority. --max-turns "exits with an error when the limit is reached". |
| https://healthchecks.io/docs/monitoring_cron_jobs/ | 2026-07-07 | official doc | WebFetch full | Dead-man's-switch = page on MISSED SUCCESS (absence), not observed failure: "notifies you as soon as a ping does not arrive on time"; Grace Time buffers late runs before declaring down; shell idiom `curl -fsS -m 10 --retry 5` chained with `&&` after real work; repeat notifications (hourly/daily reminders while down) are an explicit account-level option. |
| https://oneuptime.com/blog/post/2026-01-30-alert-deduplication/view | 2026-07-07 | vendor eng blog (2026) | WebFetch full | Dedup = stable fingerprint per underlying issue + store {first_seen, last_seen, count}; page at count 1, re-notify only at escalating thresholds (1,5,10,50...); **"When the underlying issue is resolved, we need to remove the fingerprint from our store"** — i.e. clear the latch on recovery so the next incident pages again. Fixed vs sliding windows. |
| https://markaicode.com/errors/claude-code-authentication-failed-fix/ | 2026-07-07 | practitioner guide (2026) | WebFetch full | "OAuth tokens refresh automatically before they expire"; re-auth needed only on explicit logout, refresh-token revocation, or credential store "cleared or corrupted" (= the #61912 signature). Recovery: `claude auth login`, `claude setup-token` (one-year), or delete the keychain entry. No numeric token lifetimes anywhere. Claims service name "claude-code" — WRONG on this machine (live: `Claude Code-credentials`); live inspection wins. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://platform.claude.com/docs/en/manage-claude/authentication | official doc | platform-side auth, not CLI |
| https://github.com/AndyMik90/Auto-Claude/issues/1518 | community | third-party harness hit the same expiry class; corroborates setup-token-as-fix |
| https://clawhosters.com/docs/claude-setup-token | vendor | setup-token walkthrough, redundant w/ official |
| https://www.remoteopenclaw.com/blog/claude-authorization-code-guide | blog | OAuth flow internals, out of scope |
| https://developer.puter.com/tutorials/claude-oauth/ | tutorial | generic OAuth |
| https://lalatenduswain.medium.com/claude-api-authentication-in-2026-oauth-tokens-vs-api-keys-explained-12e8298bed3d | blog (2026) | overview only |
| https://crontap.com/blog/dead-man-switch-explained-for-developers | blog | pattern covered by healthchecks.io docs |
| https://updog.watch/learn/what-is-dead-mans-switch, https://appstatus.io/docs/heartbeats, https://drumbeats.io/heartbeat-monitoring, https://domain-monitor.io/blog/what-is-heartbeat-monitoring/ | vendor | same DMS pattern, redundant |
| https://oneuptime.com/blog/post/2026-03-02-how-to-monitor-cron-job-execution-and-alerting-on-ubuntu/view | vendor (2026) | Ubuntu-specific |
| https://cronradar.com/comparisons/cron-monitoring-best-practices | vendor (2026) | listicle |
| https://www.watchflow.io/blog/10-critical-cron-jobs-to-monitor/, https://dev.to/krissv/kubernetes-cronjobs-silently-fail-more-than-you-think-2nb9, https://github.com/Kriss-V/deadmancheck | community | adjacent ("jobs that run but do nothing" = exactly our rc=1-then-exit-0 class) |
| https://incident.io/blog/alert-fatigue-solutions-for-dev-ops-teams-in-2025-what-works | vendor (2025) | org-level, not shell-level |
| https://www.logicmonitor.com/blog/network-monitoring-avoid-alert-fatigue, https://oneuptime.com/blog/post/2026-02-20-monitoring-alerting-best-practices/view, https://www.paloaltonetworks.com/cyberpedia/how-to-reduce-security-alert-fatigue, https://aiopscommunity.com/glossary/event-de-duplication-engine/, https://mindfulchase.com/..., https://torq.io/blog/cybersecurity-alert-fatigue/, https://icinga.com/blog/alert-fatigue-monitoring/ | vendor/community | dedup principles covered by OneUptime read |

### Recency scan (2024-2026)
Performed (all three searches were 2025/2026-scoped or returned 2026 hits).
Findings: (1) `claude auth status` JSON-by-default + documented 0/1 exit
contract is CURRENT docs (v2.x, 2026) — older guides show a human-readable
format that no longer matches v2.1.201; (2) setup-token/CLAUDE_CODE_OAUTH_TOKEN
is the 2026-documented headless auth path and its precedence slot 5 outranks
keychain OAuth (slot 6); (3) 2026 dedup writing (OneUptime Jan-2026) converges
on fingerprint+latch+resolve-clears-latch, matching what 66.4 needs; (4) the
"jobs that run but do nothing" failure class (deadmancheck, 2026) is exactly
our 34-session burn. No finding supersedes the internal evidence.

### Key findings (external, per-claim cites)
1. Probe exit contract exists: `claude auth status` exits 0 logged-in / 1 not
   (code.claude.com/docs/en/cli-reference, accessed 2026-07-07). But nothing in
   the docs says it VALIDATES the token remotely — combined with the live
   observation (loggedIn:true while holding an expired access token, and 17
   days of loggedIn-with-dead-refresh), treat rc=0 as "credential present",
   not "credential valid". Auth-dead detection must also consume the newest
   session JSON's `api_error_status:401`.
2. Auth-vs-network distinction: `api_error_status:401` in the session result
   is an API-delivered verdict (auth); a network-dead session produces no such
   marker (and healthcheck's existing api_health/gh_ok/adc_ok probes plus the
   curl to slack.com would fail too). `auth status` itself is local — it
   cannot produce a network false-positive (authentication docs; live run).
3. Dead-man's-switch principle: the monitor must live OUTSIDE the monitored
   process and page on missed success/observed-dead-state, with a grace
   window; optional periodic reminder while down (healthchecks.io docs).
   66.4's healthcheck-scans-session-artifacts design is this pattern
   localized (no external SaaS, consistent with local-only deployment).
4. Page-once state machine: fingerprint ("claude_auth_failed") + latch with
   first_seen/count; page on healthy->failed transition; CLEAR the latch on
   failed->healthy so the next incident pages anew; optional re-notify at
   thresholds — for a multi-week away window a daily re-page (dedupe key =
   UTC date) is a defensible middle ground (OneUptime 2026 + healthchecks.io
   reminders).
5. Recovery path confirmed: `claude setup-token` mints a 1-year
   inference-scoped token, printed once, consumed via CLAUDE_CODE_OAUTH_TOKEN
   env (precedence slot 5 > keychain OAuth slot 6) — a pre-staged escape
   hatch usable in the launchd plist env; NOT read under `--bare`
   (authentication docs).

### Consensus vs debate
Consensus: DMS/absence-detection for scheduled jobs; page-once-per-incident
with resolve-clears-latch; auto-refresh OAuth dies only on logout/revoke/
corruption. Debate/gap: refresh-token lifetime is UNPUBLISHED everywhere
(official docs give no numbers — consistent with prior 66.x briefs); strict
once-per-incident vs periodic re-notify is a policy choice, not consensus.

### Pitfalls (from literature + evidence)
- Keying failure detection on `subtype` ("success" even when is_error=true)
  or on rc alone (rc=1 is any crash/limit) — use `api_error_status:401`.
- Trusting `auth status` rc=0 as liveness (local presence check only).
- tail-1 replay dedupe (existing healthcheck idiom) re-pages every other run
  in sustained failure — an auth incident lasting weeks needs a latch or
  date-keyed dedupe, not tail-1.
- expiresAt in keychain is the 8h ACCESS token — a ">=24h pre-expiry" warning
  keyed on it would be always-firing noise; refresh-token expiry is invisible
  → criterion 3 resolves to "documented infeasibility + staleness proxy".
- New state files named `*.log` are silently gitignored (17.4); `--bare`
  would sever the Max OAuth rail (66.1); drill mode must not write p1 state
  (62.5 drill-isolation precedent).

## Application to pyfinagent (file:line mapping)
1. healthcheck.sh probe: add `claude_auth` boolean beside adc_ok/gh_ok
   (:82-83) = `claude auth status` rc test (PATH already has ~/.local/bin via
   plist) PLUS newest `handoff/away_ops/session_*.json` scan for
   `"api_error_status":401` (ls -t | head -1; grep). Auth-fail sets ok=false
   (:174-180) and appends 2 fields to the JSON line (:182-186): e.g.
   `claude_auth`, `auth_p1_raised`.
2. Dedupe: page via the EXISTING delivery chain (:130-148 python-then-bot-
   token curl) but with its own latch — scan health.jsonl for last
   auth_p1_raised=true within 24h (date-keyed once-daily page) instead of
   tail-1; drill mode reuses HEALTHCHECK_TEST_P1 isolation (:150-156).
3. run_away_session.sh: in the rc-case/cost block (:144-163), detect
   `api_error_status:401` in $OUT_JSON; on first detection page once (write
   `handoff/away_ops/auth_page_state.json` latch — dirty-tree-safe per :97),
   on subsequent runs skip-with-slog while latched; clear latch when a
   session succeeds or auth probe turns healthy. Alternative skip: check
   latch BEFORE invoking claude to stop burning the 4.4-min dead call — but
   a cheap pre-flight `claude auth status` cannot see expired-but-present, so
   latch-based skip is the reliable form.
4. Pre-expiry warning (criterion 3): INFEASIBLE for the refresh token
   (unexposed, lifetime unpublished). Document infeasibility; optional
   staleness proxy: keychain `claudeAiOauth.expiresAt` older than ~16-24h =
   refresh loop dead (lagging indicator, not pre-expiry). Read must pipe
   `security ... -w` straight into a parser, never log the payload, and
   fail-open if the keychain is locked/denied.
5. Recovery doc for the runbook: `claude setup-token` -> CLAUDE_CODE_OAUTH_TOKEN
   (slot-5 precedence beats corrupt keychain slot-6); never `--bare`.

### Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5)
- [x] 10+ unique URLs total (29)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not snippets) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (healthcheck.sh,
      run_away_session.sh, plist, session artifacts, gitignore, keychain)
- [x] Contradictions noted (markaicode service-name error; subtype:success
      vs is_error; auth-status-rc-vs-validity)
- [x] All claims cited per-claim

## JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 24,
  "urls_collected": 29,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/research_brief_66.4.md",
  "gate_passed": true
}
```
