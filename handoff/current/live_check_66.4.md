# live_check 66.4 -- Credential-expiry resilience (2026-07-07)

Required shape: "live_check_66.4.md with drill page permalink, forced-401 session.log
excerpt, and the pre-expiry feasibility verdict."

## 1. Drill page permalinks (criterion 1) -- real bot-token deliveries, read back server-side

| # | What | ts | Permalink |
|---|---|---|---|
| 1 | healthcheck `[DRILL 66.4]` forced auth-probe page (`AUTH_P1_TEST_DELIVERY=true`; latch untouched -- drill isolation) | 1783378608.623569 | https://pyfinagent.slack.com/archives/C0ANTGNNK8D/p1783378608623569 |
| 2 | wrapper AUTH-DEAD page, pre-latch-fix run (page delivered, latch write failed -- the bug the drill caught) | 1783378639.651029 | https://pyfinagent.slack.com/archives/C0ANTGNNK8D/p1783378639651029 |
| 3 | wrapper AUTH-DEAD page, post-fix run 1 (latch opened correctly) | 1783378676.110199 | https://pyfinagent.slack.com/archives/C0ANTGNNK8D/p1783378676110199 |

Auth-vs-network distinction: the incident detector keys exclusively on
`api_error_status: 401` inside the session JSON (API-delivered). The 2026-06-15..19
ECONNRESET sessions carry no such field -- network errors never open an auth incident.

## 2. Forced-401 session.log excerpt (criterion 2) -- verbatim

```
[2026-07-06T22:57:56Z] [pm] AUTH-DEAD -- 401 in session JSON (credential expired/corrupted)
[2026-07-06T22:57:56Z] [pm] AUTH-DEAD paged (delivered=true); latch OPEN -- subsequent sessions probe-and-skip
[2026-07-06T22:58:00Z] [pm] AUTH-DEAD latch active -- probing credential before launch
[2026-07-06T22:58:00Z] [pm] AUTH-DEAD latch active + probe still failing (rc=1) -- session skipped (no full launch, no page; paged once at incident open)
[2026-07-06T22:58:00Z] [pm] END session result=auth-dead-skip
```

Latch after run 1:
```json
{"incident_open": true, "opened_at": "2026-07-06T22:57:56.242654+00:00", "detail": "401 in session_pm_20260706T225755Z.json", "paged": true}
```
Run 2 sent NO Slack message (channel read-back: exactly the three messages above).
No retry-forever burn: run 2 cost one 20s-capped probe, zero full launches.

## 3. No-re-page guard proven live

After drill cleanup (`incident_open: false, cleared_at: 2026-07-06T22:58:16Z`), a
NORMAL healthcheck run with three 401 session files still on disk (all mtimes <
cleared_at) reported:
```json
{"ok": true, "auth_ok": "true", "auth_detail": "ok", "auth_p1": false}
```
exit 0, no page -- stale/drill 401s cannot re-open a cleared incident.

## 4. Pre-expiry feasibility verdict (criterion 3, OR-arm)

INFEASIBLE as specified, with storage-format evidence
(docs/runbooks/credential-expiry-monitoring.md + research_brief_66.4.md):
- Credential lives in macOS keychain item `Claude Code-credentials`;
  `~/.claude/.credentials.json` ABSENT on this machine (checked read-only).
- The payload's only expiry field is `claudeAiOauth.expiresAt` = the 8-hour
  ACCESS-token expiry (observed exactly keychain-mdat + 8h). It rotates routinely and
  predicts nothing >=24h out.
- The refresh token -- the credential that actually died -- exposes no expiry
  anywhere, its lifetime is unpublished (official authentication doc), and the
  observed 2026-06 failure matched the claude-code#61912 CORRUPTION class, which no
  timer can predict.
Compensating controls shipped instead: detection within one watchdog interval
(30 min), page-once-per-incident, probe-based automatic recovery. Structural
mitigation `claude setup-token` (1-year credential) filed as operator ask
SETUP-TOKEN in pending_tokens.json.
