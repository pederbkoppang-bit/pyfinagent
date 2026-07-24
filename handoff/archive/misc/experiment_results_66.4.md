# Experiment results -- 66.4 Credential-expiry resilience (Cycle 69, 2026-07-07)

## What was built

1. **healthcheck.sh auth probe** (criterion 1): two layers -- `claude auth status`
   (local presence) + newest `session_*.json` scanned for `api_error_status: 401`
   (the API-delivered signal that actually fired for 17 days; the pre-vacation
   ECONNRESET days carried no 401, so network errors never open an auth incident --
   that IS the auth-vs-network distinction). New JSON-line fields
   `auth_ok`/`auth_detail`/`auth_p1`; `auth_ok=false` fails the run. Page-once-per-
   INCIDENT via `handoff/away_ops/auth_page_state.json` latch (deliberately NOT the
   existing tail-1 dedupe, which re-pages every other 30-min watchdog run). A 401
   session OLDER than the latch's `cleared_at` never re-opens the incident.
   `HEALTHCHECK_TEST_AUTH_P1=1` drill mode: real delivery, zero latch writes (62.5
   drill-isolation doctrine).
2. **run_away_session.sh** (criterion 2): rc!=0 + 401-in-JSON is now `AUTH-DEAD`
   (was generic "crash or limit" -- 34 times). First detection pages ONCE (bot-token
   path) and opens the latch; with the latch open, each slot runs a single 20s-capped
   `claude -p ping` probe instead of a full launch -- probe success clears the latch
   and the session proceeds (automatic recovery); failure logs `auth-dead-skip` and
   exits. `CLAUDE_BIN` made env-overridable (`AWAY_SESSION_CLAUDE_BIN`) for drills.
3. **Criterion 3 via the OR-arm**: pre-expiry warning documented INFEASIBLE with
   evidence (docs/runbooks/credential-expiry-monitoring.md): keychain item
   `Claude Code-credentials` exposes only the 8-hour ACCESS-token `expiresAt`;
   `~/.claude/.credentials.json` absent on this machine; refresh-token expiry
   unexposed and lifetime unpublished; the observed failure matched the #61912
   corruption class, which no timer predicts. Compensating controls: detection
   within 30 min + page-once + probe-based auto-recovery. Structural mitigation
   (`claude setup-token`, 1-year) filed as pending_tokens ask SETUP-TOKEN
   (operator decision, not auto-applied).

## Drill evidence (real paging chain, server-side read-back)

- Healthcheck drill: `HEALTHCHECK_TEST_AUTH_P1=1 bash scripts/away_ops/healthcheck.sh`
  -> `AUTH_P1_TEST_DELIVERY=true`, latch file untouched. Permalink:
  https://pyfinagent.slack.com/archives/C0ANTGNNK8D/p1783378608623569
- Wrapper run 1 (stub claude emitting the 401 envelope):
  `AUTH-DEAD paged (delivered=true); latch OPEN` + latch JSON
  `{"incident_open": true, "opened_at": "2026-07-06T22:57:56...", "paged": true}`.
  Permalink: https://pyfinagent.slack.com/archives/C0ANTGNNK8D/p1783378676110199
- Wrapper run 2 (same stub): `AUTH-DEAD latch active + probe still failing (rc=1) --
  session skipped (no full launch, no page; paged once at incident open)` ->
  `END session result=auth-dead-skip`. NO new Slack message (read-back confirms).
- Cleanup + no-re-page guard proven live: latch closed
  (`cleared_by: drill_cleanup_66.4`); a NORMAL healthcheck run then reported
  `auth_ok=true, auth_detail=ok, exit 0` despite three 401 session files on disk --
  all older than `cleared_at`.

## Verbatim verification output (immutable command)

```
$ bash scripts/away_ops/healthcheck.sh --help 2>/dev/null || grep -n 'auth' scripts/away_ops/healthcheck.sh | head -5
(healthcheck ignores argv and runs fully; exited 0 on the post-cleanup run above --
the || grep leg therefore did not execute)
```
`bash -n` clean on both scripts.

## Honest disclosures

- **A first drill run found a real bug in the new code**: both latch-write one-liners
  interpolated shell `true`/`false` into Python dict literals (`NameError: name
  'true' is not defined`) -- the run-1 page delivered but the latch didn't open,
  which would have meant page-per-session (not once). Fixed (`True`/`False`),
  re-drilled: latch opens correctly. The pre-fix drill page
  (p1783378639651029, 22:57:19Z) is the extra message in the channel.
- The healthcheck drill also surfaced a TRUE stale condition: the newest real
  session (PM 2026-07-06 20:00 UTC) predates the operator /login and carries a real
  401 -- exactly what the probe should flag. The drill-cleanup `cleared_at` guards
  it; the next healthy scheduled session supersedes it naturally.
- Drill artifacts in the real session namespace: session_pm_20260706T225718Z.json +
  session_pm_20260706T225755Z.json (stub 401 envelopes, texts labeled
  "[DRILL 66.4 stub]") and auth_probe_last.json. Retained (never delete evidence),
  disclosed here.
- Three total drill pages hit the channel (one labeled [DRILL 66.4], two AUTH-DEAD
  from wrapper runs 1-prefix and 1-postfix); operator present and informed.

## File list

scripts/away_ops/run_away_session.sh (401 branch + latch-probe gate + overridable
CLAUDE_BIN), scripts/away_ops/healthcheck.sh (auth probe + latch + drill mode + JSON
fields + ok gate), docs/runbooks/credential-expiry-monitoring.md (NEW),
handoff/away_ops/pending_tokens.json (SETUP-TOKEN ask), handoff/away_ops/
auth_page_state.json (NEW runtime state), drill session artifacts + handoff step
files. NO backend python, NO sentinel.sh, NO plists.
