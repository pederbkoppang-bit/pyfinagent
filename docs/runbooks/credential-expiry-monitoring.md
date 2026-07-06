# Claude credential expiry monitoring (phase-66.4)

Why this exists: one expired/corrupted Claude OAuth credential silently killed 34
consecutive scheduled away sessions AND the cc_rail trading rail for 17 days
(2026-06-20..07-06) with zero pages. Phase-66.1 fixed in-cycle paging; this layer
lives OUTSIDE the dying process (healthchecks.io/Cronitor dead-man doctrine: the
monitor must not share the monitored process's failure domain).

## Detection layers

1. **healthcheck.sh auth probe** (away-watchdog, every 30 min):
   - `claude auth status` (local credential presence -- catches logged-out/removed).
   - Newest `handoff/away_ops/session_*.json` scanned for `api_error_status: 401` --
     the API-delivered signal that actually fired in 2026-06. Network errors
     (ECONNRESET) carry no 401 and never open an auth incident: this IS the
     auth-vs-network distinction (criterion 1).
   - JSON line fields: `auth_ok`, `auth_detail`, `auth_p1`; `auth_ok=false` fails
     the run (exit 1).
2. **run_away_session.sh 401 branch**: a session dying rc!=0 with 401 in its JSON is
   logged `AUTH-DEAD` (not "crash or limit"), pages ONCE, and opens the latch.

## Page-once-per-incident latch

State: `handoff/away_ops/auth_page_state.json` (`incident_open`, `opened_at`,
`cleared_at`, `paged`, source). First detector to see the death pages once (bot-token
chat.postMessage, C0ANTGNNK8D) and opens the incident; while open, nobody re-pages.
While the latch is open, each session slot runs a single 20s-capped `claude -p ping`
probe instead of a full launch: success clears the latch and the session proceeds
(automatic recovery); failure logs `auth-dead-skip` and exits. A healthy healthcheck
observation also closes the incident. A 401 session OLDER than `cleared_at` never
re-opens the incident (no re-page after recovery before the next session runs).

Drill modes: `HEALTHCHECK_TEST_AUTH_P1=1 bash scripts/away_ops/healthcheck.sh` sends
a real `[DRILL 66.4]` page, echoes `AUTH_P1_TEST_DELIVERY=...`, and touches NO latch
state (62.5 drill-isolation doctrine). The wrapper branch is drilled with a stub
`claude` binary emitting the 401 envelope.

## Pre-expiry warning: INFEASIBLE as specified (criterion 3 OR-arm)

Evidence (research_brief_66.4.md, inspected read-only on this machine 2026-07-07):
- Credentials live in the macOS keychain item `Claude Code-credentials`;
  `~/.claude/.credentials.json` is ABSENT on this machine.
- The keychain payload exposes only `claudeAiOauth.expiresAt` = the **8-hour
  ACCESS-token** expiry (observed exactly mdat+8h). Access tokens rotate routinely;
  their expiry says nothing about credential health >=24h out.
- The **refresh token's** expiry -- the thing that actually dies -- is NOT exposed in
  any credential surface, and its lifetime is UNPUBLISHED (official authentication
  doc carries no lifetime; the 2026-06 failure matched the claude-code#61912
  corruption class, which no timer predicts).
=> A >=24h pre-expiry warning cannot be built from the credential storage format.
The compensating controls are: detection-within-30-minutes (probe above) +
page-once + automatic probe-based recovery.

Optional lagging proxy (NOT a pre-expiry warning, documented for completeness): alert
when the keychain item's modification date exceeds ~24h (the 8h access-token rotation
implies a healthy rail touches it several times daily). Not implemented -- it
duplicates the 401 detector with more false-positive surface (idle weekends).

## Structural mitigation (operator decision, NOT auto-applied)

`claude setup-token` mints a **1-year** credential delivered via
`CLAUDE_CODE_OAUTH_TOKEN` (env precedence slot 5, above the keychain OAuth slot) --
it would have survived the entire away window. Adopting it changes the credential
model for every headless surface (sessions + cc_rail), so it is an operator decision:
run `claude setup-token` interactively, store per its instructions, and record the
decision. Filed as a pending_tokens ask (SETUP-TOKEN).

## Recovery runbook (on an auth P1)

1. On the host: `claude auth status`; if dead, `claude /login` (interactive).
2. Verify: `python -c "from backend.agents.claude_code_client import
   claude_code_health_probe; print(claude_code_health_probe())"` -> `(True, 'ok')`.
3. Nothing else to do: the next session slot's probe clears the latch automatically;
   the cc_rail guard resets per cycle (66.1).
