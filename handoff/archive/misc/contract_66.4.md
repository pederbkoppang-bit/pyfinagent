# Contract -- 66.4 Credential-expiry resilience (goal-phase66-reactivation)

Step: 66.4 | Cycle 69 | 2026-07-07 | Operator present
Sequencing disclosure: 66.4 depends_on 66.0 (done). The 66.1->66.2 P0 chain is NOT
skipped -- 66.1's only open criterion is wall-clock-gated to the 18:00 UTC scheduled
cycle; 66.2 is untouched until 66.1 closes. 66.4 fills the gap on its independent
dependency edge.

## Research-gate summary

research_brief_66.4.md (tier moderate, gate_passed: true; 5 read-in-full / 29 URLs /
recency scan / 8 internal files). Load-bearing:
- 401 signature in session JSON: `subtype:"success"` BUT `is_error:true,
  api_error_status:401, total_cost_usd:0`, CLI rc=1. The wrapper's rc-case
  (run_away_session.sh:144-149) logged "crash or limit" and exited 0 -- 34 times.
  Detector: rc!=0 AND grep '"api_error_status":401' on the session JSON; never key
  on subtype.
- Criterion 3 infeasible AS SPECIFIED (OR-arm engages): keychain payload
  (`Claude Code-credentials`; ~/.claude/.credentials.json ABSENT on this machine)
  exposes only claudeAiOauth.expiresAt = the 8h ACCESS-token expiry; refresh-token
  expiry unexposed, lifetime unpublished (official auth doc). Only a lagging
  staleness proxy is possible -> document infeasibility + proxy + `claude setup-token`
  (1-year, env precedence slot 5) as the operator-decision mitigation.
- `claude auth status` (v2.1.201): JSON, exit 0/1, but LOCAL presence check only --
  cannot detect expired-but-present refresh (the actual 17-day mode). The probe must
  ALSO scan the newest session JSON for 401. Network-vs-auth distinction comes free
  (401 is API-delivered inside a session JSON; local status can't network-fail).
- healthcheck.sh: probe slot beside adc_ok/gh_ok (:82-83); page via existing chain
  (:130-148, python attempt -> bot-token curl, C0ANTGNNK8D); drill isolation
  precedent (:150-156); JSON line currently 18 fields (:182-186). Away-watchdog
  StartInterval=1800 -> "daily probe" semantics = probe every 30min, PAGE once per
  incident: do NOT copy the tail-1 dedupe (:117-129, re-pages every other run) --
  use a latch/state file.
- State file must live under wrapper-excluded paths and not match *.log:
  handoff/away_ops/auth_page_state.json (dirty-tree-safe per wrapper :97).

## Hypothesis

The 17-day silence needed three misses to line up: the session died at turn 1 (rc=1)
with the 401 only inside its JSON; the wrapper treated every rc!=0 as generic
"crash or limit"; and no monitor outside the session process probed auth at all. A
401-aware wrapper branch (page-once latch + probe-skip) plus a healthcheck auth probe
(auth incident = newest session 401; page once per incident) closes all three, at
zero cost when healthy.

## Immutable success criteria (verbatim from .claude/masterplan.json phase-66/66.4)

1. "healthcheck (or successor) runs a daily Claude auth probe that distinguishes auth
   failure (401/expired) from network error; first auth failure produces exactly ONE
   P1 page via the bot-token path with dedupe (drill evidence: forced/invalidated
   credential -> one page, permalink)"
2. "The away wrapper treats auth failure as page-once-and-skip: a forced-401 session
   shows the skip + page with no retry-forever burn (session.log excerpt)"
3. "A pre-expiry warning (credential age/expiry surface) fires >=24h before expiry,
   OR infeasibility is documented with evidence from the credential storage format"

Verification command (immutable):
bash scripts/away_ops/healthcheck.sh --help 2>/dev/null || grep -n 'auth' scripts/away_ops/healthcheck.sh | head -5

live_check: live_check_66.4.md with drill page permalink, forced-401 session.log
excerpt, and the pre-expiry feasibility verdict.

## Plan

1. healthcheck.sh: auth probe = (a) `claude auth status` rc/JSON (local layer) +
   (b) newest session_*.json 401 scan (API layer -- the real detector). New JSON
   fields auth_ok / auth_detail. Page-once-per-incident via
   handoff/away_ops/auth_page_state.json latch (incident opens on first detection,
   clears on healthy observation); HEALTHCHECK_TEST_AUTH_P1 drill mode following the
   62.5 drill-isolation convention.
2. run_away_session.sh: after the claude exit, rc!=0 + 401-grep -> AUTH-DEAD branch:
   page once (shared latch), log the mandated skip line. On session start with latch
   ACTIVE: one 10s-capped `claude -p ping` probe -- success clears the latch and the
   session proceeds; failure logs "AUTH-DEAD latch active -- session skipped" and
   exits 0 (no full-session burn, no page).
3. Criterion 3: infeasibility section (evidence: keychain payload keys, absent
   credentials file, unpublished lifetime) + staleness proxy note + setup-token
   recommendation in docs/runbooks/credential-expiry-monitoring.md. No auto-adoption
   of setup-token (operator decision -> pending_tokens ask).
4. Drills (real paging chain, labeled): (a) healthcheck HEALTHCHECK_TEST_AUTH_P1 ->
   one page + permalink; (b) wrapper with stub `claude` emitting the 401 envelope ->
   run 1: page + AUTH-DEAD lines; run 2: skip, no page (session.log excerpt).
5. Q/A -> log Cycle 69 -> flip 66.4 if all three criteria close (criterion 3 via the
   OR-arm).

## Scope boundaries

No backend python changes (66.1 owns in-process paging); no sentinel.sh changes
(66.3); no plist changes; no setup-token adoption (operator token); state file under
handoff/away_ops/ only.

## References

research_brief_66.4.md; run_away_session.sh:97/:144-149; healthcheck.sh:61-79/
:82-83/:117-156/:174-186; session_am_20260621T053010Z.json; code.claude.com
authentication + cli-reference; healthchecks.io cron-monitoring; OneUptime alert
dedup (2026-01); 62.5 drill doctrine.
