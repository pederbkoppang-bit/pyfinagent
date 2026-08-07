# Contract — Step 85.3: unstick the away-ops auth alarm (latch freshness + reachable clear)

- **Step id:** 85.3 (P1, phase-85)
- **Tier (named field):** T3 — executor Main (Opus 5, effort max); Q/A via qa-verdict Workflow (opus/max).
- **Date:** 2026-08-07, autonomous drain, cycle 175

## Research-gate summary

`handoff/current/research_brief_85.3.md` — gate_passed: **true** (7 external sources read in full / 34 URLs / recency scan / 14 internal files). Decisive findings, two step-premises REFUTED with measurements:

1. **The close path ALREADY EXISTS and is unreachable** — `cleared_at` has a SECOND writer at healthcheck.sh:164 (`cleared_by: healthcheck_healthy`), but `if [auth_ok = false]` at :125 always wins the branch that precedes it: stale 401 → auth_ok=false → clear never runs → `cleared_at is None` at :119 keeps auth_ok=false. A closed self-sustaining cycle in ONE file. **The fix makes an existing path reachable, not a new mechanism.**
2. **The streak is 473 records over 9.85 days, not 28 days** (the step's "every record ok:false" is wrong — 34/509 are ok:true; the watchdog was itself dead 07-06..07-28). Auth IS the sole cause: re-deriving `ok` without the auth leg gives healthy for all 473.
3. **Criterion 9 answered (defect DD-1/D3)**: the away sessions never stopped — they run twice daily and skip on `incident_open`; the probe conflates "CLI exited nonzero" with "credential dead" (today's probe authenticated and billed 50K cache tokens, then exited rc=1 `error_max_turns` / rc=124 gtimeout; NEVER rc=0 in 28 days). Filed as its own step.
4. `claude auth status` rc=0 today — the stale 401 is the ONLY thing holding auth_ok=false; a freshness bound makes the live watchdog emit ok:true (replayed read-only against real state).
5. **Literature**: Nagios freshness (stale ⇒ force an ACTIVE check; explicit threshold, worked example 26h) + Yokogawa (a latch must require an explicit clear, never self-resolve) + Azure WAF (stale value = reliability issue; compare against a LIVE query) + AHRQ alarm-fatigue (72-99% false-alarm rates → desensitization; this instance is the degenerate 100%-for-28-days limit).
6. **Paging proven intact three independent ways**; the fix age-bounds only leg (a)'s INPUT — a fresh 401 is by definition inside the window.
7. **Six discovered defects DD-1..DD-6** disclosed; DD-5 (the clear arm also fires on auth_ok=='unknown' — a probe error could clear a REAL incident) is INSIDE the logic this step rewrites and is fixed in-scope; the rest are queued/disclosed per the standing rule.

## Immutable success criteria — all 9 copied verbatim into the step record (masterplan `verification.criteria`, C1-C9 as read 2026-08-07); command: `cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_85_3_auth_latch_freshness.py -q`

(Text as in `.claude/masterplan.json` 85.3 — C1 seam extraction + call-site quote; C2 tmp_path-only subprocess/import drive; C3 four fixture cases; C4 today's-exact-state regression at now-relative 27d; C5 close path without an away session; C6 fail-open on unreadable/malformed/missing state; C7 the freshness-bound mutation recorded verbatim; C8 real watchdog invocation quoted with ok:true; C9 the session-stop investigation recorded with a filed step id.)

## Explicit decisions

- **D1 — seam**: new `scripts/away_ops/auth_state.py`, identical two-token stdout contract, healthcheck.sh call site keeps the `|| echo "unknown probe_error"` fail-open wrapper verbatim. The latch TRANSITION moves into the seam behind `--apply` so C5 is test-drivable.
- **D2 — freshness window 36h (129600s)**, derivation in the code comment: sessions start 07:30/22:00 local; worst-case inter-start gap 14.5h + 4h session cap ≈ 18.5h normal mtime gap; 36h ≈ 2× tolerates one fully missed slot without flapping. The 26h Nagios-example alternative is recorded and rejected for its ~7.5h headroom.
- **D3 — latch semantics kept** (never self-resolving; Yokogawa): the fix adds (L1) the evidence max-age on the 401 INPUT, (L2) the active `claude auth status` re-check leg as load-bearing (mutation M5), and makes the EXISTING :154-166 clear arm reachable; **DD-5 fixed in-scope** — the clear fires only on auth_ok=='true', never 'unknown'.
- **D4 — paging untouched**: legs (b) `auth_status_rc_nonzero` and (c) run_away_session's independent page are not modified; leg (a) is age-bounded on input only. The sanctioned `HEALTHCHECK_TEST_AUTH_P1=1` drill proves delivery in the live_check.
- **D5 — discovered defects**: DD-1 (probe conflates exit-code with credential death) → **queued as 85.3.1**; DD-2 (health.jsonl invalid-JSON line + scheduler bare-except silently killing the digest health section) + DD-3 (formatters/healthcheck key mismatch `last_cycle_age_h` vs `cycle_age_h`) → **queued as 85.3.2**; DD-4 (**SECURITY**: away-watchdog.plist embeds a literal CLAUDE_CODE_OAUTH_TOKEN) → **queued as 85.3.3 (P1) + ask-list item**; DD-6 (runbook line falsely promises auto-clear) → docs touch in-scope, disclosed.
- **D6 — do-no-harm watch**: health.jsonl keeps being appended every 30 min (rotate_logs.sh's second watchdog pages on 2h staleness); no early-exit before the :266-270 printf.
- **D7 — live side effect stated**: clearing the latch also RESUMES the twice-daily away sessions (run_away_session.sh:143 gates on incident_open) — intended production effect; DD-1 stays armed until 85.3.1, meaning the next real 401 would re-create this outage — that is 85.3.1's urgency, recorded not hidden.

## Plan

1. Write `scripts/away_ops/auth_state.py` (derivation + freshness + latch transition + fail-open); swap the healthcheck.sh heredoc for the seam call (quote both in the handoff).
2. Test file per C2-C7: tmp_path fixtures (≥3 session files with distinct explicit mtimes via os.utime; both JSON spellings; the C4 regression built now-relative with injected --now; state-file variants incl. chmod-000/malformed/missing; DD-5 case: probe-unknown must NOT clear an open incident).
3. Mutation matrix M1-M7 (run for real, record verbatim, restore hash-verified).
4. live_check: quiet proof (direct run + one naturally-scheduled tick), latch-clear proof (auth_page_state.json after), the sanctioned paging drill, the session-resumption log line, launchctl status flip.
5. Queue 85.3.1/85.3.2/85.3.3 + ask-list entry for DD-4; runbook line fix.
6. experiment_results → qa-verdict → transcribe → harness_log → flip. Re-derive every fenced measurement after the final edit.

## References

`research_brief_85.3.md` (Nagios freshness docs, Yokogawa latch semantics, Azure WAF stale-data, AHRQ alarm fatigue, Google SRE alerting philosophy, OneUptime 2026 — URLs + access dates therein; healthcheck.sh/run_away_session.sh line-anchored audit).
