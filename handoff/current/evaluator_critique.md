# Evaluator Critique — Step 70.5 (deposit-aware Starting-capital + cron reschedule)

**Evaluator:** fresh, independent Q/A via the Workflow structured-output path (Opus 4.8, `effort: max`, $0 Max
rail, stall-immune — run wf_b9b9d9af-220). Verdict transcribed VERBATIM by Main (no-self-eval guardrail).

**VERDICT: PASS** | violated_criteria: [] | live_capture_gate.satisfied: true | do_no_harm_ok: true

## Checks
- verification_command_exit: 0 | pytest_passed: true (4 passed) | tsc_clean: true | no_cap_read_regression: true
- Harness compliance 5/5 (research-before-contract, contract-before-generate mtime-proven, results present, log-last, no-verdict-shopping — first Q/A on 70.5).
- live_capture_gate: capture_viewed true; shows_20000_deposit_truth true; satisfied true.

## Q/A notes (verbatim excerpt)

LIVE-CAPTURE GATE (BINDING) SATISFIED: viewed captures_70.5/70.5-starting-capital-deposit.png BY SIGHT. STARTING
CAPITAL renders '$20 000' (= BQ paper_portfolio.starting_capital deposit truth), NOT the stale $10,000 config;
hint "Reflects deposits (Top up fund). Total P&L % is measured against this." live_check documents method
(skip-auth :3100, @playwright/mcp@0.0.76, operator :3000 untouched -> 302).
C1 MET: manage/page.tsx renders `portfolio?.starting_capital ?? manageSettings.paper_starting_capital ?? 10000`
(BQ-first, config fallback); portfolio destructured from usePaperTradingData(); refresh() updates live.
C2 MET: reschedule_paper_job reuses _add_scheduler_job -> add_job(hour=..., id='paper_trading_daily',
replace_existing=True) (NOT modify_job, GH#234), guarded by `_scheduler and _scheduler.get_job(...)` (never
CREATES on a PUT when off), fail-open. Wired into update_settings via `if 'paper_trading_hour' in updates`
(updates = body.model_dump(exclude_none=True) -> preserves hour=0/midnight since 0!=None). paper_trading_hour now
writable (SettingsUpdate Field(None,ge=0,le=23) + _FIELD_TO_ENV PAPER_TRADING_HOUR + FullSettings). 4 tests
mutation-resistant. Bootstrapping caveat (running :8000 on pre-70.5 code -> live activation needs a one-time
restart; logic proven by unit tests) honestly documented + pre-accepted.
C3 MET: _add_scheduler_job builds only the cron trigger (reads NO caps); no settings caching; fresh-per-cycle cap
reads untouched (diff-confirmed; guard #3 honored).
DO-NO-HARM: no risk threshold moved (hour is a scheduling knob); reschedule fail-open (double try/except); $0;
paper-only; historical_macro FROZEN; read-only capture (no PUT, no config mutated).

ANTI-RUBBER-STAMP (real, minor, non-blocking): (1) no test drives update_settings(body) end-to-end into reschedule
(both legs covered separately). (2) C2's next_run is proven via a mocked add_job(hour=18) assertion, not a live
APScheduler recompute, because :8000 runs pre-70.5 code (acceptable proxy given the pre-accepted bootstrapping).
(3) the new "Daily run hour" field is not visible in the capture (empty/below fold from the old backend), so the
capture evidences C1 only. VERDICT PASS. Phase-70 closes 6/6.

## Main's disposition (recorded; not a verdict edit)
- **FO-70.5-A:** add an end-to-end `update_settings(paper_trading_hour=…) -> reschedule` integration test + (after
  the next backend restart loads 70.5) a live capture of the "Daily run hour" field + a next_run assertion.
  Deferred (both legs unit-covered; live activation is the standard one-time restart).
