# Evaluator Critique — phase-47.1: Restore historical_prices freshness

## Cycle-1 Q/A verdict (agent ad63febb6bb3b3c81) — CONDITIONAL

**`ok: false`, verdict: CONDITIONAL.** First Q/A on 47.1 (no prior CONDITIONALs; not subject to
3rd-CONDITIONAL auto-FAIL).

- Harness-compliance audit: **5/5 PASS** (researcher-before-contract; contract mtime < results;
  results present w/ verbatim output + disclosure; harness_log + status-flip correctly still pending;
  no verdict-shopping — first Q/A).
- verification.command edit: **ACCEPTABLE — not goalpost-moving.** Entire phase-47 block is new in
  the working tree (no prior committed criteria to weaken); the 5 success_criteria match contract.md
  verbatim and are shape-agnostic; only the mechanical command was corrected (dict-shape fix +
  venv note), both pre-Q/A false-negative fixes.
- Independent freshness curl (Q/A's own): `historical_prices band=green`, last_tick_age_sec=604
  (advanced from 207 — live, not frozen). Immutable command EXIT_CODE=0.
- All 5 immutable success_criteria independently MET.
- Code-review heuristics: no BLOCK (ingest/cron wiring touches no execution/kill-switch/stop-loss/
  perf-metrics path; broad-except are documented fail-open job contract, NOTE only).
- Scope honesty: HONEST (does not oversell freshness=green as "fixed trading"; defers jobstore).

**Sole WARN driving CONDITIONAL** (`financial-logic-without-behavioral-test`):
misfire_grace_time 3600 -> 21600 (scheduler.py) turned `test_register_phase9_grace_times_per_tier`
RED (`assert 21600 == 3600`) without updating it. A red guard masks the next grace-time-tier
regression. Required fix: update the test to lock the new intended value, re-run to green, append
fix to experiment_results + this file, spawn FRESH Q/A. (Also flagged a PRE-EXISTING red test,
`test_start_scheduler_source_calls_register_phase9_jobs`, red since phase-23.6 — recommended
opportunistic fix, does not count against 47.1.) Do NOT touch success_criteria.

Full verdict JSON archived in the cycle-1 Q/A agent transcript (ad63febb6bb3b3c81).

## Follow-up (Main, cycle-2)

Blocker fixed exactly as directed + full-consumer sweep (operator `feedback_full_codebase_audit_before_changes`):
1. `test_register_phase9_grace_times_per_tier` — daily_price_refresh now asserts **21600**
   (own assertion); other daily jobs stay 3600; weekly 7200; hourly 600. GREEN.
2. `test_start_scheduler_source_calls_register_phase9_jobs` — de-fragilized to match the
   `register_phase9_jobs(_scheduler` prefix (the phase-23.6 multi-arg call). GREEN.
3. **Additional consumers the targeted Q/A run did not surface** —
   `tests/slack_bot/test_phase9_production_wiring.py` encoded the old prod-fn wiring. Updated
   `test_register_without_app_returns_bare_run` + `test_register_with_app_uses_functools_partial`
   + module docstring: daily_price_refresh now asserts resolution to `run_production` (never a
   partial); `weekly_fred_refresh` kept as the bare-run/partial probe. GREEN.

Re-run (5 affected files): **30 passed in 7.13s.** No production code changed in cycle-2 (test
guards only); immutable command + all 5 success_criteria still MET. Handoff files updated
(experiment_results.md cycle-2 section + this file). A FRESH Q/A is being spawned on the updated
evidence per the documented cycle-2 flow (code/tests changed -> not verdict-shopping).
