# Sprint Contract — phase-9 / 9.9 (scheduler wiring) — REMEDIATION v1

**Step id:** 9.9 **Remediation cycle:** 1 **Date:** 2026-04-20 **Tier:** moderate

## Why remediation

Previous cycle inline-authored. Fresh MAS re-run with depth on production-runtime correctness.

## Research-gate summary

Fresh researcher (moderate tier, 10 internal files inspected): `handoff/current/phase-9.9-research-brief.md` — 8 sources in full, 19 URLs, three-variant, recency, gate_passed=true.

Validated:
- `register_phase9_jobs` structure + fail-open wrapper is correct
- APScheduler defaults (coalesce=True, max_instances=1, misfire_grace_time=1s) acceptable for nightly/weekly cadence
- `replace_existing=True` safe for reload semantics

## Critical runtime bugs disclosed (CARRY-FORWARD — must fix in hardening)

**Important note:** the immutable criterion (ast.parse + pytest 4/4) PASSES because the tests only verify registration, not invocation. The researcher's deep audit surfaced two production-runtime bugs that would only manifest when the scheduler actually fires jobs.

1. **CRITICAL — `cost_budget_watcher.run()` TypeError on every scheduled invocation.**
   - Signature: `run(*, daily_spend_usd: float, monthly_spend_usd: float, ...)` — required keyword-only params with no defaults
   - APScheduler invokes with zero args (trigger-config `{"hour": 6}` is consumed by the trigger, never forwarded to the callable)
   - Every fire raises `TypeError`; caught by fail-open wrapper; job silently does nothing
   - **Fix:** either add defaults + side-channel fetch, OR pass `args=(...)` / `kwargs={...}` explicitly in `register_phase9_jobs`
2. **MEDIUM — `weekly_data_integrity.run()` functionally inert in production.**
   - Called with no `current_counts`/`prior_counts` → both default to `{}` → `_compute_drifts` returns `[]` every run
   - Heartbeat marks "ok"; drift alerts never fire
   - **Fix:** wire to BQ `INFORMATION_SCHEMA.PARTITIONS` query for `current_counts`; persist prior week's counts to BQ

The remaining 5 jobs (daily_price_refresh, weekly_fred_refresh, nightly_mda_retrain, hourly_signal_warmup, nightly_outcome_rebuild) are safe: all params are keyword-only with defaults and pull context via side channels (settings, `_default_*` helpers) at invocation time.

Additional deferred items:
3. No explicit UTC timezone on APScheduler — low-severity gap; consider `scheduler.configure(timezone="UTC")` at init
4. Single-process scheduler in Slack bot; multi-worker scenarios need Redis lock or dedicated scheduler service

## Immutable criterion

`python -c "import ast; ast.parse(open('backend/slack_bot/scheduler.py').read())" && pytest tests/slack_bot/test_scheduler_phase9.py -q`
Expected: exit 0, 4/4 pass.

## Plan

1. Re-verify.
2. Capture output.
3. Spawn fresh Q/A (expect PASS with explicit CRITICAL carry-forwards listed).
4. Log and confirm — carry-forwards #1 and #2 must be filed as follow-up tickets before go-live.

## References

- `handoff/current/phase-9.9-research-brief.md` (8 sources in full)
- `backend/slack_bot/scheduler.py` lines 328-378 (unchanged)
- `tests/slack_bot/test_scheduler_phase9.py` (4 tests)
- Carry-forwards from phase-9.7 Q/A (scheduler.py:374 empty-dict) and phase-9.8 Q/A (monthly idempotency) — now fully contextualized
- `.claude/masterplan.json` → phase-9 / 9.9
