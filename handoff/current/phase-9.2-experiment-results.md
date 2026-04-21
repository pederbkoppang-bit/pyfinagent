# Experiment Results — phase-9 / 9.2 (daily price refresh) — REMEDIATION v1

**Step:** 9.2 **Remediation cycle:** 1 **Date:** 2026-04-20

## What was done

1. Fresh researcher subagent spawned; 7 sources read in full via WebFetch; brief at `handoff/current/phase-9.2-research-brief.md`. Gate passed.
2. Contract authored at `handoff/current/phase-9.2-contract.md`.
3. Re-verified immutable criterion on the unchanged production artifact + test suite.
4. No code changes to `backend/slack_bot/jobs/daily_price_refresh.py` or `tests/slack_bot/test_daily_price_refresh.py`.

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/slack_bot/jobs/daily_price_refresh.py').read())" && pytest tests/slack_bot/test_daily_price_refresh.py -q
...                                                                      [100%]
3 passed in 0.01s
(exit 0)
```

Tests: test_run_writes_rows_via_injected_fns, test_idempotency_dedups_same_day, test_no_live_yfinance_call.

## Artifact shape

- `backend/slack_bot/jobs/daily_price_refresh.py` — 54 lines, 1 public `run()`, 2 private defaults (`_default_fetch`, `_default_write`), `JOB_NAME="daily_price_refresh"`.
- Idempotency via `IdempotencyKey.daily(JOB_NAME, day=...)`.
- Heartbeat via `heartbeat()` ctx mgr; state-snapshot pattern inherited from phase-9.1.
- Dependency injection: `fetch_fn`, `write_fn`, `store`, `day` all injectable for tests.

## Carry-forwards (deferred, from research brief — NOT in 9.2 scope)

1. Hardcoded 5-ticker universe → settings-driven (future hardening)
2. `date.today()` vs UTC day-rollover mismatch
3. In-memory `IdempotencyStore` → wire to BQ `job_heartbeat` for prod
4. Missing retry/backoff in fetch path
5. Production `write_fn` should use BQ MERGE (not INSERT)
6. yfinance ToS risk for go-live — consider Alpaca/Polygon at live-trading time

## Success criteria re-verified

| # | Criterion | Status |
|---|---|---|
| 1 | ast.parse OK | PASS |
| 2 | pytest 3/3 | PASS |
