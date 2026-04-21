# Experiment Results — phase-9 / 9.3 (weekly FRED refresh) — REMEDIATION v1

**Step:** 9.3 **Remediation cycle:** 1 **Date:** 2026-04-20

## What was done

1. Fresh researcher subagent: 6 sources in full; brief at `handoff/current/phase-9.3-research-brief.md`; gate passed.
2. Contract authored.
3. Re-verified immutable criterion on unchanged artifact + test suite.
4. No code changes.

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/slack_bot/jobs/weekly_fred_refresh.py').read())" && pytest tests/slack_bot/test_weekly_fred_refresh.py -q
...                                                                      [100%]
3 passed in 0.01s
(exit 0)
```

Tests: test_run_writes_via_injected_fns, test_idempotency_by_iso_week, test_no_live_fredapi.

## Artifact shape

- `weekly_fred_refresh.py` — 44 lines, one `run()`, `JOB_NAME="weekly_fred_refresh"`, `_DEFAULT_SERIES=[DGS10, DGS2, VIXCLS, DFF, UNRATE, CPIAUCSL]`.
- Idempotency via `IdempotencyKey.weekly(JOB_NAME, iso_year_week=...)`.
- Heartbeat with skipped-guard; DI (`fetch_fn`, `write_fn`, `store`, `iso_year_week`) for tests.

## Carry-forwards (out of 9.3 scope)

1. Hardcoded `_DEFAULT_SERIES` → config-injectable
2. In-memory `_GLOBAL_STORE` → BQ/Redis
3. Latest-vintage only → ALFRED for look-ahead-safe backtesting
4. Missing T10Y2Y + BAMLH0A0HYM2 from default universe
5. Swap fredapi v0.5.2 → pyfredapi or FedFred for auto-throttling at go-live

## Success criteria

| # | Criterion | Status |
|---|---|---|
| 1 | ast.parse OK | PASS |
| 2 | pytest 3/3 | PASS |
