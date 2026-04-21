# Experiment Results — phase-9 / 9.5 (hourly signal warmup) — REMEDIATION v1

**Step:** 9.5 **Remediation cycle:** 1 **Date:** 2026-04-20

## What was done

1. Fresh researcher: 5 sources in full; `handoff/current/phase-9.5-research-brief.md`; gate passed.
2. Contract authored.
3. Re-verified immutable criterion.
4. No code changes.

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/slack_bot/jobs/hourly_signal_warmup.py').read())" && pytest tests/slack_bot/test_hourly_signal_warmup.py -q
...                                                                      [100%]
3 passed in 0.02s
(exit 0)
```

Tests: test_warmup_populates_injected_cache, test_watchlist_from_settings_when_not_injected, test_cache_backend_is_injectable.

## Artifact shape

- `hourly_signal_warmup.py` — 48 lines, `run()` with DI (`watchlist`, `compute_signal_fn`, `cache_backend`, `store`, `iso_hour`).
- Hourly idempotency via `IdempotencyKey.hourly(JOB_NAME, iso_hour=...)`.
- Settings-fallback via `_load_watchlist()` (try/except → static default).

## Carry-forwards

1. TTL/`warmed_at` stamp on cache entries
2. Optional market-hours gating via `exchange_calendars`
3. Document Redis upgrade path
4. Watchlist sort by position-size desc

## Success criteria

| # | Criterion | Status |
|---|---|---|
| 1 | ast.parse OK | PASS |
| 2 | pytest 3/3 | PASS |
