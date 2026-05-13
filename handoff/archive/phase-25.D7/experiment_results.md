---
step: phase-25.D7
cycle: 99
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.D7

## What was built/changed

Closed audit bucket 24.7 F-5 by adding a stale-data guard to
`preload_macro()`:

1. **`backend/backtest/cache.py`**:
   - NEW module-level constant `MACRO_MAX_AGE_DAYS = 35` (default FRED-monthly tolerance).
   - In `preload_macro()`, after the BQ query returns rows: compute the
     most-recent `date` across all series; if `(today - max_date).days > 35`,
     emit a WARNING log + return 0 + DO NOT populate `_macro_full`.
   - On fresh data, behavior is unchanged.

## Files changed

| File | Action |
|------|--------|
| `backend/backtest/cache.py` | MACRO_MAX_AGE_DAYS constant + staleness guard in preload_macro |
| `tests/verify_phase_25_D7.py` | NEW verifier (4 claims) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_D7.py

=== phase-25.D7 verification ===

[PASS] 1. macro_max_age_days_constant_35
        -> Found MACRO_MAX_AGE_DAYS=35
[PASS] 2. preload_macro_checks_max_age_days_35_before_caching
        -> compare=True refuse_msg=True
[PASS] 3. warning_log_emitted_on_stale_data_refuse_to_preload
        -> return=0 warning_records=1 cache_populated=False
[PASS] 4. fresh_data_caches_normally
        -> return=3 series_count=2

ALL 4 CLAIMS PASS
```

AST clean.

## Success criteria -> evidence

1. `preload_macro_checks_max_age_days_35_before_caching` -- Claims 1 + 2 + 4 PASS:
   constant present at 35, comparison + refuse-to-cache logic in place,
   fresh data still caches normally.
2. `warning_log_emitted_on_stale_data_refuse_to_preload` -- Claim 3 PASS:
   behavioral test with 40-day-old rows captures exactly 1 WARNING record
   AND confirms `_macro_full` is NOT populated.

## Out-of-scope / deferred

- Per-series staleness (some series like JOLTS publish quarterly; treating
  them uniformly with monthly series may be too strict). The criterion
  specifies a global 35-day default, not per-series.
- Auto-trigger of FRED re-ingestion when staleness detected: deferred to
  a future cycle that wires the alert into the meta-coordinator.

## References

- `handoff/current/research_brief.md`
- `backend/backtest/cache.py:19-26, 211-237` (constant + guard)
- `.claude/masterplan.json::25.D7`
