---
step: 25.D7
slug: preload-macro-max-age-guard
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.D7

## Step ID + masterplan reference

`25.D7` -- "preload_macro() max-age guard (35-day FRED-monthly default)"
(P2, harness_required, no dep).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`.

## Hypothesis

`preload_macro()` currently caches whatever rows BQ returns without
checking if the data is fresh. If FRED ingestion stalls (or someone
runs a backtest with a stale BQ snapshot), the macro factors flow
through the backtest silently. Adding a max-age check (35 days,
default FRED-monthly tolerance) makes the system fail-loud with a
WARNING log + refuse-to-cache instead of silently corrupting results.

## Success criteria (verbatim from masterplan.json)

> `preload_macro_checks_max_age_days_35_before_caching`
>
> `warning_log_emitted_on_stale_data_refuse_to_preload`

## Plan steps

1. **Add module-level constant** `MACRO_MAX_AGE_DAYS = 35` near the top
   of `cache.py`.
2. **In `preload_macro()`**: after fetching rows but before populating
   `_macro_full`, compute `max_date = max(r["date"] for r in rows)`.
   Compare with `today - MACRO_MAX_AGE_DAYS`.
   - If stale: `logger.warning("preload_macro: stale data ...")` + `return 0`.
   - If fresh: populate `_macro_full` and return total_rows (current behavior).
3. **Verifier** -- `tests/verify_phase_25_D7.py` with 4 claims:
   - Claim 1: source contains `MACRO_MAX_AGE_DAYS = 35`.
   - Claim 2: preload_macro contains a comparison of max date vs threshold.
   - Claim 3: behavioral -- patch `_bq_client.query` to return rows with
     `date = today - 40 days`; call preload_macro; assert returns 0 + WARNING
     log captured.
   - Claim 4: behavioral -- patch to return fresh rows (today - 5 days);
     assert preload_macro caches and returns > 0.

## Files

| File | Action |
|------|--------|
| `backend/backtest/cache.py` | Add MACRO_MAX_AGE_DAYS + max-age check |
| `tests/verify_phase_25_D7.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_D7.py
```

## Live-check

`Inject macro data with timestamp >35 days old; preload refuses with WARNING log`.
Will write `handoff/current/live_check_25.D7.md`.

## Risks + mitigations

- **Risk**: Backtests blocked when running before ingestion completes
  on a fresh deploy.
  **Mitigation**: 35-day window is generous; FRED-monthly series publish
  ~10-15 days after month end. Only stalls if ingestion is broken for
  weeks.

## References

- `handoff/current/research_brief.md`
- `backend/backtest/cache.py:184-228`
- `.claude/masterplan.json::25.D7`
