# phase-28.2 — Design: 12-quarter SUE stacking

**Step:** phase-28.2 (Candidate Picker Expansion)
**Date:** 2026-05-17
**Effort:** XS (one constant bump + one docstring + one settings sync + multi-line rationale comment)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

Module constant in `backend/services/pead_signal.py`:

```python
# BEFORE: _LOOKBACK_QUARTERS = 8
# AFTER:  _LOOKBACK_QUARTERS = 12  (+ 8-line phase-28.2 rationale comment)
```

Parallel setting in `backend/config/settings.py`:

```python
# BEFORE: pead_signal_lookback_quarters: int = Field(8, ...)
# AFTER:  pead_signal_lookback_quarters: int = Field(12, ...)
```

## Inputs / Outputs / Integration points

- **Inputs:** unchanged. `_trailing_mean_from_cache(ticker, exclude_quarter)` still takes the same args; just reads more cache files.
- **Outputs:** the returned `(mean, n_quarters)` now reflects up to 12 quarters of history instead of 8. `surprise_score = current_sentiment − trailing_mean` shifts modestly for any ticker with ≥9 cached quarters.
- **Integration points:** none changed. `compute_pead_signal_for_ticker` calls `_trailing_mean_from_cache` exactly as before.

## Weighting scheme

**Equal-weight (arithmetic mean) preserved.** Per Researcher recommendation backed by 5 sources read in full: the ScienceDirect 2025 mechanism is that *older lags GAIN importance as markets price news faster*, so EWMA would de-weight precisely the valuable observations. Every practitioner source (QuantConnect, Quantpedia) uses equal-weight.

## Test plan

1. Immutable verification command (grep + ast.parse).
2. Module constant `_LOOKBACK_QUARTERS == 12`.
3. Settings default `Settings().pead_signal_lookback_quarters == 12`.
4. Description grep for "rolling-12Q".
5. Phase-28.2 comment grep.
6. Synthetic-cache smoke: write 12 cache files in legacy format, read via `_trailing_mean_from_cache`, confirm n_quarters=12 + equal-weighted mean.
7. Q/A pass.

All seven passed; see `docs/audits/phase-28.2-smoke-test-2026-05-17.md`.

## Source rationale

- **ScienceDirect 2025** — "Beyond the last surprise: Reviving PEAD with ML and historical earnings" — documents Sharpe 0.34 → 0.63 (+85% lift) when stacking 12 quarters of SUE history vs latest-only.
- **Quantpedia** — confirms practitioner consensus on PEAD with multi-quarter aggregation.
- **QuantConnect SUE notebook** — equal-weight is the canonical implementation.

## Cache back-compat

Cache filename `pead_<TICKER>_<YYYY-MM-DD>.json` does NOT encode lookback depth. The new 12Q code reads the same files the old 8Q code wrote. No migration, no breakage. The synthetic smoke wrote 12 cache files using the legacy format and the new code read them correctly.

## Operator impact

This step CHANGES default behavior (unlike 28.5/28.1 which are feature-flagged OFF). However, the PEAD signal itself is still gated by `pead_signal_enabled=False`, so production picker is unchanged when that flag is OFF. When the flag IS on, surprise_score will shift modestly. Magnitude in the synthetic example: ~0.055 surprise_score delta for a moderately trending ticker.

## References

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md`
- `handoff/current/live_check_28.2.md`
- `handoff/current/phase-28.2-research-brief.md`
- `docs/audits/phase-28.2-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[2]`
