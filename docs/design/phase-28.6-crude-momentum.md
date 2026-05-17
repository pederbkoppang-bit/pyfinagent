# phase-28.6 — Design: Crude-oil (CL=F) cross-asset trend signal

**Step:** phase-28.6 (Candidate Picker Expansion)
**Date:** 2026-05-17
**Effort:** S (one new helper + second post-LLM hook + 6 settings fields)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

`backend/services/macro_regime.py` additions:

```python
async def _fetch_crude_momentum(
    cache_hours: int = 24,
    window_days: int = 21,
    lookback_days: int = 252,
    zscore_threshold: float = 1.0,
) -> Optional[dict]:
    """Fetch WTI crude (CL=F) 1m momentum z-score. Returns
    {current_momentum, zscore, mean, std, threshold, last_date, above_threshold, n_observations, window_days, lookback_days}
    or None on any error."""
```

REUSES `_apply_gpr_tilt(parsed, crude_info, sector_etfs_csv)` (added in phase-28.3) — the helper is generic over `above_threshold` and works identically for crude_info or gpr_info.

Second post-LLM hook inside `compute_macro_regime()` (AFTER the GPR hook at line ~476):

```python
if getattr(settings, "crude_momentum_enabled", False):
    try:
        crude_info = await _fetch_crude_momentum(...)
        if crude_info:
            parsed = _apply_gpr_tilt(parsed, crude_info, settings.crude_momentum_sector_etfs)
            logger.info("Crude momentum tilt: ...")
    except Exception as e:
        logger.warning("Crude momentum tilt application failed (non-fatal): %s", e)
```

## Data source

yfinance ticker `CL=F` (WTI continuous front-month futures). Period: 1y. Interval: daily. Already a dep (yfinance>=0.2.40). Cached JSON for 24h.

Threshold: z-score > 1.0 of the trailing 21d cumulative percent change vs the rolling 252d distribution. Equivalent to ~84th percentile under a normal assumption; calibrated for the observed ~78% crude implied vol (per Researcher).

## Sector ETFs

Configurable via `crude_momentum_sector_etfs`. Default `"XLE"` (XOM + CVX = 39% of XLE). Multi-ETF supported.

## Feature flag

`crude_momentum_enabled = False` by default. Production behavior unchanged.

## Orthogonality with phase-28.3

Per Researcher: GPR-Acts and crude momentum are ORTHOGONAL. High-GPR/flat-oil and rising-oil/low-GPR both occur in history. Today we see the high-GPR / moderate-oil regime: GPR-Acts = 285.35 > threshold 184.93 (fires); crude z-score = +0.137 < 1.0 (does NOT fire). Both can fire independently. Their outputs deduplicate naturally in `_apply_gpr_tilt`'s order-preserving dedup.

## Test plan

1. Immutable verification (syntax + grep for CL=F/crude/brent/oil_trend).
2. Settings defaults: False, 21, 252, 1.0, 24, "XLE".
3. Helpers importable.
4. Phase-28.6 markers present.
5. Live `_fetch_crude_momentum()` returns dict with numeric fields.
6. `_apply_gpr_tilt` reuse: above=True injects, above=False identity.
7. Post-LLM hook ordering: GPR → crude → save.
8. Q/A pass.

All eight passed; see `docs/audits/phase-28.6-smoke-test-2026-05-17.md`.

## References

- `handoff/current/phase-28.6-research-brief.md`
- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.6.md`
- `docs/audits/phase-28.6-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[6]`
