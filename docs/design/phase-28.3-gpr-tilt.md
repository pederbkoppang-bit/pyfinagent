# phase-28.3 — Design: GPR-triggered energy-sector tilt

**Step:** phase-28.3 (Candidate Picker Expansion)
**Date:** 2026-05-17
**Effort:** S (one new helper + post-LLM hook + 4 settings fields + 1 dep)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

`backend/services/macro_regime.py` additions:

```python
async def _fetch_gpr_acts(cache_hours: int = 24, quantile: float = 0.90) -> Optional[dict]:
    """Fetch latest GPR-Acts. Returns {current, threshold, last_date, above_threshold, rolling_n, quantile}
    or None on any error."""

def _apply_gpr_tilt(parsed: MacroRegimeOutput, gpr_info: dict, sector_etfs_csv: str) -> MacroRegimeOutput:
    """Inject configured ETFs into sector_hints.overweight when above_threshold is True. Identity otherwise."""
```

Post-LLM hook inside `compute_macro_regime()`:

```python
if getattr(settings, "gpr_signal_enabled", False):
    try:
        gpr_info = await _fetch_gpr_acts(cache_hours=..., quantile=...)
        if gpr_info:
            parsed = _apply_gpr_tilt(parsed, gpr_info, settings.gpr_signal_sector_etfs)
            logger.info("GPR tilt: ...")
    except Exception as e:
        logger.warning("GPR tilt application failed (non-fatal): %s", e)
```

## Data source

Caldara-Iacoviello GPR-Acts (GPRA) — REALIZED geopolitical events (vs GPRT = threats). Monthly Excel from matteoiacoviello.com. License: CC-BY 4.0.

URL primary: `https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls`
URL fallback: `https://www.matteoiacoviello.com/gpr_files/data_gpr.xls`

Parser: `pandas.read_excel` requires `xlrd>=2.0.1` for legacy .xls format (added to backend/requirements.txt).

## Threshold methodology

Quantile-based, not absolute. Per Researcher: 90th-percentile of the rolling 60-month (5y) GPRA series. Avoids hard-coding a value that becomes stale as the baseline drifts.

Default `gpr_signal_quantile = 0.90`. Latest data shows current GPRA = 285.35, threshold = 184.93 → above_threshold = True (May 2026).

## Sector ETFs

Configurable via `gpr_signal_sector_etfs` (comma-separated). Default = `"XLE"` (energy SPDR ETF; already in screener's SECTOR_ETFS map). Multi-ETF supported: `"XLE,XOM,CVX,COP,OXY"` would inject all five (deduped, preserving order).

## Feature flag

`gpr_signal_enabled = False` by default. Production behavior unchanged.

## Test plan

1. Immutable verification (syntax + grep).
2. 3-file syntax check.
3. Settings defaults match (False, 0.9, 24, XLE).
4. Helpers importable.
5. `_apply_gpr_tilt` unit: above-threshold inject, below-threshold identity, multi-ETF, dedup.
6. Live `_fetch_gpr_acts()` returns real data (current=285.35, threshold=184.93, above_threshold=True).
7. Q/A pass.

## Mid-cycle dependency add

First live fetch failed: `Import xlrd failed`. Added `xlrd>=2.0.1` to `backend/requirements.txt`. Production rollout must `pip install -r backend/requirements.txt`.

## References

- `handoff/current/phase-28.3-research-brief.md`
- `handoff/current/contract.md`
- `handoff/current/experiment_results.md`
- `handoff/current/live_check_28.3.md`
- `docs/audits/phase-28.3-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[3]`
