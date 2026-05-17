# Experiment Results — phase-28.6 — Crude-oil (CL=F) cross-asset trend signal

**Step ID:** phase-28.6
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified

| File | Change |
|---|---|
| `backend/services/macro_regime.py` | Added `_CRUDE_CACHE_DIR/PATH` constants with multi-line phase-28.6 rationale comment; new `async _fetch_crude_momentum()` helper (yfinance CL=F download → 1m percent change → z-score over rolling 252d); REUSES `_apply_gpr_tilt` (generic over `above_threshold` — renamed docstring to reflect dual-purpose use); added a SECOND post-LLM hook inside `compute_macro_regime()` AFTER the GPR hook. |
| `backend/config/settings.py` | Added 6 fields after the gpr block: `crude_momentum_enabled` (False), `crude_momentum_window_days` (21), `crude_momentum_lookback_days` (252), `crude_momentum_zscore_threshold` (1.0), `crude_momentum_cache_hours` (24), `crude_momentum_sector_etfs` ("XLE"). |

### Files NOT created

- No new module file (mirrors 28.3 — local to macro_regime.py).
- No new dep (yfinance>=0.2.40 already present).

---

## Verification — verbatim output

### 1. Immutable verification command

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/macro_regime.py').read()); print('syntax OK')" && grep -qE 'CL=F|crude|brent|oil_trend' backend/services/macro_regime.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. Settings defaults

```
crude_momentum_enabled = False
crude_momentum_window_days = 21
crude_momentum_lookback_days = 252
crude_momentum_zscore_threshold = 1.0
crude_momentum_cache_hours = 24
crude_momentum_sector_etfs = 'XLE'
PASS: defaults correct
```

**PASS.**

### 3. Live `_fetch_crude_momentum()` — real CL=F data

```
INFO backend.services.macro_regime: Crude momentum: current=6.685% zscore=+0.14 above=False

1m momentum: 0.0669 (+6.69%)
z-score: +0.137
rolling mean: 0.0477 std: 0.1394
threshold: 1.00
above_threshold: False
last_date: 2026-05-15 00:00:00
n_observations: 252
```

**REAL DATA:** WTI crude (CL=F) is up +6.69% over the trailing 21 trading days. The trailing 252-day distribution has mean +4.77% and std 13.94% (very volatile), giving z-score +0.137 — **just slightly above the mean, far below the 1.0 trigger threshold**. The picker would NOT inject XLE via this trigger today. This is the OPPOSITE outcome of phase-28.3 (where GPR-Acts is well above its threshold), showing the two triggers are appropriately calibrated and behave independently.

### 4. `_apply_gpr_tilt` reuse for crude_info

```
--- _apply_gpr_tilt reuse on crude_info ---
overweight before: ['XLK'] -> after: ['XLK']
```

Identity (above_threshold=False), as expected. The helper's generic-over-`above_threshold` design works as intended.

---

## Success criteria mapping

| Criterion (immutable) | Evidence | Result |
|---|---|---|
| `crude_oil_trend_signal_added_to_macro_regime` | `_fetch_crude_momentum` + second post-LLM hook in `compute_macro_regime` (gated on `settings.crude_momentum_enabled`) | PASS |
| `threshold_documented` | Settings field description cites 1.0 z-score = ~84th percentile under a normal assumption + comment in macro_regime.py explains the choice + research brief cites the 78% crude implied vol calibration | PASS |
| `fallback_when_yfinance_unavailable_does_not_break_cycle` | Helper wraps yfinance import + download in try/except; returns None on any failure; post-LLM hook wraps the call in try/except as well; identity behavior when None | PASS |
| `live_check_shows_oil_trend_value_and_resulting_sector_action` | live_check_28.6.md captures current=+6.69%, zscore=+0.137, threshold=1.0, above_threshold=False → overweight unchanged | PASS |

---

## Artifact shape

Post-edit macro_regime.py additions:

```python
# After _GPR_ROLLING_MONTHS:
_CRUDE_CACHE_DIR = _CACHE_DIR / "crude"
_CRUDE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CRUDE_CACHE_PATH = _CRUDE_CACHE_DIR / "crude_momentum.json"

# Before _apply_gpr_tilt (which is renamed to be generic):
async def _fetch_crude_momentum(cache_hours, window_days, lookback_days, zscore_threshold) -> Optional[dict]: ...

# Inside compute_macro_regime AFTER the GPR hook:
if getattr(settings, "crude_momentum_enabled", False):
    try:
        crude_info = await _fetch_crude_momentum(...)
        if crude_info:
            parsed = _apply_gpr_tilt(parsed, crude_info, settings.crude_momentum_sector_etfs)
            logger.info("Crude momentum tilt: zscore=... threshold=... above=... overweight=...")
    except Exception as e:
        logger.warning("Crude momentum tilt application failed (non-fatal): %s", e)
```

---

## Independence from phase-28.3

The two triggers are ORTHOGONAL (per Researcher): high-GPR/flat-oil and rising-oil/low-GPR both occur in history. Right now we see the high-GPR / moderate-oil regime: GPR-Acts is above its threshold (would inject XLE) but crude z-score is below (would NOT inject XLE). When `gpr_signal_enabled=True` AND `crude_momentum_enabled=True`, XLE would be injected by GPR alone today; if oil started to rally on top, both triggers would converge but the dedup in `_apply_gpr_tilt` prevents double-add.

---

## Next

Q/A pass via fresh `qa` subagent. On PASS: append Cycle 19, flip phase-28.6.
