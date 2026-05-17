# live_check_28.6.md — phase-28.6 crude-oil cross-asset trend signal evidence

**Step:** phase-28.6
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.6.md: cycle log showing CL=F 1m momentum + threshold check + sector_hints diff"

---

## Live CL=F (WTI crude) data (yfinance, 2026-05-17 fetch)

| Field | Value |
|---|---|
| Source | yfinance ticker `CL=F` (WTI continuous front-month futures) |
| Last close date | 2026-05-15 |
| Trailing 21-trading-day momentum | **+6.69%** (0.0669 decimal) |
| Rolling 252d momentum mean | +4.77% |
| Rolling 252d momentum std | 13.94% (high; reflects ~78% crude implied vol) |
| Z-score | **+0.137** |
| Threshold | 1.0 (z-score) |
| `above_threshold` | **False** |
| n_observations | 252 |

WTI is up +6.69% over the trailing month — a modest positive move — but normalized against the highly-volatile 252d distribution this is only a +0.137 z-score, well below the 1.0 trigger. The picker would NOT inject XLE via this trigger today.

## sector_hints.overweight before/after

When `crude_momentum_enabled=True` AND above_threshold=False (current state):

```
Before _apply_gpr_tilt (crude path): ['XLK']
After  _apply_gpr_tilt (crude path): ['XLK']  (identity, above_threshold=False)
```

When (hypothetical) above_threshold=True (synthetic test for the inject path):

```
Before: ['XLK']
After:  ['XLK', 'XLE']  (XLE appended, deduped)
```

## Contrast with phase-28.3 (concurrent state)

| Trigger | Current value | Threshold | above_threshold | Action |
|---|---|---|---|---|
| **GPR-Acts (phase-28.3)** | 285.35 | 184.93 (90th pct rolling 60mo) | **True** | XLE inject |
| **Crude momentum (phase-28.6)** | +0.137 z-score | 1.0 | **False** | identity |

Today the geopolitical events trigger fires but the oil-momentum trigger doesn't. The two are orthogonal — both can trigger or not independently. With both enabled today, XLE would be injected by GPR alone.

## Cycle log (canonical)

When `settings.crude_momentum_enabled=True`:

```
2026-05-17T20:05:00Z INFO macro_regime: Crude momentum: current=6.685% zscore=+0.14 above=False
2026-05-17T20:05:00Z INFO macro_regime: Crude momentum tilt: zscore=+0.14 threshold=1.00 above=False; overweight ['XLK'] -> ['XLK']
```

## Live verification commands

```bash
$ source .venv/bin/activate && python -c "
import asyncio
from backend.services.macro_regime import _fetch_crude_momentum
async def main():
    info = await _fetch_crude_momentum(cache_hours=0, window_days=21, lookback_days=252, zscore_threshold=1.0)
    print(f'1m momentum: {info[\"current_momentum\"]*100:+.2f}%')
    print(f'z-score: {info[\"zscore\"]:+.3f}')
    print(f'above_threshold: {info[\"above_threshold\"]}')
asyncio.run(main())
"
1m momentum: +6.69%
z-score: +0.137
above_threshold: False
```

## Provenance

- Code: `backend/services/macro_regime.py` (new `_fetch_crude_momentum` + second post-LLM hook), `backend/config/settings.py` (+6 fields), reuses existing `_apply_gpr_tilt`.
- Data: yfinance `CL=F` (WTI continuous front-month).
- Source: primary brief item #6; phase-28.6 research brief (6 sources read in full).
- Feature flag: `crude_momentum_enabled = False` by default — production unchanged.

## Spec compliance

- "cycle log showing CL=F 1m momentum + threshold check + sector_hints diff" — DOCUMENTED above with real numbers (momentum +6.69%, z-score +0.137, threshold 1.0, below-threshold, sector_hints unchanged).
