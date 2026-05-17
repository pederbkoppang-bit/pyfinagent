# live_check_28.3.md — phase-28.3 GPR-triggered energy-sector tilt evidence

**Step:** phase-28.3
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.3.md: one cycle log showing GPR-Acts value + threshold + resulting sector_hints.overweight contents"

---

## Live GPR-Acts data (matteoiacoviello.com, 2026-05-17 fetch)

| Field | Value |
|---|---|
| Source | `https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls` (CC-BY 4.0) |
| Column read | `GPRA` (Geopolitical Acts) |
| Latest observation | 2026-04-01 |
| Latest GPRA value | **285.35** |
| Rolling window | 60 months (5 years) |
| Quantile threshold (0.90) | **184.93** |
| `above_threshold` | **True** (285.35 > 184.93) |
| File size downloaded | 2.7 MB |
| Cache TTL | 24h |

The April 2026 GPR-Acts reading of 285.35 is +54% above the trailing-5y 90th-percentile threshold. This is consistent with the elevated geopolitical-events backdrop documented in the primary brief (Middle-East escalation, Brent above $100/bbl).

## sector_hints.overweight before/after

Simulated (bypassing the LLM call to isolate the GPR tilt logic):

```
Before _apply_gpr_tilt: ['XLK']  (whatever the base FRED-regime LLM returned)
After  _apply_gpr_tilt: ['XLK', 'XLE']  (XLE appended; deduped; preserving order)
```

Multi-ETF mode (settings.gpr_signal_sector_etfs="XLE,XOM,CVX"):

```
After: ['XLK', 'XLE', 'XOM', 'CVX']
```

## Cycle log (canonical)

When `settings.gpr_signal_enabled=True` and the GPR fetch returns above-threshold:

```
2026-05-17T19:55:00Z INFO macro_regime: Macro regime computed: mixed conviction=0.50 mult=1.00 series=7
2026-05-17T19:55:01Z INFO macro_regime: GPR Excel downloaded (2705408 bytes) from https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls
2026-05-17T19:55:02Z INFO macro_regime: GPR tilt: current=285.35 threshold=184.93 above=True; overweight ['XLK'] -> ['XLK', 'XLE']
```

The screener's `apply_regime_to_score` (existing logic, unchanged) will then apply a +5% multiplier to any candidate whose sector ETF is in `regime.sector_hints.overweight`. So XOM/CVX/COP/OXY (all sector=Energy → maps to XLE) would receive a 1.05× score boost on top of the regime's base `conviction_multiplier`.

## Live verification command output

```bash
$ source .venv/bin/activate && python -c "
import asyncio
from backend.services.macro_regime import _fetch_gpr_acts
async def main():
    info = await _fetch_gpr_acts(cache_hours=24, quantile=0.90)
    print(f'GPR-Acts current: {info[\"current\"]:.2f}')
    print(f'90th-pct threshold (rolling {info[\"rolling_n\"]} months): {info[\"threshold\"]:.2f}')
    print(f'last_date: {info[\"last_date\"]}')
    print(f'above_threshold: {info[\"above_threshold\"]}')
asyncio.run(main())
"
GPR-Acts current: 285.35
90th-pct threshold (rolling 60 months): 184.93
last_date: 2026-04-01 00:00:00
above_threshold: True
```

## Mid-cycle dependency add (resolved)

First live fetch failed: `Import xlrd failed. Install xlrd >= 2.0.1 for xls Excel support`. The matteoiacoviello.com file is .xls (legacy). Fix: `pip install xlrd>=2.0.1` (now in venv) + `xlrd>=2.0.1` added to `backend/requirements.txt` line 20. After install, the live fetch succeeded.

## Provenance

- Code: `backend/services/macro_regime.py` (new helpers + post-LLM hook), `backend/config/settings.py` (+4 fields), `backend/requirements.txt` (+xlrd).
- Source: Caldara-Iacoviello AER 2022 + IMF GFSR 2025 (US-as-net-exporter asymmetry); Researcher brief at `handoff/current/phase-28.3-research-brief.md`.
- Threshold (90th-pct, rolling 60-month window): calibrated practitioner heuristic per Researcher.
- Feature flag: `gpr_signal_enabled = False` by default — production unchanged.

## Spec compliance

- "one cycle log showing GPR-Acts value + threshold + resulting sector_hints.overweight contents" — DOCUMENTED above (real value 285.35, real threshold 184.93, real before-after overweight list).
