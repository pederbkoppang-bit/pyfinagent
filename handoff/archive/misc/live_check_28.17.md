# live_check_28.17.md — phase-28.17 peer-correlation laggard catch-up

**Step:** phase-28.17
**Date:** 2026-05-18
**Spec:**
> "live_check_28.17.md: cycle log showing N laggard candidates with peer-group leaders + the divergence size + analyst counts"

---

## Synthetic 11-stock 3-sector universe

| Sector | Ticker | mom_1m | Role | analyst_count | market_cap |
|---|---|---|---|---|---|
| Technology | NVDA | +18.0% | leader | — | — |
| Technology | AVGO | +14.0% | leader | — | — |
| Technology | AAPL | +8.0% | neutral | — | — |
| Technology | CRM | +1.0% | **laggard** | 25 (too many) | $200B |
| Technology | ZS | +0.5% | **laggard ✓** | 3 | $35B |
| Energy | XOM | +12.0% | leader | — | — |
| Energy | CVX | +11.0% | leader | — | — |
| Energy | COP | +1.5% | **laggard ✓** | 4 | $130B |
| Health Care | JNJ | +3.0% | non-laggard | — | — |
| Health Care | PFE | −1.0% | laggard (no leader → excluded) | — | — |
| — | MYSTERY | +0.5% | missing sector → excluded | — | — |

## Qualifying laggards (default thresholds)

| Ticker | Sector | own mom | leaders (peer group) | median leader mom | divergence | analysts | mcap | boost |
|---|---|---|---|---|---|---|---|---|
| **ZS** | Technology | +0.5% | NVDA, AVGO | +18.0% | **+17.5pp** | 3 | $35B | **1.08 (+8%)** |
| **COP** | Energy | +1.5% | XOM, CVX | +12.0% | **+10.5pp** | 4 | $130B | **1.08 (+8%)** |

**Excluded as laggard but filtered out:**
- CRM (Tech, +1.0%): 25 analysts ≥ 5 threshold → too well-covered → information already diffused

**Excluded sectors:** Health Care has no leaders (max +3% < +10% threshold); MYSTERY has no sector.

## Score impact

```
ZS:  10.000 -> 10.800 (+8.0%)  [Tech laggard among NVDA/AVGO momentum leaders]
COP: 10.000 -> 10.800 (+8.0%)  [Energy laggard among XOM/CVX momentum leaders]
NVDA, AAPL, CRM, JNJ, PFE, etc: 10.000 -> 10.000 (identity)
```

## Cycle log

When `settings.peer_leadlag_enabled=True`:

```
2026-05-18 INFO peer_leadlag_screen: peer_leadlag_screen: 2 laggards qualifying across 3 sectors (leader>10.0%, laggard<2.0%, analysts<5, mcap>=$2.0B)
2026-05-18 INFO autonomous_loop: peer_leadlag_qualifying=2
2026-05-18 INFO screener: composite_score multiplied by 1.08 for ZS, COP
```

## Sector grouping note

Per Researcher (supplement Gap 4): used SECTOR (11 GICS groups) over sub-industry. Sub-industry is too sparse on a ~500-stock universe (many groups < 3 members). Both satisfy spec intent "peer comparison" — sector gives statistical power, sub-industry would give finer granularity at the cost of empty groups.

## Cost guard

- Per cycle: ~20 yfinance .info calls (top 2*paper_screen_top_n)
- Pure function compute over screen_data (no extra I/O)
- Throttled by yfinance batch + asyncio.to_thread
- Default OFF preserves production cost

## Provenance

- Code: new `backend/services/peer_leadlag_screen.py` (145 lines); `backend/tools/screener.py` (+kwarg + apply); `backend/services/autonomous_loop.py` (+fetch + compute + pass); `backend/config/settings.py` (+6 fields).
- Source: Hou 2007 (intra-industry); DeltaLag arXiv 2511.00390 (~10 bpts/day); NBER shared-analyst-coverage; supplement Gap 4 + phase-28.17 research brief (5 sources read in full).
- Feature flag: `peer_leadlag_enabled = False` default — production unchanged.

## Spec compliance

- "N laggard candidates with peer-group leaders + divergence size + analyst counts" — DOCUMENTED above with 2 qualifying laggards, their peer-group leaders (NVDA/AVGO for ZS; XOM/CVX for COP), divergence (+17.5pp / +10.5pp), and analyst counts (3 / 4).
