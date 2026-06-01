# experiment_results -- phase-51.2: sector diversification (measure-first, NEGATIVE result)

**Step:** 51.2 | **Date:** 2026-06-01 | **$0 LLM** | no pip | **no live flag flip** | GENERATE complete

## Outcome in one line
Wired the sector-neutral lever so it is functional at rank time (was a silent no-op), then
MEASURED it on our own universe: **HARD sector-neutral HURTS long-only Sharpe (-0.166)** -> the
flag stays default-OFF. A rigorous negative result; "measure before fixing" prevented a regression.

## What was built / changed

| File | Change |
|------|--------|
| `backend/tools/screener.py` | **NEW `build_sector_map(tickers)`** -> {ticker: GICS sector} from the Wikipedia S&P 500 table (same source + UA as get_sp500_tickers; intl -> "" global-pool fallback). |
| `backend/services/autonomous_loop.py:369` | gated wiring: build + pass `sector_lookup` into `screen_universe` ONLY when `sector_neutral_momentum_enabled` or `multidim_momentum_enabled` is True. Flag OFF (default) -> `sector_lookup=None` -> BYTE-IDENTICAL to the prior call. Fixes the no-op (enrichment used to run AFTER ranking at :659). |
| `scripts/ablation/sector_neutral_replay.py` | **NEW** screener-level replay: production `rank_candidates` over 48 monthly rebalances (2022-2025), 503 tickers, comparing baseline / sector_neutral / vol_scaled top-N baskets by forward Sharpe + sector spread + turnover. $0. |
| `backend/tests/test_phase_51_2_sector_div.py` | **NEW** 4 tests: OFF byte-identity w/ vs w/o sector field; OFF basket tech-concentrated; ON spreads across sectors (no-op fixed); ON without sectors == OFF (documents the prior bug). |

## The measurement (criterion #2) -- evidence-based, not assumed
```
config            ann_Sharpe   avg_fwd_mo%  avg_sectors  avg_turnover
baseline               1.388         4.054         4.73         0.555
sector_neutral         1.223         2.666        10.00         0.638
vol_scaled             1.403         2.045         4.73         0.555
sector_neutral vs baseline: dSharpe=-0.166, dSectors=+5.27  -> KEEP? False
vol_scaled vs baseline: dSharpe=+0.015
```
HARD sector-neutral doubles breadth (4.73->10.0 GICS) but costs -0.166 Sharpe + ~1.4%/mo return + more turnover -- the Harvey et al. long-only caveat confirmed on OUR universe. Decision: do NOT enable; flag stays OFF.

## Research basis (gate PASSED -- two briefs, both preserved)
- `research_rotation_element2_verdict.md` (rotation gate, 8 sources): REDIRECT away from winner-take-all rotation (architecturally disconnected from live money; alt strategies lose money) -> breadth inside the working engine.
- `research_51_2_sector_div.md` (51.2 gate, 9 sources): the minimal wiring (sector_lookup at rank time); the CRITICAL long-only caveat (Harvey et al. -- the replay confirmed it); sector_neutral > multidim for breadth; the replay design.

## Verification command output (verbatim)

### Syntax (all modified files)
```
OK  backend/tools/screener.py
OK  backend/services/autonomous_loop.py
OK  scripts/ablation/sector_neutral_replay.py
OK  backend/tests/test_phase_51_2_sector_div.py
```

### pytest (phase-51.2 -- 4 tests)
```
$ python -m pytest backend/tests/test_phase_51_2_sector_div.py -q
....                                                                     [100%]
4 passed in 0.22s
```

### build_sector_map smoke (real Wikipedia)
```
AAPL= Information Technology | XOM= Energy | JPM= Financials | UNH= Health Care | SAP.DE(intl->global pool)= ''
```

### Replay -> handoff/current/live_check_51.2.md (full verbatim there)

## US byte-identity (the working engine untouched)
Flag default-OFF -> `_sector_lookup=None` -> `screen_universe(... sector_lookup=None ...)` == the prior call (no map build on the live path). `rank_candidates(sector_neutral=False)` ignores the sector field -- `test_flag_off_is_byte_identical_with_or_without_sector` asserts identical ranked order. No change to decide_trades / risk guards / sizing.

## Artifact shape
- `build_sector_map(tickers) -> {ticker: str}` (GICS sector; "" for non-S&P-500)
- replay verdict: per-config {ann_Sharpe, avg_fwd_mo%, avg_sectors, avg_turnover} + KEEP-boolean

## Scope honesty / next
- NEGATIVE result reported honestly: criterion #2 is satisfied by the MEASUREMENT + tradeoff, NOT by sector-neutral winning. The lever is wired (criterion #1) but OFF (criterion #3).
- A SOFT sector tilt/cap (vs hard replacement) is the only sector-div variant worth a future look; the wiring makes it measurable. vol-scaling (+0.015) deprioritized.
- The live near-term money lever is the now-LIVE multi-market universe (EU/KR add non-tech sectors WITHOUT neutralizing) + 51.1's resurrected overlays -- MEASURE Monday's first multi-market cycle (14:00 UTC).
