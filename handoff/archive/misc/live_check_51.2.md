# live_check -- phase-51.2: sector diversification (measure-first, NEGATIVE result)

**Step:** 51.2 | **Date:** 2026-06-01 | **Result shape:** a $0 screener-level replay
comparing sector-neutral (and vol-scaled) ranking ON vs OFF on OUR S&P 500 universe ->
the SIGN of the tradeoff. **Outcome: sector-neutral HURTS; flag stays OFF (no live change).**

**Command:**
```
source .venv/bin/activate && PYTHONPATH=. python scripts/ablation/sector_neutral_replay.py
```
48 monthly rebalances (2022-2025), 503 S&P 500 tickers, top_n=10, equal-weight forward-1mo
return. Reuses the PRODUCTION `screener.rank_candidates` (only the sector-neutral regroup
differs between configs). $0: free yfinance prices (one batch download) + Wikipedia GICS
sectors (one read_html). No LLM, no BQ writes, no live flag flip.

## Verbatim result

```
[1/4] loading S&P 500 tickers + GICS sectors from Wikipedia ...
      503 tickers; sectors: 11 distinct
[2/4] batch-downloading prices 2021-06-01..2025-12-31 (one yfinance call) ...
      usable price series: 503 tickers x 1152 days
[3/4] replaying 48 monthly rebalances, top_n=10 ...
[4/4] results

config            ann_Sharpe   avg_fwd_mo%  avg_sectors  avg_turnover
---------------------------------------------------------------------
baseline               1.388         4.054         4.73         0.555
sector_neutral         1.223         2.666        10.00         0.638
vol_scaled             1.403         2.045         4.73         0.555

--- VERDICT (masterplan 51.2 gate) ---
sector_neutral vs baseline: dSharpe=-0.166, dSectors=+5.27
KEEP sector_neutral? False  (gate: breadth +>=2 sectors AND dSharpe >= -0.05)
vol_scaled vs baseline: dSharpe=+0.015 (BETTER)

N rebalances scored: 47
```

## Interpretation (criterion #2 -- the tradeoff, evidence-based not assumed)
- **HARD sector-neutral HURTS:** Sharpe 1.388 -> 1.223 (**-0.166**) and avg forward return 4.05% -> 2.67%/mo, while breadth doubles (4.73 -> 10.0 distinct GICS) and turnover rises (0.555 -> 0.638). This is EXACTLY the Harvey et al. long-only caveat (neutralizing beats keeping across-sector in only ~22% of long-only trials) -- now confirmed on pyfinagent's OWN universe. The tech concentration is the rational outcome of momentum; forcing diversification costs ~1.4%/mo of return and 0.17 of Sharpe.
- **vol-scaling is only marginally better** (+0.015 Sharpe) -- not a compelling lever here either (the +20% engine's vol is already moderate).
- **Decision: do NOT enable sector-neutral live.** The flag stays default-OFF. "Measure before fixing" prevented a Sharpe-reducing regression.

## Criterion-by-criterion

| # | Criterion | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | candidates carry a sector at rank time (no-op fixed) | `screener.build_sector_map` (GICS from Wikipedia) + gated wiring at `autonomous_loop.py:369` passing `sector_lookup`; `test_flag_on_spreads_across_sectors` proves the basket now spreads across sectors when ON; `test_flag_on_requires_sectors_to_work` documents the prior no-op | PASS |
| 2 | backtest compares ON vs OFF + reports Sharpe/return/sector-spread tradeoff | the replay above (dSharpe -0.166, dSectors +5.27, turnover +0.083) | PASS |
| 3 | config-gated, does NOT regress the US momentum core | flag default-OFF -> `_sector_lookup=None` -> `screen_universe` called exactly as before (byte-identical); `test_flag_off_is_byte_identical_with_or_without_sector` proves the OFF ranking is unchanged by the sector field | PASS |
| 4 | live_check records the comparison + sector distribution | this file (baseline 4.73 vs sector-neutral 10.0 distinct GICS) | PASS |

## Byte-identity (US engine untouched)
With `sector_neutral_momentum_enabled=False` (default), the wiring sets `_sector_lookup=None` and calls `screen_universe` identically to before (no Wikipedia map build on the live path) -> zero live behavior change. `rank_candidates` with `sector_neutral=False` ignores the sector field (unit-tested exact-equal). The lever is now FUNCTIONAL (criterion #1) but DISABLED (criterion #3) because the measurement says it loses money.

## Recommendation / next
- Keep sector-neutral OFF. The honest finding: broad sector diversification is NOT a free lunch for a long-only momentum book -- it trades return for breadth.
- A SOFT sector tilt/CAP (vs the hard within-sector-percentile REPLACEMENT) might preserve more momentum while trimming the worst concentration -- the wiring built here makes that measurable as a future step (51.x), but it is NOT pursued now without evidence it beats baseline.
- vol-scaling (+0.015 here) is not compelling on this engine; deprioritize.
- The bigger near-term money lever remains the now-LIVE multi-market universe (EU/KR structurally adds non-tech sectors WITHOUT neutralizing) + the resurrected alpha overlays (51.1) -- MEASURE Monday's first multi-market cycle.
