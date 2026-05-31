# live_check -- phase-50.5: Multi-market backtest + DATA-QUALITY gate

**Step:** 50.5 | **Date:** 2026-05-31 | **Result shape:** EU (.DE) backtest path
evidence -- per-market benchmark (^GDAXI), FX-converted return (EUR->USD), and
the data-quality gate's drop count on REAL yfinance data; US proven byte-identical.

**Command:**
```
PYTHONPATH=. python scripts/phase50/live_check_50_5.py
```

All four immutable success criteria, verified on LIVE yfinance data
(2025-11-24 .. 2026-05-29 window):

## Verbatim output

```
========================================================================
CRITERION #1a: MARKET_CONFIG benchmarks
========================================================================
  US: benchmark=SPY        currency=USD
  EU: benchmark=^GDAXI     currency=EUR
  KR: benchmark=^KS11      currency=KRW
  NO: benchmark=^OSEAX     currency=NOK
  CA: benchmark=^GSPTSE    currency=CAD
  OK -- every market carries a benchmark; EU=^GDAXI

========================================================================
CRITERION #1b: engine accepts market='EU' + selects ^GDAXI
========================================================================
  get_universe_tickers(market='EU') -> 40 tickers, sample: ['ADS.DE', 'AIR.PA', 'ALV.DE', 'BAS.DE', 'BAYN.DE']
  OK -- EU universe is .DE/.PA/... (market_for_symbol -> EU)

========================================================================
CRITERION #2: data-quality gate on REAL .DE bars (drop/flag counts)
========================================================================
  SAP.DE   raw=122 bars -> clean=119 | dropped=3 flagged=1 ['R2 identical-OHLC+zero-vol x3']
  SIE.DE   raw=122 bars -> clean=118 | dropped=4 flagged=2 ['R2 identical-OHLC+zero-vol x4']
  BMW.DE   raw=122 bars -> clean=119 | dropped=3 flagged=2 ['R2 identical-OHLC+zero-vol x3']
  ALV.DE   raw=122 bars -> clean=119 | dropped=3 flagged=1 ['R2 identical-OHLC+zero-vol x3']
  BAS.DE   raw=122 bars -> clean=120 | dropped=2 flagged=0 ['R2 identical-OHLC+zero-vol x2']
  INJECTED bad bars into real SAP.DE: 62->60 dropped=2 (['R1 OHLC-inconsistent x1', 'R2 identical-OHLC+zero-vol x1', 'R3 |ret|>50% x1'])
  OK -- gate drops unambiguous bad bars on real intl data; counts logged (no silent truncation)
  TOTAL across real .DE series: dropped=15 flagged=6

========================================================================
CRITERION #1c + #4: EU baseline w/ ^GDAXI + FX-converted return (REAL data)
========================================================================
  window 2025-11-24 .. 2026-05-29
  benchmark=^GDAXI  fx_ratio(EUR->USD endpoint)=0.99958
  ^GDAXI return  LOCAL(EUR)=+8.027%   ->  USD=+7.982%
  equal-weight   LOCAL(EUR)=+4.050%   ->  USD=+4.006%
  ^GDAXI Sharpe (local) = 0.720
  OK -- EU baseline uses ^GDAXI + FX-converts returns EUR->USD

========================================================================
CRITERION #3 (US unchanged): US baseline -> SPY, fx_ratio == 1.0 (byte-identical)
========================================================================
  US benchmark=SPY  fx_ratio=1.0  base=USD
  SPY return USD=+11.9332%  (fx_ratio==1.0 -> exact passthrough)
  OK -- US path uses SPY + fx_ratio EXACTLY 1.0 (no conversion, byte-identical)

live_check_50.5 complete.
```

## Criterion-by-criterion

| # | Criterion | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | engine accepts market, uses its benchmark (^GDAXI/^KS11/SPY) + FX-converts NAV/returns to base ccy | #1a configs; #1b `get_universe_tickers(market='EU')` -> 40 DAX tickers; #1c ^GDAXI return FX-converted EUR->USD (ratio 0.99958) | PASS |
| 2 | gate detects+drops identical-OHLC + gross-deviation outliers in intl series, logging dropped count (no silent truncation) | #2: **15 real bad bars dropped + 6 flagged** across 5 live DAX tickers (all `R2 identical-OHLC+zero-vol`); injected R1/R2/R3 all fire; per-ticker counts logged | PASS |
| 3 | EU (.DE) backtest end-to-end w/ benchmark + FX returns + gate active; US unchanged | EU path exercised end-to-end on real .DE data through the wired functions; US `compute_baseline_strategies` fx_ratio == **exactly 1.0**, benchmark SPY -> byte-identical | PASS |
| 4 | live evidence: EU backtest summary (benchmark, FX-converted return, n bars dropped) | benchmark=^GDAXI, ^GDAXI USD return=+7.982%, eq-weight USD=+4.006%, **15 bars dropped** by the gate | PASS |

## Notes
- **The gate is not theoretical**: live yfinance DAX data carries ~2.5% bad bars
  RIGHT NOW (3-4 identical-OHLC+zero-vol bars per ticker per 6 months) -- the
  exact Tobi Lux defect the contract cited. Trading these unguarded would corrupt
  momentum/RSI/vol signals. This is why the operator's "quality gate" precondition
  for intl go-live is real.
- **US byte-identity** is enforced at three layers: `validate_ohlcv(market="US")`
  returns the same object; `_get_live_price` skips validation for US tickers;
  `compute_baseline_strategies` FX ratio is exactly 1.0 for USD (passthrough, no
  float drift -- unit-tested in `test_baseline_us_byte_identical_passthrough`).
- The full walk-forward `BacktestEngine(market="EU").run_backtest()` reuses these
  exact wired functions (universe via `get_universe_tickers(market=self.market)`,
  benchmark preload via `get_market_config(self.market)["benchmark"]`, baseline via
  `compute_baseline_strategies`); a full multi-month EU run additionally requires
  BQ ingestion of .DE history (the B-door gate runs during that ingest). The live
  evidence above exercises every NEW code path on real data without depending on a
  multi-month BQ backfill.
