# experiment_results -- phase-50.5: Multi-market backtest + DATA-QUALITY gate

**Step:** 50.5 | **Date:** 2026-05-31 | **$0 LLM** | no pip | GENERATE complete

## What was built / changed

Completes the GENERATE that was deliberately handed off mid-cycle. The on-disk
partial (price_quality.py + screener L1 + 6 tests) was audited clean by the
researcher (6/6 passing, US byte-identical); this cycle wired the remaining
doors + the market-aware backtest, fixing the **material drift** the researcher
caught (the benchmark/alpha computation lives in `analytics.py` +
`api/backtest.py`, NOT `backtest_engine.py:299` which is only a cache-preload line).

| File | Change | Status |
|------|--------|--------|
| `backend/tools/price_quality.py` | `validate_ohlcv` (R1-R4) + `is_bad_bar`; US fast-path no-op | pre-existing (audited clean) |
| `backend/tools/screener.py` | L1 door: `validate_ohlcv` before close extraction | pre-existing (audited clean) |
| `backend/backtest/markets.py` | **NEW** `benchmark` field on every MARKET_CONFIG (US=SPY, EU=^GDAXI, KR=^KS11, NO=^OSEAX, CA=^GSPTSE) | this cycle |
| `backend/backtest/backtest_engine.py` | `:283` pass `market=self.market` to `get_universe_tickers`; `:299` preload `get_market_config(self.market)["benchmark"]` instead of hardcoded SPY | this cycle |
| `backend/backtest/analytics.py` | `compute_baseline_strategies(..., benchmark="SPY", base_currency="USD", local_currency="USD")`: uses the market benchmark + FX-converts headline returns endpoint-style; US (local==base) -> fx_ratio EXACTLY 1.0 (passthrough, no drift) | this cycle |
| `backend/api/backtest.py` | pass `benchmark`+`local_currency` from `get_market_config(engine.market)` into the baseline call (US engine -> SPY/USD -> byte-identical) | this cycle |
| `backend/services/paper_trader.py` | L2 door: `_get_live_price` drops bad bars (`is_bad_bar`) for INTL tickers only; US untouched (`market_for_symbol -> US -> skip`) | this cycle |
| `backend/backtest/data_ingestion.py` | B door: `validate_ohlcv` per-ticker before BQ ingest; US (`_mkt=="US"`) -> no-op | this cycle |
| `backend/tests/test_phase_50_5_dataquality.py` | +3 tests: MARKET_CONFIG benchmarks; US baseline byte-identity (exact float); EU FX conversion (monkeypatched fx) | this cycle |
| `scripts/phase50/live_check_50_5.py` | **NEW** tiered live_check on real yfinance data | this cycle |

## Byte-identity guarantee (US path unchanged -- the regression surface)
- `validate_ohlcv(df, market="US")` returns the **same object** (screener L1 + ingestion B).
- `_get_live_price`: `market_for_symbol("AAPL") == "US"` -> validation skipped -> returns price exactly as before (L2).
- `get_universe_tickers(market="US")` == `get_universe_tickers()` (default is "US") -- proven equal at runtime.
- `compute_baseline_strategies` default benchmark="SPY", local==base=="USD" -> `fx_ratio == 1.0` -> `_to_base` returns the input float UNCHANGED (no `(1+r)*1-1` drift) -- unit-tested with exact `==`.
- `cache.preload_prices(universe + [benchmark])` with US benchmark "SPY" == the old `+ ["SPY"]`.

## Verification command output (verbatim)

### 1. Syntax (all modified files)
```
OK  backend/backtest/markets.py
OK  backend/backtest/backtest_engine.py
OK  backend/backtest/analytics.py
OK  backend/api/backtest.py
OK  backend/services/paper_trader.py
OK  backend/backtest/data_ingestion.py
OK  backend/tools/price_quality.py
OK  backend/tools/screener.py
```

### 2. pytest (phase-50.5 suite -- 9 tests)
```
$ python -m pytest backend/tests/test_phase_50_5_dataquality.py -q
.........                                                                [100%]
9 passed in 1.33s
```
New tests: `test_market_config_has_benchmarks`, `test_baseline_us_byte_identical_passthrough`
(asserts exact float equality to the raw pre-50.5 formula), `test_baseline_eu_fx_converted`
(20% local + fx 1.0->1.10 => 32.0% USD).

### 3. Regression (related existing tests + 50.5)
```
$ python -m pytest test_phase_50_3_universe.py test_phase_32_1_breakeven_ratchet.py \
      test_dod4_tier1_coverage_investment.py test_phase_50_5_dataquality.py -q
94 passed in 1.65s
```

### 4. live_check (real yfinance) -- see `handoff/current/live_check_50.5.md`
`test -f handoff/current/live_check_50.5.md` -> exists. Headline:
- Gate dropped **15 real bad bars + 6 flagged** across 5 live DAX tickers (all `R2 identical-OHLC+zero-vol`).
- EU baseline: benchmark=^GDAXI, ^GDAXI return +8.027% EUR -> +7.982% USD (fx_ratio 0.99958).
- US baseline: benchmark=SPY, fx_ratio == **1.0** (byte-identical).

## Artifact shape
- `validate_ohlcv(df, market, ticker) -> (clean_df, {dropped:int, flagged:int, reasons:[str]})`
- `is_bad_bar(o,h,l,c,volume) -> bool`
- `compute_baseline_strategies(...) -> {spy_return_pct, spy_sharpe, equal_weight_return_pct, eq_weight_sharpe, momentum_return_pct, momentum_sharpe, benchmark, fx_ratio, base_currency}`
- `get_market_config(market) -> {exchange, currency, timezone, description, benchmark}`

## Deferred (documented, NOT blocking -- re-confirmed by researcher 2026-05-31)
PIT-correct intl membership (`candidate_selector.as_of` NotImplementedError -- market-agnostic; US has the same gap; arXiv 2603.19380 quantifies survivorship at 4.94pp for EM small-caps but it's a separate reconstruct-membership project), per-bar FX inside `_compute_nav`, simultaneous mixed-currency multi-market backtest, live per-market benchmark. Sharpe stays local-currency (per-bar FX deferred).

## After 50.5: the go-live flip (operator-authorized, to be REPORTED not silently executed)
Flip `settings.paper_markets` -> `["US","EU","KR"]`. The operator authorized EU+KR + free-yfinance+quality-gate. This is the final go-live action; report it explicitly.
