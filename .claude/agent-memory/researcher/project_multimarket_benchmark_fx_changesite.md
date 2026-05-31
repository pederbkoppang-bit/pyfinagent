---
name: multimarket-benchmark-fx-changesite
description: phase-50.5 benchmark/FX-to-base-currency change site is in analytics.py + api/backtest.py, NOT backtest_engine.py — the engine's only "SPY" is a cache-preload, not the benchmark
metadata:
  type: project
---

For phase-50.5 criterion #1 ("backtest accepts a market, uses its benchmark, FX-converts NAV/returns to base currency"), the actual benchmark-return + alpha computation is NOT in `backtest_engine.py`.

**Where the benchmark RETURN actually lives:**
- `backend/backtest/analytics.py:compute_baseline_strategies` (def ~:446) hardcodes `prices_cache_fn("SPY", ...)` at ~:462; emits `spy_return_pct`/`spy_sharpe` (~:527-528).
- `generate_report` (~:536) computes alpha = `aggregate_return_pct - baselines["spy_return_pct"]` (~:580).
- `compute_baseline_strategies` is CALLED from `backend/api/backtest.py:~939` — this is where `engine.market` is in scope, so the benchmark arg must be threaded in HERE.
- `backtest_engine.py:299` `cache.preload_prices(universe_tickers + ["SPY"], ...)` is a CACHE PRELOAD, not the benchmark used for returns. (Engine `self.market` stored at ~:206; `get_universe_tickers()` called market-blind at ~:281.)

**Why:** The contract (2026-05-30) located the benchmark change at "backtest_engine.py:299 hardcodes SPY" — that anchor is a preload line; the real return/alpha path is analytics + the api caller. Patching only backtest_engine.py is INSUFFICIENT for the benchmark/FX criterion.

**How to apply:** When wiring per-market benchmarks (US=SPY, EU=^GDAXI, KR=^KS11), add a `benchmark` param to `compute_baseline_strategies` + pass it from `api/backtest.py` using `MARKET_CONFIG[engine.market]["benchmark"]`; do the FX-to-USD endpoint conversion (via `fx_rates.get_fx_rate(local,"USD",date)`) at that same analytics/api layer, not inside the engine. Add `benchmark` key to `MARKET_CONFIG` in `markets.py` (see [[project_multimarket_dataquality_gate]]). Per-bar FX is DEFERRED — endpoint conversion of a single-currency book's return is clean (QuantConnect's own DB is USD-quoted; per-bar intl FX is genuinely hard).
