---
name: project-multimarket-dataquality-gate
description: phase-50.5 research — yfinance intl data-quality gate (identical-OHLC/outlier/stale) placement + multi-market backtest benchmark/FX; 3 price doors, no shared cleaning point
metadata:
  type: project
---

phase-50.5 (researched 2026-05-30, complex tier): data-quality gate for free-yfinance international data + multi-market backtest (benchmark + FX). Operator's gating precondition for go-live.

**Why:** yfinance intl (DAX-40) has documented defects — up to 11% deviation vs XETRA, identical-OHLC (O==H==L==C) on up to 10% of days (10-24 days/yr) co-occurring with zero-volume bars; these silently corrupt momentum/vol signals. Source: Tobi Lux Medium "Data from yfinance — some Observations" (THE 11% source, read in full).

**How to apply:**
- NO shared price-cleaning point exists. THREE doors a new `backend/tools/price_quality.py` validator must guard: (L1) `screener.py:110/131` screen_universe yf.download — the LIVE signal path, operator's PRIORITY; (L2) `paper_trader.py:1200` _get_live_price — live fill/mark; (B) `data_ingestion.py:105/132` ingest_prices — backtest BQ source AND live daily-refresh (`daily_price_refresh.py:82`). `live_prices.py` is dashboard-only (NOT the loop).
- Detection rules: R1 OHLC consistency (low<=min(O,C), high>=max(O,C), >0); R2 identical-OHLC DROP + vol==0 co-signal; R3 return z>3 rolling-20 FLAG / >50%-round-trip DROP (two-tier — arXiv:2403.19735 uses HIGH 10σ to avoid flagging REAL crashes; never over-drop real volatility); R4 stale run-length>=4. Return `(clean_df, report)`; LOG per-run drop count (criterion 2 "no silent truncation" — 2026 data-quality consensus = quarantine+log not silent drop).
- US byte-identity: explicit `if market=="US": return df unchanged` fast-path (real US bars also pass all rules — defense-in-depth).
- Backtest gaps: `backtest_engine.py:281` calls get_universe_tickers() WITHOUT self.market (market-blind despite storing self.market at :206); `:299` hard-codes +["SPY"]; `analytics.py:461` SPY hard-code (called api/backtest.py:939). `MARKET_CONFIG` (markets.py:21-52) has NO benchmark field — add US:"SPY"(keep for byte-identity, NOT ^GSPC)/EU:"^GDAXI"/KR:"^KS11".
- FX in backtest NAV: `backtest_trader.mark_to_market:188` is currency-blind. For single-market EU backtest, convert ENDPOINT return + benchmark to USD via 50.1 fx_rates (mirror 50.2 _fx_local_to_usd). Per-bar FX + mixed-currency = DEFERRED.
- PIT is NOT a blocker: `candidate_selector.get_universe_tickers` NotImplementedError only fires on `as_of` (survivorship guard, market-agnostic — US has it too). Curated `.DE` list (INTL_UNIVERSE["EU"], 50.3) runs end-to-end via `run_backtest(universe_tickers=[...], market="EU")`. PIT membership/delistings feed = documented follow-on (arXiv:2603.19380: survivorship overstates ~23%).
- See [[project_multimarket_universe_wiring]], [[project_market_calendar_gating]], [[project_multimarket_scaffolding_disconnected]].
