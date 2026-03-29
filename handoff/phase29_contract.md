# Phase 2.9 Contract: Multi-Market Data Layer

**Date:** 2026-03-29 11:35 UTC

**Hypothesis:**
PyfinAgent can be extended to analyze stocks across 5+ global markets (US, NO, CA, EU, KR) without duplicating code or strategy logic. The multi-market abstraction layer will sit between data ingestion and the backtester, handling timezone differences, calendar differences, and currency conversion transparently.

**Success Criteria (Research-Backed):**

1. ✅ **BQ Schema Migration Complete**
   - All 3 tables (historical_prices, historical_fundamentals, historical_macro) have `market` column
   - Indexed for <10ms query performance at 500+ tickers

2. ✅ **Ticker Namespace Parsing**
   - Correctly parses `"US:AAPL"`, `"NO:EQNR"`, `"CA:RY"`, `"EU:ASML"`, `"KR:005930"` (Samsung)
   - Defaults to US for unprefixed tickers (backward compatible)

3. ✅ **Cache Layer Filtering**
   - All data queries include `WHERE market = ?` filter
   - No cross-contamination between markets
   - Cache hit rate ≥ 95% (single BQ query per market per run)

4. ✅ **Market Configuration**
   - 5 markets configured: US (NYSE/NASDAQ), NO (Oslo Børs), CA (TSX), EU (XETRA/Euronext), KR (KRX)
   - Timezone mappings: America/New_York, Europe/Oslo, America/Toronto, Europe/Berlin, Asia/Seoul
   - Exchange calendars: US (252 days), NO (251 days), CA (252 days), EU (250 days), KR (249 days)

5. ⏳ **Single-Seed Backtest Validation**
   - Seed 2026 backtest (running now) confirms no regression on US strategy
   - Sharpe within ±2% of baseline (1.17)
   - No crashes on multi-market queries

**Fail Conditions:**
- Any query returns cross-market data (e.g., US prices mixed with NO prices)
- Backtest crashes on Seed 2026 run
- Cache queries exceed 30s timeout
- Market config missing for any of 5 exchanges

**Acceptance Criteria:**
- All 5 tests PASS (4/5 currently PASS, 1 in progress)
- Seed 2026 result ≥ Sharpe 1.0 (confirming no regression)
- Zero log errors related to market filtering

**Handoff Files:**
- `handoff/phase29_integration_results.md` — test results
- `handoff/seed_2026_result.json` — final seed validation
- `backend/backtest/markets.py` — market config definitions

**Timeline:**
- Started: 2026-03-29 11:11 UTC (design + 4 integration tests)
- Seed 2026 test: 2026-03-29 11:31 UTC (ETA +15 min)
- Expected completion: 2026-03-29 11:50 UTC

---

**EVALUATOR SIGN-OFF (pending seed 2026 completion)**

Once Seed 2026 produces results, evaluator will verify:
1. No exceptions thrown
2. Sharpe within ±2% of baseline
3. All market columns correctly populated
4. Cache filtering working correctly
5. No performance regression

**Decision:** PASS if all 5 conditions met → Phase 2.9 COMPLETE → Ready for Phase 3
