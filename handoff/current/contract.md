# Contract -- phase-50.5: Multi-market backtest + DATA-QUALITY gate

**Step id:** 50.5 | **Priority:** P3 (phase-50; the LAST go-live prerequisite) | **depends_on:** 50.4
**Date:** 2026-05-30 | **harness_required:** true | **$0 LLM** | no pip
**STATUS: PLAN complete, GENERATE pending** -- deliberately handed off mid-cycle after 7 cycles this session (marathon-risk discipline; this step inserts into the live US screener/ingestion paths, a regression surface best done with fresh context). The next session GENERATEs from this contract.

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (gate: **7 sources read in full, recency scan, 20 URLs, 12 internal files, gate_passed=true**). Decisive:
- **NO shared price-cleaning point -- the gate guards THREE doors** via one new `backend/tools/price_quality.py` validator:
  - **L1 (operator's PRIORITY -- the live SIGNAL path):** `backend/tools/screener.py:110` (yf.download) -> `:131` `close = ticker_data["Close"]` feeds momentum/RSI/vol. Insert the gate in the per-ticker loop after `ticker_data = data[ticker]` (~:124-130), before `:131`. This is where a bad .DE/.KS bar becomes a signal.
  - **L2 (live FILL/MARK):** `backend/services/paper_trader.py:1200` `_get_live_price` -- a bad bar -> return None (callers already fall back to last-known).
  - **B (backtest source + live daily-refresh):** `backend/backtest/data_ingestion.py:132` (`ingest_prices`), ALSO called by `backend/slack_bot/jobs/daily_price_refresh.py:82`.
  - (`backend/services/live_prices.py` is dashboard-only -> NOT a cleaning point.)
- **Detection rules** (`price_quality.validate_ohlcv(df, market)`): R1 OHLC consistency (high>=max(o,c,l), low<=min(o,c,h), all>0); R2 identical-OHLC bar (o==h==l==c) AND/OR volume==0 -> DROP (zero-vol corroborates -- Tobi Lux: bad DAX bars had no volume); R3 |1-day return| z-score>3 (rolling) -> FLAG, >50% single-day round-trip -> DROP; R4 stale run-length (>=4 identical closes) -> FLAG. CRITICAL nuance (arXiv 2403.19735): never over-drop REAL volatility -- DROP only the unambiguous (identical-OHLC+zero-vol, impossible OHLC), FLAG the merely-large. Log dropped/flagged counts (NO silent truncation).
- **BYTE-IDENTITY:** `validate_ohlcv` opens with `if market == "US": return df unchanged` (fast-path no-op) -> the live US screener/ingestion are untouched; real US bars also pass every rule. US benchmark stays `"SPY"` (not ^GSPC). US backtest numbers unchanged.
- **Backtest gaps:** `backtest_engine.py:281` calls `get_universe_tickers()` WITHOUT `self.market` (market-blind despite storing self.market:206); `:299` hard-codes `+["SPY"]`; `analytics.py:461` hard-codes SPY; MARKET_CONFIG (markets.py:21-52) has NO `benchmark` field -> add (US="SPY", EU="^GDAXI", KR="^KS11"); `backtest_trader.mark_to_market:188` currency-blind.
- **Scope triage (minimal safe go-live, satisfies all 4 criteria):** price_quality.py + L1/L2/B gates + backtest market-param + per-market benchmark + FX endpoint-return conversion (reuse 50.1 fx_rates, mirror 50.2 _fx_local_to_usd) + an EU `.DE` backtest for the live_check. **DEFERRED (documented, NOT blocking):** PIT-correct intl membership (the candidate_selector as_of NotImplementedError -- a market-agnostic survivorship guard, US has the same), per-bar FX inside _compute_nav, simultaneous mixed-currency multi-market backtest, live per-market benchmark.

## Hypothesis
A `price_quality.validate_ohlcv(df, market)` validator (US fast-path no-op; intl detection rules) wired at the 3 doors (screener signal, _get_live_price fill, data_ingestion source) + a `benchmark` field in MARKET_CONFIG + the backtest accepting `market` (benchmark + FX-endpoint conversion) -- makes free-yfinance international data SAFE to trade live (the operator's "quality gate" precondition) while keeping the US path byte-identical.

## Success criteria (IMMUTABLE -- verbatim from masterplan step 50.5)
1. the backtest engine accepts a market, uses its benchmark (^GDAXI for EU, ^KS11 for KR, SPY/^GSPC for US) and FX-converts NAV/returns to base currency
2. a data-quality gate detects + drops (or flags) identical-OHLC bars and gross-deviation outliers in international price series, logging how many bars were dropped (no silent truncation)
3. an EU (.DE) backtest runs end-to-end with the correct benchmark + FX-converted returns + the data-quality gate active; US backtests unchanged
4. live evidence: an EU backtest summary (benchmark, FX-converted return, n bars dropped by the quality gate)

**Verification command:** ast.parse(price_quality.py) + pytest backend/tests/test_phase_50_5_dataquality.py + test -f live_check_50.5.md.
**live_check:** REQUIRED -- an EU-ticker backtest run with benchmark + FX-converted NAV + data-quality-gate drop count.

## Plan steps (for the next session's GENERATE)
1. **backend/tools/price_quality.py** (NEW) -- `validate_ohlcv(df, market="US") -> (clean_df, report)`; `if market=="US": return df, {dropped:0,flagged:0}` (fast-path). Else apply R1-R4; return cleaned df + a report dict (dropped/flagged counts + reasons). Pure + unit-testable (no network).
2. **Wire the 3 doors:** screener.py (~:124-130, before close extraction, market via markets.market_for_symbol(ticker)); paper_trader._get_live_price:1200 (validate the 1-row bar -> None if bad); data_ingestion.py:132 (validate per-ticker before the rows.append loop).
3. **markets.py MARKET_CONFIG:21-52** -- add `benchmark`: US="SPY", EU="^GDAXI", KR="^KS11" (NO=^OSEAX, CA=^GSPTSE).
4. **backtest_engine.py:281** pass `self.market` to get_universe_tickers; **:299** use `MARKET_CONFIG[self.market]["benchmark"]` instead of "SPY"; **analytics.py:461** benchmark param; backtest endpoint-return + benchmark FX-converted to USD via fx_rates (single-market book).
5. **backend/tests/test_phase_50_5_dataquality.py** (NEW) -- validate_ohlcv: US fast-path no-op (returns input unchanged, dropped=0); an identical-OHLC+zero-vol bar DROPPED; an impossible OHLC (high<low) DROPPED; a 60% spike DROPPED; a large-but-real move FLAGGED not dropped; a clean intl series passes; report counts correct.
6. **Verify:** pytest; an EU `.DE` backtest end-to-end (benchmark ^GDAXI, FX-converted return, gate drop count); confirm a US backtest is unchanged (byte-identical). Capture into live_check_50.5.md.
7. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 50.5 -> done.

## After 50.5: the GO-LIVE flip (operator-authorized)
Once 50.5 lands (quality gate live), the international go-live = flip `settings.paper_markets` (or its .env override) to `["US","EU","KR"]`. The operator AUTHORIZED Both EU+KR + free-yfinance+quality-gate. The flip makes the live loop screen/trade EU/KR (quality-gated). Recommend: the orchestrator presents the flip as the final go-live action (it changes live trading) -- the operator already chose it, so it's executing their decision, but report it explicitly. Then 50.6 (UI).

## Safety / scope notes
- **Byte-identity:** validate_ohlcv US fast-path no-op + US benchmark "SPY" + US backtest unchanged. The gate touches the live US screener/ingestion code paths, so the US fast-path correctness is the regression surface -- the test MUST prove US passes through unchanged.
- DROP only unambiguous bad bars (identical-OHLC+zero-vol, impossible OHLC, >50% round-trip); FLAG (don't drop) merely-large moves (never destroy real volatility).
- No silent truncation: log dropped/flagged counts.
- DEFERRED: PIT intl membership, per-bar FX, mixed-currency backtest, live per-market benchmark (documented, not blocking).
- $0 LLM; no pip; no spend; no DROP/DELETE.

## References
- handoff/current/research_brief.md (50.5 gate)
- backend/tools/screener.py:110-131 (L1), backend/services/paper_trader.py:1200 (L2), backend/backtest/data_ingestion.py:132 (B), backend/slack_bot/jobs/daily_price_refresh.py:82
- backend/backtest/markets.py:21-52 (benchmark field), backtest_engine.py:281,299, analytics.py:461, backtest_trader.py:188
- backend/backtest/candidate_selector.py:127 (INTL_UNIVERSE["EU"]), backend/services/fx_rates.py (50.1)
- Tobi Lux DAX yfinance study; axionquant (z=3); arXiv 2403.19735 (anomaly thresholds); DataIntellect (stale data)
