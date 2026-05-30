# live_check_50.3 -- international universe + suffix mapper + routing (evidence)

Verified 2026-05-30. International is BUILT but OFF (paper_markets default ['US'] = byte-identical).

## 1. Unit test (criteria #1, #2) -- backend/tests/test_phase_50_3_universe.py
`pytest -q` -> **6 passed in 0.20s**:
- suffix mapper round-trips: US:AAPL->AAPL, EU:SAP->SAP.DE, KR:005930->005930.KS; bare/already-suffixed unchanged
- market_for_symbol: AAPL->US, SAP.DE->EU, **AIR.PA->EU** (Paris-listed DAX member), 005930.KS->KR, .KQ->KR
- DAX40 >= 30, all suffixed, incl. SAP.DE + AIR.PA; KOSPI200 >= 20, all .KS, incl. 005930.KS
- KR codes are 6-digit STRINGS with leading zeros preserved (never int())
- paper_markets default == ['US']; TradeOrder.market defaults "US"

## 2. LIVE universe routing (criteria #3, #4) -- byte-identical for ['US']
```
paper_markets=['US']         -> universe=None -> get_sp500_tickers() = 503 tickers (BYTE-IDENTICAL to today)
paper_markets=['US','EU']    -> 543 tickers (+40 EU); .DE present=True; .KS present=False
paper_markets=['US','EU','KR'] -> 583 tickers; .DE=39 + .PA=1 (AIR.PA) = 40 EU; .KS=40 KR
BYTE-IDENTITY: build_universe(['US']) is None == True  (adds ZERO international tickers)
```
With ['US'] the universe is None -> `screen_universe(None)` -> `get_sp500_tickers()` (today's exact path). .DE tickers appear ONLY when EU is enabled; .KS ONLY when KR is enabled.

## 3. Files
- backend/backtest/universe_lists.py (NEW) -- curated DAX40 (.DE + AIR.PA) + KOSPI200 (.KS) yfinance symbols, static in-repo.
- backend/backtest/markets.py -- YF_SUFFIX + to_yfinance_symbol + market_for_symbol.
- backend/backtest/candidate_selector.py:127 -- non-US stub now returns INTL_UNIVERSE[market] (EU/KR).
- backend/config/settings.py -- paper_markets = Field(default_factory=lambda: ["US"]).
- backend/services/portfolio_manager.py -- TradeOrder.market field + set via markets.market_for_symbol on both BUY orders.
- backend/services/autonomous_loop.py -- universe extension (no-op for ['US']) + market=order.market on the execute_buy call.

## Success criteria mapping (all 4 met)
1. suffix mapper {market}:{ticker} -> yfinance symbol, round-trips -- YES (6 tests).
2. get_universe_tickers EU=DAX-40, KR=KOSPI-200 (documented seed) -- YES (40 each).
3. paper_markets drives universe; ['US'] byte-identical; ['US','EU'] adds .DE -- YES (503 vs 543).
4. live universe listing ['US'] vs ['US','EU'] showing .DE only when EU enabled -- YES (above).

## Scope / honesty notes
- **paper_markets default ['US'] -> the live engine is byte-identical** (international built but OFF). Go-live flip to ['US','EU','KR'] is DEFERRED to AFTER the 50.5 data-quality gate (operator's "free yfinance + quality gate" choice).
- Suffixed-symbol-as-ticker avoids the AIR.PA (Paris, NOT .DE) + KOSDAQ .KQ derivation traps. KR leading zeros preserved (STRING).
- KOSPI200 is a documented ~40-name large-cap SEED (criterion allows "or a documented subset"); expandable later.
- yfinance KR (.KS) viability + the <=11% deviation / identical-OHLC risk are handled by the 50.5 data-quality gate (mandatory before go-live).
- $0 LLM; no pip; no spend; no DROP/DELETE.
