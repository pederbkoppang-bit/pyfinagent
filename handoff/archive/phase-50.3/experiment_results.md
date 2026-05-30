# Experiment results -- phase-50.3: International universe + suffix mapper + routing

**Date:** 2026-05-30 | **Result: built + live-verified (byte-identical for paper_markets=['US'])** | $0 LLM | no pip | Operator: BOTH EU + Korea, free yfinance.

## What was built
EU (DAX-40) + KR (KOSPI-200 seed) wired into the live loop's universe capability, gated by a `paper_markets` setting that defaults to `["US"]` (byte-identical). International is BUILT but OFF until a deliberate flip after the 50.5 data-quality gate.

## Files changed/added
1. **backend/backtest/universe_lists.py** (NEW) -- curated static DAX40 (.DE + AIR.PA) + KOSPI200 (.KS, ~40-name seed) yfinance symbols. Static in-repo (can't collapse to [] on a scrape failure).
2. **backend/backtest/markets.py** -- `YF_SUFFIX` + `to_yfinance_symbol(namespaced)` + `market_for_symbol(symbol)` (derives market from the suffix: .DE/.PA->EU, .KS/.KQ->KR, bare->US).
3. **backend/backtest/candidate_selector.py:127** -- non-US `get_universe_tickers` stub now returns `INTL_UNIVERSE[market]` (was []).
4. **backend/config/settings.py** -- `paper_markets: list[str] = Field(default_factory=lambda: ["US"])`.
5. **backend/services/portfolio_manager.py** -- `TradeOrder.market` field; set via `markets.market_for_symbol(cand["ticker"])` on both BUY-order constructions (main + swap).
6. **backend/services/autonomous_loop.py** -- universe extension (append INTL_UNIVERSE per non-US paper_market; no-op for ['US']) + `market=order.market` on the execute_buy call.
7. **backend/tests/test_phase_50_3_universe.py** (NEW) -- 6 offline tests.

## Verification (live)
- `pytest backend/tests/test_phase_50_3_universe.py` -> **6 passed**. All changed modules import clean.
- `paper_markets` default == `['US']`.
- **LIVE universe routing**: ['US'] -> universe=None -> 503 S&P tickers (BYTE-IDENTICAL, zero intl added); ['US','EU'] -> 543 (+40 EU, .DE present / .KS absent); ['US','EU','KR'] -> 583 (.DE 39 + .PA 1 = 40 EU, .KS 40 KR). build_universe(['US']) is None == True.

## Success criteria mapping (all 4 met) -- see live_check_50.3.md
1. suffix mapper round-trips -- YES. 2. get_universe_tickers EU/KR non-empty + suffixed -- YES. 3. paper_markets drives universe, ['US'] byte-identical, ['US','EU'] adds .DE -- YES. 4. live universe listing -- YES.

## Scope / honesty notes
- **Byte-identical live**: paper_markets default ['US'] -> universe=None -> today's get_sp500_tickers path; every BUY TradeOrder.market = market_for_symbol(bare ticker) = "US" -> 50.2 FX x1.0. The live +20% engine is unchanged.
- Suffixed-symbol-as-ticker (vs deriving suffix from market) avoids the AIR.PA / .KQ traps; KR leading zeros preserved.
- **International is OFF by default** -- go-live (flip paper_markets to include EU/KR) is DEFERRED to AFTER the 50.5 data-quality gate per the operator's "free yfinance + quality gate" choice. 50.3 ships the capability only.
- KOSPI200 is a documented ~40-name large-cap seed (criterion allows a documented subset); expandable.
- Backtest PIT path (candidate_selector as_of) still raises NotImplementedError for non-US -- out of scope (50.5 handles intl backtest).
- $0 LLM; no pip; no spend; no DROP/DELETE.
