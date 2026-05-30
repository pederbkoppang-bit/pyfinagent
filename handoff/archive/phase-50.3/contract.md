# Contract -- phase-50.3: International universe + suffix mapper + live-loop routing

**Step id:** 50.3 | **Priority:** P3 (phase-50) | **depends_on:** 50.2
**Date:** 2026-05-30 | **harness_required:** true | **$0 LLM** | no pip (yfinance covers .DE + .KS/.KQ). Operator chose BOTH EU + Korea, free yfinance.

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (gate: **8 sources read in full, recency scan, 19 URLs, 9 internal files, gate_passed=true**). Decisive:
- **SAFETY INVARIANT holds trivially:** 50.2 already added `execute_buy(market="US")` + persists market/base_currency. The plug-in `if paper_markets==["US"]: universe=None` at `autonomous_loop.py:~321` leaves `screen_universe(None) -> get_sp500_tickers()` byte-identical; the multi-market branch is dead code until the operator flips `paper_markets`.
- **The ONE wiring gap:** `_StagedBuy` (portfolio_manager.py:25-40) has no `market` field, and the execute_buy call (autonomous_loop.py:996-1014) doesn't pass `market=`. Add `market: str = "US"` to `_StagedBuy` + `market=order.market` at the call. Default "US" -> every current buy unchanged -> 50.2 FX x1.0 -> byte-identical.
- **Suffix-mapper: STORE the suffixed symbol AS the ticker** (`SAP.DE`, `005930.KS`); `market` ("EU"/"KR") travels as a separate field. No per-call suffix derivation -> `_get_live_price`, `yf.Ticker().info`, `yf.download` all receive the ready symbol. Add `MARKET_CONFIG[m]["yf_suffix"]` + `to_yfinance_symbol()` to markets.py for build-time. Avoids: Airbus=`AIR.PA` (Paris DAX member, NOT .DE) + KOSDAQ `.KQ` vs KOSPI `.KS`.
- **No ticker-shape validators** block numeric KR tickers (grep clean). BQ keys STRING -> preserve KR leading zeros (never int()).
- **Constituent source: curated STATIC list in-repo** (`backend/backtest/universe_lists.py` NEW: DAX40 + KOSPI200 with yfinance symbols), NOT a runtime scrape (can't collapse to []). Fill the `candidate_selector.get_universe_tickers` non-US stub (:127-132, returns [] today).
- **paper_markets setting:** `paper_markets: list[str] = Field(default_factory=lambda: ["US"])` (default_factory, NOT mutable default).
- **Sector:** yfinance `.info` returns Yahoo taxonomy (not GICS) for .DE/.KS; the existing US sector cap groups on those same Yahoo strings -> NO remap; fail-open to "Unknown".
- **yfinance KR VIABLE** for KOSPI-200 large-caps; risks (20-min quote delay immaterial daily; 429 rate-limits on per-position fan-out -> batch where possible + a per-market resolve-count log; <=11% deviation -> the 50.5 data-quality gate; silent-empty -> resolve log). pykrx is the 50.5 fallback if KR resolve-rate fails.

## Hypothesis
A `paper_markets` setting (default ["US"]) gating the universe, a curated DAX40 + KOSPI200 list filling the candidate_selector non-US stub (storing suffixed yfinance symbols + market), a markets.py suffix mapper for build-time, and threading `market` through `_StagedBuy` -> `execute_buy` -- wires EU + KR into the live loop's universe capability while keeping ["US"] byte-identical (international BUILT but OFF until a deliberate flip after the 50.5 quality gate).

## Success criteria (IMMUTABLE -- verbatim from masterplan step 50.3)
1. a suffix mapper converts {market}:{ticker} to the yfinance symbol (US->bare, EU/DE->.DE, KR->.KS/.KQ); round-trips for sample tickers
2. candidate_selector.get_universe_tickers(market='EU') returns a real DAX-40 (or Xetra) universe; market='KR' returns KOSPI-200 (or a documented subset)
3. autonomous_loop's universe is driven by a `paper_markets` setting; with paper_markets=['US'] the universe + behaviour are byte-identical to today; with ['US','EU'] the universe includes .DE tickers
4. live evidence: the universe for ['US'] vs ['US','EU'] printed, showing .DE tickers added only when EU is enabled

**Verification command:** pytest backend/tests/test_phase_50_3_universe.py + paper_markets default == ['US'] + test -f live_check_50.3.md.
**live_check:** REQUIRED -- universe listing for paper_markets=['US'] (unchanged) vs ['US','EU'] (adds .DE).

## Plan steps
1. **`backend/backtest/universe_lists.py`** (NEW) -- curated `DAX40` (`.DE` + `AIR.PA`) + `KOSPI200` (`.KS`, large-cap seed) lists of yfinance symbols.
2. **`backend/backtest/markets.py`** -- add `yf_suffix` to MARKET_CONFIG (US="", EU=".DE", KR=".KS") + `to_yfinance_symbol(namespaced)` helper (build-time / round-trip).
3. **`backend/backtest/candidate_selector.py:127-132`** -- fill the non-US `get_universe_tickers(market)` stub: EU->DAX40, KR->KOSPI200 (from universe_lists).
4. **`backend/config/settings.py`** -- add `paper_markets: list[str] = Field(default_factory=lambda: ["US"])`.
5. **`backend/services/portfolio_manager.py:25-40`** -- `_StagedBuy` gets `market: str = "US"`.
6. **`backend/services/autonomous_loop.py`** -- universe plug-in: when `paper_markets==["US"]` keep today's path (byte-identical); else build the multi-market universe via candidate_selector per market. Pass `market=order.market` at the execute_buy call (:996).
7. **`backend/tests/test_phase_50_3_universe.py`** (NEW) -- to_yfinance_symbol round-trips; get_universe_tickers("EU")/"KR" non-empty + suffixed; paper_markets default ["US"]; _StagedBuy has market field; a US-only universe-selection helper returns the same set as today (byte-identical).
8. **Verify:** pytest; the live universe listing for ["US"] vs ["US","EU"] (the ["US"] set == get_sp500_tickers() today; ["US","EU"] adds .DE). Capture into live_check_50.3.md.
9. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 50.3 -> done.

## Safety / scope notes
- **paper_markets default ["US"] -> the live engine is byte-identical** (international is built but OFF). The go-live flip to ["US","EU","KR"] is DEFERRED to after the 50.5 data-quality gate (per the operator's "free yfinance + quality gate" choice -- never trade unguarded intl data).
- Suffixed-symbol-as-ticker avoids the AIR.PA / .KQ derivation traps. KR leading zeros preserved (STRING, no int()).
- No pip, no spend, no DROP/DELETE.
- 50.3 wires the UNIVERSE capability; the calendar gating (50.4), data-quality gate (50.5), and UI (50.6) follow.

## References
- handoff/current/research_brief.md (50.3 gate) + research_brief_multimarket.md
- backend/services/autonomous_loop.py:310-329 (universe), :996-1014 (execute_buy market=)
- backend/services/portfolio_manager.py:25-40 (_StagedBuy)
- backend/backtest/candidate_selector.py:98-132 (non-US stub), markets.py:21-72 (config + mapper)
- backend/config/settings.py:49-51 (paper_markets), backend/tools/screener.py (suffixed symbols)
- Yahoo suffix list; DAX/KOSPI200 Wikipedia; Tobi Lux data-quality study; yfinance #2125 rate-limit
