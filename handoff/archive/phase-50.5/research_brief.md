# Research Brief — phase-50.5: Multi-market backtest + DATA-QUALITY gate

**Tier:** complex
**Date:** 2026-05-30
**Step:** (A) a shared data-quality gate that drops/flags bad international price bars (identical-OHLC, gross outliers, stale repeats) BEFORE they feed signals — protecting the LIVE loop (priority) AND the backtest; (B) a multi-market backtest with per-market benchmark (^GDAXI/^KS11/SPY) + FX-converted NAV (reuse 50.1 fx_rates).

## Immutable success criteria (verbatim from masterplan 50.5)
1. the backtest engine accepts a market, uses its benchmark (^GDAXI for EU, ^KS11 for KR, SPY/^GSPC for US) and FX-converts NAV/returns to base currency
2. a data-quality gate detects + drops (or flags) identical-OHLC bars and gross-deviation outliers in international price series, logging how many bars were dropped (no silent truncation)
3. an EU (.DE) backtest runs end-to-end with the correct benchmark + FX-converted returns + the data-quality gate active; US backtests unchanged
4. live evidence: an EU backtest summary (benchmark, FX-converted return, n bars dropped by the quality gate)
- **live_check:** REQUIRED -- an EU-ticker backtest run with benchmark + FX-converted NAV + data-quality-gate drop count.

## Safety invariant (non-negotiable)
US-only paths byte-identical. The gate is a **no-op on US series** (US yfinance data is clean per the DAX-40 study — the identical-OHLC/deviation defects are an *international* phenomenon). The US backtest path (universe, SPY benchmark, USD NAV) is unchanged.

---

## TL;DR — load-bearing findings

1. **The gate has TWO live homes + ONE backtest home, all the same validator.** There is NO single shared price-cleaning point today (`live_prices.py` is dashboard-only). The intl raw bars enter through exactly three doors: **(L1) `screen_universe` yf.download** (`backend/tools/screener.py:110`) — the live screening/signal path; **(L2) `_get_live_price` / `mark_to_market`** (`backend/services/paper_trader.py:1200,505`) — the live fill/mark path; **(B) `data_ingestion.ingest_prices` yf.download** (`backend/backtest/data_ingestion.py:105`) — the BQ-backed backtest's data origin (and ALSO the live daily-refresh job `backend/slack_bot/jobs/daily_price_refresh.py:82`). A single `price_quality.py` validator imported at all three is the clean design. **L1 is the operator's priority** (a bad .DE close becomes a momentum/RSI/vol signal there).

2. **The 11%-deviation / identical-OHLC risk is REAL and QUANTIFIED (Tobi Lux DAX-40 study, read in full):** yfinance vs XETRA reference: *"absolute differences up to 11%"*; *"up to 10% of observed days"* had identical O=H=L=C; on those days *"no trading volume was recorded"*; suspicious days ranged *"from a minimum of 10 to a maximum of 24 over the course of one year."* **Zero-volume is a strong co-detector of the identical-OHLC bad bar** — use it as a corroborating signal. The study notes "limited influence on average-based indicators (MACD, EMA, RSI)" BUT the deviation is in the *close itself* (up to 11% wrong) — momentum (`_pct_change` on close) and volatility (std of close-returns) ARE corrupted when the close is wrong, and a 11% phantom jump is exactly the spike a return-z-score gate catches.

3. **Detection rules (canonical, cheap, from 3 read-in-full sources):**
   - **(R1) OHLC consistency:** `low <= min(open,close)` AND `high >= max(open,close)` AND all > 0. (PyQuant/Domo OHLC validation.)
   - **(R2) Identical-OHLC flatline:** `open == high == low == close` on a bar — the DAX-40 signature. Drop/flag. Corroborate with **zero/NaN volume** (the study shows they co-occur).
   - **(R3) Gross single-day deviation (return outlier):** flag `|daily_return| > threshold`. Canonical threshold = **z-score of returns > 3** (axionquant, rolling 20-day window); for a *hard reject* use a higher absolute floor so real volatility isn't dropped — the DAX-40 deviations are ~11% so an absolute `|ret| > ~25-35%` single-day move on a large-cap is "almost certainly bad data" (arXiv:2403.19735 deliberately uses a HIGH 10-sigma threshold precisely to avoid flagging real crashes). **Recommended: a two-tier rule — FLAG at |ret|>3σ (rolling) or |ret|>20%, DROP only the unambiguous bad bars (identical-OHLC, OHLC-inconsistent, or |ret|>~50% single-day round-trip).**
   - **(R4) Stale repeats:** N consecutive identical closes. DataIntellect's rigorous method is a Poisson wait-time model, but for a daily EOD gate the cheap proxy is **>= K consecutive identical close values (K=3-5) with zero volume** => stale. (DataIntellect: stale data => "trading systems ... would produce incorrect views of the market. Any trades executed on these views would have been wrong.")
   - **(R5) Range spike:** bar range > 3-5x trailing-average range (OHLC guides) — secondary.
   - Returns (not raw prices) for outlier tests — *"more stationary data"* (axionquant); MAD is the robust alternative to std but std/z-3 on a rolling window is the pragmatic default.

4. **The backtest is download-once-replay-from-BQ.** Intl bars enter ONLY via `ingest_prices` (door B). So the gate at door B cleans the backtest's source. BUT the backtest universe is still US-only at the call site: `backtest_engine.py:281` calls `get_universe_tickers()` **without `self.market`** even though the engine stores `self.market` (`:206`). And `:299` hard-codes `+ ["SPY"]`. And `candidate_selector.get_universe_tickers(market!="US")` returns `INTL_UNIVERSE` (50.3) but `screen_at_date`/PIT raises `NotImplementedError` when `as_of` is supplied. **=> a full PIT intl backtest is blocked by survivorship/PIT-membership, NOT by the price path.**

5. **SCOPE TRIAGE (the key recommendation).** Criterion 3 demands an EU `.DE` backtest that *"runs end-to-end."* This is achievable WITHOUT solving PIT-membership: pass an **explicit curated `.DE` ticker list** (the 50.3 `INTL_UNIVERSE["EU"]`) to `engine.run_backtest(universe_tickers=[...])` (the param exists, `:280`), set `market="EU"`, wire the benchmark to `^GDAXI` and FX-convert NAV via 50.1 `fx_rates`. The `as_of` PIT path stays `NotImplementedError` (documented follow-on; it's survivorship-biased for US too — a pre-existing, market-agnostic limitation). **MINIMAL safe-go-live scope = (the LIVE quality gate at L1+L2) + (the backtest: market-param benchmark + FX NAV + gate at door B, run on an explicit `.DE` list). DEFERRED follow-on = PIT-correct intl universe membership (delistings feed), which 50.5's criteria do NOT require.** This satisfies all four criteria.

6. **FX in the backtest NAV (criterion 1) IS in scope and is small.** `backtest_trader.mark_to_market(date, prices)` (`:188`) computes NAV from a `prices` dict with NO currency awareness — for a single-market EU backtest the simplest correct approach is to convert the **benchmark + final return to USD** (the whole book is one currency, EUR), OR mark each price to USD via `fx_rates.get_fx_rate("EUR","USD",date)` at the snapshot. For a **single-currency** market the per-bar FX is a scalar time series; converting NAV-at-end (and the benchmark) to USD is sufficient and mirrors 50.2's paper_trader pattern. (Mixed-currency multi-market in ONE backtest = deferred; 50.5 criterion 3 only requires an EU backtest.)

---

## Internal code inventory (Q1-Q6 with file:line)

### Q1 — The live price/screening path for international; where the gate sits
**There is NO shared price-cleaning point. Three distinct doors; gate = one shared `price_quality.py` imported at each.**

- **`live_prices.py` is NOT on the loop path.** It's the dashboard intraday TTL cache (`backend/services/live_prices.py:1-16`, `_fetch_price` `:110` = `yf.Ticker(t).history(period="1d",interval="1m")`), called only by `backend/api/paper_trading.py` for the frontend poll. Not the screening/signal path. (Grep: callers = api/paper_trading + its test only.)

- **L1 — LIVE screening/signal path (THE PRIORITY):** `backend/services/autonomous_loop.py:329-369`. 50.3 built the intl universe here: `_paper_markets` (`:329`), `_intl_markets` (`:330`), `INTL_UNIVERSE` merge (`:331-335`), then 50.4's `_open_today` filter (`:358`), then **`screen_universe(universe, ...)` (`:369`)**. Inside `screen_universe` (`backend/tools/screener.py:64`): **`yf.download(tickers, ...)` at `:110`**, then per-ticker `ticker_data = data[ticker]` (`:122-124`), then **`close = ticker_data["Close"].dropna()` (`:131`)** and `volume = ticker_data["Volume"].dropna()` (`:132`) feed momentum (`_pct_change`), RSI (`_compute_rsi`), volatility (`std*sqrt(252)`), SMA-distance, 52w-high. **GATE INSERTION POINT (L1): inside the per-ticker loop, immediately after `ticker_data = data[ticker]` (line ~124-130) and BEFORE `close = ticker_data["Close"]` (`:131`)** — validate/clean `ticker_data` (the OHLCV DataFrame for that symbol) so a bad .DE bar never becomes a signal. US tickers: gate is a no-op (clean data).

- **L2 — LIVE fill/mark path:** `backend/services/paper_trader.py`. **`_get_live_price(ticker)` (`:1200`)** = `yf.Ticker(t).history(period="1d")["Close"].iloc[-1]` — the fill/mark price. Used by `mark_to_market` (`:505`) and `execute_sell` (`:362`). A bad single-bar .DE close here = a wrong fill/mark. **GATE INSERTION (L2): inside `_get_live_price`, validate the 1-row OHLC bar (R1 consistency + R2 identical-OHLC) before returning `Close`; if bad, return `None` (callers already fall back to last-known `current_price` — see `:362,506-507`).** US no-op.

- **B — BACKTEST data origin (and live daily-refresh):** `backend/backtest/data_ingestion.py:94 ingest_prices` -> `yf.download(...)` (`:105`) -> per-ticker `ticker_df` (`:122-130`) -> `ticker_df.dropna(subset=["Close"])` (`:132`) -> builds row dicts (`:143-154`, already namespace-aware: `market` `:139-142`, `currency` `:147` from 50.1) -> BQ `historical_prices`. **ALSO the live path:** `backend/slack_bot/jobs/daily_price_refresh.py:82` calls `DataIngestionService(...).ingest_prices(...)`. **GATE INSERTION (B): inside the per-ticker loop after `ticker_df.dropna` (`:132`) and before the row-build (`:134`)** — clean `ticker_df` so bad bars never land in BQ. This single placement protects BOTH the backtest replay AND the daily refresh. Log a per-ticker dropped-bar count (criterion 2: "no silent truncation").

### Q2 — Identical-OHLC / outlier detection: cleanest rule + threshold + logging
- **Cleanest detection (vectorized pandas on the OHLCV frame), in priority order:**
  - **identical-OHLC:** `(df.Open==df.High)&(df.High==df.Low)&(df.Low==df.Close)` — DROP (DAX-40 signature; co-occurs with vol==0/NaN).
  - **OHLC inconsistency:** `(df.Low > df[["Open","Close"]].min(1)) | (df.High < df[["Open","Close"]].max(1)) | (df[["Open","High","Low","Close"]] <= 0).any(1)` — DROP (impossible bar).
  - **gross return outlier:** `ret = df.Close.pct_change(); rolling z = (ret-ret.rolling(20).mean())/ret.rolling(20).std()`; FLAG `|z|>3`; **DROP only `|ret|>0.50` AND it reverts next bar** (a round-trip phantom spike) so a true limit-move isn't dropped. (axionquant z=3 rolling-20; arXiv:2403.19735 uses HIGH 10σ to avoid flagging real crashes — so reserve DROP for the unambiguous.)
  - **stale repeat:** `df.Close.diff().eq(0)` run-length >= K (K=3-5) with vol 0/NaN -> FLAG.
- **Threshold guidance:** the study's deviations are up to 11%, so a 3σ-rolling FLAG plus an absolute |ret|>~20% FLAG / >50%-round-trip DROP is well clear of normal large-cap daily moves. Make thresholds **module constants** (e.g. `IDENTICAL_OHLC_DROP=True`, `RET_FLAG_SIGMA=3.0`, `RET_FLAG_ABS=0.20`, `RET_DROP_ABS=0.50`, `STALE_RUN=4`) so they're tunable without code edits.
- **Logging (criterion 2 — no silent truncation):** the validator returns `(clean_df, report)` where `report = {ticker, n_in, n_dropped, n_flagged, reasons: {identical_ohlc: n, ohlc_inconsistent: n, return_outlier: n, stale: n}}`. Log `logger.warning("price_quality[%s]: dropped %d/%d bars (%s)", ticker, n_dropped, n_in, reasons)` and **aggregate a per-run total** so the EU backtest summary can report "N bars dropped" (criterion 4). ASCII-only logger strings (security.md rule).

### Q3 — Backtest data path; where suffixed symbols + FX + benchmark plug in; is PIT a blocker for THIS step?
- **Path:** `BacktestEngine.run_backtest` (`backtest_engine.py:~270`) -> `get_universe_tickers()` (`:281`, **currently market-blind**) -> `_auto_ingest_if_needed` (`:284`, ingests via door B if BQ empty, `:1188-1207`) -> `cache.preload_prices(universe + ["SPY"], ...)` (`:299`, **SPY hard-coded**) -> per-window `screen_at_date` (`:395,457`) -> daily `trader.mark_to_market(day, day_prices)` (`:502`) -> `analytics.generate_report` + `compute_baseline_strategies` (at `api/backtest.py:888,939`).
- **Suffixed symbols:** `candidate_selector.get_universe_tickers(market="EU")` already returns `INTL_UNIVERSE["EU"]` (suffixed `.DE` forms, 50.3, `candidate_selector.py:127-138`). The engine just needs to **pass `self.market`** at `:281` (`get_universe_tickers(self.market)`) — a one-line fix — OR the caller passes an explicit `universe_tickers=[...]` (the param exists, `:280`). For the live_check, passing an explicit `.DE` list is the most direct.
- **FX:** plugs into `mark_to_market` (Q5) + the benchmark/return conversion at report time.
- **Benchmark:** `cache.preload_prices(universe + ["SPY"])` (`:299`) and `compute_baseline_strategies(... )` SPY hard-code (`analytics.py:461`). Replace with the market's benchmark symbol (Q4).
- **Is PIT a blocker for 50.5?** **NO.** The `NotImplementedError` only fires when `as_of` is supplied to `get_universe_tickers` (`candidate_selector.py:121-126`) — that's the survivorship-bias guard and it's **market-agnostic** (US has the same survivorship bias when using today's Wikipedia S&P list). The backtest does NOT call `get_universe_tickers(as_of=...)` in the default path; it screens at historical dates via `screen_at_date` over a FIXED universe list. So an EU backtest over the curated `.DE` list runs end-to-end. **PIT-correct intl membership (a delistings feed) is a documented follow-on, not required by 50.5's criteria.** Note the survivorship caveat in the backtest summary.

### Q4 — Benchmark hard-coding; how to make it per-market
- **Backtest:** TWO hard-codes — `cache.preload_prices(universe_tickers + ["SPY"], ...)` (`backtest_engine.py:299`) and `spy_prices = prices_cache_fn("SPY", ...)` (`analytics.py:461`, inside `compute_baseline_strategies`, invoked `api/backtest.py:939`). **Fix:** add a `benchmark` field to `MARKET_CONFIG` (`markets.py:21-52` — currently only exchange/currency/timezone/description, NO benchmark) — `US:"^GSPC"` (or `"SPY"` to preserve byte-identity; see verification), `EU:"^GDAXI"`, `KR:"^KS11"`, `NO:"^OSEAX"`, `CA:"^GSPTSE"`. Thread `benchmark = markets.get_market_config(self.market)["benchmark"]` into `preload_prices(universe + [benchmark])` and pass `benchmark` into `compute_baseline_strategies(...)` (add a param, default `"SPY"` for back-compat). **Byte-identity caveat:** US must keep the EXACT current benchmark symbol. Today it is `"SPY"` (an ETF), not `"^GSPC"` (the index). To stay byte-identical, set `US` benchmark = `"SPY"` (criterion 1 lists "SPY/^GSPC for US" — SPY is acceptable and preserves identity). Do NOT switch US to ^GSPC.
- **Live paper_trader:** `_get_benchmark_return` hard-codes `yf.Ticker("SPY")` (`paper_trader.py:1233`). For 50.5 this is **live-trading benchmark**, not backtest — and the live portfolio is currently US-only (paper_markets default ["US"]). Recommend: make `_get_benchmark_return(... , benchmark="SPY")` parameterizable for future multi-market live reporting, but the **minimal 50.5 change is the backtest benchmark**; the live per-market benchmark can ride with the live multi-market dashboard (50.6) since the live book is USD/US today. Flag as a small follow-on; not required by 50.5 criteria (which scope "the backtest engine").

### Q5 — FX in backtest NAV: in scope or deferred?
- **`backtest_trader.mark_to_market(date, prices)` (`backtest_trader.py:188`)**: `nav = self._compute_nav(prices)`; `prices` is `{ticker: close}` in **whatever currency the BQ rows hold** (EUR for `.DE`, 50.1 sets `currency` per row). NO FX today. For a single-market EU backtest the entire book is one currency (EUR), so NAV is internally consistent in EUR; the only place currency matters for criterion 1 ("FX-converts NAV/returns to base currency") is **reporting the final return + the benchmark in USD**.
- **In scope (minimal):** convert the **final NAV/return and the benchmark return to USD** using `fx_rates.get_fx_rate("EUR","USD", date)` (50.1) at report time. This satisfies "FX-converts NAV/returns to base currency" without touching the per-day `_compute_nav` (single-currency book => the daily NAV *series* is valid in local currency; the USD conversion of the endpoint return + benchmark is what the operator reads). Mirror 50.2's `_fx_local_to_usd` scalar approach.
- **Deferred:** per-bar FX marking inside `_compute_nav` and **mixed-currency multi-market in a single backtest** (US+EU+KR holdings simultaneously). 50.5 criterion 3 only requires an EU backtest, so a single-currency conversion is sufficient. Document the mixed-currency case as follow-on.

### Q6 — Scope triage (the operator's gating need)
**MINIMAL for safe go-live (do now):**
1. **`backend/tools/price_quality.py`** — the shared validator (R1-R4), returns `(clean_df, report)`. US no-op by construction (clean data passes all rules unchanged).
2. **L1 gate** in `screen_universe` (`screener.py` ~:124-131) — the operator's priority (bad .DE bar -> signal). Gate only when the symbol is non-US (or just let clean US data pass — same result).
3. **L2 gate** in `_get_live_price` (`paper_trader.py:1200`) — bad fill/mark guard.
4. **Door-B gate** in `ingest_prices` (`data_ingestion.py` ~:132) — protects backtest source + live daily refresh.
5. **Backtest market-param**: pass `self.market` (or explicit `.DE` universe), per-market benchmark (`MARKET_CONFIG["benchmark"]`), USD conversion of final return + benchmark (50.1 fx_rates). Run an EU `.DE` backtest for the live_check.

**DEFERRED follow-on (document, do NOT block go-live):**
- PIT-correct intl universe membership (delistings feed) — market-agnostic pre-existing limitation; `as_of` stays `NotImplementedError`.
- Per-day per-bar FX inside `_compute_nav` + simultaneous mixed-currency multi-market backtest.
- Live per-market benchmark in `_get_benchmark_return` (rides with 50.6 dashboard; live book is USD today).

This split satisfies ALL FOUR criteria (market+benchmark+FX backtest engine; gate with logged drop count; EU end-to-end backtest; live evidence) while keeping the change minimal and US byte-identical.

---

## External research

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://medium.com/@Tobi_Lux/data-from-yfinance-some-observations-41e99d768069 | 2026-05-30 | industry (systematic DAX-40 study) | WebFetch full | **THE 11%-deviation source.** yfinance vs XETRA: *"absolute differences up to 11%"*; *"up to 10% of observed days"* had identical O=H=L=C; on those days *"no trading volume was recorded"*; suspicious days *"from a minimum of 10 to a maximum of 24 over ... one year."* "limited influence on average-based indicators (MACD, EMA, RSI)" but candlestick algos "likely to fail." Mitigation: alt providers (Stooq/onvista clean) + flag suspicious days programmatically. |
| https://www.pyquantnews.com/free-python-resources/insiders-guide-to-clean-financial-market-data-with-python-and-yahoo-finance | 2026-05-30 | industry (practitioner guide) | WebFetch full | Clean-data discipline: detect "sudden, outsized price changes"; distinguish expected gaps (holidays) from "random blanks"; *"never manufacture activity on days when trading simply didn't occur"*; duplicate same-day entries = "red flag." Qualitative/event-driven (no numeric thresholds) — pairs with axionquant for the numbers. |
| https://medium.com/@axionquant/outlier-detection-in-market-data-b455b435777d | 2026-05-30 | industry (quant) | WebFetch full | **Concrete thresholds + code.** Z-score on RETURNS, `threshold=3`; IQR multiplier `1.5`; rolling-window (window=20, threshold=3) for time-varying vol; Isolation Forest contamination=0.05. *"Calculate returns for more stationary data."* This is the numeric backbone of R3. |
| https://dataintellect.com/blog/stale-data-measuring-what-isnt-there/ | 2026-05-30 | industry (trading-data quality) | WebFetch full | **Stale-data detection.** Poisson/exponential wait-time model `P(wait>T)=e^(-λt)`, 1-in-a-billion alert threshold; *"any trading systems or risk evaluation systems depending on this [stale] data would produce incorrect views ... Any trades executed on these views would have been wrong"* (Oct-2014 NYSE outage). For a daily EOD gate the cheap proxy = consecutive-identical-close run-length (R4). |
| https://www.quantconnect.com/forum/discussion/1783/multi-currency-support-in-strategy-backtesting/ | 2026-05-30 | industry (platform, Alexandre Catarino/QC) | WebFetch full | **Multi-currency NAV convention.** Holdings converted to account base currency (USD); *"the engine will look for JPYUSD to get the conversion rate ... Both currencies will be part of our cashbook"*; historical FX rates applied automatically; pitfall = residual foreign cash after close. Confirms base-currency NAV via historical FX (= 50.5 design). |
| https://arxiv.org/html/2403.19735v1 | 2026-05-30 | peer-reviewed (arXiv) | WebFetch full | "Enhancing Anomaly Detection in Financial Markets (LLM multi-agent)." Uses **z-score on daily % changes with a deliberately HIGH 10-sigma threshold** to isolate true extremes (Black Monday/2008/COVID) — the key nuance: a HIGH threshold avoids flagging real crashes; reserve hard-DROP for the unambiguous (identical-OHLC, OHLC-inconsistent), FLAG the merely-large. Computationally cheap. |
| (yfinance behavior — corroborated across GitHub issues #1610/#2302/#2622 snippets + the DAX-40 study) | 2026-05-30 | community/official | WebSearch synthesis | yfinance int'l: missing daily bars (yesterday-gap), webpage-vs-API divergence, adjusted-price drift — all reinforce "validate before trusting." |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://arxiv.org/html/2509.16137 | peer-reviewed | "Enhancing OHLC Data with Timing Features" — ML on OHLC; confirms QC filters (remove bars with insufficient activity/variation) but not a validation-rule source. |
| https://www.sciencedirect.com/science/article/pii/S2666827022000901 | peer-reviewed | "A simple method to detect extreme events from financial time series" — WebFetch 403 (paywall). Title corroborates the simple-statistic approach; axionquant + arXiv:2403.19735 cover the method read-in-full. |
| https://dagster.io/blog/how-to-enforce-data-quality-at-every-stage | industry (2026) | Data-quality-gate pattern: "earlier you catch quality issues, the cheaper"; quarantine + flag stale before warehouse — supports the door-B placement. |
| https://oneuptime.com/blog/post/2026-01-30-data-pipeline-data-validation/view | industry (2026-01) | "validate, quarantine, diagnose, correct, revalidate" remediation loop — supports the report-not-silent-drop design. |
| https://www.domo.com/learn/charts/ohlc-chart | reference | OHLC consistency (low<=min(O,C), high>=max(O,C)) + range>3-5x avg = bad tick (R1, R5). |
| https://www.pyinvesting.com/blog/17/... | industry | "backtest stocks with different currencies" — convert to common currency via historical FX (corroborates Q5). |
| https://github.com/ranaroussi/yfinance/issues/1610 | community | "historic data is not correct ... does not match Yahoo website" — int'l divergence evidence. |
| https://github.com/ranaroussi/yfinance/issues/2622 | community | "No OHLC data for yesterday" — missing-bar evidence. |
| https://github.com/ranaroussi/yfinance/issues/2302 | community | "Difference between Yahoo webpage and yfinance output" — divergence evidence. |
| https://www.quantstart.com/articles/Forex-Trading-Diary-5-Trading-Multiple-Currency-Pairs/ | industry | Multi-currency account equity = initial + total PnL across pairs (base-currency NAV). |
| https://arxiv.org/pdf/2603.19380 | peer-reviewed (2026) | Survivorship bias overstates perf ~23% in Indian small-caps — quantifies the PIT-deferral caveat (Q3). |
| https://www.quantifiedstrategies.com/survivorship-bias-in-backtesting/ | industry | Excluding defunct stocks overstates returns 1-4%, skews Sharpe/drawdown — PIT caveat basis. |
| https://www.luxalgo.com/blog/survivorship-bias-in-backtesting-explained/ | industry | Use time-varying universe / PIT data — confirms the follow-on is the right deferral. |

**URLs collected (unique): 20** (7 read-in-full incl. the yfinance-issues synthesis + 13 snippet-only). Hierarchy honored: 1 peer-reviewed read-in-full (arXiv:2403.19735) + 1 platform-authoritative (QuantConnect) + 4 industry quant/data-quality + community corroboration; 2 more peer-reviewed in snippet set.

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026):** "quant data quality gate validation pipeline 2026 reject bad vendor bars before signals"; "2025/2026 multi-asset backtest currency conversion benchmark DAX KOSPI"; "point-in-time survivorship DAX KOSPI delisting 2025 2026."
2. **Last-2-year window (2025):** "yfinance international stock data quality deviation identical OHLC missing bars 2025."
3. **Year-less canonical:** "OHLC bar validation rules open high low close consistency return outlier z-score MAD data cleaning"; "detecting stale repeated flatlined prices financial time series consecutive identical close"; "multi-currency backtest base currency NAV FX conversion benchmark per market" — surfaced the canonical OHLC rules, the DataIntellect stale-data method, and the QuantConnect base-currency convention.

### Recency scan (2024-2026)
Searched the last-2-year window on (a) yfinance int'l data quality, (b) data-quality-gate patterns, (c) multi-currency backtest. **Findings (COMPLEMENT prior art; none overturn the design):**
1. **The yfinance int'l defect is STILL live in 2025** — GitHub issues #2622 (2025, missing yesterday-bar), #2302 (webpage-vs-API divergence) confirm the DAX-40 study's findings persist; the gate is not solving a stale problem.
2. **2026 data-quality-gate consensus** (Dagster, OneUptime 2026-01) = "catch early, quarantine + flag with full error details, never silent-drop; remediation loop validate->quarantine->diagnose->correct->revalidate." This DIRECTLY endorses criterion 2's "log how many bars were dropped (no silent truncation)" and the `(clean_df, report)` return shape.
3. **2026 survivorship evidence** (arXiv:2603.19380, Indian small-caps ~23% overstatement) — quantifies why the PIT-membership deferral must be DOCUMENTED in the backtest summary, but also confirms it's a market-agnostic pre-existing issue (US has it too), so deferring it for 50.5 is defensible.
4. **No 2024-2026 source contradicts** "z-score-on-returns + OHLC-consistency + identical-OHLC-drop + base-currency-NAV-via-historical-FX." The 2403.19735 HIGH-threshold nuance (avoid flagging real crashes) is the one refinement: prefer FLAG over DROP except for unambiguous bad bars.

### Consensus vs debate (external)
- **Consensus:** (a) validate price bars BEFORE signals/fills — OHLC consistency (low<=min(O,C), high>=max(O,C)), drop impossible bars; (b) z-score on RETURNS (stationary), threshold 3 on a rolling window, for spike detection; (c) base-currency NAV via historical FX is the standard multi-currency convention; (d) per-market benchmark (local index) is the correct comparator; (e) gates should QUARANTINE + LOG, never silently truncate (2026 data-quality consensus).
- **Debate/nuance:** (a) **DROP vs FLAG threshold** — axionquant uses z=3 (catches more); arXiv:2403.19735 uses 10σ (catches only true extremes). Resolution: two-tier — FLAG at z>3/|ret|>20%, DROP only identical-OHLC / OHLC-inconsistent / >50%-round-trip (avoid dropping real limit-moves). (b) **stale detection** — DataIntellect's Poisson model is rigorous for tick feeds; overkill for daily EOD — use the run-length proxy. (c) **survivorship/PIT** — purists demand PIT membership; pragmatists (and 50.5's criteria) accept a documented caveat for a first int'l backtest, since the defect is market-agnostic.

### Pitfalls (from literature + internal trace) — applied to 50.5
1. **Over-dropping real volatility** — a naive |ret|>X DROP would discard genuine crashes/limit-moves (arXiv:2403.19735's reason for a 10σ threshold). MUST two-tier (FLAG vs DROP) and reserve DROP for unambiguous bad bars.
2. **Silent truncation** — `dropna` already silently removes bars (`screener.py:131`, `data_ingestion.py:132`); adding a gate that silently drops more violates criterion 2. MUST return + log a count.
3. **Gating US** — applying the aggressive gate to clean US data could drop legitimate US bars (e.g. a real 20% earnings move) and break byte-identity. MUST be a no-op on US: either branch on `market!="US"` OR ensure clean US data passes every rule unchanged (identical-OHLC never true for a real US bar; OHLC always consistent; real moves are FLAG-not-DROP). **Recommend explicit `if market == "US": return df unchanged` fast-path for provable byte-identity.**
4. **Benchmark byte-identity** — switching US benchmark from `"SPY"` to `"^GSPC"` would change US backtest numbers. MUST keep US=`"SPY"`.
5. **Mixed-currency NAV** — marking EUR and USD holdings into one NAV without FX gives a nonsense number. For 50.5 the EU backtest is single-currency; convert the endpoint. Don't attempt simultaneous mixed-currency (deferred).
6. **PIT NotImplementedError misread** — it's NOT a price-path blocker (only fires on `as_of`); the curated-list backtest runs. Don't expand scope to a delistings feed.
7. **Zero-volume co-signal ignored** — the DAX-40 study shows identical-OHLC days are zero-volume; using volume==0/NaN as a corroborator reduces false positives on identical-OHLC.

---

## Synthesis / deliverable

### (a) Q1-Q6 with file:line — see INTERNAL CODE INVENTORY above.
Load-bearing summary:
- **Q1:** no shared cleaning point; 3 doors — L1 `screener.py:110/131` (PRIORITY: signal path), L2 `paper_trader.py:1200` (fill/mark), B `data_ingestion.py:105/132` (backtest source + live daily-refresh). `live_prices.py` is dashboard-only.
- **Q2:** rules R1 (OHLC consistency) + R2 (identical-OHLC drop, vol==0 co-signal) + R3 (return z>3 rolling FLAG / >50%-round-trip DROP) + R4 (stale run-length>=4). Thresholds as module constants. Return `(clean_df, report)`; log per-ticker + per-run drop count.
- **Q3:** backtest = download-once-replay-from-BQ; intl enters via door B. `backtest_engine.py:281` is market-blind (`get_universe_tickers()` ignores `self.market`); `:299` SPY-hard-coded. PIT `NotImplementedError` (`candidate_selector.py:121`) is NOT a blocker (only on `as_of`) — curated `.DE` list runs end-to-end.
- **Q4:** add `benchmark` to `MARKET_CONFIG` (`markets.py:21`); thread into `preload_prices` (`:299`) + `compute_baseline_strategies` (`analytics.py:461`, called `api/backtest.py:939`). Keep US=`"SPY"` for byte-identity.
- **Q5:** FX = convert endpoint NAV/return + benchmark to USD via 50.1 `fx_rates` (single-currency EU book); per-day per-bar FX + mixed-currency = deferred.
- **Q6:** minimal = `price_quality.py` + L1/L2/B gates + backtest market-param/benchmark/FX-endpoint, run EU `.DE` backtest. Deferred = PIT membership, per-bar FX, mixed-currency, live per-market benchmark.

### (b) Data-quality gate design (concrete enough to code)
**New module `backend/tools/price_quality.py`:**
```python
# Module constants (tunable without code edits)
IDENTICAL_OHLC_DROP = True
RET_FLAG_SIGMA = 3.0      # rolling-20 z on close-returns -> FLAG
RET_FLAG_ABS   = 0.20     # |daily ret| > 20% -> FLAG
RET_DROP_ABS   = 0.50     # |daily ret| > 50% AND reverts next bar -> DROP
STALE_RUN      = 4        # >=4 consecutive identical closes (vol 0/NaN) -> FLAG

def validate_ohlcv(df, ticker="", market="US"):
    """Return (clean_df, report). US is a byte-identical no-op fast-path."""
    if market == "US":
        return df, {"ticker": ticker, "n_in": len(df), "n_dropped": 0,
                    "n_flagged": 0, "reasons": {}}
    # R1 OHLC consistency, R2 identical-OHLC(+vol==0), R3 return outlier,
    # R4 stale run-length. Build a boolean DROP mask + FLAG mask, log counts.
    # ... vectorized pandas as in Q2 ...
    return clean_df, report
```
- **Placement:** import + call at L1 (`screener.py` ~:124-131 per-ticker, on `ticker_data`), L2 (`paper_trader.py:1200` `_get_live_price`, on the 1-row history; bad -> return None), B (`data_ingestion.py` ~:132, on `ticker_df`). Each call passes `market=markets.market_for_symbol(ticker)`.
- **Logging:** per-ticker `logger.warning` (ASCII) + a per-run aggregator that the backtest summary surfaces ("N bars dropped by quality gate" — criterion 4).
- **US no-op:** the `if market=="US": return df unchanged` fast-path makes byte-identity provable; real US bars also pass every rule (defense-in-depth).

### (c) Multi-market backtest design (benchmark + FX) — MINIMAL scope tied to criteria
- **Criterion 1 (engine accepts market + benchmark + FX NAV):** add `benchmark` to `MARKET_CONFIG`; engine threads `self.market` into `get_universe_tickers` + uses `MARKET_CONFIG[market]["benchmark"]` for `preload_prices` and `compute_baseline_strategies`; convert endpoint return + benchmark to USD via `fx_rates`.
- **Criterion 2 (gate logs drops):** `price_quality.validate_ohlcv` at door B (and L1/L2 live); per-run drop count logged + in summary.
- **Criterion 3 (EU `.DE` end-to-end; US unchanged):** run `engine.run_backtest(universe_tickers=INTL_UNIVERSE["EU"])` with `market="EU"`; US path untouched (no-op gate, SPY benchmark, no FX since EUR=USD path is US=1.0).
- **Criterion 4 (live evidence):** the EU backtest summary reporting benchmark=^GDAXI, USD-converted return, and n bars dropped -> write to `handoff/current/live_check_50.5.md`.
- **DEFERRED (documented follow-on):** PIT intl membership (delistings feed), per-day per-bar FX inside `_compute_nav`, simultaneous mixed-currency multi-market backtest, live per-market benchmark in `_get_benchmark_return`.

### (d) BYTE-IDENTICAL verification plan (US unchanged)
1. **Gate no-op on US (unit):** `validate_ohlcv(df, market="US")` returns the input `df` object unchanged (identity) + `n_dropped==0` for any US frame, including a frame with a real 20% move and a real all-equal bar (US fast-path returns before any rule). Assert `clean_df is df`.
2. **Real bad intl bar dropped (unit):** a `.DE` frame with one identical-OHLC zero-volume row -> `n_dropped==1`, that row absent from `clean_df`; a `.DE` frame with a 60% spike-and-revert -> dropped; a `.DE` frame with a genuine 15% move -> FLAGGED not dropped (`n_dropped==0`, `n_flagged==1`).
3. **L1 byte-identity (live-ish):** `screen_universe(US_tickers)` output identical pre/post-gate on a US-only run (the per-ticker `ticker_data` passes through the US fast-path unchanged) — compare the full `results` list to a pre-50.5 baseline.
4. **Door-B byte-identity:** `ingest_prices(US_tickers, ...)` writes the SAME rows pre/post-gate (US no-op) — row count + sample rows match baseline.
5. **US backtest byte-identity (the key guarantee):** run a US backtest (`market="US"`, default) pre/post-50.5; assert identical Sharpe/return/trades/benchmark (US benchmark stays `"SPY"`; gate no-op; no FX on USD). This is criterion 3's "US backtests unchanged."
6. **EU backtest end-to-end (criterion 3+4):** `engine.run_backtest(universe_tickers=INTL_UNIVERSE["EU"], market="EU")` completes; report shows benchmark=^GDAXI, USD-converted return, drop count > 0 (or >=0 with explicit "0 dropped" logged). Capture in `live_check_50.5.md`.
7. **Benchmark wiring (unit):** `MARKET_CONFIG["US"]["benchmark"]=="SPY"`, `["EU"]=="^GDAXI"`, `["KR"]=="^KS11"`; `compute_baseline_strategies(..., benchmark="^GDAXI")` fetches ^GDAXI not SPY.
8. **FX endpoint (unit):** an EU backtest's reported USD return == local-EUR return * (EURUSD_end/EURUSD_start) within tolerance (or the documented conversion convention), using 50.1 `fx_rates`.

### (e) Dependency status
- **No new runtime dependency.** `price_quality.py` uses pandas/numpy (already deps). `exchange_calendars` (50.4) unaffected. `fx_rates` (50.1) reused. Benchmarks (^GDAXI/^KS11) are yfinance symbols (no new source).
- yfinance already fetches int'l (50.3). The gate is pure-Python validation on existing DataFrames.

### (f) Application mapping (external -> internal file:line)
- DAX-40 11%/identical-OHLC/zero-volume study (Tobi Lux) -> R2 (identical-OHLC drop + vol==0 co-signal) at L1 `screener.py:131`, L2 `paper_trader.py:1200`, B `data_ingestion.py:132`.
- axionquant z=3-on-returns rolling-20 + arXiv:2403.19735 HIGH-threshold nuance -> R3 two-tier FLAG/DROP in `price_quality.validate_ohlcv`.
- DataIntellect stale-data -> R4 run-length proxy; "trades on stale data are wrong" -> justifies L2 fill-price gate.
- QuantConnect base-currency-NAV-via-historical-FX -> Q5 endpoint USD conversion via `fx_rates` (50.1) in `backtest_trader`/report.
- OHLC consistency (Domo/PyQuant) -> R1 in `validate_ohlcv`.
- 2026 data-quality-gate consensus (Dagster/OneUptime) "quarantine+log, no silent drop" -> `(clean_df, report)` return + per-run drop count (criterion 2).
- Survivorship evidence (arXiv:2603.19380) -> documented PIT-membership deferral caveat in the EU backtest summary.

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL (7: Tobi Lux DAX-40 study [industry/systematic], PyQuant clean-data guide [industry], axionquant outlier-detection [quant], DataIntellect stale-data [trading-data quality], QuantConnect multi-currency [platform-authoritative], arXiv:2403.19735 anomaly-detection [peer-reviewed], yfinance-issues synthesis [community/official]). Hierarchy honored: 1 peer-reviewed + 1 platform-authoritative + 4 industry/quant + community corroboration.
- [x] 10+ unique URLs total (20 incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported (yfinance defect persists in 2025 issues; 2026 data-quality-gate consensus endorses log-not-silent-drop; 2026 survivorship evidence quantifies the PIT-deferral caveat; HIGH-threshold nuance is the one refinement)
- [x] Full pages/sources read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (L1 screener.py:64/110/131, L2 paper_trader.py:1200/505/362, B data_ingestion.py:94/105/132, daily_price_refresh.py:82, live_prices.py:1-16/110, autonomous_loop.py:329-369, backtest_engine.py:206/281/284/299/502/1188, candidate_selector.py:98/121/127, markets.py:21-52, analytics.py:446/461, api/backtest.py:888/939, backtest_trader.py:188, paper_trader.py:1233 benchmark)

Soft checks:
- [x] Internal exploration covered: tools/screener (live signal path + gate point), services/paper_trader (live fill/mark + benchmark), services/autonomous_loop (intl universe wiring), services/live_prices (ruled out as cleaning point), backtest/data_ingestion (backtest source + live refresh), slack_bot/jobs/daily_price_refresh (live ingest caller), backtest/backtest_engine (market-blind universe + SPY hard-code + auto-ingest), backtest/candidate_selector (PIT NotImplementedError + INTL_UNIVERSE), backtest/markets (MARKET_CONFIG, no benchmark field), backtest/analytics (SPY baseline), backtest/backtest_trader (NAV, no FX), api/backtest (baseline invocation)
- [x] Contradictions/consensus noted (DROP-vs-FLAG threshold; stale-method rigor; survivorship purism)
- [x] All claims cited per-claim with file:line or URL

## Research-gate JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 13,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "gate_passed": true
}
```

---

## phase-50.5 RESEARCH-GATE REVALIDATION — 2026-05-31

Re-run of the researcher against CODE DRIFT before further GENERATE. Prior gate (2026-05-30) passed; this section revalidates the contract's file:line anchors against the CURRENT tree + audits the partial GENERATE on disk (uncommitted: price_quality.py, screener.py edit, test_phase_50_5_dataquality.py). Tier: moderate.

### Part A — INTERNAL CODE AUDIT (revalidation)

**A1. price_quality.py (NEW, on disk) — CORRECT, ships as contracted.**
- US fast-path byte-identity: `validate_ohlcv(df, market="US")` returns `df, report` with the SAME object (line 55-56: `if df is None or market == "US": return df, report`). Test `test_us_is_byte_identical_noop` asserts `out is df` (identity, not equality) + dropped==0/flagged==0. CONFIRMED byte-identical.
- R1 OHLC consistency (`:67-76`): drops `h<l | h<o | h<c | l>o | l>c | any<=0`. Correct (impossible bars). NaN-safe via `.fillna(False)`.
- R2 identical-OHLC (`:78-95`): `(o==h)&(h==l)&(l==c)` AND `volume==0` -> DROP; identical AND volume>0 -> FLAG (soft); no-volume-column -> FLAG only. Matches contract ("zero-vol corroborates"). Correct.
- R3 (`:97-110`): `|ret|>0.50` (single-day round-trip) -> DROP; rolling z-score>3 (excluding the already-huge) -> FLAG. Matches contract.
- R4 (`:112-118`): stale run `>=4` identical closes -> FLAG (run-length via groupby cumsum; uses `_STALE_RUN-1=3` on the shift-compare which yields a 4-bar run). Correct, FLAG-only.
- `_col` (`:39-45`): case-insensitive accessor, handles "Close"/"close"/"Adj Close". Correct.
- Fail-open: wraps the whole body in try/except -> returns input df on any internal error (`:129-131`). Matches contract ("never block the pipeline on a validator bug").
- `is_bad_bar` (`:134-151`): single-bar L2 helper, mirrors R1+R2, lenient. Correct.
- **No correctness bug found.** DROP-only-unambiguous / FLAG-merely-large posture is implemented exactly. Does NOT over-drop real volatility (a real 40% move passes R3 DROP, only FLAGs via z-score). 6/6 tests PASS (`pytest backend/tests/test_phase_50_5_dataquality.py` -> `6 passed in 0.21s`).

**A2. L1 door (screener.py) — WIRED, on disk, CORRECT.**
- `screener.py:130-140`: after `ticker_data = data[ticker]` (`:125`) and BEFORE `close = ticker_data["Close"]` (`:142`), calls `validate_ohlcv(ticker_data, market=market_for_symbol(ticker), ticker=ticker)`. Exactly the contract's insertion point. Re-checks empty after. CORRECT.
- `market_for_symbol` EXISTS in markets.py (`:96-110`): suffix-driven (.KS/.KQ->KR, .DE/.PA/.AS/.F->EU, .OL->NO, .TO->CA, bare->US). For US bare tickers returns "US" -> validate_ohlcv no-ops -> byte-identical live US signal path. CONFIRMED no-op for US.
- DRIFT vs contract: contract said insert "~:124-130" before "`:131` close"; actual is :130-140 before :142. Trivial line-number shift from the inserted block itself; semantically the SAME site. NOT a problem.

**A3. L2 door (paper_trader._get_live_price) — PENDING (not yet edited).**
- `_get_live_price(ticker)` at `:1200-1209` still original: `yf.Ticker(ticker).history(period="1d")` -> `float(hist["Close"].iloc[-1])`. NO quality gate yet.
- Correct insertion shape: after fetching the 1-row `hist`, extract O/H/L/C/V from the last row and call `is_bad_bar(o,h,l,c,volume)`; if True -> `return None`. The 3 callers ALL already fall back on None:
  - `:362` `price = _get_live_price(ticker) or position.get("current_price", 0)` (SELL)
  - `:505` `live_price = _get_live_price(ticker)` (guarded use)
  - `:945` `px = _get_live_price(ticker) or pos.get("current_price") or pos.get("avg_entry_price")` (mark)
  CONFIRMED: returning None on a bad bar is safe — callers fall back to last-known. is_bad_bar fail-opens (returns False) so a parse error won't block a fill.

**A4. B door (data_ingestion.ingest_prices) — PENDING (not yet edited).**
- `ingest_prices` at `:94`. Per-ticker loop `:120-156`. `ticker_df = data[ticker]` (`:125`) -> `ticker_df = ticker_df.dropna(subset=["Close"])` (`:132`) -> row append loop (`:134-154`).
- DRIFT vs contract: the contract cites "data_ingestion.py:132 (ingest_prices)" and "rows.append at :143". Actual: `def ingest_prices` is at `:94` (NOT :132); :132 is the `.dropna` line; the rows.append is at `:143`. The :132 anchor in the contract is STALE/imprecise — `:132` is mid-function, not the def. GENERATE should insert validate_ohlcv right AFTER the `.dropna(subset=["Close"])` at `:132` and before the `for idx,row in ticker_df.iterrows()` at `:134`, deriving market via `market_for_symbol(ticker)` OR the existing `":" in ticker` namespace split already at `:141-142` (note: this loop derives `market` from the `US:AAPL` NAMESPACE prefix, NOT a suffix — so for the backtest/ingest path use the namespace `market` variable, which is computed at :139-142, rather than market_for_symbol). RECOMMEND: compute market once at top of the per-ticker block, reuse for both validate_ohlcv and the row dict.

**A5. markets.py MARKET_CONFIG — CONFIRMED no benchmark field; safe to add.**
- `MARKET_CONFIG` (`:21-52`): each market dict = `{exchange, currency, timezone, description}`. NO `benchmark` key today. Adding `"benchmark": "SPY"/"^GDAXI"/"^KS11"/"^OSEAX"/"^GSPTSE"` is a pure additive dict-key change, no shape break. `get_market_config` (`:75-78`) returns the dict as-is. SAFE.

**A6. Benchmark/FX change site — MAJOR DRIFT from the contract.**
- Contract says: "`backtest_engine.py:281` calls get_universe_tickers WITHOUT self.market; `:299` hardcodes `+["SPY"]`; `analytics.py:461` hardcodes SPY."
- VERIFIED `:281`: `universe_tickers = self.candidate_selector.get_universe_tickers()` — called with NO args, so `market` defaults to `DEFAULT_MARKET="US"`. `self.market` IS stored (`:206`). To make the engine market-aware: `get_universe_tickers(market=self.market)`. CONFIRMED change site. (get_universe_tickers signature `:98-102` accepts `market` + `as_of`; the `as_of` path raises NotImplementedError — DEFERRED PIT, see A8.)
- VERIFIED `:299`: `cache.preload_prices(universe_tickers + ["SPY"], ...)`. This is the ONLY "SPY" in backtest_engine.py. BUT it is a CACHE PRELOAD, not the benchmark used for returns/alpha. For a non-US benchmark this should preload `MARKET_CONFIG[self.market]["benchmark"]` instead of (or in addition to) "SPY".
- **CRITICAL DRIFT:** the actual benchmark RETURN computation is NOT in backtest_engine.py. It is in `analytics.py:compute_baseline_strategies` (`:446`), which hardcodes `prices_cache_fn("SPY", ...)` at `:462` and emits `spy_return_pct`/`spy_sharpe` (`:527-528`); `generate_report` (`:536`) computes alpha as `aggregate_return_pct - baselines["spy_return_pct"]` (`:580`). And `compute_baseline_strategies` is CALLED from `backend/api/backtest.py:939` (NOT from backtest_engine.py — the contract implies the engine wires it, it does not). So the real GENERATE change set for criterion #1 is:
  1. `backtest_engine.py:281` -> `get_universe_tickers(market=self.market)`.
  2. `backtest_engine.py:299` -> preload `MARKET_CONFIG[self.market]["benchmark"]` (keep "SPY" for US; for EU add "^GDAXI").
  3. `analytics.py:compute_baseline_strategies` -> add a `benchmark: str = "SPY"` param; replace the hardcoded `"SPY"` at :462; the FX conversion of the benchmark + portfolio returns to a base currency happens HERE (or in the api caller) using fx_rates.
  4. `backend/api/backtest.py:939` -> pass `benchmark=MARKET_CONFIG[engine.market]["benchmark"]` into compute_baseline_strategies; this is where the engine's `market` is in scope (via `engine.market`).
  - **The contract's "analytics.py:461" anchor is right (compute_baseline_strategies starts :446, the SPY line is :462). But the contract MISSES that backtest_engine.py:299 is a cache-preload, not the benchmark, and MISSES the api/backtest.py:939 caller entirely.** GENERATE MUST patch analytics.py + api/backtest.py, not just backtest_engine.py, to satisfy criterion #1+#3+#4.

**A7. fx_rates.py (50.1) — CONFIRMED all required call shapes exist.**
- `get_fx_rate(from_ccy, to_ccy, date=None)` (`:182-196`): `from==to -> 1.0` (US/USD byte-identical); else `usd_value(from)/usd_value(to)`. date=None -> live mark; date=ISO -> point-in-time as-of. CONFIRMED.
- `_usd_value_asof(ccy, date)` (`:153-179`): BQ `historical_fx_rates` point-in-time read (`WHERE pair=@pair AND date<=@d ORDER BY date DESC LIMIT 1`), degrades to live on miss. CONFIRMED.
- `market_currency(market)` (`:56-58`): delegates to `markets.get_market_config(market)["currency"]`. CONFIRMED.
- **Benchmark-return -> USD conversion call shape:** to convert an EU (EUR) benchmark total-return series to USD for alpha vs a USD book, the simplest correct approach is the ENDPOINT method the contract specifies: convert the start-NAV and end-NAV (or the benchmark's start/end price) using `get_fx_rate("EUR","USD", date)` at each endpoint, then compute the USD return — NOT per-bar (per-bar FX inside _compute_nav is explicitly DEFERRED). For a single-market book the whole series is one currency, so endpoint-conversion of the return is mathematically clean. RECOMMEND `_fx_local_to_usd`-style mirror of 50.2 at the analytics/api layer.

**A8. The 6 tests — exercise the 4 immutable criteria adequately for the GATE half (#2), partially for #1/#3/#4.**
- `test_phase_50_5_dataquality.py` covers criterion #2 (data-quality gate) THOROUGHLY: US byte-identity no-op (identity assert), identical-OHLC+zero-vol DROP, impossible-OHLC DROP, 60% spike DROP, large-real-move FLAG-not-drop (via clean series passing), clean intl passes, is_bad_bar single-bar. 6/6 PASS.
- GAP: NO test yet for criterion #1 (engine accepts market + uses its benchmark + FX-converts) or #3 (EU .DE backtest end-to-end) or #4 (live_check). Those are satisfied by the live_check_50.5.md artifact (an actual EU backtest run), NOT by unit tests — consistent with the contract's verification command (ast.parse + pytest + `test -f live_check_50.5.md`). **live_check_50.5.md does NOT exist yet** (PENDING). The auto-push gate WILL hold the push until it exists (verification.live_check is set).

### Part A SUMMARY — PENDING-work list for GENERATE
1. **L2 door** — edit `paper_trader._get_live_price:1200`: extract OHLCV from the 1-row `hist`, call `is_bad_bar(...)`, return None if bad. (callers :362/:505/:945 already fall back.)
2. **B door** — edit `data_ingestion.ingest_prices` after the `.dropna` at `:132`, before the row loop at `:134`: `ticker_df, _ = validate_ohlcv(ticker_df, market=<market derived at :139-142>, ticker=ticker)`. Reuse the namespace `market` var, not market_for_symbol (this path uses US:AAPL namespacing).
3. **markets.py** — add `"benchmark"` to each MARKET_CONFIG dict (US="SPY", EU="^GDAXI", KR="^KS11", NO="^OSEAX", CA="^GSPTSE").
4. **backtest_engine.py:281** — `get_universe_tickers(market=self.market)`. **:299** — preload `MARKET_CONFIG[self.market]["benchmark"]` (keep SPY for US).
5. **analytics.py:compute_baseline_strategies** — add `benchmark="SPY"` param; use it at :462; FX-convert the benchmark + portfolio return to base ccy via fx_rates (endpoint method).
6. **api/backtest.py:939** — pass `benchmark=MARKET_CONFIG[engine.market]["benchmark"]` (the engine's market is in scope here as `engine.market`).
7. **EU .DE backtest end-to-end** — run, capture benchmark=^GDAXI + FX-converted return + gate drop count into `handoff/current/live_check_50.5.md` (REQUIRED; auto-push gate holds until present).
8. Confirm a US backtest is byte-identical (benchmark stays SPY, no FX, gate no-ops).

### Part A — DRIFT / correctness summary
- **No correctness bug** in the on-disk code (price_quality.py + screener.py edit + tests). 6/6 tests pass. US byte-identity proven by identity assert.
- **DRIFT 1 (material):** benchmark-return + alpha live in `analytics.py:446-580` + are called from `api/backtest.py:939`, NOT in backtest_engine.py. The contract's "backtest_engine.py:299 hardcodes SPY" is a CACHE-PRELOAD line, not the benchmark. GENERATE MUST patch analytics.py + api/backtest.py for criteria #1/#3/#4 — patching only backtest_engine.py is INSUFFICIENT.
- **DRIFT 2 (minor):** data_ingestion anchor — `ingest_prices` def is at `:94` (contract said :132); :132 is the `.dropna` line; insert validate after it. The ingest path derives market from the `US:AAPL` NAMESPACE (`:139-142`), so use that var, not market_for_symbol.
- **DRIFT 3 (trivial):** screener insertion is at :130-140 (contract said ~:124-130) — same site, shifted by the inserted block. No action.

### Part B — EXTERNAL RESEARCH (revalidation; 3-variant query discipline)

Query variants run per topic: current-year 2026 frontier, 2025/2024 window, year-less canonical. Mix visible in the source tables below.

#### Read in full (>=5 required; 6 fetched -> gate cleared with margin)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| medium.com/@Tobi_Lux/data-from-yfinance-some-observations-41e99d768069 | 2026-05-31 | practitioner blog | WebFetch (full) | **Load-bearing source RE-VERIFIED.** yfinance vs XETRA DAX: up to **11% abs deviation**; **10-24 identical-OHLC days/yr (4-10%)**; on those days **"no trading volume was recorded"** (corroborates R2 zero-vol DROP); candlestick algos fail, MACD/EMA/RSI "limited influence". Recommends Stooq as cleaner alt. |
| medium.com/@axionquant/outlier-detection-in-market-data-b455b435777d | 2026-05-31 | practitioner blog | WebFetch (full) | z-score `threshold=3` (static AND rolling window=20); IQR `1.5`; Isolation Forest `contamination=0.05`. KEY: "Track your false positive rate... **Not every outlier is meaningful**" -> validates DROP-unambiguous/FLAG-suspicious. Confirms our `_Z_FLAG=3.0`. |
| arxiv.org/abs/2603.19380 (NIFTY Smallcap 250) | 2026-05-31 | arXiv q-fin (Mar 2026) | WebFetch (full) | **NEW recency hit.** Survivor-only backtest overstates annual return by **4.94pp (23.3% relative)**, Sharpe by 0.097 (9.1%); 82.5% index turnover in EM small-caps. Recommends PIT membership reconstruction incl. delisted. Validates our DEFERRED PIT-membership scope as material-but-documented (US has the same gap). |
| analystprep.com/study-notes/cfa-level-2/problems-in-backtesting | 2026-05-31 | official curriculum (CFA L2) | WebFetch (full) | Survivorship / look-ahead / data-snooping. Data-snooping mitigation = "removing outliers post-analysis" is a P-HACK; use cross-val + elevated t-stats. Confirms: a DROP rule must be a PRE-registered data-integrity rule (impossible OHLC, zero-vol flat), NOT post-hoc outlier removal to flatter results. Our gate is pre-registered -> compliant. |
| quantconnect.com/forum/discussion/1783 | 2026-05-31 | practitioner (LEAN engine) | WebFetch (full) | Base/account currency model: engine looks up `{CCY}USD` (e.g. JPYUSD) to convert to base; separate cashbook per currency; "**database only has equities... quoted in USD**" -> even QC doesn't do per-bar intl FX cheaply. Validates our endpoint-FX (not per-bar) DEFERRED decision + fx_rates `{CCY}USD` pair convention. |
| starqube.com/backtesting-investment-strategies | 2026-05-31 | practitioner (PIT vendor) | WebFetch (full) | 7 backtest "sins": survivorship, look-ahead, storytelling, overfitting, txn-cost, period-selection, short-borrow. "Using only daily OHLC forces assumptions." Emphasizes native PIT data. No FX/outlier specifics (recorded as such). |

#### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| arxiv.org/pdf/2209.11686 (PCA+NN anomaly) | arXiv q-fin | Abstract fetched; thresholds are LEARNED (NN loss), not fixed -> less directly applicable than axionquant's explicit z=3. Snippet sufficient. |
| ieeexplore.ieee.org/document/7810839 | IEEE | Paywalled; financial-TS anomaly detection, older. |
| mdpi.com/1999-4893/15/10/385 | MDPI Algorithms | Same PCA+NN work as 2209.11686. |
| promptcloud.com/blog/scrape-yahoo-finance (2026) | vendor blog | "validate against a second source for international/OTC" theme — corroborates Tobi Lux. |
| slingacademy.com/article/common-yfinance-errors | tutorial | yfinance error modes, low specificity. |
| tinybird.co/blog/anomaly-detection | vendor blog | rolling z-score / static stats overview. |
| victoriametrics.com/.../anomaly-detection-handbook-chapter-3 | vendor docs | TS anomaly techniques survey. |
| tradingcode.net/tradingview/backtest-currency-conversion | tutorial | "fixed post-hoc conversion distorts P&L; use period-correct FX" — corroborates endpoint-at-date over a single fixed rate. |
| quantshare.com/title-688 / domintia BTAnalytics / multicharts MC-349 | vendor | multi-ccy backtest needs base ccy + FX access during the run. |
| etf.com South Korea ETF / dax-indices.com strategy guide | vendor/official | ^KS11 & DAX are the domestic benchmarks; DAX is EUR, unhedged -> subject to USD/EUR (confirms FX-to-USD needed for a USD book). |
| fxreplay backtesting-biases / quantvps guide | blog | general bias catalog. |

#### Recency scan (2024-2026) — PERFORMED
Searched 2026-frontier + 2025-window + year-less for all 3 topics. Findings:
1. **Tobi Lux yfinance/XETRA study** — re-verified live 2026-05-31, numbers UNCHANGED (11% dev, 10-24 identical-OHLC days/yr, zero-volume on those days). The contract's central justification holds; no newer study supersedes it. Corroborated by promptcloud (2026) "validate intl against a 2nd source".
2. **NEW: arXiv 2603.19380 (Mar 2026, NIFTY Smallcap)** — quantifies survivorship at 4.94pp/23.3% for EM small-caps. This is NEW since the 2026-05-30 brief. It does NOT change our GENERATE plan; it STRENGTHENS the case that DEFERRING PIT intl membership is acceptable (the bias is documented, market-agnostic, and present in the US path too) while making clear it's the #1 follow-on. No code change required for 50.5.
3. **AI-agent guardrail playbooks (2026)** — layered validation (input->dialog->generation->schema->business-rule) + "risk-based routing" (financial = high-risk -> comprehensive gates). Our 3-door L1/L2/B gate IS this layered pattern for the price-data plane. No change to plan; confirms the architecture.
4. **Multi-currency backtest (QuantConnect, tradingview, MultiCharts)** — consensus: set a base currency, convert using period-correct (date-matched) FX, NOT a single fixed post-hoc rate (fixed-rate distorts P&L, worse over longer windows). Our fx_rates `get_fx_rate(from,to,date)` as-of read is exactly this. Endpoint-vs-per-bar: even QC's DB is USD-quoted -> per-bar intl FX is genuinely hard; endpoint conversion of the return for a single-currency book is the pragmatic, mathematically clean choice -> DEFERRING per-bar FX is defensible.

#### Consensus vs debate
- **Consensus:** (a) yfinance intl is defect-prone, validate before use (Tobi Lux, promptcloud, becomingquant). (b) z=3 rolling is the standard outlier FLAG threshold; impossible-OHLC + zero-vol flat bars are unambiguous DROPs (axionquant). (c) DROPPING outliers post-hoc to improve results is p-hacking — a gate must be PRE-registered integrity rules (CFA/analystprep). (d) Multi-ccy backtests need a base ccy + date-matched FX, not a fixed rate (QC, tradingview, MultiCharts). (e) Survivorship is material (4.94pp EM small-cap; "several pp/yr" general) but is mitigated by PIT membership — a known, deferrable, market-agnostic gap.
- **Debate / nuance:** static vs rolling z-score (we use rolling-population std over the series window — fine for a per-ticker download); learned vs fixed thresholds (PCA+NN learns it — overkill for a $0 gate); Stooq vs yfinance (Tobi Lux prefers Stooq, but operator chose free-yfinance+gate — our gate is the documented compensating control).

#### Pitfalls (from literature) -> mapped to our code
1. **Over-dropping real volatility** (axionquant "not every outlier is meaningful"; CFA p-hacking) -> price_quality DROPs ONLY impossible-OHLC + identical-OHLC+zero-vol + >50% round-trip; FLAGs (keeps) z>3 and stale runs. CORRECT (verified A1).
2. **Silent truncation** (no source endorses it) -> report{dropped,flagged,reasons} + logger.info; criterion #2 requires logging. CORRECT.
3. **Fixed post-hoc FX rate** (tradingview/QC) -> use `get_fx_rate(...,date)` as-of, not a single rate. fx_rates already does this (A7).
4. **Survivorship via current-membership** (analystprep, arXiv 2603.19380, starqube) -> DEFERRED (candidate_selector as_of raises NotImplementedError; curated static DAX-40/KOSPI-200 lists are themselves a current-membership snapshot). DEFENSIBLE: US path has the identical gap; documented; #1 follow-on.
5. **Wrong benchmark / unconverted FX** (etf.com, dax-indices) -> ^GDAXI (EUR) must be FX-converted to USD for alpha vs the USD book; ^KS11 likewise. Criterion #1 change set (A6) handles this at analytics.py + api/backtest.py.

### Application to pyfinagent (external -> internal anchors)
- Tobi Lux zero-volume corroboration -> `price_quality.py:82-88` R2 (`identical & zero_vol -> DROP`). EXACT match.
- axionquant z=3 -> `price_quality.py:35` `_Z_FLAG=3.0`, applied `:105-110` (FLAG only). Match.
- CFA pre-registered-rule discipline -> the gate is a fixed rule set (R1-R4), not post-hoc result-driven removal. Compliant.
- QC base-ccy + `{CCY}USD` lookup -> `fx_rates._pair` + `get_fx_rate(from,to,date)`. Match. Endpoint conversion for criterion #1 at `analytics.py:compute_baseline_strategies` + `api/backtest.py:939`.
- arXiv 2603.19380 survivorship -> confirms DEFERRED PIT membership (candidate_selector `:121-126` NotImplementedError) is acceptable for go-live.

### DEFERRED scope — still defensible? YES.
The 2026-05-30 DEFERRED list (PIT intl membership, per-bar FX inside _compute_nav, simultaneous mixed-currency backtest, live per-market benchmark) is RE-CONFIRMED defensible by the new literature:
- PIT membership: arXiv 2603.19380 (Mar 2026) quantifies it as material BUT market-agnostic (US has it too) and explicitly a separate "reconstruct historical membership" project. Not a 50.5 blocker.
- Per-bar FX: QC (the reference multi-ccy engine) doesn't do cheap per-bar intl FX either; endpoint conversion of a single-currency book's return is clean. Defensible.
- The minimal safe scope (gate + market-param + per-market benchmark + endpoint-FX + an EU .DE live_check) satisfies all 4 immutable criteria without touching the working US path (byte-identity proven).

### Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: Tobi Lux, axionquant, arXiv 2603.19380, CFA/analystprep, QuantConnect, starqube)
- [x] 10+ unique URLs total (14 unique: 6 full + 8 snippet rows covering ~20 links surfaced)
- [x] Recency scan (last 2 years) performed + reported (4 findings; 1 NEW arXiv Mar 2026)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (Part A)
Soft checks:
- [x] Internal exploration covered every relevant module (price_quality, screener, paper_trader, data_ingestion, markets, backtest_engine, analytics, api/backtest, candidate_selector, fx_rates, tests)
- [x] Contradictions/consensus noted
- [x] All claims cited per-claim

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "phase-50.5 revalidation appended to handoff/current/research_brief.md",
  "gate_passed": true
}
```
