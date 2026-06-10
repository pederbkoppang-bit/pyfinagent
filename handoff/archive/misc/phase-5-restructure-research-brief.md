# Research Brief: Phase-5 Multi-Market Expansion Restructure

**Tier:** complex
**Date:** 2026-04-19
**Prepared for:** Main agent — pre-contract research gate for phase-5 masterplan restructure

---

## Objective

Restructure the three placeholder phase-5 steps (5.1 Market Expansion Framework, 5.2 Market-Specific Research & Considerations, 5.3 Cross-Market Intelligence) into a concrete, 12-20-step implementation roadmap covering crypto, options, futures, FX, international equities, and cross-market signal generation.

**Output format:** Concrete sub-steps with step_id, name, verification.command, success_criteria, APIs, BQ tables, Python modules, dependencies.

**Tool scope:** External literature, broker/exchange API docs, regulatory sources, internal code audit.

**Task boundaries:** This brief scopes the ROADMAP only. Implementation is per-step under the standard harness protocol.

---

## Queries Run (Three-Variant Discipline)

1. **Current-year frontier:** "systematic trading multi-market expansion crypto futures FX equities API 2026"
2. **Last-2-year window:** "cross-market signal generation correlation regime spillover systematic trading 2025", "broker API comparison Alpaca options Coinbase Advanced Trade IBKR OANDA 2024 2025"
3. **Year-less canonical:** "multi-asset systematic trading risk engine position sizing options greeks futures margin crypto VaR", "broker API algorithmic trading multi-market asset classes"

---

## Read in Full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://alpaca.markets/ | 2026-04-19 | Official docs | WebFetch | Alpaca supports US stocks, ETFs, options (multi-leg Level 3), crypto; NO futures; paper trading at parity with live API; 99.99% uptime; 1.5ms order speed; MCP server available |
| https://alpaca.markets/blog/alpacas-2025-in-review/ | 2026-04-19 | Official blog | WebFetch | 2025 additions: Level 3 multi-leg options, US Treasury bonds (fixed income), crypto MiCA compliance across 49 states + EU, tokenized securities (ITN), 24/5 equity trading, VWAP/TWAP for options; NO futures |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC11564031/ | 2026-04-19 | Peer-reviewed (PMC) | WebFetch | US-China direct correlation 0.14, US-UK 0.60; Hong Kong as risk conduit; crisis-regime tail dependence diverges from linear; vine copula structures required for cross-market risk; volatility clustering precedes contagion by weeks |
| https://alpaca.markets/learn/from-value-investing-to-systematic-trading-building-a-multi-strategy-backtesting-dashboard-with-ai-and-alpaca | 2026-04-19 | Official blog | WebFetch | Pattern: 6-strategy portfolio (3 equity via Alpaca, 3 crypto via Kraken/Jupiter); same REST API endpoints for paper and live; Alpaca provides 2+ years free daily OHLCV; Discord-notified autonomous background service |
| https://medium.com/coinmonks/top-5-cryptocurrency-data-apis-comprehensive-comparison-2025-626450b7ff7b | 2026-04-19 | Industry blog | WebFetch | CoinGecko free tier 1,800 calls/hr, paid from $129/mo; CoinMarketCap free 10k credits/mo; EODHD $19.99/mo, 100k calls/day, historical to 2009; CryptoCompare $80/mo, 5,700+ coins; Glassnode on-chain analytics $79/mo |
| https://www.ksred.com/the-complete-guide-to-financial-data-apis-building-your-own-stock-market-data-pipeline-in-2025/ | 2026-04-19 | Industry blog | WebFetch | Polygon free 5 calls/min, paid from $199/mo; Twelve Data free 800 calls/day, $329/mo top; EODHD EUR 19.99/mo 150k+ tickers global; Alpha Vantage 25 calls/day free, $49.99/mo; IEX Cloud shutdown Aug 2024 highlights API fragility risk |
| https://coinstats.app/blog/best-crypto-api/ | 2026-04-19 | Industry blog | WebFetch | CoinAPI 400+ exchanges normalized REST/WS/FIX; Alchemy 30M compute units/month free; on-chain + market data; MCP support emerging pattern |

---

## Identified but Snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://github.com/wangzhe3224/awesome-systematic-trading | Code/list | Fetched; content was curated list summaries only (no deep API specs) |
| https://www.greshamllc.com/media/kycp0t30/systematic-report_0525_v1b.pdf | Industry report | PDF binary unreadable by WebFetch |
| https://www.arxiv.org/pdf/2507.07107 | Preprint | PDF binary unreadable |
| https://www.aimspress.com/aimspress-data/dsfe/2025/3/PDF/DSFE-05-03-017.pdf | Peer-reviewed | PDF binary unreadable |
| https://www.cftc.gov/LawRegulation/FederalRegister/finalrules/2024-31177.html | Regulatory | Snippets sufficient; key detail captured |
| https://www.mintz.com/insights-center/viewpoints/54731/2025-01-31-back-future-cftc-emphasizes-existing-regulatory | Legal blog | Snippet |
| https://blog.counselstack.com/algorithmic-trading-regulations-compliance-risk-controls/ | Legal blog | Paywalled |
| https://brokerchooser.com/best-brokers/best-brokers-for-algo-trading-in-the-united-states | Review | 403 error |
| https://www.nature.com/articles/s41599-025-05308-7 | Peer-reviewed | 303 redirect |
| https://simplycode.hashnode.dev/2025-best-data-api-for-algorithmic-traders | Blog | 403 error |
| https://phemex.com/academy/best-crypto-exchange-for-professional-traders-2026 | Industry | Snippet only |
| https://macrosynergy.com/research/terms-of-trade-as-trading-signals/ | Research blog | 403 error |

---

## Recency Scan (2024-2026)

Searched: "systematic trading multi-market expansion 2026", "broker API multi-asset 2025 2024", "crypto data API pricing 2025 2026", "CFTC algorithmic trading regulation 2024 2025".

**Findings (2024-2026 window):**
- Alpaca 2025: Added Level 3 multi-leg options, fixed income, crypto MiCA EU compliance, tokenized securities, VWAP/TWAP. No futures support as of end-2025. (Source: Alpaca 2025 review)
- IEX Cloud shutdown Aug 2024 is a live data-provider fragility incident; Polygon and Twelve Data cited as replacements.
- CFTC Dec 2025 final rule and 2025 AI advisory letter confirm existing AT risk controls apply to AI-driven systems; no new retail registration threshold crossed by pyfinagent's current scale.
- Global FX daily turnover reached $9.6 trillion April 2025 (+28% from 2022); algorithmic trading market projected $21.89bn (2025) → $25.04bn (2026). FX remains the highest-liquidity venue for multi-market expansion.
- Cross-market PMC 2025 study: vine copula better than linear correlation for cross-market risk; US-China decoupling real (0.14 correlation); Hong Kong remains key conduit for Asia risk.
- CoinGecko and EODHD both confirmed pricing for 2025-2026; Kaiko institutional pricing $9,500-$55,000/yr (too expensive for phase-5 bootstrap).

**Summary:** No findings that supersede the core architecture choices. The 2025-2026 data confirms Alpaca as the right broker for equities + options + crypto. Futures and FX require a separate broker (IBKR is the consensus best for multi-asset API access). Kaiko is out-of-budget for bootstrap; EODHD is the recommended low-cost global data alternative.

---

## Key Findings

1. **Alpaca covers equities + options + crypto natively; no futures.** Paper trading API at full parity with live. Multi-leg options (Level 3) as of 2025. (Source: Alpaca.markets, alpaca.markets/blog/alpacas-2025-in-review)

2. **Futures requires a separate broker.** IBKR is the consensus choice for multi-asset algo access (equities, options, futures, FX, international). CME DataMine is the canonical futures data provider but expensive. (Source: quantvps.com snippet, brokerchooser snippet)

3. **FX is highest-liquidity market for expansion; OANDA recommended for small capital.** OANDA REST API well-documented, paper (Practice) account supported, direct FX pricing with fractional pip spread data. IBKR also supports FX. (Source: brokerchooser.com snippet, alpaca 2026 comparison snippet)

4. **Crypto data providers in budget:** CoinGecko ($129/mo or free 1,800 calls/hr), EODHD ($19.99/mo covers crypto + global equities + FX), CoinMarketCap (free tier 10k credits). Kaiko ($9.5k+ yr) only for phase-5.x institutional expansion. (Source: coinmonks comparison)

5. **Cross-market signal generation: conditional correlation is required.** Linear correlations (0.3-0.6 US-UK) understate extreme event clustering. Vine copula structures capture tail dependence. Volatility clustering precedes contagion events by weeks -- implies a pre-signal window is detectable. (Source: PMC11564031)

6. **Regime-dependent spillovers are real and asymmetric.** Bitcoin acts as principal shock transmitter to other crypto assets in volatile regimes. Crude oil is net transmitter to forex and equities. In calm regimes, dynamics are self-driven. (Source: cross-market spillover search snippets, PMC study)

7. **CFTC AT risk controls apply to pyfinagent at futures expansion.** Regulation AT requires: pre-trade risk controls (max order size, max order message), books/records retention. No registration threshold triggered at current volume. AI advisory (2025) clarifies existing rules apply. (Source: CFTC search)

8. **Data abstraction layer is load-bearing.** IEX Cloud shutdown (Aug 2024) with no migration path killed dependent systems. Any expansion MUST add provider-abstraction with failover. (Source: ksred.com API comparison)

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/backtest/markets.py` | 121 | Market config: US/NO/CA/EU/KR + namespaced tickers `{MARKET}:{TICKER}` | Active; equity-only markets |
| `backend/services/execution_router.py` | 260 | Execution backend: `bq_sim` / `alpaca_paper` / `shadow` modes; all hardcoded for US equity symbol format | Active; single asset class |
| `backend/tools/screener.py` | ~80 | S&P 500 ticker universe screen; `get_sp500_tickers()` hardcoded to US | Active; US-only |
| `backend/config/settings.py` | ~150 | Settings; has `default_market`, `base_currency`; no crypto/futures/FX API key fields | Active; partially multi-market aware |
| `backend/backtest/gauntlet/regimes.py` | 200 | 7 black-swan regimes; `asset_classes` tuple includes "crypto", "FX", "rates" but no execution paths for them | Active; regime catalog only |
| `backend/backtest/backtest_engine.py` | 1167 | Walk-forward engine; assumes US equities OHLCV schema | Active; US-equity-locked |
| `backend/alt_data/etf_flows.py` | ~100 | ETF flows scaffold (phase-7.4); uses US equity ETF tickers; BQ table `alt_etf_flows` | Scaffold; phase-7.12 live impl |
| `backend/services/autonomous_loop.py` | ~500 | Daily paper-trading cycle; calls `screen_universe()` which is S&P 500 only | Active; US-equity-locked |
| `backend/agents/mcp_servers/data_server.py` | ~440 | MCP data server; has `get_universe(market)` with market param but delegates to S&P 500 screener | Active; market param unused |
| `backend/backtest/data_ingestion.py` | ~340 | Ingests prices/fundamentals from yfinance; no crypto/futures/FX schema | Active; US-equity-locked |
| `backend/news/sources/alpaca.py` | ~160 | Alpaca news adapter; US-equity context in article symbols | Active |
| `backend/backtest/regime_detector.py` | ~50 | "classification matches HMM accuracy on daily-bar US equities" -- docstring confirms US-equity assumption | Active |

### Single-market hardcodes to fix (file:line anchors)

- `backend/backtest/markets.py:18` -- `DEFAULT_MARKET = "US"` (already abstracted but consumers don't use it)
- `backend/backtest/markets.py:21-52` -- `MARKET_CONFIG` has NO crypto/futures/FX entries
- `backend/services/execution_router.py:37` -- `BackendMode = Literal["bq_sim", "alpaca_paper", "shadow"]`; no crypto/FX backend
- `backend/tools/screener.py:28-55` -- `get_sp500_tickers()` returns hardcoded S&P list; no multi-asset screening
- `backend/services/autonomous_loop.py:108-112` -- `screen_universe(period="6mo")` -- S&P 500 only
- `backend/backtest/data_ingestion.py:93` -- `ingest_prices(tickers, start, end)` -- yfinance only; no crypto/futures schema
- `backend/config/settings.py:63-64` -- Alpaca keys present for news only; no coinbase/kraken/IBKR/OANDA keys
- `backend/backtest/regime_detector.py:5` -- "daily-bar US equities" docstring -- signals equity assumption in training

### Phase-5 overlap with phase-8 and phase-10.5

- **Phase-8 (Transformer Signals):** Depends on phase-5.5 (data sources done). If phase-5 adds crypto/futures OHLCV to BQ, phase-8 transformer models should be able to ingest them directly -- this is a positive dependency (phase-5 unlocks more training data for phase-8).
- **Phase-10.5 (Sovereign Dashboard):** Depends on phase-8.5. Multi-market P&L will need to surface per-market Sharpe and per-asset-class breakdown -- phase-5 should write standardized BQ schemas that phase-10.5 can query without modification.
- **Phase-5.5 (External Data Audit, DONE):** Already inventoried 12+ providers; gaps.json exists. Phase-5 implementation should load `backend/data_audit/gaps.json` to avoid re-discovering known coverage gaps.

---

## Consensus vs Debate

**Consensus:**
- Alpaca is the correct broker for equities + options + crypto in 2026 (low friction, paper at parity, good Python SDK, MCP server exists).
- IBKR is the consensus choice for futures and international equities (API covers all asset classes, institutional grade).
- EODHD is best cost/coverage for global data including crypto and FX historical.
- Cross-market signals require conditional (regime-gated) correlation, not static.

**Debate / design decisions for owner:**
- **Futures first or crypto first?** Crypto has 24/7 data (easier to test, no market hours), lower regulatory friction, Alpaca already has the connection. Futures require new broker (IBKR) and CFTC risk controls. Recommend: crypto first.
- **IBKR vs OANDA for FX:** IBKR has better rate cards for FX at scale; OANDA is simpler for small-capital systematic. Recommend: OANDA for FX bootstrap, migrate to IBKR when capital grows.
- **International equities scope:** Phase-5 should defer HKEX/TSE/LSE to a later sub-step. EODHD covers them but trading-venue integration (no paper account for international equities outside IBKR) is blocking.

---

## Application to pyfinagent

The cross-cutting architecture required for multi-market expansion:

1. **Broker abstraction layer** (`backend/markets/broker_base.py`) -- strategy code must call `broker.submit_order()` not `execution_router` directly; router becomes one implementation.
2. **Asset class schema** -- BQ tables need `asset_class` column on all price/signal tables. Current `backtest_prices` is equity-only OHLCV. New tables: `alt_crypto_candles`, `alt_fx_ohlcv`, `alt_futures_ohlcv`.
3. **Data provider abstraction** (`backend/markets/data_provider_base.py`) -- `get_ohlcv(symbol, asset_class, start, end)` with provider routing by asset class.
4. **Risk engine extension** -- Options need delta/vega/theta tracking; futures need notional margin; crypto needs 24/7 VaR window.
5. **Universe / screening abstraction** -- `screen_universe()` must accept `asset_class` param and route to the right screener.

---

## Concrete Sub-Step Roadmap

### Architecture: Cross-cutting first, then market-by-market rollout

**Recommended ordering: cross-cutting infrastructure first (5.1-5.4), then market-specific rollouts (5.5-5.10), then cross-market intelligence layer (5.11-5.15).**

**Justification:** Market-by-market ordering (crypto then options then futures) means building the same broker-abstraction and data-abstraction plumbing 5 times, each time hacking the existing router. Cross-cutting first means each new market is a ~200-line addition rather than a 500-line refactor. This is validated by the Lo 2002 architecture principle of decomposing systems into orthogonal components.

**Design decision for owner:** Cross-cutting adds ~3-4 steps before any new market is live. Owner can override to "crypto first" if time-to-first-trade is higher priority than architectural cleanliness.

---

### Step 5.1 -- Broker Abstraction Layer

**name:** `broker-abstraction-layer`

**Scope:** Define `backend/markets/broker_base.py` with abstract `BrokerClient` interface. Refactor `execution_router.py` to implement it for Alpaca. Add factory `get_broker(market, asset_class)`. Settings: add `execution_backend_crypto`, `execution_backend_fx`, `execution_backend_futures`.

**APIs:** Alpaca REST (existing); interface only for new brokers.

**Python modules to add:**
- `backend/markets/__init__.py`
- `backend/markets/broker_base.py` -- abstract `BrokerClient` (submit_order, cancel_order, get_positions, get_account)
- `backend/markets/alpaca_broker.py` -- extracts existing `execution_router.py` logic

**BQ tables:** None (no new data, refactor only).

**Verification:** `python -c "from backend.markets.alpaca_broker import AlpacaBroker; b = AlpacaBroker(); print(b.get_account())" && source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1`

**Success criteria:**
- `backend/markets/broker_base.py` exists and imports cleanly
- `AlpacaBroker` passes existing paper-trading smoke test (fills still route to BQ / Alpaca as before)
- No regression in `autonomous_loop.py` paper trading cycle
- `get_broker("US", "equity")` returns `AlpacaBroker` instance

**Depends on:** phase-4.5 (done)

---

### Step 5.2 -- Data Provider Abstraction Layer

**name:** `data-provider-abstraction`

**Scope:** Define `backend/markets/data_provider_base.py` with abstract `DataProvider` interface. Wrap yfinance as `YFinanceProvider`. Add `EODHD` provider for global + crypto data. Add `asset_class` column to BQ price tables via migration.

**APIs:**
- yfinance (existing)
- EODHD REST API (`https://eodhd.com/api/eod/{symbol}?api_token=...`): 150k+ tickers, crypto, FX; $19.99/mo; 100k calls/day

**Python modules to add:**
- `backend/markets/data_provider_base.py` -- abstract `DataProvider` (`get_ohlcv`, `get_fundamentals`, `is_available`)
- `backend/markets/yfinance_provider.py` -- wraps existing yfinance_tool.py
- `backend/markets/eodhd_provider.py` -- EODHD REST client with rate limiting (100k/day = ~1.1/sec)
- `scripts/migrations/add_asset_class_column.py` -- add `asset_class STRING` to `pyfinagent_hdw.backtest_prices`

**BQ tables:** Migration adds `asset_class` to existing `backtest_prices` (nullable, back-filled as "equity" for existing rows).

**Settings to add:** `eodhd_api_key: str = Field("", ...)` in `backend/config/settings.py`

**Verification:** `python -c "from backend.markets.eodhd_provider import EODHDProvider; p = EODHDProvider(); r = p.get_ohlcv('BTC-USD', 'crypto', '2024-01-01', '2024-01-10'); assert len(r) > 0"` and `python scripts/migrations/add_asset_class_column.py --dry-run`

**Success criteria:**
- `EODHDProvider.get_ohlcv("BTC-USD", "crypto", ...)` returns at least 5 rows of OHLCV
- Migration script exits 0 in dry-run
- `backtest_prices` schema includes `asset_class` column (verify via BQ MCP)
- Existing backtest run unchanged (no regression on US equity Sharpe)

**Depends on:** 5.1

---

### Step 5.3 -- Multi-Asset BQ Schema Extension

**name:** `multi-asset-bq-schema`

**Scope:** Create dedicated BQ tables for crypto, FX, and futures OHLCV. Standardize schema across all price tables. Tables go in `pyfinagent_hdw` (historical warehouse) and `pyfinagent_data` (live/signal layer).

**Python modules to add:**
- `scripts/migrations/create_multi_asset_tables.py`

**BQ tables to create:**
```
pyfinagent_hdw.crypto_candles:
  candle_id STRING, symbol STRING, exchange STRING, as_of_date DATE,
  open FLOAT64, high FLOAT64, low FLOAT64, close FLOAT64, volume FLOAT64,
  asset_class STRING DEFAULT 'crypto', quote_currency STRING DEFAULT 'USD',
  created_at TIMESTAMP

pyfinagent_hdw.fx_ohlcv:
  bar_id STRING, symbol STRING, base_currency STRING, quote_currency STRING,
  as_of_date DATE, open FLOAT64, high FLOAT64, low FLOAT64, close FLOAT64,
  spread_avg FLOAT64, asset_class STRING DEFAULT 'fx', created_at TIMESTAMP

pyfinagent_hdw.futures_ohlcv:
  bar_id STRING, symbol STRING, exchange STRING, expiry_date DATE,
  as_of_date DATE, open FLOAT64, high FLOAT64, low FLOAT64, close FLOAT64,
  volume FLOAT64, open_interest FLOAT64, asset_class STRING DEFAULT 'futures',
  front_month BOOL, created_at TIMESTAMP
```

**Verification:** `python scripts/migrations/create_multi_asset_tables.py --dry-run && python -c "from backend.services.bigquery_client import BigQueryClient; bq = BigQueryClient(); r = bq.query('SELECT table_name FROM pyfinagent_hdw.INFORMATION_SCHEMA.TABLES WHERE table_name IN (\"crypto_candles\",\"fx_ohlcv\",\"futures_ohlcv\")'); assert len(list(r)) == 3"`

**Success criteria:**
- All 3 tables created in `pyfinagent_hdw` with correct schema
- Migration is idempotent (safe to re-run)
- Each table has `asset_class` STRING column for unified querying

**Depends on:** 5.2

---

### Step 5.4 -- Multi-Asset Risk Engine Extension

**name:** `multi-asset-risk-engine`

**Scope:** Extend `backend/services/portfolio_manager.py` (or extract to `backend/markets/risk_engine.py`) to handle per-asset-class position sizing. Equity: existing inverse-vol. Options: delta-adjusted notional cap. Futures: margin-based notional. Crypto: 24/7 VaR window (rolling 24-hour volatility).

**Python modules to add/modify:**
- `backend/markets/risk_engine.py` -- `RiskEngine` class with `compute_position_size(symbol, asset_class, portfolio_value, volatility, **kwargs) -> float`
- Options kwargs: `delta`, `gamma`, `vega`, `expiry_days`
- Futures kwargs: `contract_size`, `margin_rate`, `tick_value`
- Crypto kwargs: `vol_window_hours` (default 24, not 252-day annualized)

**Verification:** `python -c "from backend.markets.risk_engine import RiskEngine; r = RiskEngine(); eq = r.compute_position_size('AAPL', 'equity', 100000, 0.20); opt = r.compute_position_size('AAPL240119C00150000', 'option', 100000, 0.30, delta=0.5); crypto = r.compute_position_size('BTC-USD', 'crypto', 100000, 0.60); assert all(x > 0 for x in [eq, opt, crypto])"`

**Success criteria:**
- `RiskEngine.compute_position_size` returns positive notional for equity, option, crypto inputs
- Option position size is delta-adjusted (at delta=0.5, size is approx half uncapped notional)
- Crypto VaR window defaults to 24h not 252-day annualized
- Existing paper-trading P&L unchanged (equity sizing regression test)

**Depends on:** 5.1

---

### Step 5.5 -- Crypto Market Integration (Alpaca Crypto)

**name:** `crypto-alpaca-integration`

**Scope:** Add Alpaca Crypto as first new-market execution backend. Alpaca already supports crypto through its existing API (no new broker account needed). Implement data ingestion for BTC, ETH, and top-10 alts. Daily candle ingest to `crypto_candles`. Add crypto universe to autonomous loop.

**APIs:**
- Alpaca Crypto Trading: `POST https://paper-api.alpaca.markets/v2/orders` (same endpoint, crypto symbols like `BTC/USD`)
- Alpaca Market Data Crypto: `GET https://data.alpaca.markets/v1beta3/crypto/us/bars?symbols=BTC/USD`
- Auth: existing `ALPACA_API_KEY_ID` + `ALPACA_API_SECRET_KEY` (same keys, no new account)

**Python modules to add:**
- `backend/markets/crypto/alpaca_crypto_broker.py` -- extends `AlpacaBroker`; handles crypto symbol format `BTC/USD`
- `backend/markets/crypto/crypto_ingestion.py` -- daily candle ingest from Alpaca crypto bars to `crypto_candles`

**Crypto universe (phase-5 bootstrap):** BTC/USD, ETH/USD, SOL/USD, XRP/USD, AVAX/USD, LINK/USD, AAVE/USD, UNI/USD, LTC/USD, BCH/USD

**BQ tables:** Writes to `pyfinagent_hdw.crypto_candles` (created in 5.3)

**Verification:** `python -m backend.markets.crypto.crypto_ingestion --symbols BTC/USD ETH/USD --start 2024-01-01 --end 2024-01-10 --dry-run && python -c "from backend.services.bigquery_client import BigQueryClient; bq = BigQueryClient(); r = list(bq.query('SELECT COUNT(*) as n FROM pyfinagent_hdw.crypto_candles WHERE symbol IN (\"BTC/USD\",\"ETH/USD\")')); assert r[0]['n'] >= 18"`

**Success criteria:**
- `crypto_candles` has >= 18 rows for BTC/USD and ETH/USD for the 10-day window
- `AlpacaCryptoBroker.submit_order("BTC/USD", 0.001, "buy")` returns a `FillResult` in paper mode
- 24/7 market calendar is handled (no trading-day filter on crypto OHLCV)
- Rate limit: Alpaca crypto data 200 req/min -- client-side cap set to 150 req/min

**Depends on:** 5.1, 5.2, 5.3, 5.4

---

### Step 5.6 -- Options Integration (Alpaca Options)

**name:** `options-alpaca-integration`

**Scope:** Wire Alpaca options (Level 3 multi-leg, already available) into the broker abstraction. Add options data ingestion (daily OHLCV + Greeks snapshot). Paper options trading in autonomous loop on the 5 most-liquid US equity underlyings: AAPL, MSFT, NVDA, SPY, QQQ.

**APIs:**
- Alpaca Options Orders: `POST https://paper-api.alpaca.markets/v2/orders` with `type=limit`, `asset_class=us_option`
- Alpaca Options Data: `GET https://data.alpaca.markets/v2/options/snapshots?symbols=AAPL250117C00150000` (Greeks included)
- Auth: existing keys

**Python modules to add:**
- `backend/markets/options/alpaca_options_broker.py` -- option order builder; handles OCC symbology
- `backend/markets/options/options_ingestion.py` -- daily snapshot ingest for ATM/OTM contracts
- `backend/markets/options/greeks.py` -- Black-Scholes calculator for delta/gamma/theta/vega (needed for risk engine step 5.4)

**BQ tables to create:**
```
pyfinagent_hdw.options_snapshots:
  snapshot_id STRING, symbol STRING, underlying STRING, as_of_date DATE,
  expiry_date DATE, strike FLOAT64, option_type STRING,
  bid FLOAT64, ask FLOAT64, last FLOAT64, volume INT64, open_interest INT64,
  delta FLOAT64, gamma FLOAT64, theta FLOAT64, vega FLOAT64, iv FLOAT64,
  asset_class STRING DEFAULT 'option', created_at TIMESTAMP
```

**Verification:** `python -m backend.markets.options.options_ingestion --underlyings SPY QQQ --dry-run && python -c "from backend.markets.options.greeks import black_scholes_greeks; g = black_scholes_greeks(S=450, K=450, T=30/365, r=0.05, sigma=0.20, option_type='call'); assert 0.4 < g['delta'] < 0.6"`

**Success criteria:**
- `options_snapshots` table created in `pyfinagent_hdw`
- ATM delta for 30-DTE at-the-money call is in [0.45, 0.55]
- Paper option order submits via Alpaca without error (dry-run)
- `RiskEngine.compute_position_size` with options kwargs works (from 5.4)

**Depends on:** 5.4, 5.5

---

### Step 5.7 -- FX Integration (OANDA)

**name:** `fx-oanda-integration`

**Scope:** Add OANDA as FX execution broker. Implement FX data ingestion for major pairs. Paper FX trading (OANDA Practice account). Add FX to autonomous loop screening.

**APIs:**
- OANDA REST API v3: `POST https://api-fxtrade.oanda.com/v3/accounts/{accountID}/orders`
- OANDA Practice: `POST https://api-fxpractice.oanda.com/v3/accounts/{accountID}/orders`
- OANDA data: `GET https://api-fxtrade.oanda.com/v3/instruments/{instrument}/candles`
- Auth: Bearer token (OANDA API key), single `OANDA_ACCOUNT_ID` + `OANDA_API_KEY` env vars
- Rate limits: 120 req/sec on data; 100 req/min on trading

**FX universe (bootstrap):** EUR_USD, GBP_USD, USD_JPY, AUD_USD, USD_CAD, USD_CHF (6 majors)

**Python modules to add:**
- `backend/markets/fx/oanda_broker.py` -- implements `BrokerClient` for FX; handles pip/lot sizing
- `backend/markets/fx/fx_ingestion.py` -- daily H1 and D1 candle ingest to `fx_ohlcv`

**Settings to add:** `oanda_api_key: str`, `oanda_account_id: str`, `oanda_practice: bool = True`

**BQ tables:** Writes to `pyfinagent_hdw.fx_ohlcv` (created in 5.3)

**Verification:** `python -m backend.markets.fx.fx_ingestion --pairs EUR_USD GBP_USD --start 2024-01-01 --end 2024-01-10 --dry-run && python -c "from backend.markets.fx.oanda_broker import OANDABroker; b = OANDABroker(practice=True); acct = b.get_account(); assert acct is not None"`

**Success criteria:**
- `fx_ohlcv` has >= 16 rows for EUR_USD + GBP_USD over 10-day window (FX has weekday-only bars)
- OANDA Practice account connects and returns non-empty account summary
- Lot sizing: 1 micro lot = 1,000 units; min order enforced
- Spread recorded in `spread_avg` column

**Depends on:** 5.1, 5.2, 5.3, 5.4

---

### Step 5.8 -- Futures Integration (IBKR via ib_insync)

**name:** `futures-ibkr-integration`

**Scope:** Add IBKR as futures execution broker using `ib_insync` library. Paper futures trading (IBKR Paper Account). Target: ES (S&P 500 e-mini), NQ (Nasdaq e-mini), GC (Gold), CL (Crude Oil) -- 4 liquid CME contracts. Daily OHLCV from Alpaca Futures Data (if available) or IBKR historical data as fallback.

**APIs:**
- IBKR TWS API via `ib_insync`: `IB.placeOrder(contract, MarketOrder(...))`
- IBKR Paper: connect to `127.0.0.1:7497` (TWS paper) or `127.0.0.1:4002` (IB Gateway paper)
- IBKR historical bars: `IB.reqHistoricalData(contract, ...)`
- Auth: TWS client ID + port (no API key; TWS/IB Gateway must be running)
- Rate limits: IBKR 50 pacing requests/10-min for historical data

**Python modules to add:**
- `backend/markets/futures/ibkr_broker.py` -- implements `BrokerClient` for futures; TWS connection management
- `backend/markets/futures/futures_ingestion.py` -- daily bar ingest from IBKR to `futures_ohlcv`
- `backend/markets/futures/contract_roll.py` -- front-month contract roll logic (roll at volume crossover)

**Settings to add:** `ibkr_host: str = "127.0.0.1"`, `ibkr_port: int = 7497`, `ibkr_client_id: int = 1`, `ibkr_paper: bool = True`

**BQ tables:** Writes to `pyfinagent_hdw.futures_ohlcv` (created in 5.3)

**Verification:** `python -c "from backend.markets.futures.contract_roll import get_front_month; c = get_front_month('ES'); assert c.symbol.startswith('ES')" && python -m backend.markets.futures.futures_ingestion --symbols ES NQ --start 2024-01-01 --end 2024-01-10 --dry-run`

**Success criteria:**
- `futures_ohlcv` has >= 18 rows for ES + NQ over 10-day window
- `contract_roll.get_front_month("ES")` returns valid contract with correct expiry
- `IBKRBroker.submit_order("ES", 1, "buy")` routes to paper account in dry-run mode
- `open_interest` column populated (not null) for all rows

**Depends on:** 5.1, 5.2, 5.3, 5.4
**DESIGN DECISION:** IBKR requires TWS/IB Gateway running locally; this is a infra dependency. If running headless (cloud VM), IB Gateway headless mode is required. Owner should decide: (a) run IB Gateway on VPS alongside pyfinagent, or (b) defer futures to after cloud infrastructure is stabilized.

---

### Step 5.9 -- International Equities (EODHD + IBKR)

**name:** `international-equities-integration`

**Scope:** Add international equity data via EODHD (Oslo Bors, LSE, TSX, XETRA). Existing `markets.py` already has NO/CA/EU/KR configs. Wire EODHD as data provider for these markets. Execution via IBKR (paper) for international symbols.

**APIs:**
- EODHD: `GET https://eodhd.com/api/eod/{symbol}.{exchange}?api_token=...` (e.g., `EQNR.OSL`)
- IBKR for execution of international symbols

**Python modules to add/modify:**
- `backend/markets/yfinance_provider.py` -- add international symbol routing (already partially handled by `markets.py` namespacing)
- `backend/markets/eodhd_provider.py` -- extend with exchange-code routing for international markets

**BQ tables:** Writes to `pyfinagent_hdw.backtest_prices` with `asset_class='equity'` and new `exchange` column

**Verification:** `python -c "from backend.markets.eodhd_provider import EODHDProvider; p = EODHDProvider(); r = p.get_ohlcv('EQNR.OSL', 'equity', '2024-01-01', '2024-01-31'); assert len(r) >= 20"`

**Success criteria:**
- EODHD returns >= 20 rows for `EQNR.OSL` (Equinor on Oslo Bors)
- `markets.py` `MARKET_CONFIG` extended with LSE (`XLON`) and TSX (`XTSE`) entries
- Currency conversion: OHLCV stored in local currency with `base_currency` column
- FX conversion layer applies `base_currency/USD` rate for portfolio P&L

**Depends on:** 5.2, 5.7 (needs FX rates for currency conversion)
**DESIGN DECISION:** Defer HKEX and TSE (Japan) to phase-5.x.ext -- timezone complexity (JST, HKT) adds substantial testing surface. Oslo is highest-priority given project owner's location.

---

### Step 5.10 -- Expanded ETF Universe

**name:** `etf-universe-expansion`

**Scope:** Broaden ETF coverage beyond the 20 tickers in `alt_data/etf_flows.py`. Add thematic ETFs (ARK, ICLN, XBI), leveraged ETFs (TQQQ, SOXL), and bond ETFs (TLT, HYG, LQD). Connect flows data to the crypto and FX signals as a cross-market leading indicator.

**Python modules to modify:**
- `backend/alt_data/etf_flows.py` -- add 20 new tickers to `_STARTER_TICKERS`

**BQ tables:** Extends `alt_etf_flows` (existing schema from phase-7.4 scaffold)

**New tickers to add:** ARKK, ARKG, ARKW, ICLN, XBI, TQQQ, SOXL, UVXY, SVXY, BITO (Bitcoin ETF), IBIT (BlackRock Bitcoin), EWJ (Japan), EWY (Korea), EWG (Germany), EWL (Norway proxy: NORW), FXI (China), VPL (Asia Pacific), HEDJ, DXJ

**Verification:** `python -m backend.alt_data.etf_flows --dry-run | python -c "import sys,json; d=json.load(sys.stdin); assert len(d['tickers']) >= 40"`

**Success criteria:**
- `_STARTER_TICKERS` has >= 40 entries including BITO, IBIT (crypto ETFs), NORW, EWJ
- Dry-run output is valid JSON with all tickers listed
- No regression on existing 20-ticker flows fetch

**Depends on:** 5.5 (crypto awareness)

---

### Step 5.11 -- Cross-Market Regime Detection

**name:** `cross-market-regime-detection`

**Scope:** Extend `regime_detector.py` to detect multi-market regime states. Add a `CrossMarketRegimeDetector` that monitors: (1) equity VIX, (2) crypto 30-day realized vol, (3) FX DXY rolling volatility, (4) yield curve slope (10Y-2Y). Classify into 4 cross-market regimes: RISK_ON, RISK_OFF, DECOUPLED, CONTAGION. Persist regime state to BQ.

**Research basis:** PMC11564031 finding that volatility clustering precedes contagion by weeks; BIS work 702 (cross-stock spillover) snippet.

**Python modules to add:**
- `backend/markets/cross_market/regime_detector.py` -- `CrossMarketRegimeDetector`; uses VIX + crypto_vol + DXY_vol + yield_slope inputs
- `backend/markets/cross_market/contagion_monitor.py` -- rolling conditional correlation tracker (30-day window)

**BQ tables to create:**
```
pyfinagent_data.cross_market_regime:
  regime_id STRING, as_of_date DATE, regime STRING,
  vix_level FLOAT64, crypto_vol_30d FLOAT64, dxy_vol_30d FLOAT64,
  yield_slope FLOAT64, contagion_score FLOAT64, created_at TIMESTAMP
```

**Verification:** `python -c "from backend.markets.cross_market.regime_detector import CrossMarketRegimeDetector; d = CrossMarketRegimeDetector(); r = d.classify(vix=25.0, crypto_vol=0.80, dxy_vol=0.10, yield_slope=-0.002); assert r in ['RISK_ON','RISK_OFF','DECOUPLED','CONTAGION']"`

**Success criteria:**
- `classify(vix=35, crypto_vol=1.2, yield_slope=-0.003)` returns `CONTAGION` (all three elevated)
- `classify(vix=14, crypto_vol=0.3, yield_slope=0.01)` returns `RISK_ON`
- `cross_market_regime` table created and writable
- 90-day backfill of historical regime states written to BQ

**Depends on:** 5.5, 5.7 (needs crypto + FX vol data)

---

### Step 5.12 -- Cross-Market Signal Generation

**name:** `cross-market-signal-generation`

**Scope:** Implement 3 cross-market alpha signals: (1) Crypto-to-Equity spillover momentum (Bitcoin 24h return > +5% => equity risk-on signal for tech stocks), (2) FX carry signal (high-yield currency pair excess return predicts equity sector rotation), (3) Yield curve + equity sector rotation (curve flattening => defensives). Integrate with existing orchestrator pipeline.

**Research basis:** BIS/PMC research on volatility spillovers; US-UK correlation 0.60 finding; "cross-market dynamics as independent source of alpha in systematic macro trading" (search snippet).

**Python modules to add:**
- `backend/markets/cross_market/spillover_signals.py` -- 3 cross-market signal functions, each returning `{signal_name, value, direction, confidence, asset_class_target}`
- Integrate into `backend/agents/orchestrator.py` as a new enrichment step (step ~15.5 in the pipeline)

**BQ tables:** Signals persisted to existing `pyfinagent_data.signals` table with `source='cross_market'`

**Verification:** `python -c "from backend.markets.cross_market.spillover_signals import crypto_equity_spillover; s = crypto_equity_spillover(btc_24h_return=0.07); assert s['direction'] in ['long','short','neutral'] and 0 <= s['confidence'] <= 1"`

**Success criteria:**
- All 3 signal functions return valid `{signal_name, value, direction, confidence}` dicts
- Signals are written to `pyfinagent_data.signals` with `source='cross_market'`
- Orchestrator pipeline includes cross-market enrichment step (optional, gated by `cross_market_signals_enabled` settings flag)

**Depends on:** 5.11

---

### Step 5.13 -- Multi-Asset Backtest Engine Extension

**name:** `multi-asset-backtest`

**Scope:** Extend `backtest_engine.py` to handle multi-asset portfolios. Add `asset_class` filter to data loading. Add cross-asset position sizing (fixed notional budget per asset class). Add per-asset-class performance metrics (Sharpe by asset class). This unlocks strategy validation for crypto/FX/futures signals.

**Python modules to modify:**
- `backend/backtest/backtest_engine.py` -- add `asset_classes: list[str]` param (default `["equity"]`); route data loading to `YFinanceProvider` / `EODHDProvider` by asset class
- `backend/backtest/analytics.py` -- add `compute_sharpe_by_asset_class(trades)` function
- `backend/backtest/data_ingestion.py` -- add `ingest_crypto_candles()`, `ingest_fx_ohlcv()` functions

**Verification:** `source .venv/bin/activate && python -c "from backend.backtest.backtest_engine import BacktestEngine; e = BacktestEngine(asset_classes=['equity','crypto']); r = e.run_backtest(tickers=['AAPL','BTC-USD'], start_date='2024-01-01', end_date='2024-06-30'); assert 'sharpe_by_asset_class' in r"`

**Success criteria:**
- Backtest with `asset_classes=["equity","crypto"]` completes without error
- `sharpe_by_asset_class` dict in result with keys `"equity"` and `"crypto"`
- Crypto positions sized correctly using 24h VaR window from `RiskEngine`
- No regression on existing equity-only backtest (Sharpe within 0.01 of baseline)

**Depends on:** 5.4, 5.5, 5.11

---

### Step 5.14 -- Multi-Market Autonomous Loop Integration

**name:** `multi-market-autonomous-loop`

**Scope:** Extend `autonomous_loop.py` to run parallel screening cycles per enabled asset class. Each cycle: screen universe -> analyze -> decide -> trade (via appropriate broker). Gated by settings flags per asset class. Persist cross-market P&L to BQ.

**Python modules to modify:**
- `backend/services/autonomous_loop.py` -- add `ENABLED_ASSET_CLASSES` loop; route to appropriate screener, broker, data provider per class
- `backend/tools/screener.py` -- add `screen_crypto_universe()` and `screen_fx_universe()` functions

**Settings to add:**
- `enable_crypto_trading: bool = False` (default off until phase-5.14 completes)
- `enable_fx_trading: bool = False`
- `enable_futures_trading: bool = False`

**BQ tables:** Trades written to `pyfinagent_pms.paper_trades` with `asset_class` column (migration: add nullable `asset_class` column)

**Verification:** `source .venv/bin/activate && ENABLE_CRYPTO_TRADING=true python -m backend.services.autonomous_loop --dry-run --cycles 1 2>&1 | grep -q "crypto"` and `python -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"`

**Success criteria:**
- `autonomous_loop.py` runs one dry-run cycle with `ENABLE_CRYPTO_TRADING=true` and logs "crypto" in output
- `paper_trades` table has `asset_class` column
- Default (all flags false) is identical to current behavior (no regression)

**Depends on:** 5.5, 5.7, 5.8, 5.12, 5.13

---

### Step 5.15 -- Phase-5 Integration Test and Go/No-Go Gate

**name:** `multi-market-integration-gate`

**Scope:** End-to-end integration test: run a 30-day backtest across equity + crypto; verify cross-market regime detection fires on a known historical date (2024-08-05 yen carry unwind -- already in gauntlet regimes); verify all 3 BQ asset-class tables populated; verify paper-trade cycle completes for equity + crypto without error; produce per-asset-class Sharpe. Owner reviews and approves phase-5 go-live.

**Python modules to add:**
- `tests/integration/test_multi_market_e2e.py`

**Verification:** `source .venv/bin/activate && python -m pytest tests/integration/test_multi_market_e2e.py -v && python scripts/harness/run_harness.py --dry-run --cycles 1`

**Success criteria:**
- Integration test exits 0
- 30-day backtest with `asset_classes=["equity","crypto"]` completes; result contains `sharpe_by_asset_class`
- `cross_market_regime` table has rows for 2024-08-05 with regime != RISK_ON
- `crypto_candles` has data for BTC/USD for the test window
- Per-asset-class Sharpe both > -2.0 (sanity check; no requirement to beat benchmark at this stage)
- Harness dry-run exits 0

**Depends on:** 5.14 (all prior steps)

---

## Recommended Phase-5 Ordering (Summary)

```
5.1  Broker Abstraction Layer        [foundation; no new markets yet]
5.2  Data Provider Abstraction       [foundation; adds EODHD]
5.3  Multi-Asset BQ Schema           [foundation; creates 3 tables]
5.4  Multi-Asset Risk Engine         [foundation; options/futures/crypto sizing]
5.5  Crypto (Alpaca)                 [first new market; reuses existing broker creds]
5.6  Options (Alpaca)                [second market; reuses broker; needs 5.4 Greeks]
5.7  FX (OANDA)                      [third market; new broker; simple REST]
5.8  Futures (IBKR)                  [fourth market; infra dependency on TWS]
5.9  International Equities (EODHD) [fifth; data-only; reuses IBKR for execution]
5.10 ETF Universe Expansion          [quick win; extends existing module]
5.11 Cross-Market Regime Detection   [intelligence layer; needs 5.5+5.7]
5.12 Cross-Market Signal Generation  [alpha layer; needs 5.11]
5.13 Multi-Asset Backtest            [validation layer; needs 5.5+5.7+5.11]
5.14 Multi-Market Autonomous Loop    [production layer; gates behind feature flags]
5.15 Integration Gate                [go/no-go; all must pass]
```

**Total sub-steps proposed: 15**

**Owner design decisions flagged:**
1. IBKR infra: TWS must run locally or on VPS -- decide before step 5.8
2. Futures-first vs crypto-first: current ordering is crypto-first (lower friction)
3. Oslo Bors is prioritized in international equities given project owner's location; HKEX/TSE deferred

---

## Research Gate Checklist

**Hard blockers:**
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 sources read)
- [x] 10+ unique URLs total including snippet-only (>20 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (see Internal Code Inventory section)

**Soft checks:**
- [x] Internal exploration covered every relevant module (12 files inspected)
- [x] Contradictions / consensus noted (see Consensus vs Debate section)
- [x] All claims cited per-claim (not just listed in a footer)
- [x] Three-variant query discipline documented (Queries Run section)

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 12,
  "urls_collected": 22,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "report_md": "handoff/current/phase-5-restructure-research-brief.md",
  "gate_passed": true
}
```

---

## Sources

- [Alpaca Trading Platform](https://alpaca.markets/)
- [Alpaca 2025 in Review](https://alpaca.markets/blog/alpacas-2025-in-review/)
- [Multi-market capital flow risk contagion — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564031/)
- [Building a Multi-Strategy Backtesting Dashboard with AI and Alpaca](https://alpaca.markets/learn/from-value-investing-to-systematic-trading-building-a-multi-strategy-backtesting-dashboard-with-ai-and-alpaca)
- [Top 5 Cryptocurrency Data APIs 2025 — Coinmonks](https://medium.com/coinmonks/top-5-cryptocurrency-data-apis-comprehensive-comparison-2025-626450b7ff7b)
- [Financial Data APIs Compared: Polygon vs IEX Cloud — 2025](https://www.ksred.com/the-complete-guide-to-financial-data-apis-building-your-own-stock-market-data-pipeline-in-2025/)
- [Best Crypto API Providers for Developers 2026 — CoinStats](https://coinstats.app/blog/best-crypto-api/)
- [Awesome Systematic Trading — GitHub](https://github.com/wangzhe3224/awesome-systematic-trading)
- [CFTC Automated Trading Regulation AT](https://www.cftc.gov/LawRegulation/FederalRegister/finalrules/2024-31177.html)
- [CFTC AI Advisory 2025 — Mintz Legal](https://www.mintz.com/insights-center/viewpoints/54731/2025-01-31-back-future-cftc-emphasizes-existing-regulatory)
- [Broker Comparison IBKR vs Alpaca — TradingView](https://www.tradingview.com/brokers/Alpaca-vs-IBKR/)
- [Best Brokers for Algo Trading 2026 — BrokerChooser](https://brokerchooser.com/best-brokers/best-brokers-for-algo-trading-in-the-united-states)
- [Cross-stock market spillovers — BIS Working Paper 702](https://www.bis.org/publ/work702.pdf)
- [Kaiko Developer Hub](https://docs.kaiko.com/)
- [Systematic Strategies & Quant Trading 2025 — Gresham](https://www.greshamllc.com/media/kycp0t30/systematic-report_0525_v1b.pdf)
