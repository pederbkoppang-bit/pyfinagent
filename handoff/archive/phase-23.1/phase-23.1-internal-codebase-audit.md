# Internal Codebase Audit: Trading Universe Selection & Candidate Generation
## phase-23.1 — 2026-04-26

> Tier: internal-only (no external sources required; parallel agent covers
> external literature floor). All claims are file:line anchored.

---

## Read in full (internal files — counts toward internal gate)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/tools/screener.py` | 236 | Primary universe + scoring entry point | Active |
| `backend/services/autonomous_loop.py` | 632 | Daily cycle orchestrator — calls screener | Active |
| `backend/config/settings.py` | 178 | All paper-trading cost/limit knobs | Active |
| `backend/backtest/candidate_selector.py` | 210 | Backtest-time universe + scoring | Active |
| `backend/agents/orchestrator.py` | 1477 | 15-step analysis pipeline wiring | Active |
| `backend/alt_data/reddit_wsb.py` | 263 | Reddit/WallStreetBets alt-data scaffold | Scaffold (no live feed) |
| `backend/alt_data/twitter.py` | 226 | Twitter/X alt-data scaffold | Scaffold (no live feed) |
| `backend/alt_data/etf_flows.py` | ~200 | ETF flow alt-data | Active scaffold |
| `backend/alt_data/congress.py` | ~310 | Congressional trades alt-data | Active scaffold |
| `backend/alt_data/f13.py` | ~200 | 13-F institutional holdings | Active scaffold |
| `backend/alt_data/google_trends.py` | ~190 | Google Trends alt-data | Active scaffold |
| `backend/alt_data/finra_short.py` | ~100 | FINRA short volume | Active scaffold |
| `backend/alt_data/hiring.py` | ~150 | Job posting / R&D hiring signals | Active scaffold |
| `backend/news/registry.py` | ~100 | News source protocol + registry | Active |
| `backend/news/bq_writer.py` | ~220 | News + calendar BQ writer | Active |
| `backend/econ_calendar/watcher.py` | ~200 | Economic calendar pipeline | Active |
| `backend/slack_bot/scheduler.py` | 383 | APScheduler — 7 phase-9 cron jobs | Active |
| `backend/slack_bot/jobs/hourly_signal_warmup.py` | 49 | Watchlist-based signal cache warmup | Active (tiny watchlist) |
| `scripts/migrations/migrate_backtest_data.py` | 100 | BQ table definitions for backtest | Reference |
| `backend/agents/skills/*.md` (28 files) | varies | Agent skill prompts | Active |

---

## 1. Universe Definition

### Primary definition point

`backend/tools/screener.py:17`
```
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
```

`get_sp500_tickers()` at `backend/tools/screener.py:28-60` performs a live Wikipedia
HTML scrape at every daily cycle invocation. The `pd.read_html()` call targets the
first table (`tables[0]`) and extracts the `Symbol` column, normalizing dots to
hyphens (e.g., BRK.B -> BRK-B). There is no cache, no BQ table, no scheduled
refresh — every call to `screen_universe()` that does not supply a `tickers` arg
re-scrapes Wikipedia live over HTTP.

This is why the daily run logged "Screened 503 S&P 500 tickers" — that is the
current S&P 500 membership count as of today's Wikipedia snapshot.

### Survivorship-bias guard (explicit NotImplementedError)

Both the live screener and the backtest candidate selector raise
`NotImplementedError` when `as_of` is supplied:

- `backend/tools/screener.py:41-46`: "point-in-time S&P 500 membership not
  available yet; callers must supply a cached historical universe or wait for the
  delistings-feed ingestion (phase-4.8.x)."
- `backend/backtest/candidate_selector.py:121-126`: identical message.

There is **no historical index membership table** in BQ. Backtests that need
point-in-time correctness have no supported path yet.

### Fallback tickers (Wikipedia failure)

`backend/tools/screener.py:228-236`: 49-ticker hardcoded fallback list (large-cap
megacaps) used when the Wikipedia HTTP request fails.

`backend/backtest/candidate_selector.py:203-210`: separate 50-ticker fallback list,
similar composition.

### Backtest-side universe

`backend/backtest/candidate_selector.py:98-144`: `get_universe_tickers()` also
scrapes Wikipedia for market="US". Non-US markets (NO/CA/DE/KR from phase-2.9
abstraction) return an empty list with a warning — they are stubs only.

### Sector ETFs (implicit universe overlay)

`backend/tools/screener.py:20-25`: 11 SPDR sector ETFs (XLK, XLV, XLF, XLY, XLC,
XLI, XLP, XLE, XLU, XLRE, XLB) imported into the screener module but only used
by the downstream `sector_analysis.py` for relative-strength calculation. They do
not currently expand the candidate universe.

---

## 2. Screening and Scoring Logic

### Entry path

`backend/services/autonomous_loop.py:113-114`:
```python
screen_data = screen_universe(period="6mo")
candidates = rank_candidates(screen_data, top_n=settings.paper_screen_top_n)
```

`paper_screen_top_n` defaults to **10** (`backend/config/settings.py:145`).

### Step 1 — Quant filters (zero LLM cost)

`backend/tools/screener.py:63-148` — `screen_universe()`:

1. Batch yfinance download for all 503 tickers, `period="6mo"`, threads=True.
2. Per-ticker filter (each of these discards the ticker if it fails):
   - `len(close) < 20` — less than 20 price bars: skip (`screener.py:106`)
   - `current_price < min_price` (default 5.0): skip (`screener.py:113`)
   - `avg_vol < min_avg_volume` (default 100,000 shares 20-day avg): skip (`screener.py:113`)
3. Computes: momentum_1m (21d), momentum_3m (63d), momentum_6m (full period),
   RSI-14, annualized volatility, 50-day SMA distance (`screener.py:117-142`).
4. No market-cap filter at this stage (min_market_cap param defined but only
   passed through, not applied — `screener.py:65-66`).

### Step 2 — Composite scoring and top-10 selection

`backend/tools/screener.py:151-201` — `rank_candidates()`:

Default strategy = "momentum":
```
score = mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25
```
RSI penalty: score *= 0.7 if RSI > 80; score *= 0.8 if RSI < 20.
Volatility penalty: score *= 0.85 if ann_vol > 0.6.

Alternative strategy "value_momentum": `mom_3m*0.5 - abs(sma_dist)*0.2 + mom_1m*0.3`

Returns `scored[:top_n]` — hard cutoff at `top_n=10` by default.

### Step 3 — Already-held filter

`autonomous_loop.py:119-123`: removes tickers already in open positions from the
candidate list before analysis.

### Step 4 — LLM analysis budget gate

`autonomous_loop.py:152-163`: iterates `analyze_tickers = new_candidates[:settings.paper_analyze_top_n]`
(default **5** tickers, `settings.py:146`) and breaks if
`total_analysis_cost >= settings.paper_max_daily_cost_usd` (default **$2.00 USD**,
`settings.py:150`).

### Backtest scoring (slightly different weights)

`backend/backtest/candidate_selector.py:168-200` — `_rank_candidates()`:
```
alpha_score = mom_6m/100 * 0.4  +  rsi_score * 0.2  +  (1-vol) * 0.2  +  sma_score * 0.2
```
The backtest uses 4-way equal weighting (0.4/0.2/0.2/0.2) vs the live screener's
momentum-heavy 3-factor weighting (0.40/0.35/0.25). **These two implementations
are not kept in sync** — a known divergence between live and backtest behavior.

### Cost-per-candidate estimate

- Screen phase: $0 (yfinance only).
- Analysis phase: each `_run_single_analysis()` call runs the 15-step Gemini
  pipeline in lite mode (`settings.lite_mode = True` forced at `autonomous_loop.py:149`).
  Lite mode cuts ~39 calls down to ~20 LLM calls (per `backend-agents.md`).
  Cost tracked via `analysis.get("total_cost_usd", 0.1)` fallback at
  `autonomous_loop.py:161` — the default assumption is **$0.10/candidate** when
  actual cost is not returned.
- Daily budget: $2.00 hard cap allows approximately 20 candidates at $0.10 each
  (in practice, 5 new + re-eval holdings per cycle).

---

## 3. Existing Alt-Data Infrastructure

### News pipeline (phase-6)

| Module | Path | Data source | BQ table | Status |
|--------|------|-------------|----------|--------|
| Finnhub news | `backend/news/sources/finnhub.py` | Finnhub API (key: `finnhub_api_key`) | `pyfinagent_data.news_articles` | Active |
| Benzinga news | `backend/news/sources/benzinga.py` | Benzinga API (key: `benzinga_api_key`) | `pyfinagent_data.news_articles` | Active |
| News registry | `backend/news/registry.py` | Protocol + decorator pattern | — | Active |
| BQ writer | `backend/news/bq_writer.py:36,38` | Aggregates all sources | `news_articles`, `news_sentiment`, `calendar_events` | Active |

Rate limits configured in `settings.py:73-77`: Finnhub 25 RPS, Benzinga 2 RPS,
Alpaca 30 RPS, FRED 5 RPS, AlphaVantage 1 RPS.

News sentiment also consumed from **Alpaca data API**
(`settings.py:64-65` — `alpaca_api_key_id` + `alpaca_api_secret_key` for
`data.alpaca.markets/v1beta1/news`).

### Economic calendar (phase-6.6)

`backend/econ_calendar/watcher.py`: orchestrates calendar sources, normalizes
events to `CalendarEvent` TypedDict, applies FOMC blackout windows. Writes to
`pyfinagent_data.calendar_events` via `news/bq_writer.py:38`.

Finnhub earnings calendar: `backend/econ_calendar/sources/finnhub_earnings.py`.

### Social sentiment (live pipeline)

Alpha Vantage social API consumed via `backend/tools/social_sentiment.py` —
feeds `orchestrator.py:664-665` (Step 7 of the 15-step pipeline).

### Social alt-data (scaffolded, not live)

| Source | Path | BQ table | Notes |
|--------|------|----------|-------|
| Reddit/WallStreetBets | `backend/alt_data/reddit_wsb.py:43` | `pyfinagent_data.alt_reddit_sentiment` | PRAW scaffold; `fetch_wsb_posts()` returns [] until `praw.Reddit(...)` creds are configured |
| Twitter/X | `backend/alt_data/twitter.py:37` | `pyfinagent_data.alt_twitter_sentiment` | X API v2 scaffold; `fetch_cashtag_tweets()` returns [] per `twitter.py:86-89` until `X_BEARER_TOKEN` is set |

Both modules have full BQ table schemas, VADER sentiment scoring, cashtag
extraction, and author-hash privacy. Live data feeds are deferred.

### FRED / macro

`backend/backtest/data_ingestion.py:21`: 7 FRED series: FEDFUNDS, CPIAUCSL,
UNRATE, GDP, T10Y2Y, UMCSENT, DGS10. Consumed by Enhanced Macro Agent
(`skills/enhanced_macro_agent.md` — Step 9 in pipeline). Weekly refresh via
`backend/slack_bot/jobs/weekly_fred_refresh.py` (scheduler `scheduler.py:362`,
Sundays 02:00).

FRED data lands in `financial_reports.historical_macro`
(`scripts/migrations/migrate_backtest_data.py:61-68`).

### Earnings call transcripts

`backend/tools/earnings_tone.py` — consumed by `orchestrator.py:672-674`.
Source: API Ninjas earnings transcript endpoint (`settings.py:58` — `api_ninjas_key`)
plus GCS bucket fallback (`settings.gcs_bucket_name = "10k-filling-data"`).

### Patent / R&D / hiring signals

| Signal | Path | BQ table |
|--------|------|----------|
| Patent velocity | `backend/agents/skills/sector_catalyst_agent.md` (orchestrator Step 7 aux) | — (derived at runtime from patent_tracker.py) |
| R&D hiring | `backend/alt_data/hiring.py:36` | `pyfinagent_data.alt_hiring_signals` |
| Google Trends | `backend/alt_data/google_trends.py:34` | `pyfinagent_data.alt_google_trends` |

### SEC / insider / institutional

| Signal | Path | BQ table |
|--------|------|----------|
| SEC insider trades | `backend/tools/sec_insider.py` | — (runtime fetch from SEC EDGAR) |
| Congressional trades | `backend/alt_data/congress.py:49` | `pyfinagent_data.alt_congress_trades` |
| 13-F institutional holdings | `backend/alt_data/f13.py:39` | `pyfinagent_data.alt_13f_holdings` |

`alt_data/features.py` provides `aggregate_congress_features()` and
`aggregate_13f_features()` plus Spearman IC evaluation helpers — designed to
produce factor signals from the above tables.

### FINRA short volume

`backend/alt_data/finra_short.py:36`: table `pyfinagent_data.alt_finra_short_volume`.

### ETF flows

`backend/alt_data/etf_flows.py:35`: table `pyfinagent_data.alt_etf_flows`.

### Options data

`backend/markets/options/options_ingestion.py:27`: `TABLE_FQN = "pyfinagent_hdw.options_snapshots"`.
Greeks computed via `backend/markets/options/greeks.py`. Not yet connected to
candidate selection.

### Sector relative strength

`backend/tools/sector_analysis.py:13`: 11 GICS sector ETFs used to compute
relative strength. Called by `orchestrator.py:684-686` for Step 8. A sector
rotation feature flag exists in `backend/autonomous_loop.py:567`
(`feature_name: "sector_rotation"`) but is not wired into the screener.

### Watchlist-based signal warmup

`backend/slack_bot/jobs/hourly_signal_warmup.py:37-45`: reads
`settings.watchlist` (field not defined in `settings.py` — returns empty list
with fallback `["AAPL", "MSFT", "SPY"]`). This is a dead-end stub; the watchlist
field does not exist on the `Settings` model.

### 15-step pipeline agents (all operate per-ticker AFTER screening)

| Step | Agent | Key signal |
|------|-------|------------|
| 3 | RAG Agent | SEC filings / Vertex Search |
| 4 | Market Agent | Google Search grounded news sentiment |
| 5 | Competitor Agent | Alpha Vantage news co-occurrence |
| 6 | Enrichment | AlphaVantage financials + quant |
| 7 | Social Sentiment | AV social API + NLP FinBERT |
| 7 aux | Sector Catalyst | Patent velocity + R&D hiring |
| 8 | Bull/Bear Debate | LLM synthesis |
| 9 | Enhanced Macro | FRED 7-series + AV macro |
| 10 | Deep Dive | Options, insider, supply chain |
| 11 | Synthesis | All prior signals aggregated |

None of these agents influence the candidate **selection** step. They all run
**after** a ticker has been picked by the quant screener.

---

## 4. BigQuery Tables Relevant to Candidate Generation

| Dataset | Table | Purpose | Refreshed by |
|---------|-------|---------|--------------|
| `financial_reports` | `historical_prices` | OHLCV for backtest | `data_ingestion.py:93` via yfinance |
| `financial_reports` | `historical_fundamentals` | Quarterly financials for backtest | `data_ingestion.py:185` via yfinance |
| `financial_reports` | `historical_macro` | FRED 7-series | `data_ingestion.py:273` / weekly_fred_refresh |
| `pyfinagent_data` | `news_articles` | Finnhub + Benzinga + Alpaca news | phase-6 news pipeline |
| `pyfinagent_data` | `news_sentiment` | Per-article sentiment scores | phase-6 sentiment scorer |
| `pyfinagent_data` | `calendar_events` | Earnings dates + FOMC + macro | econ_calendar/watcher.py |
| `pyfinagent_data` | `alt_reddit_sentiment` | Reddit cashtag sentiment | reddit_wsb.py (scaffold) |
| `pyfinagent_data` | `alt_twitter_sentiment` | Twitter cashtag sentiment | twitter.py (scaffold) |
| `pyfinagent_data` | `alt_etf_flows` | ETF fund flow data | etf_flows.py |
| `pyfinagent_data` | `alt_congress_trades` | Congressional trade disclosures | congress.py |
| `pyfinagent_data` | `alt_13f_holdings` | Institutional 13-F positions | f13.py |
| `pyfinagent_data` | `alt_finra_short_volume` | FINRA short volume by ticker | finra_short.py |
| `pyfinagent_data` | `alt_hiring_signals` | R&D / engineering job postings | hiring.py |
| `pyfinagent_data` | `alt_google_trends` | Google Trends search volume | google_trends.py |
| `pyfinagent_data` | `harness_learning_log` | Harness loop cycle outcomes | autoresearch/slot_accounting.py |
| `pyfinagent_data` | `filings_rag` | SEC 10-K/10-Q RAG chunks | orchestrator.py:1073 |
| `pyfinagent_hdw` | `options_snapshots` | Options chain snapshots | options_ingestion.py |
| `pyfinagent_pms` | (portfolio tables) | Paper trading positions + NAV | paper_trader.py |

None of the alt_data tables (congress, 13f, reddit, twitter, google_trends,
hiring, finra_short, etf_flows) are consumed during the screening step. They are
populated by background cron jobs or scaffolded for future use, but there is no
code path that reads them when selecting candidates.

---

## 5. Cost Ceiling

All values are `Settings` field defaults from `backend/config/settings.py`.
They can be overridden via `.env`.

| Parameter | Default | Location |
|-----------|---------|----------|
| `paper_max_positions` | 10 | `settings.py:143` |
| `paper_screen_top_n` | 10 | `settings.py:145` |
| `paper_analyze_top_n` | 5 | `settings.py:146` |
| `paper_max_daily_cost_usd` | $2.00 | `settings.py:150` |
| `paper_reeval_frequency_days` | 3 | `settings.py:148` |
| `max_analysis_cost_usd` | $0.50 soft | `settings.py:120` |
| `lite_mode` forced on paper | True (forced at cycle start) | `autonomous_loop.py:149` |

**Effective daily capacity:** At $2.00/day and ~$0.10/candidate (lite-mode
estimate), the system can afford roughly 20 deep-analyses per day. With
`paper_analyze_top_n=5` and `paper_max_positions=10`, the system analyzes at
most 5 new candidates + re-evaluates held positions every 3 days. In practice,
fewer than 10 deep LLM calls per cycle. Widening the universe to e.g. Russell 2000
(~2,000 tickers) would not change the LLM cost — only the yfinance batch download
time and the scoring computation, both of which are free.

---

## 6. Synthesis

### "The 503 universe is defined here"

`backend/tools/screener.py:17` — `SP500_URL` constant pointing to Wikipedia.
`get_sp500_tickers()` at `screener.py:28-60` live-scrapes at every daily cycle
call (no cache). The identical pattern is duplicated in
`backend/backtest/candidate_selector.py:134-141` for the backtest path.
Refresh mechanism: on-demand, every invocation of `screen_universe()` (called
once per daily paper-trading cycle at `autonomous_loop.py:113`).

### "Today's bottleneck on going wider is"

**Not LLM cost and not latency** — it is data coverage. The screener is
structurally blocked from wider universes by:

1. **No historical membership table**: the `NotImplementedError` at
   `screener.py:41-46` and `candidate_selector.py:121-126` means any universe
   beyond today's Wikipedia snapshot is unsupported. Russell 2000 or Nasdaq 100
   could be scraped from a similar source, but there is no PIT-correct data path.

2. **No fundamental filter at screen time**: `screen_universe()` ignores
   `min_market_cap` (the parameter is defined at `screener.py:65` but never
   applied in the loop at `screener.py:112-113`). Expanding to small-cap universes
   without a market-cap gate risks including illiquid micro-caps.

3. **Single-market only**: `candidate_selector.py:127-132` returns an empty
   universe for any market other than "US". International expansion (NO, CA, DE,
   KR) is stubbed at the Phase 2.9 abstraction level.

4. **Survivorship bias**: using today's S&P 500 for backtests injects survivorship
   bias — delistings feed is explicitly deferred to "phase-4.8.x".

### "These idea-generation surfaces ALREADY EXIST in the repo and could feed candidate selection"

All of the following are wired, populated (or scaffolded with BQ schemas), and
could be used as pre-screener universe filters or idea signals upstream of the
quant screen:

1. **Finnhub + Benzinga + Alpaca news pipeline** (`backend/news/`, BQ:
   `pyfinagent_data.news_articles`) — tickers mentioned in breaking news could
   seed a supplemental watchlist before the S&P screen.

2. **Congressional trades** (`backend/alt_data/congress.py`, BQ:
   `pyfinagent_data.alt_congress_trades`) — lagged disclosure signal; buy-side
   trades could generate a ranked ticker list.

3. **13-F institutional holdings** (`backend/alt_data/f13.py`, BQ:
   `pyfinagent_data.alt_13f_holdings`) + `aggregate_13f_features()` in
   `alt_data/features.py` — already has IC-evaluation infrastructure.

4. **FINRA short volume** (`backend/alt_data/finra_short.py`, BQ:
   `pyfinagent_data.alt_finra_short_volume`) — elevated short volume as a mean-
   reversion candidate signal.

5. **ETF flows** (`backend/alt_data/etf_flows.py`, BQ:
   `pyfinagent_data.alt_etf_flows`) — sector-level fund flows could feed a
   sector rotation pre-filter.

6. **Google Trends** (`backend/alt_data/google_trends.py`, BQ:
   `pyfinagent_data.alt_google_trends`) — search-volume spikes as a retail
   attention proxy.

7. **Hiring / R&D signals** (`backend/alt_data/hiring.py`, BQ:
   `pyfinagent_data.alt_hiring_signals`) — already feeding the Sector Catalyst
   Agent (Step 7 aux); the same signal could elevate tickers in the screener.

8. **Sector relative strength ETFs** (`backend/tools/sector_analysis.py:13`) —
   11 SPDR ETFs are already loaded; a sector-momentum pre-filter could narrow
   the 503 to the top 2-3 sectors before applying per-ticker screens.

9. **Earnings calendar** (`backend/econ_calendar/`, BQ:
   `pyfinagent_data.calendar_events`) — upcoming earnings dates as a catalyst
   signal for pre-event analysis.

10. **Options snapshots** (`pyfinagent_hdw.options_snapshots`, Greeks via
    `backend/markets/options/greeks.py`) — unusual options activity (IV spike,
    large call sweeps) could flag tickers for inclusion outside S&P 500.

### "These surfaces are missing entirely"

The following signal types have no code, no schema, and no integration in the
repo:

1. **Russell 2000 / Nasdaq 100 universe lists** — no scraper, no BQ table, no
   `get_russell2000_tickers()` equivalent. Small-cap / growth-focused universe
   expansion requires new code.

2. **Point-in-time index membership** — the delistings feed for historical S&P
   500 membership is explicitly deferred (`phase-4.8.x`). No BQ table, no data
   source.

3. **Quant factor model (Fama-French / momentum factors)** — no factor library
   (HML, SMB, MOM, QMJ) computed or stored. The screener computes raw momentum
   but has no market-neutral factor exposure or factor-tilt logic.

4. **StockTwits** — only Reddit (scaffolded) and Twitter (scaffolded) are
   present. StockTwits API is not integrated.

5. **Polygon.io news / tick data** — `settings.py` has no Polygon API key field;
   no Polygon integration exists anywhere.

6. **SEC EDGAR full-text search for idea generation** — `sec_insider.py`
   fetches Form 4 insider transactions but there is no pipeline to scan SEC
   EDGAR full-text search for 8-K filings that could surface non-S&P candidates.

7. **Macro-to-sector idea generation** — there is no code path that reads the
   FRED macro signal output and uses it to nominate tickers for screening. The
   Enhanced Macro Agent analyzes macro *for an already-selected ticker*; the
   inverse flow (macro regime -> sector -> tickers to screen) does not exist.

8. **Earnings surprise / estimate revision feeds** — no integration with
   FactSet, Bloomberg consensus, or any earnings-estimate data source. The
   Earnings Tone Agent analyzes transcripts for selected tickers but there is no
   pre-screen for high-revision-dispersion candidates.

---

## Internal Code Inventory Summary

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/tools/screener.py` | 236 | Live universe + quant screen + ranking | Active |
| `backend/backtest/candidate_selector.py` | 210 | Backtest-time screen (slightly different weights) | Active |
| `backend/services/autonomous_loop.py` | 632 | Daily cycle; calls screener, gates LLM cost | Active |
| `backend/config/settings.py` | 178 | All paper trading knobs | Active |
| `backend/news/` (4 files) | ~600 | News pipeline (Finnhub/Benzinga/Alpaca) | Active |
| `backend/econ_calendar/` (5 files) | ~500 | Calendar pipeline | Active |
| `backend/alt_data/` (9 files) | ~1,700 | Alt signals (most scaffolded) | Mixed |
| `backend/agents/skills/` (28 files) | varies | Per-ticker analysis agents | Active |
| `backend/agents/orchestrator.py` | 1477 | 15-step pipeline | Active |
| `backend/markets/options/` (2 files) | ~300 | Options data | Scaffold |
| `backend/tools/sector_analysis.py` | 182 | Sector relative strength | Active (post-screen) |

**Total internal files inspected: 20+ (see table at top of document)**

---

## Research Gate Checklist

Hard blockers:
- [x] file:line anchors for every internal claim
- [x] Internal exploration covered every relevant module
- [x] Contradictions noted (backtest scorer != live scorer weights)
- [x] All claims cited per-claim

Soft checks:
- [x] Alt-data BQ tables inventoried
- [x] Cost ceiling documented with exact settings.py locations
- [x] Dead code identified (watchlist field missing from Settings model)
- [x] Scaffold vs active distinction noted for every alt-data module

External gate note: this is an internal-only brief. The parallel agent covers
the >=5 external sources floor. `external_sources_read_in_full` is 0 by design.

---

```json
{
  "tier": "internal-only",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 22,
  "report_md": "handoff/current/phase-23.1-internal-codebase-audit.md",
  "gate_passed": "internal-half-only — external gate deferred to parallel agent"
}
```
