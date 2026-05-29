# Contract -- phase-50.1: FX data layer

**Step id:** 50.1 | **Priority:** P3 (phase-50 international expansion -- FREE foundation) | **depends_on:** 49.3
**Date:** 2026-05-30 | **harness_required:** true | **$0 LLM** | No new pip deps (yfinance/pandas/exchange_calendars already installed)

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher gate: **7 sources read in full, recency scan, 16 URLs, 11 internal files, gate_passed=true**). Decisive:
- **FX ticker direction LOCKED:** yfinance `EURUSD=X` = USD per 1 EUR (matches FRED `DEXUSEU` "USD to One Euro"); yfinance `KRW=X` = KRW per 1 USD (matches FRED `DEXKOUS` "Won to One USD"). **NEVER `KRWUSD=X`** (inverse -- silent KRW inversion is pitfall #1).
- **BQ:** dataset = `financial_reports` (sibling of historical_prices/_macro; **us-central1 -- migration must NOT pin `--location US`**). Mirror `historical_macro` (unpartitioned, tiny, `date` as STRING). Idempotent via `CREATE TABLE IF NOT EXISTS` (mirror `scripts/migrations/create_data_source_events_table.py:43-98`).
- **markets.py** `MARKET_CONFIG:21-52` is the currency source of truth (US->USD, EU->EUR, KR->KRW). Verified `from backend.backtest import markets` imports cleanly (no circular dep). `fx_rates.market_currency()` delegates, does not duplicate.
- **data_ingestion.py:146 stub fix:** `"USD" if market=="US" else "USD"` -> `markets.get_market_config(market)["currency"]`. (Flag: markets keys Germany as `EU` not `DE`; `DE`->USD fallback -- the namespace reconciliation is a 50.3 concern, NOT 50.1.)
- **Consumers (50.2/50.5):** paper_trader `mark_to_market` (live, no date -> "today") + backtest_trader `mark_to_market(date: str)` (point-in-time). So `get_fx_rate(from, to, date: str | None = None)` -- None=live, str=as-of; both speak str. `if from==to: return 1.0` keeps US-only byte-identical.
- **Caching:** BOTH -- BQ `historical_fx_rates` for backtest/point-in-time (as-of query `WHERE pair=? AND date<=? ORDER BY date DESC LIMIT 1`, forward-fills weekend/holiday gaps naturally) + `api_cache` TTL for the live daily mark, write-through (today's live mark becomes tomorrow's history). FRED (`DEXUSEU`/`DEXKOUS`, `FRED_API_KEY` present in .env) is the robustness fallback to yfinance.
- **Best practices:** point-in-time as-of storage (no look-ahead), forward-fill gaps with last-available, mid-rate (daily close) for NAV.

## Hypothesis
A `backend/services/fx_rates.py` exposing `get_fx_rate(from_ccy, to_ccy, date=None)` (BQ as-of for historical + api_cache write-through for live, yfinance primary + FRED fallback, direction-correct EURUSD=X/KRW=X) + a `market_currency(market)` delegating to markets.py + a `historical_fx_rates` BQ table (financial_reports, us-central1) + the data_ingestion.py:146 currency-stub fix, gives every downstream multi-currency calc a correct, look-ahead-free FX rate, with `from==to -> 1.0` keeping the US-only path byte-identical.

## Success criteria (IMMUTABLE -- verbatim from masterplan step 50.1)
1. backend/services/fx_rates.py exists: get_fx_rate(base, quote, date) + a daily-refresh path; sources EUR/USD and KRW/USD from yfinance (EURUSD=X, KRW=X) with a cache; USD->USD returns 1.0
2. historical_fx_rates BQ table (or a documented store) holds dated FX rates; backfilled for EUR/USD + KRW/USD over the backtest window
3. data_ingestion.py currency stub (line ~146) fixed: writes the correct ISO currency per market (US->USD, EU->EUR, KR->KRW), not 'USD' unconditionally
4. live evidence: a fetched EUR/USD + KRW/USD rate for a recent date captured verbatim

**Verification command (finalized post-research; the planning placeholder was intentional):** `ast.parse(fx_rates.py)` + `get_fx_rate('USD','USD')==1.0` + `market_currency('EU'/'KR'/'US')` == EUR/KRW/USD + `test -f live_check_50.1.md`. (The live EUR/USD + KRW/USD network fetch is the live_check evidence, kept out of the deterministic command to avoid network flakiness.)
**live_check:** REQUIRED -- verbatim fetched EUR/USD + KRW/USD rates + a BQ read showing dated FX rows.

## Plan steps
1. **`scripts/migrations/create_historical_fx_rates_table.py`** -- idempotent `CREATE TABLE IF NOT EXISTS financial_reports.historical_fx_rates` (cols: `pair` STRING e.g. "EURUSD", `date` STRING, `rate` FLOAT64, `source` STRING; mirror historical_macro shape; NO --location US pin). `--apply` dry-run guard.
2. **`backend/services/fx_rates.py`** -- `market_currency(market)` (delegates to markets.get_market_config); `get_fx_rate(from_ccy, to_ccy, date=None)`: from==to->1.0; else resolve the pair (USD-base; EURUSD=X gives USD/EUR, KRW=X gives KRW/USD -> invert as needed for the requested direction); date=None -> live (yfinance Ticker.history period=1d, FRED fallback) + api_cache write-through + persist to BQ; date=str -> BQ as-of query (forward-fill); a `backfill_fx(pairs, start, end)` daily-refresh path (yf.download EURUSD=X + KRW=X, FRED fallback) writing to BQ. ASCII logs, encoding utf-8.
3. **`backend/backtest/data_ingestion.py:146`** -- replace the `'USD' if market=='US' else 'USD'` stub with `markets.get_market_config(market)["currency"]` (guard unknown market -> 'USD').
4. **Verify:** ast.parse; the masterplan command (USD->USD=1.0 + market_currency); run the migration (`--apply`); backfill a small EUR/USD + KRW/USD window; a LIVE get_fx_rate('EUR','USD') + ('KRW','USD') fetch; a BQ read of historical_fx_rates rows -> capture verbatim into live_check_50.1.md.
5. **EVALUATE:** fresh qa (no self-eval). Then harness_log.md (LAST), then flip masterplan 50.1 -> done.

## Safety / scope notes
- 50.1 is PURELY ADDITIVE + market-agnostic: a new service + a new BQ table + a 1-line stub fix. NO change to paper_trader/backtest NAV math (that's 50.2). `from==to->1.0` means US-only/USD flows are untouched.
- Direction correctness is the #1 risk (KRW inversion) -- the live_check MUST show EUR/USD ~1.1-1.2 and KRW/USD ~0.0007 (1/1300) so an inversion is caught.
- The `DE` vs `EU` namespace mismatch in markets.py is OUT OF SCOPE (50.3); 50.1 uses the market codes markets.py already defines (US/EU/KR).
- No owner approval needed (yfinance + FRED free; no pip; financial_reports table create is not a DROP/DELETE).

## References
- handoff/current/research_brief.md (gate brief) + research_brief_multimarket.md (phase context)
- backend/backtest/markets.py:21-52 (MARKET_CONFIG currency map)
- backend/backtest/data_ingestion.py:146 (currency stub), :104-114 (yf.download pattern)
- scripts/migrations/create_data_source_events_table.py:43-98 (idempotent DDL mirror), migrate_backtest_data.py:61-68 (historical_macro shape)
- backend/services/paper_trader.py:432/446/480/1127 + backtest_trader.py:188/233 (FX consumers, 50.2/50.5)
- backend/services/api_cache.py (TTL cache, write-through), settings.py:43 (bq_dataset_reports)
- FRED DEXUSEU / DEXKOUS; yfinance EURUSD=X / KRW=X; Glassnode point-in-time; ECB/Kantox mid-rate
