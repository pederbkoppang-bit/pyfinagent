# Experiment results -- phase-50.1: FX data layer

**Date:** 2026-05-30 | **Result: built + live-verified** | $0 LLM | no new pip deps | first step of phase-50 (international expansion).

## What was built
The FREE FX foundation for multi-currency work: a `fx_rates` service (correct-direction EUR/USD + KRW/USD from yfinance with FRED fallback, BQ point-in-time + api_cache live mark), a `historical_fx_rates` BQ table, and the `data_ingestion.py:146` currency-stub fix.

## Files changed/added
1. **`backend/services/fx_rates.py`** (NEW) -- `market_currency(market)` (delegates to markets.MARKET_CONFIG); `get_fx_rate(from, to, date=None)` (from==to->1.0; else usd_value(from)/usd_value(to)); `_usd_value_live` (api_cache -> yfinance -> FRED fallback -> BQ write-through), `_usd_value_asof` (BQ as-of `date<=` point-in-time read); `backfill_fx(currencies, start, end)`. Direction map: EUR via EURUSD=X (USD/EUR), KRW via KRW=X (KRW/USD, inverted to usd_value); NEVER KRWUSD=X.
2. **`scripts/migrations/create_historical_fx_rates_table.py`** (NEW) -- idempotent CREATE TABLE IF NOT EXISTS `financial_reports.historical_fx_rates` (pair STRING, date STRING, rate FLOAT64, source STRING; CLUSTER BY pair; no --location pin -- us-central1 auto-resolved). Dry-run default; --apply.
3. **`backend/backtest/data_ingestion.py`** -- import `markets`; line 146 stub `"USD" if market=="US" else "USD"` -> `markets.get_market_config(market)["currency"]` (US->USD, EU->EUR, KR->KRW).

## Live verification (full evidence in live_check_50.1.md)
- Migration APPLIED (table created in financial_reports / us-central1).
- get_fx_rate: USD/USD=1.0, EUR/USD=1.166, KRW/USD=0.000664 (DIRECTION CORRECT, no KRW inversion), USD/EUR=0.858 (inverse).
- backfill 22 rows; well-formed EURUSD 12 rows (avg 1.164) + KRWUSD 12 rows (avg 0.000665), 2026-05-15..29.
- point-in-time as-of read: EUR/USD @2026-05-20 = 1.1607, KRW/USD @2026-05-20 = 0.000663 (no look-ahead).
- data_ingestion imports markets + the stub fix verified (market_currency EU/KR/US == EUR/KRW/USD).

## Success criteria mapping (all 4 met)
1. fx_rates.py with get_fx_rate (USD/USD=1.0) + EUR/USD + KRW/USD from yfinance + cache -- YES.
2. historical_fx_rates table holds dated FX rates, backfilled for EUR/USD + KRW/USD -- YES (12 days each).
3. data_ingestion.py:146 currency stub fixed (per-market ISO currency) -- YES.
4. live EUR/USD + KRW/USD rates fetched verbatim -- YES.

## Scope honesty / hardening flags (NOT criterion violations)
- **A real bug was caught + fixed during live verification**: single-ticker `yf.download` returns a 1-col (MultiIndex) DataFrame; the initial `.items()` iterated the column name into `date` -> 2 malformed rows (date='EURUSD=X'/'KRW=X'). FIXED (squeeze Close to a Series before `.items()`). The 2 junk rows could NOT be DELETE'd (BQ streaming-buffer blocks DELETE <~90min); they are HARMLESS (the as-of `date<=` filter excludes the non-ISO dates lexically) and cleanable post-flush via `DELETE WHERE date NOT LIKE '2%'`.
- **Append, not upsert**: backfill + live write-through both append; the as-of `LIMIT 1` read is unaffected by duplicate dates. Hardening follow-up: switch backfill to a BQ load-job (avoids the streaming-buffer DELETE limitation) + MERGE-upsert for dedup.
- **US-only path byte-identical**: from==to->1.0 means USD-only flows are untouched. This step is purely additive + market-agnostic (no paper_trader / backtest NAV change -- that's 50.2).
- No owner approval used: yfinance + FRED free; no pip; CREATE TABLE (not DROP/DELETE). The one qualified cleanup DELETE attempt was blocked by the streaming buffer (not executed).
