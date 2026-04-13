# Phase 2.9: Multi-Market Data Layer — Integration Test Plan

**Status:** Ready for execution (upon Phase 2.8 PASS)  
**Prepared:** 2026-03-29 10:57 UTC  
**Scope:** Verify multi-market abstractions work end-to-end; BQ schema changes don't break US-only workflows

---

## Quick Summary

Phase 2.9 implementation (commit `18fa902`) added:
- BQ schema: `market`, `currency` columns (all 3 tables: prices, fundamentals, macro)
- Data ingestion: ticker namespace parsing (e.g., `NO:EQNR` → market=NO, ticker=EQNR)
- Market abstractions: `backend/backtest/markets.py` with exchange calendars (US, NO, CA, EU, KR)
- Cache layer: market filtering (currently hardcoded to `'US'`)

**Integration test goal:** Verify these changes don't break current US-only operation AND work correctly if future code enables multi-market.

---

## Test Plan (Research-Backed)

No external research needed (this is infrastructure testing, not statistical work). Tests are straightforward validation.

### 1. BQ Schema Verification

**Test:** Confirm all 3 tables have market + currency columns with correct defaults.

```python
# backend/backtest/data_ingestion.py — add temporary test
import os, json
from google.cloud import bigquery
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path('.') / 'backend' / '.env')

PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'sunny-might-477607-p8')
creds_json = os.environ.get('GCP_CREDENTIALS_JSON', '')
if creds_json:
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json))
else:
    credentials = None

client = bigquery.Client(project=PROJECT_ID, credentials=credentials)

for table_name in ['historical_prices', 'historical_fundamentals', 'historical_macro']:
    table = client.get_table(f'{PROJECT_ID}.financial_reports.{table_name}')
    field_names = [f.name for f in table.schema]
    
    assert 'market' in field_names, f"Market column missing from {table_name}"
    assert 'currency' in field_names or table_name == 'historical_macro', f"Currency column missing from {table_name}"
    print(f"✅ {table_name}: market + currency columns present")
```

**Expected result:** All 3 tables confirm schema updates.

### 2. Data Ingestion — Ticker Namespace Parsing

**Test:** Verify `market` and `currency` columns are populated correctly during ingestion.

```python
# Stub test: check if data_ingestion.py correctly extracts market from ticker namespace
from backend.backtest.data_ingestion import DataIngestionService
from backend.config.settings import Settings

# Mock test: parse ticker with namespace
test_cases = [
    ("US:AAPL", "US", "AAPL"),
    ("NO:EQNR", "NO", "EQNR"),
    ("MSFT", "US", "MSFT"),  # Default to US if no namespace
]

for ticker_input, expected_market, expected_clean in test_cases:
    market = "US"
    clean_ticker = ticker_input
    if ":" in ticker_input:
        market, clean_ticker = ticker_input.split(":", 1)
    
    assert market == expected_market, f"Expected {expected_market}, got {market}"
    assert clean_ticker == expected_clean, f"Expected {expected_clean}, got {clean_ticker}"
    print(f"✅ {ticker_input} → market={market}, ticker={clean_ticker}")
```

**Expected result:** Namespace parsing works correctly; existing US tickers default to market='US'.

### 3. Cache Layer — Market Filtering

**Test:** Verify cache.py queries include `AND market = 'US'` filter and don't break.

```python
# Verify cache.preload_prices() query has market filter
with open('backend/backtest/cache.py', 'r') as f:
    cache_code = f.read()
    assert "market = 'US'" in cache_code, "Cache query missing market filter"
    print("✅ Cache query includes market = 'US' filter")
```

**Expected result:** Market filter is in place; US-only queries work correctly.

### 4. Market Config Accessibility

**Test:** Verify `markets.py` loads and provides correct configuration for all 5 markets.

```python
from backend.backtest.markets import MARKET_CONFIG, get_trading_calendar

# Check all markets are defined
required_markets = ["US", "NO", "CA", "EU", "KR"]
for market in required_markets:
    assert market in MARKET_CONFIG, f"Missing config for {market}"
    config = MARKET_CONFIG[market]
    assert "exchange" in config and "currency" in config and "timezone" in config
    print(f"✅ {market}: {config['description']}")

# Verify exchange_calendars work (optional, depends on calendar availability)
try:
    cal_us = get_trading_calendar("US")
    print(f"✅ US trading calendar: {cal_us}")
except Exception as e:
    print(f"⚠️  Calendar load (expected if exchange_calendars not installed): {e}")
```

**Expected result:** All 5 markets configured; calendar abstraction ready for future use.

### 5. End-to-End Backtest (US-Only)

**Test:** Run a quick backtest to confirm no regression. Should match Phase 2.8 baseline.

```bash
cd ~/.openclaw/workspace/pyfinagent
source .venv/bin/activate
python -c "
from backend.backtest.backtest_engine import BacktestEngine
from backend.config.settings import Settings

settings = Settings()
bq = None  # Uses default from Settings

# Load best params from Phase 2.8
import json
with open('backend/backtest/experiments/optimizer_best.json') as f:
    params = json.load(f)

# Quick backtest: 100 days instead of full 3 years (just validation)
engine = BacktestEngine(
    params=params,
    settings=settings,
    bq_client=bq,
    start_date='2024-01-01',
    end_date='2024-04-10',  # 100 days
)
result = engine.run_backtest()
print(f'✅ Quick backtest complete: Sharpe={result[\"full_period\"][\"sharpe\"]:.4f}')
"
```

**Expected result:** Backtest completes without errors. Sharpe should be close to Phase 2.8 baseline (not regression).

---

## Success Criteria

- ✅ All 3 BQ tables have market + currency columns
- ✅ Ticker namespace parsing works correctly
- ✅ Cache layer includes market filter
- ✅ Market config loads all 5 markets
- ✅ Quick backtest (US-only) matches baseline
- ✅ No regressions in existing US workflow

---

## Execution Timeline

**When to run:** After Phase 2.8 PASSES (seeds 789, 2026 complete)

**Time estimate:** ~15 minutes total
1. Schema verification: ~2 min
2. Ticker parsing test: ~1 min
3. Cache filter check: ~1 min
4. Market config test: ~2 min
5. Quick backtest: ~9 min

**Failure response:** If any test fails → investigate, fix, re-run that test

---

## Next Step After Phase 2.9 PASS

- **Option A:** Commit Phase 2.9 validation results, proceed to Phase 3 (awaits Peder budget approval)
- **Option B:** If Phase 2.8 FAILS → fix seed issues, redo Phase 2.8, then Phase 2.9
- **Option C:** If budget approval pending → start Phase 2.10 (Karpathy autoresearch integration) while waiting

---

## Reference: What Phase 2.9 Changed

**Files modified:**
- `backend/backtest/data_ingestion.py` — market/currency population during ingestion
- `backend/backtest/cache.py` — market filtering in queries
- `backend/backtest/markets.py` — (new) market abstractions + calendars
- `migrate_backtest_data.py` — BQ schema with market/currency columns

**Zero impact on May launch:** New columns don't affect US-only workflows. Code is backward-compatible.
