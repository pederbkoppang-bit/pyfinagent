# Phase 2.9: Multi-Market Abstractions (Lightweight)

## Goal
Prepare data layer for future market expansion (CA, EU, NO, KR) without building actual multi-market support. Zero risk to May US launch. ~4 hours work.

## Current State
- All data is US market only (implicit)
- Tables: historical_prices, historical_fundamentals, historical_macro, paper_portfolio, paper_positions
- Tickers are plain (e.g., "AAPL", not "US:AAPL")
- Currency is always USD (implicit)

## Changes Required

### 1. Schema Additions (Safe — backward compatible)

#### historical_prices
```sql
ALTER TABLE financial_reports.historical_prices
ADD COLUMN IF NOT EXISTS market STRING DEFAULT 'US',
ADD COLUMN IF NOT EXISTS currency STRING DEFAULT 'USD';
```

#### historical_fundamentals
```sql
ALTER TABLE financial_reports.historical_fundamentals
ADD COLUMN IF NOT EXISTS market STRING DEFAULT 'US',
ADD COLUMN IF NOT EXISTS currency STRING DEFAULT 'USD';
```

#### historical_macro
```sql
ALTER TABLE financial_reports.historical_macro
ADD COLUMN IF NOT EXISTS market STRING DEFAULT 'US',
ADD COLUMN IF NOT EXISTS base_currency STRING DEFAULT 'USD';
```

#### paper_portfolio
```sql
ALTER TABLE financial_reports.paper_portfolio
ADD COLUMN IF NOT EXISTS market STRING DEFAULT 'US',
ADD COLUMN IF NOT EXISTS base_currency STRING DEFAULT 'USD';
```

#### paper_positions
```sql
ALTER TABLE financial_reports.paper_positions
ADD COLUMN IF NOT EXISTS market STRING DEFAULT 'US',
ADD COLUMN IF NOT EXISTS base_currency STRING DEFAULT 'USD';
```

### 2. Code Changes

#### Backend: markets.py
Create `backend/backtest/markets.py`:
```python
from enum import Enum

class Market(Enum):
    US = ("US", "USD", "NYSE")  # ticker, currency, exchange
    CA = ("CA", "CAD", "TSX")
    EU = ("EU", "EUR", "XETRA")
    NO = ("NO", "NOK", "OSE")
    KR = ("KR", "KRW", "KRX")

DEFAULT_MARKET = Market.US
```

#### Backend: BQ queries
- Add `market` filter to all `WHERE` clauses (default `'US'`)
- Add `currency` to return columns where relevant
- Example: `WHERE market = @market` with param

#### Backend: Backtester
- Add `market: str = "US"` param to BacktestEngine.__init__
- Pass to candidate_selector and cache

#### Frontend: Settings
- Add market selector (dropdown: US, CA, EU, NO, KR)
- Store in portfolio settings
- Default: US

### 3. Tests
- Add unit test for Market enum
- Add BQ query test: filter by market returns correct rows
- Add integration test: backtest with market='US' matches current behavior

### 4. Documentation
- Create MARKETS.md with:
  - Market codes and currencies
  - Exchange calendars (OSE, KRX, TSX, XETRA)
  - Timezone mapping
  - Historical data availability per market
- Add to Phase 5 expansion checklist

## Timeline
- Schema migration: 30 min
- Code changes: 2 hours
- Tests: 1 hour
- Docs: 30 min
- Review + buffer: 30 min
**Total: ~4.5 hours (can pause for Phase 2.8 completion)**

## Risk Assessment
- **Zero risk to May US launch** — default to 'US', existing data unaffected
- **Backward compatible** — columns are optional, default to US/USD
- **Non-blocking** — no code depends on these columns yet
- **Reversible** — can drop columns if needed (unlikely)

## Success Criteria
- [ ] All 5 tables have market + currency columns
- [ ] BacktestEngine accepts market param
- [ ] Unit tests pass
- [ ] Integration test: backtest('US') == current behavior
- [ ] MARKETS.md complete
- [ ] Code changes committed and pushed

## Start
Ready to begin: 2026-03-29 10:26 UTC
