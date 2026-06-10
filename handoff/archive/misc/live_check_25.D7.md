# Live-check placeholder -- phase-25.D7

**Step:** 25.D7 -- preload_macro() max-age guard (35-day FRED-monthly default)
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Inject macro data with timestamp >35 days old; preload refuses with WARNING log"

## Pre-deployment evidence
- 4/4 verifier PASS.
- Claim 3 behavioral test: mocked BQ rows dated 40-45 days ago -> preload
  returns 0, captures 1 WARNING log record, `_macro_full` stays empty.
- Claim 4 behavioral test: fresh rows (5-10 days) -> cache populated +
  3 rows returned across 2 series.

## Post-deployment operator workflow
1. Pull main + restart backend:
   ```
   git pull origin main
   source .venv/bin/activate
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   ```
2. Simulate stale data via Python:
   ```
   python -c "
   from datetime import date, timedelta
   from unittest.mock import MagicMock
   import backend.backtest.cache as c
   today = date.today()
   stale = [{'series_id': 'CPIAUCSL', 'value': 295.5, 'date': today - timedelta(days=40)}]
   mq = MagicMock()
   mq.result.return_value = iter(stale)
   mc = MagicMock()
   mc.query.return_value = mq
   c._bq_client = mc
   c._project = 'test'
   c._macro_full.clear()
   n = c.preload_macro()
   print('return:', n, 'cache size:', len(c._macro_full))
   "
   ```
3. Expected stderr WARNING:
   `preload_macro: stale data, refusing to cache (max_date=2026-04-03 age=40 days threshold=35 days)`

## Closes audit basis
bucket 24.7 F-5 RESOLVED.

**Audit anchor for next bucket:** 25.F3, 25.B10.1 (lesser secrets), follow-ups (25.C9.1, 25.D9.1, 25.S.1).
