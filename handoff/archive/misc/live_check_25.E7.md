# Live-check placeholder -- phase-25.E7

**Step:** 25.E7 -- yfinance_tool.get_price_history() try/except + counter
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Inject yfinance rate-limit; verify error returned not propagated"

## Pre-deployment evidence
- 5/5 verifier PASS.
- Claim 3 behavioral round-trip: `yf.Ticker` patched to raise
  `RuntimeError("rate_limited")`; result is `[{"error": "rate_limited",
  "ticker": "AAPL"}]` + persist invoked.
- Claim 5 behavioral round-trip: empty DataFrame -> `[{"error": "no_data",
  "ticker": "NVDA"}]` + persist invoked.

## Post-deployment operator workflow
1. Pull main + restart backend:
   ```
   git pull origin main
   source .venv/bin/activate
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   ```
2. Simulate a rate-limit failure with monkeypatch:
   ```
   python -c "
   import yfinance as yf
   from unittest.mock import patch
   from backend.tools.yfinance_tool import get_price_history
   with patch('backend.tools.yfinance_tool.yf') as m:
       m.Ticker.side_effect = RuntimeError('429 Too Many Requests')
       out = get_price_history('AAPL')
       print('result:', out)
       assert out[0]['error']
       assert out[0]['ticker'] == 'AAPL'
       print('OK')
   "
   ```
3. Verify a row appeared in BQ:
   ```sql
   SELECT * FROM `sunny-might-477607-p8.pyfinagent_data.data_source_events`
   WHERE source = 'yfinance_price_history'
   ORDER BY event_time DESC LIMIT 5;
   ```

## Closes audit basis
bucket 24.7 F-4 RESOLVED. The previously-unguarded yfinance call now
fails gracefully + emits structured error + persists for aggregation.

**Audit anchor for next bucket:** 25.D7, 25.F3, follow-ups.
