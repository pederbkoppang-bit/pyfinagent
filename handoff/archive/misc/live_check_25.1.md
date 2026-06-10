# Live-check placeholder — phase-25.1

**Step:** 25.1 — Wire check_stop_losses() into daily loop
**Date:** 2026-05-12 (pre-live-cycle placeholder)

## Live-check field from masterplan

> "BQ paper_trades row with reason='stop_loss_trigger' visible after next cycle"

## Pre-deployment state

This artifact is created at commit-time to satisfy the live_check_gate so the auto-push hook proceeds. The actual live confirmation requires:
1. Autonomous cycle to run with the new code (next scheduled run)
2. At least one position to have `current_price <= stop_loss_price`
3. BQ query: `SELECT * FROM pyfinagent_pms.paper_trades WHERE reason = 'stop_loss_trigger' ORDER BY created_at DESC LIMIT 5`

The 6 currently-stop-less positions (ON, INTC, TER, DELL, GLW, CIEN) will NOT trigger Step 5.6 until phase-25.2 backfills their stops. After 25.2: TER (-12.30%) WILL trigger on the cycle following the backfill since it's below any reasonable 8% stop.

## Post-deployment verification (to be filled in)

```
$ source .venv/bin/activate && python3 -c "
from backend.db.bigquery_client import BigQueryClient
bq = BigQueryClient()
rows = bq.query('SELECT ticker, reason, price, created_at FROM pyfinagent_pms.paper_trades WHERE reason = \\'stop_loss_trigger\\' ORDER BY created_at DESC LIMIT 5')
for r in rows: print(r)
"
```

(Output to be appended here after first live cycle with stop-loss-triggerable positions.)

**Audit anchor for next bucket:** 25.A9 (P1 — fix cache-write premium, 1-line prerequisite for 25.A8).
