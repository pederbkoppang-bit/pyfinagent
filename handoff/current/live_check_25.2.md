# Live-check placeholder — phase-25.2

**Step:** 25.2 — Backfill missing stops with same-cycle re-check
**Date:** 2026-05-12

## Live-check field
> "BQ paper_positions WHERE status='OPEN' AND stop_loss_price IS NULL returns 0 rows"

## Pre-deployment evidence
- 10/10 verifier PASS including behavioral round-trip (100 × 0.92 = 92.0 confirmed)
- Idempotent: subsequent calls skip already-stop-set positions
- Persistence path: `bq.save_paper_position` for each backfilled row

## Post-deployment operator workflow
1. `source .venv/bin/activate`
2. `python scripts/maintenance/backfill_stops.py` — review proposed stops, confirm `y`
3. Wait for next autonomous cycle OR trigger via `python -c "import asyncio; from backend.services.autonomous_loop import run_daily_cycle; print(asyncio.run(run_daily_cycle()))"`
4. Verify BQ: `SELECT * FROM pyfinagent_pms.paper_positions WHERE status='OPEN' AND stop_loss_price IS NULL` → empty
5. Verify TER sold: `SELECT * FROM pyfinagent_pms.paper_trades WHERE ticker='TER' AND reason='stop_loss_trigger' ORDER BY created_at DESC LIMIT 1`

**Audit anchor for next bucket:** 25.6 (no-stop-on-entry hard block in execute_buy).
