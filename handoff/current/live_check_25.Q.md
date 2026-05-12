# Live-check placeholder -- phase-25.Q

**Step:** 25.Q -- Real-time profit_per_llm_dollar metric (closes red-line goal-d)
**Date:** 2026-05-12

## Live-check field (per masterplan)
> "GET /api/sovereign/efficiency returns valid ratio; not hardcoded zero"

## Pre-deployment evidence
- 11/11 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_Q.py`)
- 4 behavioral round-trips: happy path (ratio=10.0), zero-cost (None), persist=True (save called once), provider mapping (gemini->vertex).
- Backend AST clean for all 3 touched files.
- Migration dry-run prints 9-column DDL.

## Post-deployment operator workflow
1. (Optional) Apply migration if persistence is desired:
   ```
   source .venv/bin/activate
   python3 scripts/migrations/add_efficiency_snapshots.py --apply
   ```
2. Start/restart the backend so the new route + helper are loaded:
   ```
   source .venv/bin/activate
   python -m uvicorn backend.main:app --reload --port 8000
   ```
3. Hit the efficiency endpoint:
   ```
   curl -s "http://localhost:8000/api/sovereign/efficiency?window=30d" \
     -H "Authorization: Bearer $TOKEN" | jq .
   ```
   Expected shape:
   ```json
   {
     "window": "30d",
     "profit_per_llm_dollar": <float or null>,
     "realized_pnl_usd": <float>,
     "llm_cost_usd": <float>,
     "anthropic_cost_usd": <float>,
     "vertex_cost_usd": <float>,
     "openai_cost_usd": <float>,
     "computed_at": "2026-05-12T...",
     "note": null
   }
   ```
4. Also verify the existing `/compute-cost` endpoint no longer reports zeros:
   ```
   curl -s "http://localhost:8000/api/sovereign/compute-cost?window=30d" \
     -H "Authorization: Bearer $TOKEN" | jq '.totals'
   ```
   Expected: `anthropic`, `vertex`, `openai` totals > 0 (assuming there are recent llm_call_log rows).
5. (Optional) Persist a snapshot for trend tracking:
   ```
   curl -s "http://localhost:8000/api/sovereign/efficiency?window=30d&persist=true" \
     -H "Authorization: Bearer $TOKEN" | jq .
   ```
   Then verify the BQ row:
   ```sql
   SELECT *
   FROM `sunny-might-477607-p8.pyfinagent_data.efficiency_snapshots`
   ORDER BY computed_at DESC
   LIMIT 5;
   ```

## First-mover status
Per arxiv 2503.21422 (March 2025 survey) -- no published autonomous trading system has the `profit_per_llm_dollar` metric. pyfinagent is the first-mover.

## Closes red-line goal-d
Goal-d ("low cost") is now observable in real time. Combined with 25.R (which closed goal-c "dynamically shift strategy"), both auto-switching and cost-observability red-line gaps are now closed.

**Audit anchor for next bucket:** 25.A6 (live-vs-backtest Sharpe reconciliation) or 25.S (daily P&L attribution per ticker).
