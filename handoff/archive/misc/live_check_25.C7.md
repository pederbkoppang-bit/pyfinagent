# Live-check placeholder -- phase-25.C7

**Step:** 25.C7 -- Unified /api/observability/data-freshness endpoint + page
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "GET /api/observability/data-freshness returns per-table ages with SLA bands"

## Pre-deployment evidence
- 5/5 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_C7.py`).
- Behavioral round-trip in claim 5 invokes `get_observability_data_freshness`
  with `compute_freshness` mocked and confirms the handler returns the mocked
  payload (sources dict + overall_band + computed_at).
- Backend AST clean; frontend `tsc --noEmit` clean (excluding pre-existing
  25.A12 Playwright noise).
- Sidebar nav entry "Data Freshness" registered under the System group.

## Post-deployment operator workflow
1. Pull main, restart backend, rebuild frontend:
   ```
   git pull origin main
   source .venv/bin/activate
   python -m uvicorn backend.main:app --reload --port 8000 &
   cd frontend && npm run dev &
   ```
2. Hit the new endpoint:
   ```
   curl -s http://localhost:8000/api/observability/data-freshness | jq '.sources | keys'
   # Expect 6 sources: paper_trades, paper_portfolio_snapshots,
   # historical_prices, historical_fundamentals, historical_macro, signals_log.
   ```
3. Confirm overall_band reflects the worst-band aggregation:
   ```
   curl -s http://localhost:8000/api/observability/data-freshness | jq '.overall_band'
   ```
4. Visit http://localhost:3000/observability -- 6-row table with band pills.

## Closes audit basis
bucket 24.7 F-1 + F-7 RESOLVED. Observability dashboard now has a single
URL that surfaces per-table freshness without operators having to
remember the legacy /freshness alias name.

**Audit anchor for next bucket:** 25.B6, 25.B7, 25.D7, 25.E7 (P2 backlog).
