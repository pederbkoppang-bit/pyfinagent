# Live-check placeholder -- phase-25.C12

**Step:** 25.C12 -- Cross-tab Sharpe KPI reconciliation (backend authoritative)
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Home and /paper-trading show identical Sharpe ratio for same window"

## Pre-deployment evidence
- 11/11 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_C12.py`)
- 3 behavioral round-trips: happy-path (60 noisy snapshots -> finite Sharpe), no-data (empty snapshots -> None/0.0 graceful), fail-open (snapshot fetch raises -> sharpe_ratio=None + rest of response intact).
- Backend AST clean; frontend TS clean (`npx tsc --noEmit`).
- Researcher's formula-divergence finding confirmed: frontend kpiSharpe skips the RFR subtraction that backend Sharpe applies (~0.16 unit gap at 4% RFR).

## Post-deployment operator workflow
1. Restart backend + frontend dev server:
   ```
   source .venv/bin/activate
   python -m uvicorn backend.main:app --reload --port 8000
   # (in another shell)
   cd frontend && npm run dev
   ```
2. Compare values via curl:
   ```
   curl -s "http://localhost:8000/api/paper-trading/portfolio" \
     -H "Authorization: Bearer $TOKEN" | jq '.portfolio.sharpe_ratio'

   curl -s "http://localhost:8000/api/paper-trading/performance" \
     -H "Authorization: Bearer $TOKEN" | jq '.sharpe_ratio'
   ```
   Both must yield the SAME number (modulo identical snapshot freshness).
3. Visit `/` (home) and `/paper-trading` in the browser. The "Sharpe (90d)" KPI tile on the home page and the Sharpe display on `/paper-trading` must show the same value (rounded to 2 decimals).
4. (Optional negative test) Stop the backend; reload the home page. The KPI should still render via the local `kpiSharpe(navSeries)` fallback -- value will be slightly different (no RFR subtraction) but the page does not crash.

## Closes audit basis
phase-24.12 F-4 RESOLVED. Cross-tab Sharpe divergence eliminated by construction:
- Both pages now consume the same backend-authoritative value (computed by the same `compute_sharpe_from_snapshots` helper on the same BQ snapshot series).
- `kpiMetrics.ts::sharpe` retained for backwards-compat + rolling-deploy fallback only; JSDoc `@deprecated` warns new consumers.

**Audit anchor for next bucket:** 25.A12 (Playwright visual regression CI baseline) OR 25.B9 (system prompt cache threshold) OR 25.B (P2 cosmetic-patch cleanup).
