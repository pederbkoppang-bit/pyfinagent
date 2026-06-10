# Live-check placeholder -- phase-25.A11

**Step:** 25.A11 -- Wire /paper-trading/learnings backend endpoint
**Date:** 2026-05-12

## Live-check field (per masterplan)
> "GET /api/paper-trading/learnings?window_days=30 returns valid VirtualFundLearningsData"

## Pre-deployment evidence
- 10/10 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_A11.py`)
- Behavioral round-trip (claim 10) executes `_compute_learnings(MockBQ(), 30)` with a deliberately-missing audit JSONL and asserts the dict shape -- proves the endpoint returns the locked VirtualFundLearningsData shape end-to-end, not just grep-level matches.
- TS clean (`npx tsc --noEmit` exit 0 from frontend/)
- ESLint 0 errors, 0 warnings on touched files
- Existing VirtualFundLearnings.test.tsx 5/5 pass (type promotion is source-compatible)
- Backend AST clean for `paper_trading.py`, `bigquery_client.py`, `api_cache.py`

## Post-deployment operator workflow (capture-after-restart)
1. Restart backend so the new route is registered:
   ```
   source .venv/bin/activate && python -m uvicorn backend.main:app --reload --port 8000
   ```
2. Hit the endpoint:
   ```
   curl -s "http://localhost:8000/api/paper-trading/learnings?window_days=30" \
     -H "Authorization: Bearer $TOKEN" | jq .
   ```
3. Expected shape:
   ```json
   {
     "reconciliation_divergences": [...],
     "kill_switch_triggers": [{"reason": "uat-16.6-drill", "count": 1}, ...],
     "regime_buckets": [],
     "window_days": 30,
     "collected_at": "2026-05-12T..."
   }
   ```
4. Visit `/paper-trading/learnings` in the browser -- page should now show:
   - Top page header "Virtual-Fund Learnings"
   - Reconciliation Divergences table populated (or empty-state row "No divergences recorded yet." if window is dry)
   - Kill-Switch Trigger Distribution bars (from `handoff/kill_switch_audit.jsonl`)
   - Regime Underperformance section with "No regime buckets computed yet." (documented first-pass behavior; future step can add a per-trade regime column)

## Documented first-pass behavior
`regime_buckets` returns `[]` because `paper_trades` has no per-trade `macro_regime` column. The component's empty state already renders this gracefully. Closing the gap (joining by date to the macro_regime cache) is intentionally deferred to a follow-up step -- not silently broken.

**Audit anchor for next bucket:** 25.A6 (live-vs-backtest Sharpe reconciliation) or 25.C12 (cross-tab Sharpe KPI reconciliation), depending on operator priority.
