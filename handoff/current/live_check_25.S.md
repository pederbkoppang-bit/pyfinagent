# Live-check placeholder -- phase-25.S

**Step:** 25.S -- Daily P&L attribution report per ticker
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "GET /api/paper-trading/attribution?window=7d returns per-ticker pnl_usd, llm_cost_usd, pnl_per_cost_usd"

## Pre-deployment evidence
- 10/10 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_S.py`).
- 4 behavioral round-trips: happy-path proportional split, zero-cost None ratio, empty-trades [] response, exact ratio computation.
- Backend AST clean for all 3 touched files.

## Post-deployment operator workflow
1. Restart backend so the new endpoint + cycle hook load:
   ```
   source .venv/bin/activate
   python -m uvicorn backend.main:app --reload --port 8000
   ```
2. Hit the endpoint:
   ```
   curl -s "http://localhost:8000/api/paper-trading/attribution?window_days=7" \
     -H "Authorization: Bearer $TOKEN" | jq .
   ```
3. Expected shape:
   ```json
   {
     "window_days": 7,
     "computed_at": "2026-05-13T...",
     "per_ticker": [
       {
         "ticker": "AAPL",
         "realized_pnl_usd": 245.67,
         "llm_cost_usd": 0.12,
         "pnl_per_cost_usd": 2047.25,
         "n_round_trips": 3,
         "n_analyses": 7
       },
       ...
     ],
     "totals": {
       "realized_pnl_usd": 800.0,
       "llm_cost_usd": 1.5,
       "pnl_per_cost_usd": 533.33
     },
     "note": "LLM cost split proportionally by analysis count per ticker (first pass; per-ticker tagging in llm_call_log is a follow-up step)."
   }
   ```
4. Cross-check: `totals.realized_pnl_usd` must equal `/api/sovereign/efficiency?window=7d` `realized_pnl_usd` (same numerator, same window).
5. Spot-check a single ticker's `pnl_per_cost_usd` -- a value of, e.g., 2000 means $2000 of realized P&L for every $1 of LLM cost on that ticker (good); a value below 100 suggests the LLM cost is large relative to the alpha captured.

## Approximation disclosure
The `note` field documents that LLM cost is split proportionally by analysis count per ticker -- a first-pass approximation. Per-call ticker tagging in `llm_call_log` (a column addition + writer updates) would give exact attribution; that's a follow-up step (25.S.1 / future audit anchor).

## Closes audit basis
phase-24.13 F-6 RESOLVED. Operators can now answer the SHARP-arxiv-required "which tickers earned the most relative to LLM cost?" question. Direct operationalization of red-line goal-c ("dynamically shift to whichever strategy is making the most money") at the per-ticker granularity.

**Audit anchor for next bucket:** 25.C9 (P1 Batch API for non-interactive pipeline steps; 50% savings) OR 25.S.1 (per-call ticker tagging follow-up for exact cost attribution).
