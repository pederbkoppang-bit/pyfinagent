# Live-check placeholder -- phase-25.A7

**Step:** 25.A7 -- Per-table freshness endpoint covering all 5 data tables
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "GET /api/observability/freshness response includes historical_prices + historical_fundamentals + historical_macro + signals_log + paper_portfolio_snapshots with SLA bands"

## Pre-deployment evidence
- 11/11 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_A7.py`)
- 5 behavioral round-trips: worst-band priority, 6-table coverage, happy-green/no-alert, red-band/P1-alert, fail-open on Slack failure.
- Backend AST clean for `cycle_health.py`.
- grep confirms no frontend/backend consumer reads the old `paper_snapshots` key (key renamed to `paper_portfolio_snapshots` to match BQ table name).

## Post-deployment operator workflow
1. Restart backend so the new helper + extended `compute_freshness` are loaded:
   ```
   source .venv/bin/activate && python -m uvicorn backend.main:app --reload --port 8000
   ```
2. Hit the canonical or alias endpoint:
   ```
   curl -s "http://localhost:8000/api/observability/freshness" \
     -H "Authorization: Bearer $TOKEN" | jq '.sources | keys, .overall_band'
   ```
3. Expected `sources` keys (in any order):
   - `paper_trades`
   - `paper_portfolio_snapshots`
   - `historical_prices`
   - `historical_fundamentals`
   - `historical_macro`
   - `signals_log`
4. Expected each entry shape:
   ```json
   {
     "last_tick_age_sec": <float or null>,
     "interval_sec": <float>,
     "ratio": <float or null>,
     "band": "green" | "amber" | "red" | "unknown"
   }
   ```
5. Verify the top-level `overall_band` is set to the worst of the 6 source bands.
6. (Optional negative test) Force a critical band by sleeping a table source long enough to exceed 2x its `expected_max_age_sec`; verify a P1 Slack alert lands in the configured channel. The `AlertDeduper` ensures repeats within the dedup window don't spam.

## SLA intervals (research-backed)
- `historical_prices`: 26h (nightly + T+1 US market)
- `historical_fundamentals`: 95 days (quarterly + filing lag)
- `historical_macro`: 35 days (monthly FRED release + lag)
- `paper_portfolio_snapshots`: 26h (daily snapshot)
- `signals_log` + `paper_trades`: per-cycle (caller's `cycle_interval_sec`)

WARN_RATIO = 1.5x interval -> amber. CRITICAL_RATIO = 2.0x -> red.

## Closes audit basis
phase-24.7 F-1 RESOLVED. The 4 historical/log tables are now monitored alongside the 2 paper tables. Operator visibility into ingestion lag is uniform across the data plane.

**Audit anchor for next bucket:** 25.D6 (P1 planner plateau lock-file) or 25.B (P2 cosmetic-patch cleanup).
