# Live-check placeholder -- phase-25.A6

**Step:** 25.A6 -- Explicit live-vs-backtest Sharpe reconciliation
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Reconciliation report shows explicit live_sharpe - backtest_sharpe gap on next cycle"

## Pre-deployment evidence
- 11/11 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_A6.py`)
- 6 behavioral round-trips covering all four fallback tiers (optimizer_best / shadow_curve / proxy_fallback / no_data) + compute_gate integration.
- Backend AST clean for both touched files.
- SR_GAP_THRESHOLD = 0.30 mirrored in both `perf_metrics.py` and `paper_go_live_gate.py` (industry benchmark per Jacquier et al. arxiv 2501.03938).

## Post-deployment operator workflow
1. Restart backend so the new helper + wiring are loaded:
   ```
   source .venv/bin/activate && python -m uvicorn backend.main:app --reload --port 8000
   ```
2. Trigger the paper-trading gate endpoint:
   ```
   curl -s "http://localhost:8000/api/paper-trading/gate" \
     -H "Authorization: Bearer $TOKEN" | jq .
   ```
3. Expected `details` shape now includes explicit Sharpe diagnostics:
   ```json
   {
     ...
     "details": {
       ...,
       "live_sharpe": <float or null>,
       "backtest_sharpe": <float or null>,
       "sharpe_gap_rel": <float or null>,
       "sharpe_gap_source": "optimizer_best" | "shadow_curve" | "proxy_fallback" | "no_data",
       "sharpe_gap_proxy_fallback": <bool>,
       "sharpe_gap_note": <string or null>,
       "latest_reconciliation_divergence_pct": <float>,   # legacy sibling preserved
       ...
     }
   }
   ```
4. `booleans.sr_gap_le_30pct` now reflects the EXPLICIT Sharpe gap (or stays red when both backtest sources are unavailable).

## Fallback chain (behaviorally verified)
1. `optimizer_best.json["sharpe"]` -> `source="optimizer_best"`.
2. Reconciliation shadow NAV curve (>=6 points) -> `source="shadow_curve"`.
3. Reconciliation divergence proxy -> `source="proxy_fallback"`, `proxy_fallback=True` (preserves legacy behavior; visible to operators).
4. Nothing usable -> `source="no_data"`, `gap_within_threshold=None`, gate stays red.

## Closes audit basis
phase-24.6 F-3: "paper_go_live_gate.py:91-94 uses NAV-divergence proxy honestly disclosed; no explicit Sharpe comparison" -> RESOLVED. The gate now performs the canonical Sharpe-gap measurement; the NAV divergence remains as a sibling diagnostic.

**Audit anchor for next bucket:** 25.A7 (per-table freshness endpoint) OR 25.B (P2 cosmetic-patch removal).
