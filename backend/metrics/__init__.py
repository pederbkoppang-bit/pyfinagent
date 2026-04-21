"""phase-10.5 canonical risk-adjusted metrics.

Separate from `backend/services/perf_metrics.py` because:
- masterplan phase-10.5 mandates `backend/metrics/sortino.py`
- `perf_metrics.compute_sortino` uses a divergent formula (std on negatives,
  not LPM_2) and has existing callers that can't churn right now
- `backend-services.md`'s "single metric source" rule targets Sharpe /
  drawdown / alpha specifically; Sortino is not enumerated
"""
