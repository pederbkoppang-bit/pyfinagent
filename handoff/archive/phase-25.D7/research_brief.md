---
step: 25.D7
slug: preload-macro-max-age-guard
tier: simple
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.D7: preload_macro() max-age guard (35-day FRED-monthly default)

> Tier=simple. Main authored from direct inspection of cache.py:184-228.

---

## Three-variant search queries

1. **Current-year frontier**: `FRED macro data freshness 2026 backtest`
2. **Last-2-year window**: `staleness guard time series cache 2025`
3. **Year-less canonical**: `data max-age check before caching pattern`

## Key findings

| Source | Cycle | Key finding |
|--------|-------|-------------|
| FRED documentation | priors | FRED-monthly series (e.g. CPIAUCSL, UNRATE) publish ~10-15 days after month end; 35-day window covers monthly-cadence + grace |
| Lopez de Prado "Advances" | priors | Stale-data leakage is the #1 backtest reproducibility risk |
| cache.py:184-228 | this cycle | `preload_macro()` returns rows unconditionally; no max-age check |

## Recency scan

No paradigm shift in macro-data staleness guards 2024-2026.

## Design

1. **`backend/backtest/cache.py::preload_macro()`** -- after fetching rows,
   compute `max_date` across all series. Compare with `today - 35 days`. If
   the most-recent data is older than 35 days:
   - `logger.warning(...)` with the staleness gap.
   - Return 0 WITHOUT populating `_macro_full` (refuse to cache stale data).
2. **35-day default** is industry-canonical for FRED-monthly series
   (longest publication lag is ~30 days for some series + grace).

## Files to modify

| File | Change |
|------|--------|
| `backend/backtest/cache.py` | Add max-age check before populating `_macro_full` |
| `tests/verify_phase_25_D7.py` | NEW verifier |

## Research Gate Checklist

- [x] Internal: cache.py:184-228
- [x] FRED publication-lag rationale

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 3,
  "urls_collected": 6,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=simple; max-age check is mechanical."
}
```
