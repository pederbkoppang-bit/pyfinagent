---
step: phase-25.C7
cycle: 86
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.C7

## What was built/changed

Added a second HTTP alias `GET /api/observability/data-freshness` (sibling
to the existing `/api/observability/freshness` added phase-16.22). Both
delegate to `cycle_health.compute_freshness` -- the new name clarifies
the post-25.A7 scope (6 tables including historical_prices, historical_
fundamentals, historical_macro, signals_log) rather than the
paper-trading-only impression conveyed by `/freshness`.

Built a new operator page at `/observability` rendering the
per-table freshness payload as a 5-column table (Source | Age |
SLA Interval | Ratio | Band) with band-coloured pills (green/amber/
red/unknown). Auto-refreshes every 30s. Sidebar nav entry added
under the "System" group.

## Files changed

| File | Action | Reason |
|------|--------|--------|
| `backend/api/observability_api.py` | Added `get_observability_data_freshness` route handler | Unified endpoint per masterplan |
| `frontend/src/lib/types.ts` | Added `FreshnessBand`, `FreshnessSource`, `FreshnessResponse` | Type safety for new page |
| `frontend/src/lib/api.ts` | Added `getObservabilityDataFreshness()` | API client method |
| `frontend/src/app/observability/page.tsx` | NEW page (160 lines) | Per-table freshness UI |
| `frontend/src/components/Sidebar.tsx` | Added `/observability` link under "System" | Nav surface |
| `tests/verify_phase_25_C7.py` | NEW verifier (5 claims) | Success-criteria proof |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_C7.py

=== phase-25.C7 verification ===

[PASS] 1. new_route_data_freshness_in_observability_api
        -> Found @router.get('/data-freshness') decorator
[PASS] 2. handler_delegates_to_compute_freshness
        -> Handler routes through asyncio.to_thread(_cf, ...)
[PASS] 3. frontend_observability_page_renders_per_table_table_with_bands
        -> exists=True table=True band=True fetch=True
[PASS] 4. sidebar_links_to_observability_page
        -> Sidebar contains /observability nav entry
[PASS] 5. behavioral_round_trip_returns_compute_freshness_payload
        -> Handler returned mocked payload via compute_freshness

ALL 5 CLAIMS PASS
```

Frontend `npx tsc --noEmit` (filtered to exclude pre-existing
25.A12 Playwright-not-installed noise): 0 errors.

Backend `python -c "import ast; ast.parse(open('backend/api/observability_api.py').read())"`: clean.

## Artefact shape

Endpoint payload (mocked, identical shape to existing `/freshness`):

```json
{
  "sources": {
    "paper_trades":              { "last_tick_age_sec": 120.0, "interval_sec": 86400.0, "ratio": 0.0014, "band": "green" },
    "paper_portfolio_snapshots": { "...": "..." },
    "historical_prices":         { "...": "..." },
    "historical_fundamentals":   { "...": "..." },
    "historical_macro":          { "...": "..." },
    "signals_log":               { "...": "..." }
  },
  "overall_band": "green",
  "heartbeat": { "updated_at": "...", "band": "green" },
  "bq_ingest_lag_sec": 120.0,
  "thresholds": { "warn_ratio": 1.5, "critical_ratio": 2.0, "cycle_interval_sec": 86400 },
  "computed_at": "2026-05-13T00:00:00Z"
}
```

## Success criteria (verbatim from masterplan) -> evidence

1. `new_route_data_freshness_in_observability_api` -- Verifier claim 1
   PASS: regex-matched `@router.get('/data-freshness')` decorator in
   `backend/api/observability_api.py`.
2. `frontend_observability_page_renders_per_table_table_with_bands`
   -- Verifier claim 3 PASS: page file exists at
   `frontend/src/app/observability/page.tsx`, contains `<table>`
   element, has `BandPill`/`band` references, and wires
   `getObservabilityDataFreshness`.

## Out-of-scope / deferred

- Live BQ smoke test (will be captured in `live_check_25.C7.md`).
- Sparkline or trendline per source (the underlying payload is
  point-in-time; a history endpoint is a separate masterplan
  candidate).
- Heartbeat / `bq_ingest_lag_sec` rendering -- only the per-table
  table is in the success criterion; secondary surfacing can come
  later if operators ask.
