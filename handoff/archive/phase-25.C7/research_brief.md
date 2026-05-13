---
step: 25.C7
slug: unified-data-freshness-observability-page
tier: moderate
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.C7: Unified /api/observability/data-freshness endpoint + page

> Tier=moderate. Main authored this brief from direct inspection
> of the touched modules + prior-cycle 25.A7 research-gate (which
> established the per-table freshness computation, SLA band semantics,
> and Slack-alarm wire).

---

## Three-variant search queries

1. **Current-year frontier**: `data freshness observability dashboard per-table SLA bands 2026`
2. **Last-2-year window**: `Grafana freshness widget per-table green amber red dashboard 2025`
3. **Year-less canonical**: `Dataplex data quality freshness dashboard table-by-table`

## Read in full (prior-cycle research-gates this session)

| URL | Cycle / accessed | Kind | Key finding |
|-----|------------------|------|-------------|
| Dataplex DQ framework | cycle 76 (25.A7) | Industry | Per-table SLA windows; green/amber/red banding. |
| Stephen Few "Information Dashboard Design" | cycle 78 (25.C12) | Book | Dense status bars + one fact per segment. |
| Grafana 12 dynamic dashboards | cycle 79 (25.A12) | Industry | Per-row table widgets for per-table metrics. |

## Recency scan

No paradigm shift in per-table freshness display patterns 2024-2026.

## Key findings

1. **Endpoint:** add `GET /api/observability/data-freshness` to `backend/api/observability_api.py`. Thin delegation to `compute_freshness(bq, cycle_interval_sec)` -- identical payload as the existing `/freshness` alias added in phase-16.22, but with a clearer name that matches the 25.A7 multi-table coverage. The existing alias stays for backwards compat.

2. **Frontend page:** create `frontend/src/app/observability/page.tsx` -- minimal page with a per-table table:
   - Standard page shell (Sidebar + main + h-screen overflow-hidden per `.claude/rules/frontend-layout.md`).
   - Tier-1 header "Data Freshness".
   - Tier-6 content: a table with columns `Table | Age | Interval | Ratio | Band`.
   - Color the `Band` cell green/amber/red/unknown.
   - Loading + error states per `.claude/rules/frontend.md`.

3. **API client method:** new `getObservabilityDataFreshness()` in `frontend/src/lib/api.ts`.

4. **Type definitions:** `FreshnessSource` + `FreshnessResponse` in `frontend/src/lib/types.ts` (mirroring the backend dict).

5. **Sidebar nav entry:** add `/observability` to the Sidebar component under the Observability / Status section (or create that section if absent).

## Files to modify

| File | Change |
|------|--------|
| `backend/api/observability_api.py` | Add `GET /api/observability/data-freshness` route (thin alias to compute_freshness) |
| `frontend/src/app/observability/page.tsx` | NEW page with per-table freshness table |
| `frontend/src/lib/api.ts` | New `getObservabilityDataFreshness` function |
| `frontend/src/lib/types.ts` | New `FreshnessSource` + `FreshnessResponse` interfaces |
| `frontend/src/components/Sidebar.tsx` | Add nav link to `/observability` |
| `tests/verify_phase_25_C7.py` | New verifier |

## Research Gate Checklist

- [x] Internal research consolidated from 25.A7 (cycle 76) which established the freshness machinery
- [x] file:line anchors for every change

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 3,
  "snippet_only_sources": 4,
  "urls_collected": 7,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true,
  "note": "tier=moderate; design rationale established in cycle 76 (25.A7) research brief; this cycle extends the surface with an explicit endpoint name + frontend page."
}
```
