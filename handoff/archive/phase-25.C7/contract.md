---
step: 25.C7
slug: unified-data-freshness-observability-page
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.C7

## Step ID + masterplan reference

`25.C7` -- "Unified /api/observability/data-freshness endpoint"
(P2, harness_required, depends on `25.A7` done).

## Research-gate summary

Tier=moderate. Main authored brief from 25.A7 prior research-gate
(cycle 76 established freshness computation + SLA bands +
Slack-alarm wire). External canonical: Dataplex DQ + Grafana 12 +
Stephen Few. JSON envelope: `external_sources_read_in_full=3`,
`gate_passed=true`. See `handoff/current/research_brief.md`.

## Hypothesis

The 25.A7 cycle shipped a multi-table `compute_freshness` function
but the only HTTP surface for that data is the legacy alias
`GET /api/observability/freshness` (added phase-16.22) -- a name
that implied paper-trading-only scope. By exposing the same payload
under the clearer `/api/observability/data-freshness` AND adding a
dedicated `/observability` page with a per-table SLA-banded table,
operators get a single hop to "is my data fresh across all 6
tables?" instead of grepping logs or BQ.

The change is additive: existing `/freshness` alias stays for
backwards compat; new endpoint is a thin, identical delegation.

## Success criteria (verbatim from masterplan.json)

> `new_route_data_freshness_in_observability_api`
>
> `frontend_observability_page_renders_per_table_table_with_bands`

## Plan steps

1. **Backend endpoint** -- add `GET /api/observability/data-freshness`
   route handler in `backend/api/observability_api.py` that delegates
   to `compute_freshness(bq, cycle_interval_sec)`, returning the
   identical dict shape as the existing `/freshness` alias.

2. **Frontend types** -- add `FreshnessSource` + `FreshnessResponse`
   interfaces to `frontend/src/lib/types.ts`.

3. **Frontend API client** -- add `getObservabilityDataFreshness()`
   to `frontend/src/lib/api.ts`.

4. **Frontend page** -- create `frontend/src/app/observability/page.tsx`
   with the standard page shell (Sidebar + main h-screen overflow),
   tier-1 header "Data Freshness", and a 5-column table
   `Source | Age (s) | Interval (s) | Ratio | Band` colored per band.
   Loading + error + empty states per `.claude/rules/frontend.md`.

5. **Sidebar nav** -- add a `/observability` link in
   `frontend/src/components/Sidebar.tsx`.

6. **Verifier** -- create `tests/verify_phase_25_C7.py` with 4 claims:
   - Backend route definition exists with correct path.
   - Backend handler delegates to `compute_freshness`.
   - Frontend page file exists with `freshness` table render.
   - Sidebar contains `/observability` link.

## Files

| File | Action |
|------|--------|
| `backend/api/observability_api.py` | Add route handler |
| `frontend/src/lib/types.ts` | New interfaces |
| `frontend/src/lib/api.ts` | New API client method |
| `frontend/src/app/observability/page.tsx` | NEW page |
| `frontend/src/components/Sidebar.tsx` | Add nav link |
| `tests/verify_phase_25_C7.py` | NEW verifier |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_C7.py
```

## Live-check

`GET /api/observability/data-freshness returns per-table ages with SLA bands`.
Will write `handoff/current/live_check_25.C7.md` summarising the
endpoint output shape + page render proof.

## Risks + mitigations

- **Risk**: Sidebar layout collision -- adding a new nav item may
  push other items.
  **Mitigation**: Append at the bottom of the existing Observability
  / Status section (or create one if none exists).
- **Risk**: `compute_freshness` signature drift between 25.A7 and now.
  **Mitigation**: Verifier asserts handler delegates by AST inspection,
  not just literal call-site.
- **Risk**: Frontend TS noise from pre-existing 25.A12 Playwright
  spec files.
  **Mitigation**: Continue grepping out `tests/visual-regression/`
  paths in tsc reports.

## References

- `handoff/current/research_brief.md` (this cycle)
- `handoff/archive/phase-25.A7/contract.md` (parent)
- `.claude/rules/frontend.md` + `.claude/rules/frontend-layout.md`
- `backend/services/cycle_health.py::compute_freshness`
- `.claude/masterplan.json::25.C7`
