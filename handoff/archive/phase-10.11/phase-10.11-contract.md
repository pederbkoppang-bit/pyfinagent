# Sprint Contract — phase-10.11 (Integration: backend endpoint + dashboard wiring)

**Step id:** 10.11 **Date:** 2026-04-21 **Tier:** moderate **Harness-required:** true

## Why

Phase-10.9 shipped `HarnessSprintTile.tsx` as a pure presentational component. Phase-10.8 shipped the BQ sink. Phase-10.11 wires them together: add `GET /api/harness/sprint-state?week_iso=...` backend endpoint that reads the phase-10.8 `harness_learning_log` rows and projects them into the `HarnessSprintWeekState` shape; add a frontend fetcher + wire the tile into `HarnessDashboard.tsx`.

## Research-gate summary

Fresh researcher (moderate): `handoff/current/phase-10.11-research-brief.md` — 6 in full, 15 URLs, gate_passed=true.

Key grounding:
- Existing API pattern: `backend/api/backtest.py:25` uses `APIRouter(prefix="/api/backtest", tags=...)` + `get_api_cache()` + `BigQueryClient`. Mirror for `/api/harness`.
- Route registration: add one line to `backend/main.py:292` for the new router.
- ISO week: use `date.today().isocalendar()` named-tuple (not `strftime("%Y-W%V")` which has Dec/Jan boundary bugs).
- No-data weeks: return HTTP 200 with `null` body (not 404) — frontend `apiFetch` throws on 404.
- 60s TTL via `api_cache` singleton (consistent with other dashboard endpoints).
- BQ row shape from phase-10.8: `{week_iso, slot_id ∈ {thu_batch, fri_promotion, monthly_gate, rollback}, phase, result_json, logged_at}`. One row per slot per week; take latest per slot.
- Frontend: `useEffect + Promise.all` pattern (no SWR — project-wide idiom).

## Immutable success criteria (masterplan-verbatim)

Test: `python scripts/harness/phase10_integration_test.py`

1. `backend_endpoint_returns_harness_sprint_week_state_shape` — GET response body matches `HarnessSprintWeekState` interface (weekIso, thu, fri, monthly); camelCase keys; nullable fields
2. `endpoint_reads_from_harness_learning_log` — SQL string contains `harness_learning_log` and `week_iso` parameter binding
3. `dashboard_renders_tile_when_data_present` — `HarnessDashboard.tsx` imports `HarnessSprintTile` and passes populated `data` prop (grep-checkable in the source)
4. `dashboard_renders_empty_state_when_data_null` — `HarnessDashboard.tsx` handles null state; tile's own empty-state renders (inherited from phase-10.9's empty-state test; this is an integration-level assertion on the JSX)

## Plan

1. Create `backend/api/harness_autoresearch.py`:
   - `APIRouter(prefix="/api/harness", tags=["harness"])`
   - `GET /sprint-state?week_iso={optional}` → `HarnessSprintWeekState | None`
   - Default `week_iso`: `date.today().isocalendar()` → `f"{iso.year}-W{iso.week:02d}"`
   - Query BQ `pyfinagent_data.harness_learning_log` for latest row per `slot_id` for the requested `week_iso` with `phase = 'phase-10'`
   - Parse `result_json` per slot; project to camelCase `HarnessSprintWeekState` shape
   - 60s cache via `api_cache` with key `harness:sprint_state:{week_iso}`
   - Injectable `bq_query_fn` keyword (optional; only used by tests) for hermetic testing
   - Fail-open: BQ errors return `None` with warning log; no 5xx to frontend
2. Register router in `backend/main.py`: add `from backend.api.harness_autoresearch import router as harness_autoresearch_router`; `app.include_router(harness_autoresearch_router)` after line 291
3. Add Pydantic response models `HarnessSprintWeekState`, `HarnessSprintThu`, `HarnessSprintFri`, `HarnessSprintMonthly`
4. Frontend:
   - Add `getHarnessSprintState(weekIso?: string): Promise<HarnessSprintWeekState | null>` to `frontend/src/lib/api.ts`
   - Edit `frontend/src/components/HarnessDashboard.tsx` to import `HarnessSprintTile`, add state `sprintState`, fetch in existing pattern, render `<HarnessSprintTile data={sprintState} />` somewhere appropriate
5. Create `scripts/harness/phase10_integration_test.py`:
   - 4 cases matching success_criteria verbatim
   - Case 1: construct endpoint with stub `bq_query_fn` returning sample row set; call the handler; assert response shape
   - Case 2: inspect the SQL string directly (capture via stub); assert `harness_learning_log` + `week_iso` both present
   - Case 3: read `HarnessDashboard.tsx` source; assert it `import`s `HarnessSprintTile` AND renders it with a `data={` binding
   - Case 4: same source read; assert the null-path is wired (tile's own empty state handles it; grep for `sprintState` state var and the tile usage)
6. Add pytest `tests/api/test_harness_autoresearch.py` — ≥5 cases
7. Run: ast + immutable CLI + pytest + frontend `npm run test -- --filter=HarnessDashboard` (if a test exists; create a minimal one if needed)
8. Spawn fresh Q/A. Cycle-2 flow if gaps surfaced.
9. Log, flip, close task #95.

## References

- `handoff/current/phase-10.11-research-brief.md`
- `backend/api/backtest.py:25, 391, 480, 697` (APIRouter + api_cache idiom)
- `backend/main.py:280-291` (router registration)
- `backend/autoresearch/slot_accounting.py` (row schema source of truth)
- `backend/db/bigquery_client.py` (BQ query idiom)
- `frontend/src/lib/api.ts` (fetcher idiom)
- `frontend/src/components/HarnessDashboard.tsx` (integration target)
- `frontend/src/components/HarnessSprintTile.tsx` (phase-10.9 component to import)

## Carry-forwards (out of scope)

- Wire 10.3/10.4/10.6/10.7 routines to actually CALL `log_slot_usage` on their success paths (real-data pipeline; requires scheduler integration)
- Backend authentication on the new endpoint (inherits the existing auth middleware from `main.py`)
- Fix `HarnessDashboard.tsx` pre-existing icon-import violation (separate ticket)
