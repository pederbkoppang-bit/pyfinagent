# Experiment Results — phase-10.11 (Integration: backend endpoint + dashboard wiring)

**Step:** 10.11 (final phase-10 integration) **Date:** 2026-04-21

## What was done

1. Fresh researcher (moderate): 6 in full, 15 URLs, gate_passed=true. Brief at `handoff/current/phase-10.11-research-brief.md`. Found existing `APIRouter` idiom at `backend/api/backtest.py:25`; ISO-week safety pattern; project-wide `useEffect + Promise.all` (no SWR); null-body + HTTP 200 convention.
2. Contract authored at `handoff/current/phase-10.11-contract.md`.
3. Created `backend/api/harness_autoresearch.py` (~165 lines):
   - `APIRouter(prefix="/api/harness", tags=["harness"])`
   - Pydantic response models: `HarnessSprintWeekState`, `HarnessSprintThu`, `HarnessSprintFri`, `HarnessSprintMonthly` (camelCase wire shape)
   - Pure `fetch_sprint_state(*, week_iso, bq_query_fn, table) -> HarnessSprintWeekState | None` — injectable BQ function for hermetic tests
   - `_build_sql()` module-level helper so tests can inspect the SQL string
   - `_default_bq_query()` runs parameterized BQ with `@week_iso` ScalarQueryParameter; fail-open to `[]`
   - ISO week default via `date.today().isocalendar()` (avoids `strftime("%V")` Dec/Jan pitfall)
   - Latest-row-per-slot semantics (SQL orders DESC; dict insertion order wins)
   - Fail-open on malformed `result_json`, BQ errors, etc.
4. Registered router in `backend/main.py:293-294` (one import + `include_router` call).
5. Added `getHarnessSprintState(weekIso?)` fetcher to `frontend/src/lib/api.ts` (after `getHarnessLog`).
6. Wired `frontend/src/components/HarnessDashboard.tsx`:
   - Imported `HarnessSprintTile` + `HarnessSprintWeekState`
   - Added `sprintState` state var
   - Added `getHarnessSprintState()` to the existing `Promise.all` fetch group with `.catch(() => null)` fail-open
   - Rendered `<HarnessSprintTile data={sprintState} />` at top of the dashboard's scrollable content (before Current Contract)
7. Created `scripts/harness/phase10_integration_test.py` — 4 cases matching masterplan verbatim; cases 3-4 inspect `HarnessDashboard.tsx` source via regex (integration-level assertion without firing Vitest runtime).
8. Created `tests/api/test_harness_autoresearch.py` — 7 pytest cases covering: all-three-slots projection, empty-returns-None, partial data, SQL-inspection, ISO-week format, latest-row-wins, malformed-JSON fail-open.

## Verification (verbatim)

```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/api/harness_autoresearch.py','scripts/harness/phase10_integration_test.py','tests/api/test_harness_autoresearch.py']]; print('OK')"
OK

$ python -c "from backend.api.harness_autoresearch import router; print('router:', router.prefix)"
router: /api/harness

$ python scripts/harness/phase10_integration_test.py
[PASS] backend_endpoint_returns_harness_sprint_week_state_shape  (shape=HarnessSprintWeekState, fields=thu/fri/monthly)
[PASS] endpoint_reads_from_harness_learning_log  (table_in_sql=True, param_bound={'week_iso': '2026-W25'})
[PASS] dashboard_renders_tile_when_data_present  (import=True, render=True, fetch=True)
[PASS] dashboard_renders_empty_state_when_data_null  (null_typed=True, catch_null=True, always_rendered=True)

ALL PASS  (4/4)
(exit 0)

$ pytest tests/api/test_harness_autoresearch.py -q
.......                                                                  [100%]
7 passed in 0.06s

$ pytest tests/autoresearch/ tests/slack_bot/ tests/housekeeping/ tests/api/ backend/metrics/ -q
........................................................................ [ 58%]
....................................................                     [100%]
124 passed in 1.95s

$ cd frontend && npx tsc --noEmit 2>&1 | grep -E "HarnessSprint|HarnessDashboard"
(empty -- no new type errors)
```

## Success criteria (masterplan, immutable)

| # | Criterion | Status |
|---|---|---|
| 1 | `backend_endpoint_returns_harness_sprint_week_state_shape` | PASS — pydantic-validated `HarnessSprintWeekState` with populated thu/fri/monthly; camelCase keys; nullable fields work |
| 2 | `endpoint_reads_from_harness_learning_log` | PASS — SQL contains `harness_learning_log` + `@week_iso` parameter; params dict = `{"week_iso": "2026-W25"}` |
| 3 | `dashboard_renders_tile_when_data_present` | PASS — `HarnessSprintTile` imported from `@/components/HarnessSprintTile`; rendered with `data={sprintState}`; `getHarnessSprintState` called in the Promise.all |
| 4 | `dashboard_renders_empty_state_when_data_null` | PASS — `sprintState` is `HarnessSprintWeekState \| null`; `.catch(() => null)` fail-open; tile always rendered (phase-10.9 tile handles null internally) |

## Side-channel check

- `/api/harness` route is live at boot (`router.prefix == '/api/harness'` on import).
- `tsc --noEmit` shows no new errors for either `HarnessSprintTile` or `HarnessDashboard`.
- Phase-10.9's `HarnessSprintTile.test.tsx` (5 tests) still passes since the tile itself is unchanged.

## Carry-forwards (out of scope)

- Wire 10.3/10.4/10.6/10.7 routines to actually CALL `log_slot_usage` at their success paths (real-data pipeline; requires scheduler integration step)
- Add weekly-selector dropdown so users can view past weeks' sprint state (UX enhancement)
- Fix `HarnessDashboard.tsx` pre-existing `@phosphor-icons/react` direct-import violation (separate ticket)
- Auth/session: endpoint inherits existing auth middleware — no changes needed here
