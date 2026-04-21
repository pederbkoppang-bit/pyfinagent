# Research Brief — phase-10.11
# Integration: backend endpoint + dashboard wiring for autoresearch sprint-state tile
# Tier: moderate | Date: 2026-04-20

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://fastapi.tiangolo.com/tutorial/query-param-models/ | 2026-04-20 | official doc | WebFetch | "Use `Annotated[ModelName, Query()]` to declare query parameter models; feature available since FastAPI 0.115.0+" |
| https://swr.vercel.app/docs/getting-started | 2026-04-20 | official doc | WebFetch | "Request deduplication across components sharing the same data; `{ data, error, isLoading }` states returned from useSWR" |
| https://docs.python.org/3/library/datetime.html | 2026-04-20 | official doc | WebFetch | "`date(2003, 12, 29).isocalendar()` returns year=2004, week=1 — ISO year can differ from calendar year at boundaries; pair `%G` with `%V`, never `%Y` with `%V`" |
| https://apipark.com/techblog/en/fastapi-how-to-return-null-none-responses-correctly/ | 2026-04-20 | blog | WebFetch | "Return 404 when the resource itself is missing; return 200 OK with null fields when resource exists but a field legitimately has no value" |
| https://nextjs.org/docs/pages/building-your-application/data-fetching/client-side | 2026-04-20 | official doc | WebFetch | "Next.js team highly recommends SWR for client-side fetching; handles caching, revalidation, focus tracking, refetching on intervals" |
| https://blog.greeden.me/en/2026/02/03/fastapi-performance-tuning-caching-strategy-101-a-practical-recipe-for-growing-a-slow-api-into-a-lightweight-fast-api/ | 2026-04-20 | blog (2026) | WebFetch | "Start with HTTP cache headers on read-only endpoints; layer application-level TTL cache (in-process or Redis) for BQ-backed endpoints with 'rankings, trending, etc.' data" |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://medium.com/@ThinkingLoop/12-react-19-data-fetching-patterns-that-kill-waterfalls-6782ef923fc0 | blog | Snippet sufficient; useEffect + apiFetch is already the project pattern |
| https://medium.com/algomart/fastapi-request-and-response-models-with-pydantic-made-simple-329fe6eb0c80 | blog | Pydantic response_model pattern covered by official doc above |
| https://unfoldai.com/fastapi-evolution/ | blog | FastAPI 0.115 Pydantic query model covered by official doc |
| https://weeknumber.com/how-to/python | doc | ISO week Python covered by CPython docs above |
| https://pypi.org/project/iso-week-date/ | package | stdlib isocalendar is sufficient; no external dep needed |
| https://medium.com/@komalbaparmar007/caching-for-fastapi-tiny-layer-big-p95-win-acf6e0cd37b8 | blog | In-process TTL already implemented in `backend/services/api_cache.py` |
| https://github.com/fastapi/fastapi/discussions/10454 | forum | Query param models covered by official doc |
| https://dev.to/emmanueloloke/reacts-useeffect-vs-useswr-exploring-data-fetching-in-react-2j60 | blog | Covered by Next.js official doc |
| https://pypi.org/project/pytest-subprocess/ | package | subprocess invocation of vitest covered in scope-resolution below |

---

## Recency scan (2024-2026)

Searched: "FastAPI BigQuery read endpoint Pydantic Optional response model 2026",
"React 19 data fetching patterns client component useEffect SWR 2025",
"FastAPI performance caching strategy 101 2026".

**Findings (2024-2026):**
- FastAPI 0.115.0 (late 2024) introduced native Pydantic models for query parameters via `Annotated[Model, Query()]`, superseding the earlier workaround of using `Depends()` with a class. The harness project uses FastAPI 0.115+ per `backend/api/` imports — this feature is available.
- Next.js 15 (2024) makes App Router the default; `"use client"` components still use `useEffect` + manual fetch or SWR. The project uses `"use client"` pattern throughout `HarnessDashboard.tsx` — no change needed.
- The greeden.me caching post (Feb 2026) confirms the in-process TTL cache approach used by `backend/services/api_cache.py` is the recommended lightweight alternative to Redis for single-instance deployments. TTL of 60s for dashboard-style read endpoints is within the recommended "seconds to minutes" band for frequently-checked data.
- No new findings that supersede the existing `api_cache.py` pattern for this step.

---

## Key findings

1. **Router registration pattern** — `backend/main.py:279-291` registers all routers via `app.include_router()`. A new `harness_autoresearch` router should be imported and registered in the same block. The existing harness endpoints (`/api/backtest/harness/log`, etc.) live in `backend/api/backtest.py:1227+` under the backtest router. The new sprint-state endpoint belongs in its own module `backend/api/harness_autoresearch.py` with prefix `/api/harness` (no backtest dependency). (Source: `backend/main.py:279-291`)

2. **Harness endpoint idiom** — `backend/api/backtest.py:1227-1330` shows the canonical harness endpoint pattern: `def` (not `async def`) for BQ reads wrapped in `asyncio.to_thread` or a plain sync function auto-run by FastAPI's threadpool; no Pydantic response model (returns raw dicts); 60s TTL via `get_api_cache()` (see `backend/services/api_cache.py:49`). The `performance_api.py:56-96` BQ query pattern uses `bq.client.query(sql).result()` with parameterized queries. (Source: `backend/api/backtest.py:1227`, `backend/api/performance_api.py:56-96`)

3. **BQ table schema for `harness_learning_log`** — `backend/autoresearch/slot_accounting.py:60-70` writes these columns: `logged_at`, `row_id`, `week_iso`, `slot_id` (one of `thu_batch`, `fri_promotion`, `monthly_gate`, `rollback`), `phase`, `routine`, `result_json` (JSON string), `status`, `error_msg`. The `result_json` field is the JSON blob written by each phase-10 routine. To reconstruct `HarnessSprintWeekState`, the endpoint must query rows WHERE `week_iso = @w AND phase = 'phase-10'`, then parse `result_json` per `slot_id`. (Source: `backend/autoresearch/slot_accounting.py:27,60-70`)

4. **HarnessSprintWeekState shape** — `frontend/src/lib/types.ts:936-945` defines: `weekIso: string; thu: {batchId,candidatesKicked}|null; fri: {promotedIds,rejectedIds}|null; monthly: {sortinoDelta,approvalPending,approved}|null`. The backend response model must map 1:1 with this. camelCase keys match because the project uses direct JSON key naming between Python and TS (no snake_case conversion layer observed in existing endpoints). (Source: `frontend/src/lib/types.ts:936-945`)

5. **Frontend fetching idiom** — `frontend/src/components/HarnessDashboard.tsx:195-218` fetches all harness data in a single `Promise.all()` on mount, with `.catch(() => fallback)` per call. Sprint-state should follow the same pattern: add `getHarnessSprintState()` to the `Promise.all` array, catch to `null`. SWR is not currently used in this project's dashboard components; do not introduce it for consistency. (Source: `frontend/src/components/HarnessDashboard.tsx:195-218`)

6. **ISO week derivation safety** — Python `date.today().isocalendar()` returns `(iso_year, iso_week, iso_weekday)` as a named tuple (Python 3.9+). Use `f"{r.year}-W{r.week:02d}"` where `r = date.today().isocalendar()`. Do NOT use `strftime("%Y-W%V")` — `%Y` is the Gregorian year and diverges from `%V` (ISO week) in the last days of December / first days of January. Safe pattern: `date.today().isocalendar()` named tuple, then format with `iso_year` (not Gregorian `year`). (Source: CPython docs — `date(2003,12,29).isocalendar()` returns year=2004)

7. **Scope of the verification test** — `scripts/harness/phase10_integration_test.py` is a Python CLI test. The four success_criteria name both backend and dashboard. Resolution: backend criteria 1+2 verified directly in Python (mock BQ, assert response shape + SQL). Frontend criteria 3+4 verified by the Python test invoking `npm run test -- --filter=HarnessDashboard` via `subprocess.run()` with `check=True`. This is deterministic, avoids introducing a separate test runner, and matches the pattern used in `scripts/harness/run_harness.py` for subprocess calls. The HarnessSprintTile tests already exist at `frontend/src/components/HarnessSprintTile.test.tsx`; a new `HarnessDashboard.test.tsx` (or augment existing) must assert tile is rendered when data is non-null and not rendered / shows empty-state when data is null.

8. **200 + null vs 404** — For the sprint-state endpoint, if no rows found for a requested `week_iso`: return `{"data": null}` with HTTP 200. Rationale: the resource (the endpoint) exists; the data for that week is legitimately absent (sprint hasn't run yet). Returning 404 would cause the frontend's `apiFetch` at `frontend/src/lib/api.ts:131-133` to throw "Endpoint not found" — misleading. Return shape: `{"data": HarnessSprintWeekState | null}`. (Source: `apipark.com` null vs 404 guidance; `frontend/src/lib/api.ts:131-133`)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/main.py` | 335 | FastAPI app; registers all routers at lines 280-291 | Active — add new router import+include here |
| `backend/api/backtest.py` | 1330+ | Backtest + harness file-reader endpoints; `/api/backtest/harness/*` | Active — do NOT add sprint-state here; use new module |
| `backend/api/performance_api.py` | ~200 | Perf/cache/LLM endpoints; BQ query pattern at lines 56-96 | Active — reference BQ pattern |
| `backend/services/api_cache.py` | ~100 | Thread-safe TTL cache singleton; `get_api_cache().get/set()` | Active — use for 60s TTL on new endpoint |
| `backend/autoresearch/slot_accounting.py` | 150 | BQ sink for all phase-10 activity; defines table schema | Active — column map at lines 60-70 |
| `backend/autoresearch/weekly_ledger.py` | 118 | TSV ledger; columns map phase-10 fields | Active — secondary source; BQ is primary |
| `frontend/src/components/HarnessDashboard.tsx` | 350+ | Harness tab component; `Promise.all` fetch pattern | Active — add sprint-state fetch here |
| `frontend/src/components/HarnessSprintTile.tsx` | 190 | Read-only sprint tile; accepts `HarnessSprintWeekState\|null` | Active — no changes needed to tile itself |
| `frontend/src/components/HarnessSprintTile.test.tsx` | 99 | Vitest tests for tile; 5 test cases already passing | Active — passes; new HarnessDashboard test needed |
| `frontend/src/lib/api.ts` | 500+ | API client; add `getHarnessSprintState()` here | Active — add fetcher function |
| `frontend/src/lib/types.ts` | 982 | `HarnessSprintWeekState` at lines 936-945 | Active — type already defined; no change needed |

---

## Consensus vs debate (external)

**Consensus:**
- Pydantic `Optional` fields + `response_model` for nullable responses is the canonical FastAPI pattern (official docs + community).
- Return HTTP 200 with `null` data field when the endpoint exists but no data is present for that week; 404 only when the endpoint itself is invalid.
- `useEffect` + `Promise.all` (project's existing idiom) is valid for this non-polling, one-shot dashboard fetch. SWR adds revalidation and deduplication that are not needed here.

**Debate:**
- SWR vs `useEffect`: SWR is recommended by Next.js for most cases, but the project consistently uses `useEffect`+`Promise.all` and this tile is not a polling component. Introducing SWR only for this one tile would be inconsistent with the codebase. Decision: stay with `useEffect`.
- 60s vs 5min TTL: BQ sprint-state data changes at most twice a week (Thursday batch, Friday promotion). A 5min TTL is defensible. However, the project's existing `ENDPOINT_TTLS` in `api_cache.py` uses 60s for most dashboard endpoints; consistency wins over marginal savings. Use 60s.

---

## Pitfalls (from literature + internal audit)

1. **ISO year boundary (Python stdlib)** — `strftime("%Y-W%V")` returns wrong year for dates in ISO week 1 that fall in December (e.g., Dec 29 2003 = ISO 2004-W01). Use `date.today().isocalendar()` named tuple and format with `iso_cal.year` (not the Gregorian `.year`).

2. **camelCase vs snake_case** — The backend normally uses snake_case in Python dicts; the TS interface uses camelCase. Looking at `slot_accounting.py:60-70`, the BQ columns are snake_case (`week_iso`, `slot_id`, `result_json`). The response endpoint must manually map `week_iso` → `weekIso`, `batch_id` → `batchId`, etc. Do NOT rely on Pydantic's `alias_generator` unless the existing codebase uses it (it does not — existing harness endpoints return raw dicts with explicit keys).

3. **`result_json` is a JSON string** — `slot_accounting.py:55-56` stores the routine result as a JSON string (`json.dumps(result)`). The endpoint must `json.loads(row["result_json"])` before projecting fields. If `result_json` is malformed, fail-open: set that slot's field to `null`.

4. **Multiple rows per slot_id** — If `log_slot_usage` is called twice for the same `week_iso` + `slot_id` (idempotency guard is upstream, but BQ streaming doesn't deduplicate), the query may return two rows for `thu_batch`. Use `ORDER BY logged_at DESC LIMIT 1` per slot_id, or take the MAX/last row. Safest: `SELECT slot_id, result_json FROM ... WHERE week_iso=@w AND phase='phase-10' ORDER BY logged_at DESC` then take first occurrence of each slot_id in Python.

5. **Frontend `apiFetch` 404 throw** — `frontend/src/lib/api.ts:131-133` throws `"Endpoint not found"` on HTTP 404. If the endpoint returns 404 for "no data", the frontend will surface a misleading error instead of the empty-state tile. Must return 200 + `{"data": null}`.

6. **Subprocess vitest in Python test** — `subprocess.run(["npm", "run", "test", "--", "--filter=HarnessDashboard"], cwd="frontend/", check=True)` must use the absolute path to `frontend/` to avoid CWD-relative issues in CI. Pass `timeout=120`.

---

## Application to pyfinagent — design recommendation

### Backend endpoint: `GET /api/harness/sprint-state`

**File:** `backend/api/harness_autoresearch.py` (new module)
**Router prefix:** `/api/harness` (not `/api/backtest/harness` — that's for file-reader endpoints)
**Registration:** `backend/main.py` — add import + `app.include_router(harness_autoresearch_router)` at line ~291

```python
# backend/api/harness_autoresearch.py  (pseudocode for generator)
from fastapi import APIRouter, Query
from typing import Optional
from datetime import date
import json
import logging

router = APIRouter(prefix="/api/harness", tags=["harness-autoresearch"])
logger = logging.getLogger(__name__)

@router.get("/sprint-state")
def get_sprint_state(week_iso: Optional[str] = Query(None)):
    """Return HarnessSprintWeekState for a given ISO week from harness_learning_log."""
    from backend.services.api_cache import get_api_cache
    
    # Default to current ISO week using safe isocalendar() pattern
    if not week_iso:
        r = date.today().isocalendar()
        week_iso = f"{r.year}-W{r.week:02d}"   # r.year is ISO year, not Gregorian
    
    cache_key = f"sprint_state:{week_iso}"
    cached = get_api_cache().get(cache_key)
    if cached is not None:
        return cached
    
    # Query BQ
    result = _query_sprint_state(week_iso)
    get_api_cache().set(cache_key, result, ttl_seconds=60)
    return result


def _query_sprint_state(week_iso: str) -> dict:
    """Query harness_learning_log and project into HarnessSprintWeekState shape."""
    try:
        from google.cloud import bigquery
        import os
        project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
        client = bigquery.Client(project=project)
        sql = """
            SELECT slot_id, result_json, logged_at
            FROM `pyfinagent_data.harness_learning_log`
            WHERE week_iso = @week_iso
              AND phase = 'phase-10'
            ORDER BY logged_at DESC
        """
        params = [bigquery.ScalarQueryParameter("week_iso", "STRING", week_iso)]
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        rows = list(client.query(sql, job_config=cfg).result())
    except Exception as exc:
        logger.warning("sprint_state: BQ query fail-open: %r", exc)
        rows = []
    
    # Take first (most recent) row per slot_id
    seen: dict[str, dict] = {}
    for row in rows:
        sid = row["slot_id"]
        if sid not in seen:
            seen[sid] = row

    def parse_slot(slot_id: str) -> dict | None:
        if slot_id not in seen:
            return None
        try:
            return json.loads(seen[slot_id]["result_json"])
        except Exception:
            return None

    thu_raw = parse_slot("thu_batch")
    fri_raw = parse_slot("fri_promotion")
    monthly_raw = parse_slot("monthly_gate")

    thu = None
    if thu_raw:
        thu = {
            "batchId": str(thu_raw.get("batch_id", "")),
            "candidatesKicked": int(thu_raw.get("candidates_kicked", 0)),
        }

    fri = None
    if fri_raw:
        fri = {
            "promotedIds": list(fri_raw.get("promoted_ids", [])),
            "rejectedIds": list(fri_raw.get("rejected_ids", [])),
        }

    monthly = None
    if monthly_raw:
        monthly = {
            "sortinoDelta": float(monthly_raw.get("sortino_delta", 0.0)),
            "approvalPending": bool(monthly_raw.get("approval_pending", False)),
            "approved": bool(monthly_raw.get("approved", False)),
        }

    data = None
    if thu or fri or monthly:
        data = {
            "weekIso": week_iso,
            "thu": thu,
            "fri": fri,
            "monthly": monthly,
        }

    return {"data": data}
```

### Frontend wiring

**`frontend/src/lib/api.ts`** — add after the existing `getSeedStability` function:
```ts
import type { HarnessSprintWeekState } from "./types";  // already imported via apiFetch types

export function getHarnessSprintState(
  weekIso?: string
): Promise<{ data: HarnessSprintWeekState | null }> {
  const qs = weekIso ? `?week_iso=${encodeURIComponent(weekIso)}` : "";
  return apiFetch(`/api/harness/sprint-state${qs}`);
}
```

**`frontend/src/components/HarnessDashboard.tsx`** — add state + fetch:
1. Add `const [sprintState, setSprintState] = useState<HarnessSprintWeekState | null>(null);`
2. Import `getHarnessSprintState` and `HarnessSprintWeekState`
3. Add `getHarnessSprintState().catch(() => ({ data: null }))` to the `Promise.all` array
4. In the `.then()` handler, add `setSprintState(sprint.data);`
5. Render `<HarnessSprintTile data={sprintState} />` inside the `return` JSX

### Verification test: `scripts/harness/phase10_integration_test.py`

**Scope resolution — BOTH backend and frontend are in scope.** The 4 success criteria are:
1. `backend_endpoint_returns_harness_sprint_week_state_shape` — unit test with mock BQ; assert response keys
2. `endpoint_reads_from_harness_learning_log` — inspect SQL string for `harness_learning_log` + `week_iso` + `phase = 'phase-10'`
3. `dashboard_renders_tile_when_data_present` — vitest via subprocess
4. `dashboard_renders_empty_state_when_data_null` — vitest via subprocess

**Test structure:**
```python
# scripts/harness/phase10_integration_test.py
import sys, json, subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]

def test_backend_shape():
    # Import endpoint function, inject mock BQ rows, assert shape
    ...

def test_sql_references():
    # Read backend/api/harness_autoresearch.py source, assert SQL contains required strings
    src = (ROOT / "backend/api/harness_autoresearch.py").read_text()
    assert "harness_learning_log" in src
    assert "week_iso" in src
    assert "phase = 'phase-10'" in src or "phase='phase-10'" in src

def test_frontend_vitest():
    result = subprocess.run(
        ["npm", "run", "test", "--", "--filter=HarnessDashboard"],
        cwd=str(ROOT / "frontend"),
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr

if __name__ == "__main__":
    failures = []
    for fn in [test_backend_shape, test_sql_references, test_frontend_vitest]:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
            failures.append(fn.__name__)
    sys.exit(len(failures))
```

The frontend vitest test (`HarnessDashboard.test.tsx`) must be created alongside the component changes and must cover:
- `dashboard_renders_tile_when_data_present`: mock `getHarnessSprintState` to return sample data; assert `<HarnessSprintTile>` renders
- `dashboard_renders_empty_state_when_data_null`: mock to return `{data: null}`; assert tile renders with "No sprint activity yet"

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (15 collected: 6 full + 9 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (11 files inspected)
- [x] Contradictions / consensus noted (SWR vs useEffect, 60s vs 5min TTL)
- [x] All claims cited per-claim

---

## Search queries run (three-variant discipline)

1. **Current-year frontier (2026):** "FastAPI read-only endpoint Pydantic response model query parameter 2026"; "FastAPI BigQuery read endpoint Pydantic Optional response model 2026"; "FastAPI performance tuning caching strategy 101 2026"
2. **Last-2-year window (2025):** "React 19 data fetching patterns client component useEffect SWR 2025"; "Python subprocess pytest vitest integration test 2025 2026"; "BigQuery FastAPI caching strategy TTL read-only endpoint 2025"
3. **Year-less canonical:** "Python ISO week date.isocalendar year boundary pitfall strftime %G-W%V"; "FastAPI nullable response HTTP 200 null body vs 404 empty state REST API design"

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 9,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "handoff/current/phase-10.11-research-brief.md",
  "gate_passed": true
}
```
