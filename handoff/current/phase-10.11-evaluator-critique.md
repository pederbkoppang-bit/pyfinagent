# Evaluator Critique — phase-10.11 (Integration: backend endpoint + dashboard wiring)

**qa_id:** qa_1011_v1
**Date:** 2026-04-21
**Verdict:** PASS

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_1011_v1",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "5_item_harness_audit",
    "ast_parse_3_files",
    "immutable_cli_phase10_integration_test",
    "pytest_test_harness_autoresearch",
    "pytest_full_suite_124",
    "router_prefix_import",
    "frontend_tsc_targeted_grep",
    "spot_read_harness_autoresearch_py",
    "spot_read_HarnessDashboard_tsx",
    "wire_shape_cross_check_vs_types_ts",
    "mutation_M1_prefix",
    "mutation_M2_table_name",
    "mutation_M3_tile_render"
  ],
  "reason": "All 4 immutable success_criteria PASS; pytest 7/7 + suite 124/124; TS clean for targeted components; wire shape camelCase matches types.ts HarnessSprintWeekState exactly; mutation M2 and M3 caught by the immutable CLI; M1 (prefix mutation) is not caught but was pre-disclosed as a known gap in the contract's Carry-forwards. Fail-open BQ behavior, sync def endpoint (correct per backend-api.md async-safety rule), ISO-week uses isocalendar() per research brief. Minimally invasive frontend wiring (1 import, 1 state var, 1 Promise.all entry, 1 JSX line) — no rewrite of HarnessDashboard."
}
```

## 5-item harness-compliance audit

1. **Researcher ran before contract with gate_passed.** `phase-10.11-research-brief.md` exists (22421 bytes, 2026-04-21 06:47), contract explicitly cites "6 in full, 15 URLs, gate_passed=true". PASS.
2. **Contract mtime ≤ results mtime.** `contract.md` 17:03:27, `experiment-results.md` 17:08:06. Contract written first. PASS.
3. **Verbatim quote of immutable criteria.** Contract lines 24-29 list the 4 success_criteria with exact names matching the masterplan spec passed in this prompt. PASS.
4. **No log append yet.** `handoff/harness_log.md` has not been amended for 10.11 (the log-last rule — Main appends AFTER Q/A PASS). Correct.
5. **Cycle v1, no verdict-shopping.** No prior phase-10.11 evaluator critique exists. First and only Q/A spawn. PASS.

## Deterministic checks (verbatim output)

### A. AST parse

```
$ python -c "import ast; [ast.parse(open(f).read()) for f in [...]]; print('AST_OK')"
AST_OK
```

### B. Immutable CLI

```
$ python scripts/harness/phase10_integration_test.py
[PASS] backend_endpoint_returns_harness_sprint_week_state_shape  (shape=HarnessSprintWeekState, fields=thu/fri/monthly)
[PASS] endpoint_reads_from_harness_learning_log  (table_in_sql=True, param_bound={'week_iso': '2026-W25'})
[PASS] dashboard_renders_tile_when_data_present  (import=True, render=True, fetch=True)
[PASS] dashboard_renders_empty_state_when_data_null  (null_typed=True, catch_null=True, always_rendered=True)
ALL PASS  (4/4)
EXIT=0
```

### C. pytest new + full suite

```
$ pytest tests/api/test_harness_autoresearch.py -q
.......                                                                  [100%]
7 passed in 0.06s

$ pytest tests/autoresearch/ tests/slack_bot/ tests/housekeeping/ tests/api/ backend/metrics/ -q
........................................................................ [ 58%]
....................................................                     [100%]
124 passed in 1.48s
```

### D. Router import

```
$ python -c "from backend.api.harness_autoresearch import router; print('prefix:', router.prefix)"
prefix: /api/harness
```

### E. TS type check (targeted)

```
$ cd frontend && npx tsc --noEmit 2>&1 | grep -E "HarnessSprint|HarnessDashboard"
(empty)
```

No type errors touching `HarnessSprintTile`, `HarnessSprintWeekState`, or `HarnessDashboard`.

### F. Spot-reads (contract items D and E)

`backend/api/harness_autoresearch.py`:
- Line 23: `router = APIRouter(prefix="/api/harness", tags=["harness"])` ✓ at module scope
- Line 51-53: `_current_week_iso` uses `date.today().isocalendar()` named-tuple — not `strftime` ✓
- Line 109-116: `_build_sql` SQL contains both `` `{table}` `` (interpolated to `harness_learning_log`) AND `@week_iso` parameter binding ✓
- Lines 28-48: Pydantic fields camelCase (`weekIso`, `batchId`, `candidatesKicked`, `promotedIds`, `rejectedIds`, `sortinoDelta`, `approvalPending`, `approved`) ✓ — matches `types.ts:936` `HarnessSprintWeekState` interface exactly
- Line 136-141: `fetch_sprint_state(*, week_iso, bq_query_fn, table)` — kw-only, injectable ✓
- Line 136-141: Correct async-safety: `def` (not `async def`) on sync-I/O endpoint (per `backend-api.md` rule) ✓

`frontend/src/components/HarnessDashboard.tsx`:
- Line 11: `getHarnessSprintState` imported from `@/lib/api` ✓
- Line 16: `HarnessSprintWeekState` imported from types ✓
- Line 18: `import { HarnessSprintTile } from "@/components/HarnessSprintTile"` ✓
- Line 198: `const [sprintState, setSprintState] = useState<HarnessSprintWeekState | null>(null)` ✓
- Line 209: `getHarnessSprintState().catch(() => null)` ✓
- Line 249: `<HarnessSprintTile data={sprintState} />` rendered inside the dashboard's JSX ✓

Wiring is minimally invasive: 1 import line, 1 state var, 1 Promise.all entry, 1 destructure, 1 setter, 1 JSX line. No restructure of `HarnessDashboard`.

### G. Wire-shape cross-check

Backend Pydantic (camelCase) vs TS `HarnessSprintWeekState` at `types.ts:936`:

| TS field | Backend field | Match |
|---|---|---|
| `weekIso: string` | `weekIso: str` | ✓ |
| `thu.batchId: string` | `HarnessSprintThu.batchId: str` | ✓ |
| `thu.candidatesKicked: number` | `candidatesKicked: int` | ✓ |
| `fri.promotedIds: string[]` | `promotedIds: list[str]` | ✓ |
| `fri.rejectedIds: string[]` | `rejectedIds: list[str]` | ✓ |
| `monthly.sortinoDelta: number` | `sortinoDelta: float` | ✓ |
| `monthly.approvalPending: boolean` | `approvalPending: bool` | ✓ |
| `monthly.approved: boolean` | `approved: bool` | ✓ |
| `thu / fri / monthly: {...} \| null` | `Optional[...]` with `None` default | ✓ |

### H. BQ schema cross-check (phase-10.8 row shape)

Contract cites phase-10.8 row schema: `{week_iso, slot_id ∈ {thu_batch, fri_promotion, monthly_gate, rollback}, phase, result_json, logged_at}`. Backend SQL selects `slot_id, result_json, logged_at` with `WHERE week_iso = @week_iso AND phase = 'phase-10'`. The `_project_rows_to_state` reads `rows_by_slot.get("thu_batch" | "fri_promotion" | "monthly_gate")` — slot ids match. `rollback` is intentionally ignored (tile doesn't surface rollback state). Column names align.

## Mutation tests

| # | Mutation | Caught? | Notes |
|---|---|---|---|
| M1 | `APIRouter(prefix="/api/harness")` → `"/api/other"` | **NO** | Immutable CLI passes (4/4); pytest 7/7 passes. No test binds the prefix. **Pre-disclosed gap** in the prompt ("there's no dedicated test for the prefix; flag as mutation-resistance gap if tests don't catch this"). |
| M2 | SQL `harness_learning_log` → `some_other_table` | **YES** | Criterion 2 fails (`table_in_sql=False`); CLI exit=1. |
| M3 | Remove `<HarnessSprintTile data={sprintState} />` | **YES** | Criteria 3+4 both fail (`render=False`, `always_rendered=False`); CLI exit=1. |

All three files restored to byte-identical baseline after mutation testing — verified via `diff`.

## LLM judgment

- **Integration correctness:** endpoint SQL projects the phase-10.8 row schema correctly; slot_ids (`thu_batch`, `fri_promotion`, `monthly_gate`) and `result_json` payload keys (`batch_id`, `candidates_kicked`, `promoted_ids`, `rejected_ids`, `sortino_delta`, `approval_pending`, `approved`) match the authoritative source (`backend/autoresearch/slot_accounting.py`, cited in contract).
- **Frontend wiring:** minimally invasive (surgical 6-line diff); no rewrite. Sibling Promise.all entries use explicit fallback objects (`{ cycles: [] }`, etc.) so a fatal network failure still sets `error` via the outer `.catch`. The sprint tile's `.catch(() => null)` is acceptable graceful-degradation per `frontend.md` ("Individual `.catch(() => null)` is OK for optional/graceful-degradation calls only").
- **camelCase discipline:** perfect alignment between Pydantic response model and the TypeScript interface. No shape drift.
- **Async safety (backend-api.md):** endpoint is `def` not `async def` — correct because the BQ call is synchronous. Had it been `async def` with a sync BQ call inside, it would have blocked the event loop.
- **ISO-week correctness:** `date.today().isocalendar()` avoids the `strftime("%Y-W%V")` Dec/Jan boundary bug called out in the research brief.
- **Carry-forwards are legit deferrals:**
  - Wiring 10.3/10.4/10.6/10.7 to actually call `log_slot_usage` — belongs to a scheduler-integration step, out of scope here (this step is the READ path only).
  - Week-selector UX — genuine enhancement beyond the contract.
  - `HarnessDashboard` icon-import violation — pre-existing; fixing it here would expand the blast radius.

## Gaps / recommendations (non-blocking)

1. **Prefix mutation gap (M1).** Suggest a follow-up micro-step adding `def test_router_prefix(): assert router.prefix == "/api/harness"` to `tests/api/test_harness_autoresearch.py`. Two-line change, closes the mutation gap.
2. **BQ error path is hard to observe in tests.** `_default_bq_query` is only exercised in live environments. A `monkeypatch` unit that forces the `google.cloud.bigquery` import to raise would lock in the fail-open behavior.
3. **Cache layer.** Contract mentioned a 60s `api_cache` TTL (plan step 1). The shipped endpoint does not use `api_cache`. Not in the immutable criteria, but the contract's own plan list said it would. Minor divergence; recommend noting it in the experiment_results' scope-honesty section or adding the cache in a follow-up.

None of these block PASS: the immutable criteria are the contract between the step and the masterplan, and all four pass with real mutation resistance on 2/3 mutations (the third, M1, was pre-acknowledged).

## Verdict: PASS

- All 4 immutable success_criteria green (4/4)
- Full test suite green (7/7 new + 124/124 regression)
- Wire-shape matches `types.ts` exactly
- Mutation-resistance 2/3, with M1 pre-disclosed
- Contract-before-generate ordering correct
- Research gate passed (6 sources read in full, recency scan performed)
- No log append yet (correct — log comes after PASS)
