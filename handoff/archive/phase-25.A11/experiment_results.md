---
step: phase-25.A11
cycle: 68
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_A11.py'
title: Wire /paper-trading/learnings backend endpoint (P1)
audit_basis: phase-24.11 F-1 (orphan UI page wired to live backend)
---

# Experiment Results -- phase-25.A11

## Code changes

### `backend/api/paper_trading.py`
- New imports: `json`, `timedelta` (added to existing `datetime` import), `Path`
- New module-level constant: `_KILL_SWITCH_AUDIT_PATH` resolving to `handoff/kill_switch_audit.jsonl` (mirrors `backend/services/kill_switch.py::_AUDIT_PATH`)
- New helper: `_compute_learnings(bq, window_days) -> dict` builds the 3-section response (reconciliation_divergences, kill_switch_triggers, regime_buckets) with per-section try/except so one failure does not nuke the others
- New route: `GET /api/paper-trading/learnings?window_days=30` (`window_days: int = Query(30, ge=1, le=365)`), mirrors `/trades` sibling pattern -- cache check, `asyncio.to_thread`, cache set with `f"paper:learnings:{window_days}"` key

### `backend/db/bigquery_client.py`
- New method: `get_paper_trades_in_window(self, window_days: int) -> list[dict]` -- parameterized `SELECT * FROM paper_trades WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @window_days DAY) ORDER BY created_at DESC LIMIT 2000`, with `result(timeout=30)` per CLAUDE.md rule

### `backend/services/api_cache.py`
- `ENDPOINT_TTLS` gains `"paper:learnings": 300.0` (5-minute TTL, same as `/performance`)

### `frontend/src/lib/types.ts`
- Appended: `ReconciliationDivergence`, `KillSwitchTrigger`, `RegimeBucket`, `VirtualFundLearningsData` (promoted from `VirtualFundLearnings.tsx`)

### `frontend/src/components/VirtualFundLearnings.tsx`
- Removed local `export interface` declarations for the four types
- Replaced with `import type { VirtualFundLearningsData } from "@/lib/types"`

### `frontend/src/components/VirtualFundLearnings.test.tsx`
- Updated import to pull `VirtualFundLearningsData` from `@/lib/types` (was from `./VirtualFundLearnings`)

### `frontend/src/lib/api.ts`
- New export: `getPaperLearnings(windowDays = 30): Promise<VirtualFundLearningsData>` (default 30-day window)

### `frontend/src/app/paper-trading/learnings/page.tsx`
- Converted from static wrapper to fetch wrapper: `useEffect` calls `getPaperLearnings(30)` on mount; passes `data`, `loading`, `error` props to `<VirtualFundLearnings />`
- Removed stale "// Live data hookup lands in a follow-up backend step" comment (this WAS the follow-up step)

### `tests/verify_phase_25_A11.py` (new file)
- 10 immutable claims covering: route registration, signature, response shape, BQ helper + timeout, types.ts placement, component import, page wiring, api.ts export, ENDPOINT_TTLS, behavioral round-trip when audit JSONL is absent

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_A11.py
PASS: new_get_api_paper_trading_learnings_endpoint_registered
PASS: get_learnings_signature_window_days_query_30_ge1_le365
PASS: compute_learnings_returns_required_keys
PASS: bq_helper_get_paper_trades_in_window_with_timeout_30
PASS: virtualfundlearningsdata_type_in_types_ts
PASS: vfl_component_imports_type_from_lib_types_no_local_define
PASS: frontend_learnings_page_renders_non_empty_states
PASS: api_ts_exports_getPaperLearnings_default_30
PASS: api_cache_endpoint_ttls_has_paper_learnings
PASS: compute_learnings_handles_missing_audit_jsonl_gracefully

10/10 claims PASS, 0 FAIL
```

## Frontend gates

- `npx tsc --noEmit` from `frontend/` -- EXIT=0 (clean across the whole project)
- `npx eslint --max-warnings 0` on all touched files (`learnings/page.tsx`, `VirtualFundLearnings.tsx`, `VirtualFundLearnings.test.tsx`, `types.ts`) -- EXIT=0 (zero warnings, zero errors)
- `npx vitest run src/components/VirtualFundLearnings.test.tsx` -- 5/5 tests pass (verifies the type-promotion didn't break the component)

## Backend gates

- `python -c "import ast; ast.parse(open('backend/api/paper_trading.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/db/bigquery_client.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/services/api_cache.py').read())"` -- OK
- Behavioral round-trip (claim 10): `_compute_learnings(MockBQ(), 30)` with a deliberately-missing `_KILL_SWITCH_AUDIT_PATH` returns `{reconciliation_divergences: [], kill_switch_triggers: [], regime_buckets: [], window_days: 30, collected_at: <iso>}` -- the graceful-degradation contract holds.

## Hypothesis verdict

CONFIRMED. The component contract was already locked (4 interfaces with exactly the shape used by the rendering code). The smallest possible wiring -- promote the types to `types.ts`, add a single backend endpoint that mirrors `/trades`, and wire the page via `useEffect` -- satisfies all 3 immutable success criteria + the live-check shape. The 3-section payload-builder is intentionally permissive (per-section try/except + empty-array fallbacks) because the component's empty states already render correctly.

## Live-check

Per masterplan: `GET /api/paper-trading/learnings?window_days=30 returns valid VirtualFundLearningsData`.

Live evidence pending operator capture into `handoff/current/live_check_25.A11.md` -- the auto-push gate enforces this on status flip. Expected shape (curl with backend on port 8000):

```json
{
  "reconciliation_divergences": [...],
  "kill_switch_triggers": [{"reason": "uat-16.6-drill", "count": 1}, ...],
  "regime_buckets": [],
  "window_days": 30,
  "collected_at": "2026-05-12T..."
}
```

`regime_buckets: []` is the documented first-pass behavior (paper_trades has no per-trade regime column; component handles empty array with "No regime buckets computed yet.").

## Non-regressions

- Existing `VirtualFundLearnings.test.tsx` (5 cases) still passes -- type promotion was source-compatible.
- No change to existing `paper_trading.py` routes -- new endpoint is purely additive.
- No BQ schema migration.

## Next phase

Q/A pending.
