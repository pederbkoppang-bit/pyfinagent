# Sprint Contract -- phase-25.A11 -- Wire /paper-trading/learnings backend endpoint

**Cycle:** phase-25 cycle 12 (P1 sprint)
**Date:** 2026-05-12
**Step ID:** 25.A11
**Priority:** P1
**Audit basis:** bucket 24.11 F-1 (orphan page with "// Live data hookup lands in a follow-up backend step")

## Research-gate

Researcher spawned this cycle (agent ac1fa845643b9a419). Brief at
`handoff/current/research_brief.md`. Gate envelope: tier=moderate,
external_sources_read_in_full=6, urls_collected=14, recency_scan_performed=true,
internal_files_inspected=10, gate_passed=true.

Key research conclusions:
- Component contract is already locked at `VirtualFundLearnings.tsx:26-44`. Backend MUST return `{reconciliation_divergences, kill_switch_triggers, regime_buckets, window_days, collected_at}` exactly.
- Mirror sibling endpoint pattern `paper_trading.py:216-228` (`/trades`): `async def` + `Query(ge=1, le=365)` + `asyncio.to_thread` + `get_api_cache()` with `f"paper:learnings:{window_days}"` key.
- Reconciliation divergences: query `paper_trades` in window, pair via existing `pair_round_trips()` -- per-trade `drift_pct = (sell_price - avg_entry_price) / avg_entry_price * 100`.
- Kill-switch triggers: read `handoff/kill_switch_audit.jsonl` in Python (no BQ needed). Filter `event == "pause"`, `Counter(row["trigger"])`.
- Regime buckets: `paper_trades` has NO per-trade `regime_tag` column (grep-confirmed). Return `regime_buckets: []` for first pass -- the component's empty state handles this gracefully. Future step can add a regime column to the trade row.

## Hypothesis

Adding a single new GET endpoint plus a thin BQ helper plus promoting the
existing interfaces to `types.ts` will close phase-24.11 F-1 without
schema migrations. The component already renders the empty case correctly,
so the smallest possible wiring satisfies all three immutable success
criteria.

## Success criteria (verbatim from masterplan)

1. `new_get_api_paper_trading_learnings_endpoint_returns_data`
2. `frontend_learnings_page_renders_non_empty_states`
3. `virtualfundlearningsdata_type_in_types_ts`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_A11.py`

Live check (per masterplan):
`GET /api/paper-trading/learnings?window_days=30 returns valid VirtualFundLearningsData`

## Plan

1. **Backend** -- `backend/api/paper_trading.py`:
   - Add `GET /learnings` route with `window_days: int = Query(30, ge=1, le=365)`.
   - Skeleton mirrors `/trades` exactly: cache check, settings/BQ init, `await asyncio.to_thread(_compute_learnings, bq, window_days)`, cache set, return.
   - New module-level helper `_compute_learnings(bq, window_days) -> dict` builds the response.
2. **BQ helper** -- `backend/db/bigquery_client.py`:
   - Add `get_paper_trades_in_window(self, window_days: int) -> list[dict]` returning ticker, action, price, quantity, created_at for the window. Pass `timeout=30` on `result()` per CLAUDE.md rule.
3. **Kill-switch JSONL aggregation** -- inside `_compute_learnings` (no new service file; the audit file is small and the read is cheap):
   - Path: `handoff/kill_switch_audit.jsonl` (`backend/services/kill_switch.py:36`).
   - If file missing: return `kill_switch_triggers: []` (do not raise).
   - Filter `event == "pause"`, `timestamp >= cutoff`, `Counter(row.get("trigger", "unknown"))`, sort desc.
4. **Reconciliation divergences** -- inside `_compute_learnings`:
   - Pass window trades to existing `pair_round_trips(trades)` from `backend/services/paper_round_trips.py`.
   - For each round-trip: `paper_fill = exit_price`, `sim_fill = entry_price`, `drift_pct = (paper_fill - sim_fill) / sim_fill * 100` when sim_fill > 0. Side = "sell" (the exit leg). Symbol = ticker. ts = exit_date ISO string.
   - Cap to top 100 by abs(drift_pct) to bound payload; the component sorts to top-10 client-side.
5. **Regime buckets** -- return `[]` with a logger.info note (no per-trade regime column today). Documented in brief, key 5.
6. **TS type promotion** -- `frontend/src/lib/types.ts`:
   - Append `ReconciliationDivergence`, `KillSwitchTrigger`, `RegimeBucket`, `VirtualFundLearningsData` (exactly the four interfaces currently at `VirtualFundLearnings.tsx:5-32`).
7. **TS component import update** -- `frontend/src/components/VirtualFundLearnings.tsx`:
   - Remove local `export interface` definitions for the four types.
   - Replace with `import type { VirtualFundLearningsData, ReconciliationDivergence, KillSwitchTrigger, RegimeBucket } from "@/lib/types";`.
8. **API client** -- `frontend/src/lib/api.ts`:
   - Add `getPaperLearnings(windowDays = 30): Promise<VirtualFundLearningsData>` mirroring sibling functions (`getPaperPerformance` shape).
9. **Page wiring** -- `frontend/src/app/paper-trading/learnings/page.tsx`:
   - Convert from static wrapper to fetch wrapper. `useEffect` fetches `getPaperLearnings(30)` on mount; passes `data`, `loading`, `error` props to `<VirtualFundLearnings />`. Remove the stale "// Live data hookup lands in a follow-up backend step" comment.
10. **ENDPOINT_TTLS** -- `backend/services/api_cache.py`: add `"paper:learnings": 300` (5 min, same as `/performance`).
11. **Verifier** -- `tests/verify_phase_25_A11.py` -- 8+ immutable claims:
    - Claim 1: route registered as `GET /api/paper-trading/learnings`.
    - Claim 2: route signature has `window_days: int = Query(30, ge=1, le=365)`.
    - Claim 3: `_compute_learnings` returns dict with three required array keys + `window_days` + `collected_at`.
    - Claim 4: BQ helper `get_paper_trades_in_window` exists with `timeout=30`.
    - Claim 5: `VirtualFundLearningsData` declared in `frontend/src/lib/types.ts`.
    - Claim 6: `VirtualFundLearnings.tsx` imports `VirtualFundLearningsData` from `@/lib/types` (not local define).
    - Claim 7: `learnings/page.tsx` calls `getPaperLearnings` and passes data/loading/error props.
    - Claim 8: `api.ts` exports `getPaperLearnings(windowDays = 30)`.
    - Claim 9: Kill-switch JSONL absent -> returns `kill_switch_triggers: []` (behavioral round-trip via mocked Path).

## Non-goals

- No new BQ table or migration. Regime bucketing intentionally returns
  `[]` for first pass; a follow-up step can add a regime column to
  `paper_trades`.
- No discriminated-union refactor of the component props (research
  noted this as a nice-to-have; not in immutable criteria).
- No frontend visual changes beyond the page becoming live. Component
  internals untouched.

## References

- `handoff/current/research_brief.md` -- full research brief (this cycle)
- `docs/audits/phase-24-2026-05-12/24.11-frontend-data-wiring-findings.md:17-41,148-158` -- F-1 audit basis
- `frontend/src/components/VirtualFundLearnings.tsx:5-44` -- locked props contract
- `backend/api/paper_trading.py:216-228` -- sibling endpoint pattern
- `backend/services/kill_switch.py:36` -- audit JSONL path
- `backend/services/paper_round_trips.py::pair_round_trips` -- round-trip pairing
- CLAUDE.md `Critical Rules` -- 30s BQ timeout; no_emojis
