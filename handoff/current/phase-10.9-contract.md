# Sprint Contract — phase-10.9 (Harness-tab sprint-state tile)

**Step id:** 10.9 **Date:** 2026-04-20 **Tier:** moderate **Harness-required:** true
**Final phase-10 step.**

## Why

phase-10.3-10.8 built the autoresearch sprint machinery backend. phase-10.9 is the one frontend tile that surfaces weekly+monthly sprint state on the phase-4.7 Harness tab. Read-only by construction — no mutation controls.

## Research-gate summary

Fresh researcher (moderate): `handoff/current/phase-10.9-research-brief.md` — 6 in full, 13 URLs, recency, gate_passed=true.

Key findings:
- **`--filter=HarnessSprintTile` resolution:** `frontend/scripts/run-test.mjs:22` strips the `--filter=` prefix and passes the value as a positional substring to `vitest run`. Vitest 4.x uses positional filename substring as its filter mechanism. No tooling mismatch. Test filename must CONTAIN `HarnessSprintTile` as substring.
- **Test stack:** Vitest 4.1.4 + React Testing Library + jsdom. `cleanup()` in `afterEach` is the existing convention.
- **Component path:** `frontend/src/components/HarnessSprintTile.tsx` (flat, sibling to `HarnessDashboard.tsx`). Name overrides researcher's `SprintStateTile` suggestion to ensure the `--filter` substring matches.
- **Icons:** import from `@/lib/icons.ts` only (frontend.md rule). `HarnessDashboard.tsx` violates this but we don't replicate the mistake.

## Immutable success criteria (masterplan-verbatim)

Test command: `cd frontend && npm run test -- --filter=HarnessSprintTile`

1. `tile_renders_weekly_state` — renders `data-section="weekly-state"` with Thu batch + Fri promotion details when `data.thu` and `data.fri` are non-null
2. `tile_renders_monthly_sortino_delta` — renders `data-cell="sortino-delta"` with the numeric delta when `data.monthly` is non-null
3. `read_only_no_mutation_controls` — `screen.queryAllByRole('button')` returns `[]`; no `<input>`, `<select>`, or `<textarea>`

## Plan

1. Add `HarnessSprintWeekState` interface to `frontend/src/lib/types.ts` (after line 931).
2. Create `frontend/src/components/HarnessSprintTile.tsx`:
   - Props `{ data: HarnessSprintWeekState | null }` — parent owns fetching; no `useEffect`
   - Uses BentoCard pattern per frontend.md
   - Icons from `@/lib/icons.ts` only
   - Empty state: "No sprint activity yet" when `data === null`
   - Data sections marked with `data-section="weekly-state"`, `data-section="monthly-state"`, `data-cell="sortino-delta"`
   - Colors: green for `approved`, amber for `approval_pending`, slate for `null`/missing
   - No `<button>`, no form inputs
3. Create `frontend/src/components/HarnessSprintTile.test.tsx` with ≥3 tests mapping to success_criteria verbatim + edge cases (null data, partial data).
4. Run: `cd frontend && npm run test -- --filter=HarnessSprintTile`
5. Spawn fresh Q/A.
6. Log, flip, close task #69.

## References

- `handoff/current/phase-10.9-research-brief.md`
- `frontend/src/components/AutoresearchLeaderboard.tsx` + `.test.tsx` (existing pattern to mirror)
- `frontend/src/components/HarnessDashboard.tsx` (sibling context for the tile)
- `frontend/src/lib/types.ts:928-931` (insertion point for new type)
- `frontend/src/lib/icons.ts` (icon source)
- `.claude/rules/frontend.md` + `.claude/rules/frontend-layout.md` (BentoCard, empty-state, scrollbar, read-only conventions)

## Carry-forwards (out of scope)

- Wire the tile into `HarnessDashboard.tsx` (backend fetcher + parent container) — separate integration step
- Backend API endpoint that returns `HarnessSprintWeekState` from `harness_learning_log` (phase-10.8 sink) — deferred to go-live ticket
- Fix `HarnessDashboard.tsx` icon-import violation (separate ticket)
