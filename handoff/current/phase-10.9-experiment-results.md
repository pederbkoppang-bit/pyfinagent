# Experiment Results — phase-10.9 (Harness-tab sprint-state tile)

**Step:** 10.9 (FINAL phase-10 step) **Date:** 2026-04-20

## What was done

1. Fresh researcher (moderate): 6 in full, 13 URLs, recency 2026, gate_passed=true. Brief at `handoff/current/phase-10.9-research-brief.md`. Critical resolution: `--filter=HarnessSprintTile` is handled by `frontend/scripts/run-test.mjs` which strips the prefix and passes the value as a positional substring to `vitest run`. File name MUST contain `HarnessSprintTile` as substring.
2. Contract authored at `handoff/current/phase-10.9-contract.md`.
3. Added `HarnessSprintWeekState` interface to `frontend/src/lib/types.ts` (after `HarnessValidation`, line 933):
   ```typescript
   export interface HarnessSprintWeekState {
     weekIso: string;
     thu: { batchId: string; candidatesKicked: number } | null;
     fri: { promotedIds: string[]; rejectedIds: string[] } | null;
     monthly: { sortinoDelta: number; approvalPending: boolean; approved: boolean } | null;
   }
   ```
4. Created `frontend/src/components/HarnessSprintTile.tsx` (~160 lines):
   - Props: `{ data: HarnessSprintWeekState | null }` — parent owns fetching; no `useEffect`
   - BentoCard pattern per `frontend.md` (rounded-2xl, zinc borders, dark:bg-zinc-900)
   - Empty state with `IconTimer` + guidance text when `data === null`
   - Weekly state (Thu + Fri) in a 2-col grid under `data-section="weekly-state"`
   - Monthly Sortino delta in its own card under `data-section="monthly-state"` with `data-cell="sortino-delta"`
   - Color coding: emerald (approved), amber (pending), slate (null/missing) — matches `frontend-layout.md` §9
   - All icons imported from `@/lib/icons.ts` only (no direct `@phosphor-icons/react` import)
   - **Zero `<button>`, `<input>`, `<select>`, `<textarea>`, `<form>` elements** — read-only by construction
5. Created `frontend/src/components/HarnessSprintTile.test.tsx` — 5 Vitest cases:
   - `tile_renders_weekly_state`
   - `tile_renders_monthly_sortino_delta`
   - `read_only_no_mutation_controls` — canonical `screen.queryAllByRole('button').toHaveLength(0)` guard plus `querySelectorAll` checks for `input/select/textarea/form`
   - `renders empty state when data is null`
   - `handles partial data (fri null, monthly null)`

## Verification (verbatim)

```
$ cd frontend && npm run test -- --filter=HarnessSprintTile

 RUN  v4.1.4 /Users/ford/.openclaw/workspace/pyfinagent/frontend

 Test Files  1 passed (1)
      Tests  5 passed (5)
   Start at  23:47:28
   Duration  905ms (transform 20ms, setup 26ms, import 547ms, tests 28ms, environment 219ms)

$ npx tsc --noEmit 2>&1 | grep -E "HarnessSprint|error TS"
(empty — no type errors)
```

## Success criteria (masterplan, immutable)

| # | Criterion | Status |
|---|---|---|
| 1 | `tile_renders_weekly_state` | PASS — `data-section="weekly-state"` present; `thu-candidates=128`, `fri-promoted-count=2` |
| 2 | `tile_renders_monthly_sortino_delta` | PASS — `data-cell="sortino-delta"` renders `+0.420` and `+0.350` for approved + pending |
| 3 | `read_only_no_mutation_controls` | PASS — `queryAllByRole('button') === []`; zero `input/select/textarea/form` in DOM |

## Key design decisions

- **Filename-matched substring:** `HarnessSprintTile.tsx` (not researcher's `SprintStateTile.tsx`) so `--filter=HarnessSprintTile` matches
- **Parent-owned fetching:** tile is a pure view component — no API, no state, no effects. Makes testing trivial and prevents accidental mutation paths.
- **Empty state present:** per `frontend.md` "never show blank space" rule; `data=null` renders guidance text
- **Color semantics:** emerald/amber/slate match dashboard convention (approved/pending/missing)

## Carry-forwards (out of scope)

- Wire the tile into `HarnessDashboard.tsx` — requires a backend fetcher that queries the phase-10.8 `harness_learning_log` for the current week; separate integration ticket
- Backend API endpoint `GET /api/harness/sprint-state?week=YYYY-Www` returning `HarnessSprintWeekState` — needs backend implementation
- Fix `HarnessDashboard.tsx` icon-import violation (imports from `@phosphor-icons/react` directly) — unrelated to this tile
