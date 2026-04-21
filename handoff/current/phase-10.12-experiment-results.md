# Experiment Results — phase-10.12 (HarnessSprintTile visual alignment)

**Step:** 10.12 **Date:** 2026-04-21

## What was done

1. Fresh researcher (simple): 6 in full, 13 URLs, gate_passed=true. Brief at `handoff/current/phase-10.12-research-brief.md`. Confirmed the app uses custom `navy-*` hex tokens (defined in `tailwind.config.js`), NOT generic Tailwind zinc. All 5 phase-10.9 test assertions target `data-*` attrs + `.textContent` — zero tests touch CSS class strings, so styling changes are zero-risk.
2. Contract authored at `handoff/current/phase-10.12-contract.md`.
3. Edited `frontend/src/components/HarnessSprintTile.tsx`:
   - Outer `<section>` (both empty and data paths): `rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900` → `rounded-xl border border-navy-700 bg-navy-800/60 p-5`
   - Empty-state container: `py-12` → `py-8`; icon `size={40}` → `size={32}`
   - All three inner sub-cards (Thu, Fri, Monthly): `rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950/40` → `rounded-lg border border-navy-700/50 bg-navy-900/40 p-4`
   - Preserved ALL `data-section`, `data-cell`, `data-week-iso`, `aria-label` attributes
   - Preserved ALL text color classes (`text-slate-*`, `text-emerald-400`, `text-amber-400`, `text-rose-*`) — already correct
   - No structural changes; no behavior changes; no new or removed DOM elements
4. Did NOT edit `HarnessSprintTile.test.tsx` — all 5 cases still pass as-is.

## Verification (verbatim)

```
$ grep -c zinc frontend/src/components/HarnessSprintTile.tsx
0

$ grep -c "navy-" frontend/src/components/HarnessSprintTile.tsx
5

$ grep -c "py-8" frontend/src/components/HarnessSprintTile.tsx
1

$ grep -c "size={32}" frontend/src/components/HarnessSprintTile.tsx
1

$ cd frontend && npm run test -- --filter=HarnessSprintTile
 RUN  v4.1.4

 Test Files  1 passed (1)
      Tests  5 passed (5)
   Start at  17:32:35
   Duration  1.27s

$ cd frontend && npx tsc --noEmit 2>&1 | grep -E "HarnessSprint"
(empty — no type errors)
```

## Success criteria (masterplan, immutable)

| # | Criterion | Status |
|---|---|---|
| 1 | `existing_5_tests_still_pass` | PASS — 5/5 Vitest cases (tile_renders_weekly_state, tile_renders_monthly_sortino_delta, read_only_no_mutation_controls, empty state, partial data) |
| 2 | `tile_uses_navy_palette_not_zinc` | PASS — `zinc` count = 0; `navy-` count = 5 |
| 3 | `empty_state_not_oversized` | PASS — `py-8` (was py-12); icon `size={32}` (was 40) |
| 4 | `read_only_preserved` | PASS — the `queryAllByRole('button')` test in phase-10.9 still green; no buttons/inputs/forms added |

## Class-change audit

| Element | Before | After |
|---|---|---|
| Outer `<section>` (×2 branches) | `rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900` | `rounded-xl border border-navy-700 bg-navy-800/60 p-5` |
| Empty-state centering | `py-12` | `py-8` |
| Empty-state icon | `size={40}` | `size={32}` |
| Thu sub-card | `rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950/40` | `rounded-lg border border-navy-700/50 bg-navy-900/40 p-4` |
| Fri sub-card | (same as Thu) | (same) |
| Monthly sub-card | (same as Thu) | (same) |

Zero changes to: text color classes, grid layout (`grid-cols-1 sm:grid-cols-2`), DOM structure, `data-*` attributes, `aria-label`, icon weights, monthly color-state logic.

## Carry-forwards

- `BentoCard.tsx` itself uses the same light-theme defaults — affects other components too; separate ticket
- If Vitest ever gets visual-regression snapshots, add one for this tile
