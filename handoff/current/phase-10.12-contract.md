# Sprint Contract — phase-10.12 (HarnessSprintTile visual alignment)

**Step id:** 10.12 **Date:** 2026-04-21 **Tier:** simple **Harness-required:** true

## Why

User-reported visual regression: the phase-10.9 `HarnessSprintTile` uses the frontend.md BentoCard light-theme defaults (`bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 rounded-2xl p-6`). The rest of the Harness tab uses the project's custom navy tokens (`bg-navy-800/60`, `border-navy-700`, `text-slate-*`). Empty state renders as an oversized ~400px white void.

## Research-gate summary

Fresh researcher (simple): `handoff/current/phase-10.12-research-brief.md` — 6 in full, 13 URLs, gate_passed=true. Key finding: tailwind.config.js defines custom navy-900..navy-500 hex tokens — the tile must use these, not generic zinc. All 5 phase-10.9 test assertions target `data-*` attributes + `.textContent`, NEVER CSS class strings — styling changes are zero-risk to the test suite.

## Immutable success criteria (masterplan-verbatim)

Test: `cd frontend && npm run test -- --filter=HarnessSprintTile`

1. `existing_5_tests_still_pass` — all 5 phase-10.9 Vitest cases still pass (no behavior change)
2. `tile_uses_navy_palette_not_zinc` — `grep -c "bg-navy-" HarnessSprintTile.tsx > 0` AND `grep -c "zinc" HarnessSprintTile.tsx == 0`
3. `empty_state_not_oversized` — empty state container uses `py-8` (not `py-12`); icon `size={32}` (not `size={40}`)
4. `read_only_preserved` — phase-10.9's `read_only_no_mutation_controls` test still green (no `<button>`, `<input>`, etc.)

## Plan

1. Edit `frontend/src/components/HarnessSprintTile.tsx`:
   - Outer `<section>`: `rounded-xl border border-navy-700 bg-navy-800/60 p-5`
   - Empty state: `py-8` + icon `size={32}`
   - Inner sub-cards: `rounded-lg border border-navy-700/50 bg-navy-900/40 p-4`
   - Preserve ALL `data-section`/`data-cell`/`aria-label` attributes
   - Preserve ALL existing `text-*` color classes (they're already correct)
2. Run `cd frontend && npm run test -- --filter=HarnessSprintTile` — expect 5/5
3. Run `cd frontend && npx tsc --noEmit | grep -E "HarnessSprint"` — expect empty
4. Visual smoke: reload http://localhost:3000 Harness tab — confirm tile matches surrounding BentoCards
5. Spawn fresh Q/A
6. Log, flip, close task

## References

- `handoff/current/phase-10.12-research-brief.md`
- `frontend/src/components/HarnessSprintTile.tsx` (edit target)
- `frontend/src/components/BentoCard.tsx` (design token source)
- `frontend/tailwind.config.js` (navy palette definition)
- `.claude/rules/frontend-layout.md` §4 (summary card anatomy)

## Carry-forwards (out of scope)

- BentoCard.tsx itself has light-mode classes — might want to fix globally later
- If Vitest test infrastructure ever gets visual-regression snapshots, add one here (current tests are behavior-only)
