---
step: phase-16.55
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/components/RedLineMonitor.tsx (revert: h-48/h-40 -> h-64 original)
  - frontend/src/app/sovereign/page.tsx (add h-full to AlphaLeaderboard wrapper, remove items-start)
  - frontend/src/components/AlphaLeaderboard.tsx (BentoCard h-full flex-col + table flex-1 overflow-auto)
---

# Experiment Results -- phase-16.55

## What was done -- REVERSED DIRECTION mid-cycle

Operator clarified mid-implementation: "Red Line Monitor could be bigg
but Alpha Leaderboard should just match the hight of Red Line Monitor."

Initial cycle plan (shrink Red Line further to h-40 + add items-start)
was REVERSED in favor of:
- Restore Red Line to its original height (`h-64`)
- Make Alpha Leaderboard CARD fill its grid cell to match Red Line's height
- Make the table inside Alpha Leaderboard scroll vertically when content overflows

Net effect: NO MORE DEAD SPACE in the two-hero row, without shrinking Red Line.

## Deliverables

### `frontend/src/components/RedLineMonitor.tsx` (L107)

```tsx
- className={compact ? "h-72" : "h-40"}    # was h-48 in 16.54, then h-40 mid-16.55
+ className={compact ? "h-72" : "h-64"}    # restored to original
```

### `frontend/src/app/sovereign/page.tsx` (L139)

Removed `items-start` (added earlier in this cycle, now reversed) +
added `h-full` to the AlphaLeaderboard wrapper div:

```tsx
- <div className="mb-6 grid grid-cols-1 items-start gap-4 lg:grid-cols-5">
+ <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-5">
    <div className="lg:col-span-3"> ... RedLineMonitor ... </div>
-   <div className="lg:col-span-2"> ... AlphaLeaderboard ... </div>
+   <div className="lg:col-span-2 h-full"> ... AlphaLeaderboard ... </div>
```

### `frontend/src/components/AlphaLeaderboard.tsx` (L146 + L190)

```tsx
- <BentoCard>
+ <BentoCard className="flex h-full flex-col">

- <div data-testid="alpha-leaderboard" className="overflow-x-auto scrollbar-thin">
+ <div data-testid="alpha-leaderboard" className="flex-1 overflow-auto scrollbar-thin">
```

`overflow-auto` covers BOTH x and y -- existing horizontal scroll
preserved + new vertical scroll for tall content.

## Verification (verbatim, immutable from masterplan)

```
$ cd frontend && npx tsc --noEmit
(exit 0; no output)

$ cd frontend && npm run lint
0 errors, 34 warnings (all pre-existing)
```

## Files touched

| Path | Action | Note |
|------|--------|------|
| `frontend/src/components/RedLineMonitor.tsx` | revert | back to original h-64 (256px chart) |
| `frontend/src/app/sovereign/page.tsx` | edit (L139) | removed items-start, added h-full to AlphaLeaderboard wrapper |
| `frontend/src/components/AlphaLeaderboard.tsx` | edit (L146, L190) | BentoCard h-full flex-col + table flex-1 overflow-auto |
| `handoff/current/contract.md` | rewrite (rolling) | reflects reversed direction |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-16.55-research-brief.md` | created | originally for shrink approach; findings still apply |

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | RedLineMonitor non-compact restored to h-64 (no shrink) | PASS |
| 2 | AlphaLeaderboard CARD fills its grid cell height | PASS (BentoCard h-full + grid-cell h-full wrapper) |
| 3 | AlphaLeaderboard TABLE scrolls inside the card when overflowing | PASS (flex-1 overflow-auto) |
| 4 | items-start removed (let cards stretch naturally) | PASS |
| 5 | Compact branch unchanged | PASS |
| 6 | tsc + lint clean | PASS |
| 7 | Operator visual confirmation | DEFERRED (operator will refresh) |

## Honest disclosures

1. **Mid-cycle reversal.** The operator's clarification arrived AFTER I had already implemented the SHRINK direction (h-40 + items-start). I reversed the changes within the same cycle (16.55) rather than spawning a new cycle. The harness handoff files (contract.md + experiment_results.md) reflect the FINAL state, not the intermediate.

2. **No shrink to Red Line.** Restored to `h-64` (256px chart, ~440px card). Operator said "could be bigg" so leaving at original size; if they want it even bigger in a follow-up, h-72 or h-80 are the next steps.

3. **AlphaLeaderboard table now scrolls vertically when needed.** Currently 2 strategies fit comfortably so no scroll appears. With 10+ strategies, the table will scroll within the card instead of pushing the card taller than RedLineMonitor.

4. **No regression risk on homepage.** Compact branch of RedLineMonitor unchanged (`h-72`). AlphaLeaderboard not used on homepage.

5. **Cycle-2 not needed.** Final state TS+lint clean.

## Closes

Net-new task #84 (UAT-16.55). Adds new step phase-16.55 to masterplan.

## Next

Spawn Q/A.
