---
step: phase-16.57
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/app/sovereign/page.tsx (revert grid: 2+3 -> 3+2)
---

# Experiment Results -- phase-16.57

## What was done

Operator post-16.56 deploy reversed the grid proportion ask: "thats to
wide Red Line Monitor make red line montiner tak 60 % and aplha the
rest". Reverted only the grid column-span swap from 16.56. All other
16.55/16.56 changes (height-fill, flex layout, BentoCard h-full,
cell padding `px-2.5`) preserved.

## Deliverable

### `frontend/src/app/sovereign/page.tsx` (L140 + L148)

```tsx
- <div className="lg:col-span-2">          # was 16.56 (RedLine 40%)
+ <div className="lg:col-span-3">          # restored (RedLine 60%)
    ... RedLineMonitor ...
- <div className="lg:col-span-3 h-full">   # was 16.56 (Alpha 60%)
+ <div className="lg:col-span-2 h-full">   # restored (Alpha 40%)
    ... AlphaLeaderboard ...
```

`h-full` on Alpha wrapper preserved (16.55 carry-over). Cell padding
`px-2.5` in AlphaLeaderboard preserved (16.56 carry-over).

## Verification (verbatim, immutable from masterplan)

```
$ cd frontend && npx tsc --noEmit
(exit 0; no output)
```

Grid revert confirmed via grep: L140 = `lg:col-span-3`, L148 = `lg:col-span-2 h-full`.

## Files touched

| Path | Action | Note |
|------|--------|------|
| `frontend/src/app/sovereign/page.tsx` | edit (L140 + L148) | grid col-span values reverted |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-16.57-research-brief.md` | created | -- |

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | RedLine restored to lg:col-span-3 (60%) | PASS |
| 2 | Alpha restored to lg:col-span-2 with h-full preserved (40%) | PASS |
| 3 | Cell padding px-2.5 from 16.56 retained | PASS (no change to AlphaLeaderboard.tsx this cycle) |
| 4 | tsc clean | PASS |
| 5 | Operator visual confirmation | DEFERRED |

## Honest disclosures

1. **Tradeoff reintroduced.** At 40% width, Alpha's 7 columns will
   horizontal-scroll again. Operator implicitly accepted this -- they
   prefer wider RedLine over a non-scrolling table. Cell padding
   `px-2.5` (saved from 16.56) gives ~14px back vs the original
   `px-3` so the scroll appears slightly later.

2. **Smallest possible scope.** Just the 2-line revert. No other files
   modified.

3. **All 16.55 height/flex work preserved** -- Alpha card still fills
   row height and the table still has flex-1 overflow-auto for
   vertical scrolling on many-strategy cases.

4. **Cycle-2 not needed.** First-pass clean.

## Closes

Net-new task #86 (UAT-16.57). Adds new step phase-16.57 to masterplan.

## Next

Spawn Q/A.
