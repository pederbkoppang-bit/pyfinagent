---
step: phase-16.55
title: Sovereign two-hero balance round 2 -- AlphaLeaderboard fills RedLine height
cycle_date: 2026-04-26
harness_required: true
verification: cd frontend && npx tsc --noEmit
research_brief: handoff/current/phase-16.55-research-brief.md
---

# Contract -- phase-16.55

## Step ID

`phase-16.55` -- "Sovereign two-hero balance round 2: Alpha Leaderboard
fills Red Line height". REVERSED direction from earlier in this cycle:
operator clarified the new ask AFTER initial implementation: "Red Line
Monitor could be bigg but Alpha Leaderboard should just match the
hight of Red Line Monitor."

So the prior 16.54 shrink (h-64 -> h-48) AND the in-flight 16.55 shrink
(h-48 -> h-40) are BOTH reverted to the original `h-64` (256px). Instead,
AlphaLeaderboard is made to fill its grid cell so its card height
matches RedLineMonitor's (no dead space inside the grid cell either).

## Research-gate summary

Internal-only brief at `handoff/current/phase-16.55-research-brief.md`.
The brief was written for the SHRINK direction; this contract reflects
the operator's clarification mid-cycle. Findings still apply (4 internal
files inspected). `gate_passed: true`.

Key revised findings:
- Operator wants Red Line BIGGER (or at least at original h-64 size), not smaller
- Alpha Leaderboard's CARD should fill the grid cell to match Red Line's height
- The TABLE inside should scroll vertically when content overflows (defense for many-strategy future state)

## Hypothesis

Three coordinated changes will eliminate the dead space WITHOUT shrinking Red Line:

1. **Restore RedLineMonitor non-compact** to original `h-64` (256px chart). Card naturally fills its column.
2. **Wrap AlphaLeaderboard's grid cell** with `h-full` so the inner card knows its target height.
3. **Restructure AlphaLeaderboard interior** to be a flex column (`h-full flex flex-col`) with the table area as `flex-1 overflow-auto scrollbar-thin` -- the card now FILLS its grid cell and any future table overflow scrolls within the card.

## Immutable success criteria

```
verification: cd frontend && npx tsc --noEmit
```

Plus operator visual confirmation: AlphaLeaderboard card extends to the
SAME bottom edge as RedLineMonitor card; no dead space below either card
in the two-hero row.

## Plan steps (FINAL, after operator clarification)

1. `frontend/src/components/RedLineMonitor.tsx` L107: revert `h-40` -> `h-64` (original).

2. `frontend/src/app/sovereign/page.tsx`:
   - Remove `items-start` from L139 (let cards stretch naturally).
   - Add `h-full` to the AlphaLeaderboard wrapper div: `<div className="lg:col-span-2 h-full">`.

3. `frontend/src/components/AlphaLeaderboard.tsx`:
   - Change `<BentoCard>` -> `<BentoCard className="flex h-full flex-col">`
   - Change table container: `className="overflow-x-auto scrollbar-thin"` -> `className="flex-1 overflow-auto scrollbar-thin"`

4. Verify:
   - `cd frontend && npx tsc --noEmit` -- exit 0
   - `cd frontend && npm run lint` -- 0 errors

## References

- `handoff/current/phase-16.55-research-brief.md` (internal-only)
- `handoff/archive/phase-16.54/` (the shrink approach the operator rejected)
- `frontend/src/components/BentoCard.tsx` -- already accepts `className` prop, no shim needed
- `.claude/rules/frontend.md` "no equal-height rows mixing short and tall widgets" -- here we're explicitly making BOTH cards equal height by filling the shorter one, NOT stretching the taller one

## Out of scope

- Compact branch (`h-72` for homepage hero) -- preserved
- RedLineMonitor going BIGGER than `h-64` (operator said "could be" big -- treating "original size" as fine)
- Horizontal scroll on AlphaLeaderboard (separate; widening the column would help but column-width changes break grid proportions)
- New tests
