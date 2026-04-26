---
step: phase-16.56
title: AlphaLeaderboard wider -- swap sovereign grid proportions to 2+3
cycle_date: 2026-04-26
harness_required: true
verification: cd frontend && npx tsc --noEmit
research_brief: handoff/current/phase-16.56-research-brief.md
---

# Contract -- phase-16.56

## Step ID

`phase-16.56` -- "AlphaLeaderboard wider so the table doesn't horizontal-scroll".
Operator screenshot 2026-04-26 18:28:42 confirmed AlphaLeaderboard
table still scrolls horizontally even after the 16.55 height-fill fix.
Operator: "make Alpha Leaderboard wider so we dont have to scroll
within to see all the data."

## Research-gate summary

Internal-only brief at `handoff/current/phase-16.56-research-brief.md`.
gate_passed=true.

## Hypothesis

Swapping the sovereign two-hero grid proportions from `5 (3+2)` -- where
RedLine takes 60% and Alpha 40% -- to `5 (2+3)` gives Alpha 60% width
(50% more horizontal space). All 7 columns should fit without scroll
at typical laptop widths. Tightening cell `px-3` -> `px-2.5` adds
~14px more for safety margin.

## Immutable success criteria

```
verification: cd frontend && npx tsc --noEmit
```

Plus operator visual confirmation: AlphaLeaderboard table fits all 7
columns with no horizontal scrollbar at typical viewport widths.

## Plan steps

1. `frontend/src/app/sovereign/page.tsx`:
   - L140: `lg:col-span-3` -> `lg:col-span-2` (RedLine = 40%)
   - L148: `lg:col-span-2 h-full` -> `lg:col-span-3 h-full` (Alpha = 60%)

2. `frontend/src/components/AlphaLeaderboard.tsx`: replace_all `px-3 py-2.5` -> `px-2.5 py-2.5` (8 occurrences -- 7 cells + 1 header).

3. Verify: `cd frontend && npx tsc --noEmit && npm run lint`.

## References

- `handoff/current/phase-16.56-research-brief.md`
- `handoff/archive/phase-16.55/` (the height-fill prerequisite)
- `handoff/archive/phase-16.54/` (the prior shrink approach -- superseded)

## Out of scope

- RedLineMonitor chart sizing (h-64 retained per 16.55)
- AlphaLeaderboard column hiding / row truncation
- Compact branch / homepage hero
- New tests
