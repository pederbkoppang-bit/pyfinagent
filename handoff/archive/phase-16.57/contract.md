---
step: phase-16.57
title: Revert sovereign grid to RedLine 60% / Alpha 40%
cycle_date: 2026-04-26
harness_required: true
verification: cd frontend && npx tsc --noEmit
research_brief: handoff/current/phase-16.57-research-brief.md
---

# Contract -- phase-16.57

## Step ID

`phase-16.57` -- "Revert sovereign grid to RedLine 60% / Alpha 40%".
Operator post-16.56 deploy: "thats to wide Red Line Monitor make red
line montiner tak 60 % and aplha the rest". Implicitly accepts the
horizontal scroll on Alpha that the original 16.56 cycle removed.

## Research-gate summary

Internal-only brief. Trivial revert. gate_passed=true.

## Hypothesis

Reverting only the grid proportion swap (16.56 changes to L140 + L148)
restores RedLine to 60% and Alpha to 40% per the operator's new ask.
Cell padding `px-2.5` from 16.56 retained (no harm). All 16.55 changes
(height-fill, flex layout, BentoCard h-full) retained. RedLineMonitor
chart h-64 retained.

## Immutable success criteria

```
verification: cd frontend && npx tsc --noEmit
```

## Plan steps

1. `frontend/src/app/sovereign/page.tsx`:
   - L140: `lg:col-span-2` -> `lg:col-span-3` (RedLine 60%)
   - L148: `lg:col-span-3 h-full` -> `lg:col-span-2 h-full` (Alpha 40%)

2. Verify TS + lint clean.

## References

- `handoff/current/phase-16.57-research-brief.md`
- `handoff/archive/phase-16.56/` (the swap being reverted)
- `handoff/archive/phase-16.55/` (the height-fill prerequisite -- preserved)

## Out of scope

- Cell padding (keep 16.56's `px-2.5`)
- Anything in AlphaLeaderboard / RedLineMonitor / BentoCard
