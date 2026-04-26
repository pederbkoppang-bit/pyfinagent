---
step: phase-16.54
title: Sovereign two-hero balance -- shrink RedLineMonitor non-compact height
cycle_date: 2026-04-26
harness_required: true
verification: cd frontend && npx tsc --noEmit
research_brief: handoff/current/phase-16.54-research-brief.md
---

# Contract -- phase-16.54

## Step ID

`phase-16.54` -- "Sovereign two-hero balance: shrink RedLineMonitor
non-compact height". User flagged via screenshot 2026-04-26 15:31:38:
sovereign page has ~160px dead space below Alpha Leaderboard because
Red Line Monitor card is ~440px tall while Alpha Leaderboard is ~280px
naturally.

## Research-gate summary

Internal-only brief at `handoff/current/phase-16.54-research-brief.md`
(simple tier; pure-UI cycle precedent 16.43/16.46/16.47/16.48/16.49/16.52/16.53).
5 internal files inspected. `gate_passed: true`.

Decisive findings:
- RedLineMonitor chart container is `h-64` (256px) for non-compact
- Total card height ~440px (256 chart + 50 header + 30 footer + 48 BentoCard padding + 60 window-selector)
- AlphaLeaderboard natural height ~280px (header + 2 strategy rows + horizontal scrollbar)
- Difference ~160px = the operator-flagged dead space
- `frontend-layout.md` §4.6 mandates `min-h-[55svh]` only for the HOMEPAGE hero embed (uses compact=true), NOT for /sovereign route
- Compact branch (`h-72`, used by homepage) must NOT change
- Horizontal scroll on AlphaLeaderboard is a SEPARATE issue, out of scope

## Hypothesis

Reducing the non-compact chart height from `h-64` (256px) to `h-48`
(192px) reduces the RedLineMonitor card to ~376px, much closer to
AlphaLeaderboard's ~280px. Dead space below shrinks from ~160px to
~96px. Compact branch (homepage hero) preserved.

## Immutable success criteria

```
verification: cd frontend && npx tsc --noEmit
```

Plus operator visual confirmation at next /sovereign load: the
dead space below Alpha Leaderboard is materially reduced.

## Plan steps

1. Edit `frontend/src/components/RedLineMonitor.tsx` L107:
   - `className={compact ? "h-72" : "h-64"}` -> `className={compact ? "h-72" : "h-48"}`

2. Verify:
   - `cd frontend && npx tsc --noEmit` -- exit 0
   - `cd frontend && npm run lint` -- 0 errors

## References

- `handoff/current/phase-16.54-research-brief.md`
- `frontend/src/components/RedLineMonitor.tsx` L107 (the line being edited)
- `frontend/src/app/sovereign/page.tsx` L139-155 (the consumer; unchanged)
- `frontend/src/app/page.tsx` (homepage hero; uses compact=true via next/dynamic; unchanged)
- `.claude/rules/frontend-layout.md` §4.6 (sovereign two-hero pattern; min-h-[55svh] only for homepage)

## Out of scope

- Compact branch height (`h-72` for homepage hero) -- intentionally preserved
- Horizontal scroll on AlphaLeaderboard (separate issue; user did not flag)
- Grid-column proportions on `/sovereign` (lg:col-span-3 / lg:col-span-2 stays)
- Adding `items-start` to the grid (cards aren't being stretched; only intrinsic height differs)
- New tests (pure-UI; visual change verified by operator)
