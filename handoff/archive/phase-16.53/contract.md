---
step: phase-16.53
title: Settings full-width content fix -- drop max-w-4xl from 3 tab grids
cycle_date: 2026-04-26
harness_required: true
verification: cd frontend && npx tsc --noEmit
research_brief: handoff/current/phase-16.53-research-brief.md
---

# Contract -- phase-16.53

## Step ID

`phase-16.53` -- "Settings full-width content fix". User flagged via
screenshot 2026-04-26 15:18:54: settings page wastes ~30% of viewport
width on the right side because tab content is constrained to
`max-w-4xl`. Net-new step under phase-16 continuation series.

## Research-gate summary

Internal-only brief at `handoff/current/phase-16.53-research-brief.md`
(simple tier; pure-UI cycle precedent 16.43/16.46/16.47/16.48/16.49/16.52).
5 internal files inspected. `gate_passed: true` on internal-only basis
(`.claude/rules/frontend{,-layout}.md` are the authoritative source;
user-screenshot evidence is unambiguous).

Decisive findings:
- 3 occurrences of `max-w-4xl` on grid wrappers in settings/page.tsx (L601 Models, L794 Cost & Weights, L978 Performance)
- No other page route uses `max-w-4xl` on tab content
- `frontend-layout.md` does NOT prescribe a max-width on tab content
- `max-w-fit` on the tab bar (L580) is correct -- keep it
- Fix: 3 single-line edits dropping `max-w-4xl` from the wrappers; cards naturally fill the scrollable zone

## Hypothesis

Removing `max-w-4xl` from the 3 grid wrappers in `settings/page.tsx`
allows the 2-column BentoCard layout to fill the full scrollable zone
width, eliminating the dead whitespace on the right.

## Immutable success criteria

```
verification: cd frontend && npx tsc --noEmit
```

End-of-cycle: `cd frontend && npm run lint` -> 0 errors.

Manual visual check (operator): settings page tab content fills the
full width (no large dead zone on the right at 1500px+ viewport).

## Plan steps

1. Edit `frontend/src/app/settings/page.tsx`:
   - L601: `grid max-w-4xl grid-cols-1 gap-6 lg:grid-cols-2` -> `grid grid-cols-1 gap-6 lg:grid-cols-2`
   - L794: same
   - L978: same

2. Verify:
   - `cd frontend && npx tsc --noEmit` -- exit 0
   - `cd frontend && npm run lint` -- 0 errors

## References

- `handoff/current/phase-16.53-research-brief.md`
- `.claude/rules/frontend-layout.md` §1 (canonical shell -- already in place from 16.52), §4 (metric grids fill natural width)
- `frontend/src/app/settings/page.tsx` (the file being edited)
- 16.52 cycle: `handoff/archive/phase-16.52/` (which established the two-zone shell that exposed this max-w-4xl as the next bottleneck)

## Out of scope

- The `max-w-fit` on the tab bar (L580) -- intentionally kept; pills hug content
- `max-w-4xl` on individual cards inside the grids -- those are fine; only the WRAPPER is wrong
- Adding icons to SETTINGS_TABS (deferred from 16.52)
- Reports tab bg color drift (deferred from 16.52)
- New tests (pure-UI; visual change verified by operator at the next browser load)
