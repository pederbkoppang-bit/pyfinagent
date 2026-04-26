---
step: phase-16.52
title: UX audit pass C -- settings two-zone refactor + tab pattern alignment + backtest banner relocation
cycle_date: 2026-04-26
harness_required: true
verification: cd frontend && npx tsc --noEmit
research_brief: handoff/current/phase-16.52-research-brief.md
---

# Contract -- phase-16.52

## Step ID

`phase-16.52` -- "UX audit pass C: settings deep-dive + tab-pattern regression". Net-new step under phase-16 (UAT continuation series).

## Research-gate summary

Internal-heavy brief at `handoff/current/phase-16.52-research-brief.md`
(simple tier). 10 internal files inspected (rules + 7 page routes + shared
ReportTabs.tsx). Per established pure-UI cycle precedent (16.43, 16.46,
16.47, 16.48, 16.49), external research not required: project rules
files (`.claude/rules/frontend.md` + `frontend-layout.md`) are the
authoritative source.

**Key finding:** the settings page (the user's named "canonical
reference") does NOT actually follow the two-zone shell that became
canonical in 16.48/16.49 -- it's the outlier. 7 violations total.
Severity:
- 1 SEVERE (settings single-zone — tabs scroll off-screen)
- 2 MODERATE (settings loading-state same single-zone bug; SETTINGS_TABS missing icon field — defer)
- 1 MINOR (backtest ingestResult banner in fixed-header zone — should be in scrollable)
- 3 COSMETIC (active-tab color mismatch; max-w-fit; reports tab bar bg color drift -- defer all 3)

## Hypothesis

Refactoring `settings/page.tsx` to the canonical two-zone shell (header
+ tab bar in `flex-shrink-0`; tab content in `flex-1 overflow-y-auto
scrollbar-thin`) AND moving `backtest/page.tsx` ingestResult banner
into the scrollable zone makes settings a true canonical reference and
closes the only remaining 16.49 residual violation.

## Immutable success criteria

```
verification: cd frontend && npx tsc --noEmit
```

(End-of-cycle: also run `cd frontend && npm run lint` for ESLint
compliance, but TS clean is the gate.)

Manual visual check: settings page header + tab bar stay pinned when
scrolling tab content (same pattern as reports/backtest/agents).

## Plan steps

1. Refactor `frontend/src/app/settings/page.tsx` (L534-1242):
   - Wrap with two-zone shell: `<main className="flex flex-1 flex-col overflow-hidden">`
   - Move header (L538-565) + tab bar (L567-582) into `flex-shrink-0` zone
   - Move tab content (L584-end) into `flex-1 overflow-y-auto scrollbar-thin` zone
   - Apply same shell to loading early-return (L509-532)
   - Fix active-tab color from `bg-slate-700 text-slate-100` to `bg-sky-500/10 text-sky-400` (per canonical pattern)

2. Refactor `frontend/src/app/backtest/page.tsx`:
   - Move `ingestResult` banner (L687-704 per research brief) from fixed-header zone to scrollable content zone

3. Verify:
   - `cd frontend && npx tsc --noEmit` -- exit 0
   - `cd frontend && npm run lint` -- exit 0

## References

- `handoff/current/phase-16.52-research-brief.md` (research gate)
- `.claude/rules/frontend-layout.md` §1 (canonical shell), §3 (page anatomy), §5 (tab bar canonical color), §New-Page-Template (skeleton)
- `.claude/rules/frontend.md` (scrollbar-thin, no emoji, BentoCard)
- `frontend/src/app/reports/page.tsx` (reference implementation -- same pattern target)
- `frontend/src/app/settings/page.tsx` L509-1242 (the file being refactored)

## Out of scope

- Adding `icon` field to SETTINGS_TABS (cosmetic, defer)
- Changing reports tab bar bg color from navy-800/60 to slate-800/50 (cosmetic, defer)
- Removing `max-w-fit` on settings tab bar (harmless, defer)
- Refactoring shared ReportTabs.tsx component (not currently used by any page route; deferred)
- Adding new tests (this is pure-UI; verification is `npx tsc --noEmit`)
