---
step: phase-16.43
title: Home polish -- gate bar on top, equal-height columns, RedLine chart sizing
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/src/components/RedLineMonitor.tsx (chart container h-full -> h-72)
  - frontend/src/app/page.tsx (reorder OpsStatusBar to top, remove min-h-[55svh], add lg:items-stretch + h-full)
  - frontend/src/components/RecentReportsTable.tsx (h-full flex flex-col)
  - frontend/src/components/HomeQuickActionsPanel.tsx (h-full flex flex-col)
---

# Sprint Contract -- phase-16.43

Tight follow-up cycle on phase-16.42 for 4 user-reported visual bugs.

## User feedback (verbatim)

1. "Quick actions box should match in high with recent reports"
2. "All of suden plenty space between the boxes and red line monitor"
3. "Doesnt show any data" (Red Line Monitor empty despite "31 points")
4. "Gate bar should always be on top"

## Concrete fixes

### Fix 1: gate bar at top (Bug #4)
In `page.tsx`, move `<OpsStatusBar />` to the FIRST position in the
scrollable content zone, BEFORE `<RedLineMonitor>`.

### Fix 2: chart sizing (Bug #3)
In `RedLineMonitor.tsx:107`, change chart container className from
`compact ? "h-full min-h-[16rem]" : "h-64"` to
`compact ? "h-72" : "h-64"`. Explicit pixel height (288px) for
ResponsiveContainer regardless of parent shape (recharts issue #172).

### Fix 3: remove empty-space floor (Bug #2)
In `page.tsx:139`, remove `min-h-[55svh]` from the RedLineMonitor
wrapper div. Replace `<div className="mb-6 min-h-[55svh]">` with
`<div className="mb-6">`. The chart's own `h-72` is sufficient.

### Fix 4: equal-height grid (Bug #1)
In `page.tsx:178`, change the grid wrapper:
`<div className="grid grid-cols-1 gap-6 lg:grid-cols-3 lg:items-stretch">`
Both column wrappers get `h-full`:
`<div className="lg:col-span-2 h-full">` and
`<div className="lg:col-span-1 h-full">`.

Then in `RecentReportsTable.tsx` outer wrapper, add `h-full flex flex-col`
to the root div so it stretches AND its inner table area flexes.

Same in `HomeQuickActionsPanel.tsx` outer wrapper.

## Success Criteria (verbatim, immutable)

```
cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
npx tsc --noEmit && \
grep -q "compact ? \"h-72\"" src/components/RedLineMonitor.tsx && \
grep -q "lg:items-stretch" src/app/page.tsx && \
grep -q "h-full flex flex-col" src/components/RecentReportsTable.tsx && \
grep -q "h-full flex flex-col" src/components/HomeQuickActionsPanel.tsx && \
! grep -q "min-h-\[55svh\]" src/app/page.tsx
```

Plus:
- `gate_bar_first`: in page.tsx scrollable zone, OpsStatusBar appears
  BEFORE RedLineMonitor (textual order in source).
- `tsc_clean`: exit 0.
- `lint_clean`: 0 phosphor warnings.
- `live_chart_renders`: with the fix, the live chart on /dashboard
  shows axes + grid + line (manual user verification).

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. OpsStatusBar source-textual position is BEFORE RedLineMonitor
   in page.tsx.
3. Chart container in RedLineMonitor is `h-72` (288px) when compact.
4. Grid uses `lg:items-stretch` AND both column wrappers have `h-full`.
5. Table + Panel outer wrappers have `h-full flex flex-col`.
6. NO `min-h-[55svh]` remains in page.tsx.
7. tsc + lint clean, no new errors.
8. No backend code touched.
