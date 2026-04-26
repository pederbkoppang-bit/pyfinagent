---
step: phase-16.44
title: KPI scorecards under gate bar with comparison sub-text + Last/Next gate-bar segments
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/src/lib/kpiMetrics.ts (new, pure helpers)
  - frontend/src/app/page.tsx (reorder + new KPI tile shape with subText)
  - frontend/src/components/OpsStatusBar.tsx (Last + Next segments)
---

# Sprint Contract -- phase-16.44

## User feedback (verbatim)

1. "the scorecards should be right under gate"
2. "also have the same camarioson in the scorecard as the picture" (sub-text below value)
3. "gate bar is missing scheduler date last and next run"

## Concrete plan

### 1. `frontend/src/lib/kpiMetrics.ts` (new, ~85 LOC)

Pure helpers, all return `null` on insufficient data (NaN-safe):
- `dailyDelta(series)` -> `{dollars, pct} | null`
- `sharpe(navSeries, periodsPerYear=252)` -> `number | null`
- `sortino(navSeries, periodsPerYear=252)` -> `number | null`
- `maxDrawdownPct(navSeries)` -> `number | null`
- `categorizePositions(positions)` -> `{long, short, total}`

### 2. `frontend/src/app/page.tsx` reorder + KPI rewrite

New scrollable-zone order:
1. KillSwitchShortcut (invisible)
2. OpsStatusBar (gate bar) -- with `nextRunAt={ptStatus?.next_run}` prop
3. Error banner (conditional)
4. KPI grid (6 tiles with sub-text) -- moved here from below
5. Red Line Monitor
6. Two-column grid (Recent Reports + Quick Actions)

Each KPI tile gets a `subText` slot for the comparison line:
- NAV: value=fmtUsd(nav); subText=null
- P&L: value=daily$; subText=daily%
- VS SPY: value=alpha%; subText=`SPY ${benchmark}%`
- SHARPE: value=sharpe(redLineSeries) toFixed(2); subText=`Sortino ${sortino}`
- MAX DD: value=maxDD; subText=`bounded 8.0%` (8% is the kill-switch trailing-DD limit, a real config value from settings)
- POSITIONS: value=count; subText=`{long} long · {short} short`

All "—" when computation returns null (which it will today since the NAV series is flat pre-inception).

### 3. `frontend/src/components/OpsStatusBar.tsx`

Replace single `SchedulerSegment` with TWO segments:
- `LastSegment` -- uses `latestCycle?.started_at` + `formatRelativeTime` (the helper from 16.42). Shows "—" if no cycles have run yet.
- `NextSegment` -- the existing logic, now under its own component.

Both with Divider separators, NextSegment with `ml-auto` to right-align (existing behavior preserved).

## Success Criteria (verbatim, immutable)

```
cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
test -f src/lib/kpiMetrics.ts && \
npx tsc --noEmit && \
grep -q "nextRunAt={ptStatus" src/app/page.tsx && \
grep -q "kpiMetrics" src/app/page.tsx && \
grep -q "LastSegment\|Last run\|Last:" src/components/OpsStatusBar.tsx && \
grep -q "subText" src/app/page.tsx
```

Plus order check (KPI grid line < Red Line Monitor line < two-column grid line in page.tsx source).

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. Reorder verified via source-line awk: OpsStatusBar < KPI grid < RedLineMonitor < two-column grid.
3. KPI sub-text computed from real data, NOT hardcoded (grep ensure no literal 1.42 / 2.08 / -3.12 / 14 / "6 long" / "8 hedge").
4. `nextRunAt={ptStatus?.next_run}` actually passed to OpsStatusBar.
5. SchedulerSegment expanded to show LAST + NEXT (two segments).
6. `kpiMetrics.ts` helpers are PURE (no I/O, no React imports).
7. tsc + lint clean.
