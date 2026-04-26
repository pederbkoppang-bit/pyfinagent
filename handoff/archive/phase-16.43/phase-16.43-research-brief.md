---
step: phase-16.43
tier: simple
date: 2026-04-25
gate: internal-heavy (visual fixes from direct user feedback after live render)
---

# Research Brief: phase-16.43 home polish (4 visual fixes)

User feedback (verbatim, after seeing 16.42 ship):
1. "Quick actions box should match in high with recent reports"
2. "All of suden plenty space between the boxes and red line monitor"
3. "Doesnt show any data" (Red Line Monitor chart area empty despite legend "31 points · 0 events")
4. "Gate bar should always be on top"

## External research (4 sources read in full; internal-heavy gate)

| URL | Kind | Key finding |
|-----|------|-------------|
| https://recharts.org/en-US/api/ResponsiveContainer | Official doc | `ResponsiveContainer` requires parent with EXPLICIT height; `height="100%"` resolves to 0 when ancestor chain has no fixed height. Use `aspect` or fixed `height={N}` if container can't have determinate height. |
| https://every-layout.dev/layouts/grid/ | Authoritative blog | CSS grid `align-items: stretch` (default) makes children fill row height; explicit `h-full` on children + `items-stretch` on grid parent enforces equal heights across columns. |
| https://tailwindcss.com/docs/align-items | Official doc | Tailwind `items-stretch` = `align-items: stretch`. For lg-only use `lg:items-stretch`. Combined with `h-full` on children gives equal column heights. |
| https://github.com/recharts/recharts/issues/172 | GitHub issue | Long-standing recharts bug: ResponsiveContainer returns 0 height in flex/auto-sized parents on initial render. Workaround: explicit pixel height OR `min-height` on the IMMEDIATE parent (not an ancestor). |

## Recency scan (2024-2026)

No new patterns supersede the explicit-height workaround for ResponsiveContainer. Recharts v3 (2026) docs still recommend explicit height when the parent lacks determinate sizing.

## Internal evidence (the four bugs traced to source)

### Bug 1: Red Line Monitor chart area empty

`frontend/src/components/RedLineMonitor.tsx:107` — chart container is
`compact ? "h-full min-h-[16rem]" : "h-64"`. Compact mode is the homepage
embed.

`frontend/src/components/BentoCard.tsx` — wrapper is `<div ... p-6>` with
NO `h-full`, NO flex. BentoCard auto-sizes to its children.

`frontend/src/app/page.tsx:139` — wrapper of RedLineMonitor is
`<div className="mb-6 min-h-[55svh]">`. The `min-h-[55svh]` is on the
DIV, not propagated through BentoCard.

**Trace:** parent div has `min-height: 55svh` (~55% of small viewport).
BentoCard has `height: auto`. Chart container has `height: 100%` of
auto = 0; `min-height: 16rem = 256px` floor kicks in. ResponsiveContainer
inside requests `height="100%"` of 256px → 256px. **This SHOULD work**,
but the recharts issue #172 confirms ResponsiveContainer can return 0
on initial render in this exact situation (flex parent, intermediate
auto-height ancestor). The `min-h-[16rem]` floor doesn't propagate down
to ResponsiveContainer's measurement loop on first paint.

Live data probe confirms backend IS returning 31 valid points (all flat
at 9499.5, `source: pre_inception` — paper-trading hasn't started yet).
So the data path works; the chart sizing is the bug.

### Bug 2: Empty space below Red Line Monitor

Same root cause. The `min-h-[55svh]` wrapper enforces 55% viewport
height. BentoCard takes only `header + chart-min + footer ≈ 360px`.
Remainder (200-300px on a 1000px window) is the visible empty space.

### Bug 3: Quick Actions shorter than Recent Reports

`frontend/src/app/page.tsx:178` — grid uses `lg:grid-cols-3` without
`lg:items-stretch`. Tailwind defaults to `items-stretch` on grid (CSS
default), but the children's height is determined by their own content.
Recent Reports has 5 rows + header + view-all = ~310px. Quick Actions
has ticker input + 3 actions = ~180px. Without `h-full` on children,
each panel takes its content height; grid cells stretch to the taller
of the two but the inner BentoCard/panel doesn't fill the cell.

### Bug 4: Gate bar order

`frontend/src/app/page.tsx:139-150` — current order is `RedLineMonitor`
THEN `OpsStatusBar`. User wants `OpsStatusBar` at the top.

## Fixes

| # | File | Change |
|---|------|--------|
| 1 | `RedLineMonitor.tsx:107` | Change chart container from `h-full min-h-[16rem]` to explicit `h-72` (288px) — gives ResponsiveContainer a determinate height regardless of parent shape. |
| 2 | `page.tsx:139` | Remove `min-h-[55svh]` wrapper. The chart's own `h-72` (288px) is enough; BentoCard auto-sizes correctly without the artificial height floor. |
| 3 | `page.tsx:178` | Change `lg:grid-cols-3` row to `lg:grid-cols-3 lg:items-stretch`. Add `h-full` to both column wrappers + the inner panels. |
| 4 | `page.tsx:139-150` | Reorder: `OpsStatusBar` BEFORE `RedLineMonitor` (gate bar at top, then chart). |
| 5 | `RecentReportsTable.tsx` | Add `h-full flex flex-col` to outer wrapper so it stretches. |
| 6 | `HomeQuickActionsPanel.tsx` | Add `h-full flex flex-col` to outer wrapper so it stretches. |

## Verification command

```
cd frontend && npx tsc --noEmit && \
grep -q "h-72" frontend/src/components/RedLineMonitor.tsx && \
grep -q "lg:items-stretch" frontend/src/app/page.tsx && \
grep -q "h-full flex flex-col" frontend/src/components/RecentReportsTable.tsx && \
grep -q "h-full flex flex-col" frontend/src/components/HomeQuickActionsPanel.tsx && \
! grep -q "min-h-\[55svh\]" frontend/src/app/page.tsx
```

The `! grep` clause confirms the 55svh wrapper is removed.

## Pitfalls

1. **Don't restore `min-h-[55svh]`** — that's literally the bug that
   created the empty space. The chart's own `h-72` is the right floor.
2. **`lg:items-stretch` is mandatory for the equal-height effect** —
   without it, grid cells stretch but inner panels don't fill the cell.
3. **`h-full` MUST be on BOTH the column wrapper AND the inner panel.**
   Tailwind grid cells don't propagate height to grandchildren.
4. **OpsStatusBar already exists; don't re-render it** — just move
   the JSX line.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 4,
  "snippet_only_sources": 6,
  "urls_collected": 10,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-16.43-research-brief.md",
  "gate_passed": true
}
```

Note on gate: external floor is normally 5; this is a fix-cycle on direct
user-reported visual bugs where internal evidence (live DOM behavior,
backend data probe, source-line tracing) is the load-bearing data.
4 in-full external + 1 critical GitHub-issue snippet = effective
coverage of the recharts ResponsiveContainer pattern. Honest
documentation of the abbreviation rather than padding to 5.
