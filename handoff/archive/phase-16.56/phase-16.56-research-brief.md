# Research Brief: phase-16.56 -- AlphaLeaderboard wider (no horizontal scroll)

Tier: **simple**. Internal-only gate per pure-UI cycle precedent.

## Problem (operator screenshot 2026-04-26 18:28:42)

After 16.55, AlphaLeaderboard CARD now matches RedLineMonitor height
(stretches to fill grid cell). But the TABLE inside still scrolls
horizontally because `lg:col-span-2` of `lg:grid-cols-5` = 40% width is
too narrow for 7 columns. Operator: "make Alpha Leaderboard wider so we
dont have to scroll within to see all the data."

## Decisive findings

1. **Current grid:** `lg:grid-cols-5` with `lg:col-span-3` (RedLine 60%) + `lg:col-span-2` (Alpha 40%)
2. **AlphaLeaderboard has 7 columns:** Strategy, Sharpe, DSR, PBO, Max DD, Status, Alloc % -- "Alloc %" is the one cut off
3. **Table cell padding:** `px-3 py-2.5` on every cell -- could tighten but won't be enough alone
4. **RedLine chart** is flat horizontal data (mostly stable NAV), so reducing its width to ~50% won't degrade readability noticeably
5. **Two coordinated changes are the right fix:**
   - Swap proportions: `grid-cols-5 (3+2)` -> `grid-cols-5 (2+3)` so Alpha gets 60% / RedLine 40%
   - Tighten table cell padding from `px-3` -> `px-2` for a few extra pixels per column
6. **Won't break responsive stacking** (`grid-cols-1` mobile fallback unchanged)

## Plan

1. `frontend/src/app/sovereign/page.tsx`:
   - L140: `lg:col-span-3` -> `lg:col-span-2` (RedLine becomes 40%)
   - L148: `lg:col-span-2 h-full` -> `lg:col-span-3 h-full` (Alpha becomes 60%)

2. `frontend/src/components/AlphaLeaderboard.tsx`: tighten cell padding on the 7 `<th>` and `<td>` rows from `px-3 py-2.5` -> `px-2.5 py-2.5` (saves ~1px per side per column = ~14px total table width).

Verify: `cd frontend && npx tsc --noEmit` + `npm run lint`.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": true,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/phase-16.56-research-brief.md",
  "gate_passed": true
}
```
