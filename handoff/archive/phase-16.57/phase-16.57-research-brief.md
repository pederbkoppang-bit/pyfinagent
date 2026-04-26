# Research Brief: phase-16.57 -- Revert sovereign grid to RedLine 60% / Alpha 40%

Tier: **simple**. Internal-only gate. Trivial revert of 16.56's swap.

## Problem

Operator post-16.56 deploy screenshot 2026-04-26: "thats to wide Red
Line Monitor make red line montiner tak 60 % and aplha the rest"

Translation: revert the grid proportions. RedLineMonitor should be 60%
(lg:col-span-3), AlphaLeaderboard should be 40% (lg:col-span-2). The
table's horizontal scroll (the original 16.56 motivation) is implicitly
accepted by the operator -- they prefer the wider RedLine over a
non-scrolling table.

## Decisive findings

- Revert `frontend/src/app/sovereign/page.tsx` L140 + L148 to pre-16.56
- Keep cell padding `px-2.5` from 16.56 (no harm, tiny visual improvement)
- Keep all 16.55 changes (height-fill via h-full + flex layout) -- Alpha card still matches RedLine height
- Keep RedLineMonitor h-64 (16.55 restoration)

## Plan

1. `frontend/src/app/sovereign/page.tsx`:
   - L140: `lg:col-span-2` -> `lg:col-span-3` (RedLine = 60%)
   - L148: `lg:col-span-3 h-full` -> `lg:col-span-2 h-full` (Alpha = 40%)

Verify: `cd frontend && npx tsc --noEmit`.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": true,
  "internal_files_inspected": 1,
  "report_md": "handoff/current/phase-16.57-research-brief.md",
  "gate_passed": true
}
```
