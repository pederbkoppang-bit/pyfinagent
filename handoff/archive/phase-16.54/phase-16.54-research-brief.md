# Research Brief: phase-16.54 -- Sovereign two-hero row height balance

Tier: **simple** (pure-UI cleanup, single-file 1-line edit). Internal-only
gate per established pure-UI cycle precedent (16.43, 16.46, 16.47, 16.48,
16.49, 16.52, 16.53).

## Problem (operator screenshot 2026-04-26 15:31:38)

Sovereign page two-hero row has the Red Line Monitor (~440px tall card)
side-by-side with the Alpha Leaderboard (~280px tall card). The cards
are in a `grid-cols-5 (3+2)` row -- they don't share `items-stretch`
forcing equal heights, but the visual mismatch creates a clear ~160px
dead area below Alpha Leaderboard.

User explicit ask: "make Red Line Monitor smaller so we dont need to
scroll on Alpha Leaderboard."

## Internal sources read in full

| File | Lines | Role |
|------|-------|------|
| `frontend/src/app/sovereign/page.tsx` L139-155 | -- | Two-hero grid: `grid grid-cols-1 gap-4 lg:grid-cols-5` with `lg:col-span-3` (RedLine) + `lg:col-span-2` (Leaderboard) |
| `frontend/src/components/RedLineMonitor.tsx` L107 | -- | Chart container: `className={compact ? "h-72" : "h-64"}` |
| `frontend/src/components/AlphaLeaderboard.tsx` | -- | Naturally short (header + table rows ~210px depending on entry count) |
| `.claude/rules/frontend-layout.md` §4.6 | -- | Sovereign two-hero pattern: prescribes `min-h-[55svh]` on the WRAPPER for the homepage hero, but `/sovereign` itself uses static import (no min-h-[55svh] anywhere on this page) |
| `.claude/rules/frontend.md` (BentoCard guidance) | -- | "No equal-height rows mixing short and tall widgets" -- short cards collapse to natural height; cards should not be forced to fill |

## Decisive findings

1. **Chart container is `h-64`** = 256px fixed. Plus header ~50px + bottom legend ~30px + BentoCard padding ~48px = ~384px total card. With 30d-window flat data (NAV ranges 9497-9501), the y-axis dominates with 5+ ticks, making the chart visually over-scaled.

2. **AlphaLeaderboard has natural ~210-280px** depending on entry count. With 2 strategies + horizontal-scroll bar visible, ~280px observed.

3. **Difference: ~100-160px** of dead space below AlphaLeaderboard.

4. **`frontend-layout.md` §4.6 Sovereign two-hero pattern** does NOT mandate `min-h-[55svh]` for the `/sovereign` route -- only for the HOMEPAGE hero embed. So we can reduce the chart height freely on `/sovereign`.

5. **Horizontal scroll on AlphaLeaderboard** is a SEPARATE issue (table wider than column). User did not flag it; out of scope this cycle.

6. **Fix:** Reduce the non-compact `h-64` -> `h-48` (192px) for RedLineMonitor. New total card height = ~320px which is much closer to AlphaLeaderboard's ~280px. Compact variant (used by homepage) stays `h-72` -- not affected.

7. **Side effect to verify:** the homepage uses `compact={true}` (per RedLineMonitor docs), so the `h-72` branch is unchanged. No homepage regression.

## Pitfalls

1. **Don't touch `compact` branch** -- homepage hero uses it via `next/dynamic` with explicit `min-h-[55svh]` wrapper.
2. **Don't change grid proportions** (lg:col-span-3 / lg:col-span-2) -- that's correct per layout. Only the chart inner-height needs adjustment.
3. **Don't add `items-start`** -- the cards aren't forced equal height (no stretch class), so the dead space comes purely from RedLine being intrinsically tall, not from grid stretching.

## Plan

Single 1-line edit in `frontend/src/components/RedLineMonitor.tsx` L107:

```tsx
className={compact ? "h-72" : "h-64"}
                          ^^^^
```

Change `"h-64"` -> `"h-48"`. The compact branch ("h-72") is preserved.

Verify: `cd frontend && npx tsc --noEmit` + `npm run lint`.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-16.54-research-brief.md",
  "gate_passed": true,
  "gate_passed_basis": "internal-only per pure-UI cycle precedent (16.43, 16.46, 16.47, 16.48, 16.49, 16.52, 16.53); .claude/rules/frontend{,-layout}.md authoritative for layout decisions; operator-screenshot evidence is unambiguous; single-line 1-file fix"
}
```
