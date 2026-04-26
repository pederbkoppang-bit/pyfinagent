# Research Brief: phase-16.53 -- Settings full-width content fix

Tier: **simple** (pure-UI cleanup; single-file 3-line edit + 1 wrapper relax).
Internal-only gate per established pure-UI cycle precedent (16.43, 16.46,
16.47, 16.48, 16.49, 16.52).

## Problem (from user screenshot 2026-04-26 15:18:54)

The settings page now uses the canonical two-zone shell (16.52 fix
landed) but the tab CONTENT inside the scrollable zone is constrained
to `max-w-4xl` (~896px), leaving the right ~30% of the page as dead
whitespace on a typical 1500px+ wide laptop viewport. User explicitly
flagged: "you are not using the whole page on settings".

## Internal sources read in full

| File | Lines | Role |
|------|-------|------|
| `.claude/rules/frontend.md` | 48 | Frontend conventions authority |
| `.claude/rules/frontend-layout.md` | 496 | Layout blueprint authority |
| `frontend/src/app/settings/page.tsx` | 1255 | The page being fixed |
| `frontend/src/app/reports/page.tsx` | 604 | Reference: how reports tab content fills width |
| `frontend/src/app/backtest/page.tsx` | 1300+ | Reference: how backtest tab content fills width |

## Decisive findings

1. **3 occurrences of `max-w-4xl`** on `<div class="grid max-w-4xl grid-cols-1 gap-6 lg:grid-cols-2">` wrappers, one per tab:
   - L601: Models & Analysis tab grid wrapper
   - L794: Cost & Weights tab grid wrapper
   - L978: Performance tab grid wrapper

2. **No other page uses `max-w-4xl`** on tab content. `reports`, `backtest`, `agents`, etc. let content fill the scrollable zone's width naturally.

3. **`frontend-layout.md` does NOT prescribe a max-width** on tab content. §4 Metric Grids show responsive `grid-cols-2 sm:grid-cols-4 lg:grid-cols-6` — full width by design.

4. **The `max-w-fit` on the tab bar at L580** is correct/intentional (the pill tabs should hug content, not stretch). Keep that.

5. **`max-w-4xl` was likely a copy-paste from a single-card-per-row pattern** that no longer applies once we have the two-column `lg:grid-cols-2` layout. With 2-col + wide cards, `max-w-4xl` truncates the second column at ~430px each which is below the natural BentoCard width — wasteful.

6. **Fix:** drop `max-w-4xl` from the 3 grid wrappers. Cards will fill the scrollable zone's natural width inside the `flex-1 overflow-y-auto` container. `lg:grid-cols-2` continues to keep them as 2 columns on wide screens; `grid-cols-1` collapses to single column on narrow.

## Pitfalls

1. **Don't relax `max-w-fit` on the tab bar** (L580) -- pills should hug content per canonical `frontend-layout.md` §5.
2. **Don't remove `max-w-4xl` from individual cards** -- they're fine; only the grid WRAPPER constraint is wrong.
3. **No new tests** -- pure-UI; verification is `cd frontend && npx tsc --noEmit`. Visual regression is verified by user (already flagged the issue).

## Plan

3 single-line edits in `frontend/src/app/settings/page.tsx`:
- L601: `grid max-w-4xl grid-cols-1 gap-6 lg:grid-cols-2` -> `grid grid-cols-1 gap-6 lg:grid-cols-2`
- L794: same change
- L978: same change

Verify: `cd frontend && npx tsc --noEmit && npm run lint`.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-16.53-research-brief.md",
  "gate_passed": true,
  "gate_passed_basis": "internal-only per pure-UI cycle precedent (16.43, 16.46, 16.47, 16.48, 16.49, 16.52); .claude/rules/frontend.md + frontend-layout.md are sole authority for this layout decision; user-screenshot evidence is unambiguous"
}
```
