# Research Brief: phase-16.55 -- Sovereign two-hero balance round 2

Tier: **simple**. Internal-only gate per pure-UI cycle precedent.

## Problem (operator screenshot 2026-04-26 15:39:53, 8 minutes after 16.54 push)

The 16.54 fix (h-64 -> h-48) reduced RedLineMonitor card from ~440px
to ~376px, but the operator confirms ~100px dead space below Alpha
Leaderboard remains. Operator: "deadspace is still there!"

## Measurements from screenshot

| Card | Top | Bottom | Height |
|------|-----|--------|--------|
| Red Line Monitor | 140px | 440px | ~300px |
| Alpha Leaderboard | 140px | 340px | ~200px |
| Dead space | 340px | 440px | ~100px |

The 16.54 fix DID work (cards are closer in height than before), but
not enough. Operator wants the dead space ELIMINATED, not reduced.

## Decisive findings

1. **The chart container is the only height knob.** No wrapper min-h on
   sovereign page or BentoCard.

2. **Alpha Leaderboard's natural height is determined by content** -- 2
   strategy rows + header + horizontal scrollbar = ~200px. We can't
   change that without changing the data or hiding the scrollbar.

3. **Need to reduce RedLine card to ~270px** to match (220 chart area
   would leave ~50px header+footer overhead, but that's getting too cramped).

4. **Better solution per `.claude/rules/frontend.md`:** "no equal-height
   rows mixing short and tall widgets". Use the bento/sidebar pattern --
   stack a SECOND short widget below Alpha Leaderboard on the right
   column. But that's scope creep for this user request.

5. **User's explicit ask is "make Red Line Monitor smaller"** -- so go
   smaller. Try `h-40` (160px). Card height becomes ~248px. Slightly
   shorter than Alpha Leaderboard at ~200px so dead space MOVES under
   Red Line instead of Alpha. That's not great either.

6. **Best compromise: `h-44` (176px)** -- card ~264px which matches
   Alpha Leaderboard's ~200-280px range depending on entry count. Or
   `h-40` (160px) - card ~248px which is slightly shorter than
   AlphaLeaderboard. Either is much closer to balance.

7. **Alternative: use `items-start` on the grid** to remove any
   stretching behavior. Default CSS grid does NOT force equal heights
   on row tracks, but `items-stretch` (default for items in a row)
   stretches each item to fill its row. So adding `items-start` would
   ensure each card stays at its natural height. Combined with smaller
   chart, this is the cleanest fix.

## Plan

Two coordinated changes:

1. `frontend/src/components/RedLineMonitor.tsx` L107: `h-48` -> `h-40` (192px -> 160px)
2. `frontend/src/app/sovereign/page.tsx` L139: add `items-start` to the grid so cards don't stretch even when one is intrinsically taller

Plus update the docstring comment at L49 to reflect the new value.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "report_md": "handoff/current/phase-16.55-research-brief.md",
  "gate_passed": true
}
```
