---
step: phase-16.47
tier: simple
date: 2026-04-26
gate: internal-only (continued visual feedback on home cockpit Quick Actions box)
---

# Research Brief: phase-16.47 fix Quick Actions overflow

## User feedback (verbatim)

> "that is not wokring look at QUICK ACTIONS box"

Screenshot shows two problems in the Quick Actions panel at the new 20% width:
1. **Analyze button cropped** — only "Analyz" visible at the right edge of the box
2. **Action labels wrapping** — "Run morning cycle" wraps to 2 lines ("Run morning / cycle"); "Halt all trading" wraps similarly

## Trace

**Grid context** (page.tsx after 16.46): `lg:grid-cols-5` with col-span `2/2/1`. Quick Actions = 1/5 = 20%. On a 1400px viewport that's ~280px minus padding. Too narrow.

**Inside HomeQuickActionsPanel (Section A: input + button):**
```tsx
<div className="flex gap-2">
  <input className="flex-1 ... px-3 py-2 ..." />   {/* flex-1, no min-width */}
  <button className="... px-4 py-2 ...">Analyze</button>  {/* no shrink-0 */}
</div>
```
At narrow widths, `flex-1` input shrinks toward 0; button keeps its `px-4 py-2 + text` width which exceeds remaining space. Browser overflow clips the button on the right.

**Inside Section B (action rows):**
```tsx
<button className="flex w-full items-center gap-3 px-4 py-3 text-left ...">
  <Icon />
  <span className="flex-1 text-sm">{label}</span>   {/* "Run morning cycle" */}
  <Kbd>{shortcut}</Kbd>                              {/* "Ctrl+Shift+R" */}
</button>
```
At narrow widths, label and kbd compete for space. Label has `flex-1` so it gets remaining space, but with `gap-3 + Icon + Kbd` the remaining space forces label to wrap. Specifically:
- "Run morning cycle" (~140px) + "Ctrl+Shift+R" kbd (~80px) + icon + gap = ~250px minimum
- Container has ~240px after padding → label wraps

## Two-part fix

### Part 1: widen Actions column

Change grid from `lg:grid-cols-5` (2/2/1) to `lg:grid-cols-6` (2/2/2).
- Reports: 33% (was 40%) — still wide enough for COMPANY column
- Transactions: 33% (was 40%) — still wide enough for the 5 narrow columns
- Actions: 33% (was 20%) — gives input+button room AND action labels room

Equal-thirds is the simplest stable layout. All three boxes have similar
information density per row.

### Part 2: harden internal layout (defense-in-depth)

Even at 33%, narrower mobile/tablet breakpoints could squeeze. Add:
- Input wrapper: `min-w-0` so flex-1 can shrink below content min-width
  cleanly
- Analyze button: `shrink-0` so it never gets cropped (input shrinks instead)
- Action label span: `min-w-0` + keep `flex-1` (allows wrap if truly cramped)
- Kbd badges: `shrink-0` + `whitespace-nowrap` so they never wrap

If at very narrow viewport the label still wraps, that's now intentional
graceful degradation — but at 33% on a desktop viewport this won't happen
anymore.

## Verification

```
cd frontend && npx tsc --noEmit && \
  grep -q "lg:grid-cols-6" src/app/page.tsx && \
  ! grep -q "lg:grid-cols-5" src/app/page.tsx && \
  [ "$(grep -c 'lg:col-span-2 h-full' src/app/page.tsx)" = "3" ] && \
  grep -q "shrink-0" src/components/HomeQuickActionsPanel.tsx
```

3 col-span-2's (Reports + Transactions + Actions) = 6 = grid-cols-6 ✓.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/phase-16.47-research-brief.md",
  "gate_passed": true
}
```

Pure CSS-grid + flex-shrink rebalance from direct user feedback. Tailwind
`min-w-0` / `shrink-0` semantics are MDN-canonical; no fresh external
research needed.
