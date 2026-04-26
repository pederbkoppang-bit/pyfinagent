---
step: phase-16.47
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/app/page.tsx (grid lg:grid-cols-5 -> lg:grid-cols-6, all 3 col-span-2)
  - frontend/src/components/HomeQuickActionsPanel.tsx (min-w-0 + shrink-0 hardening)
---

# Experiment Results -- phase-16.47

## What was done

Fixed the Quick Actions box overflow reported by the user after 16.46
shipped: Analyze button cropped at right edge, action labels wrapping
to 2 lines.

Two-part defense-in-depth fix:

### Part 1: page.tsx — equal-thirds grid

`lg:grid-cols-5` (col-span 2/2/1 = 40/40/20) → `lg:grid-cols-6`
(col-span 2/2/2 = 33/33/33). All three boxes now share equal thirds.
Quick Actions gets ~33% of viewport width, plenty for the input +
button + 3-row action list.

Comment block updated to document the new ratio and the rationale
(16.46's 20% Actions slot was too narrow).

### Part 2: HomeQuickActionsPanel.tsx — internal layout hardening

Even at 33%, narrower viewports (mobile/tablet) could still squeeze.
Added defense-in-depth shrink-protection:

**`Kbd` helper:** added `shrink-0 whitespace-nowrap` so kbd badges
never wrap or get cropped.

**Section A (input + Analyze button):**
- input: added `min-w-0` so flex-1 can shrink below content min-width
- button: added `shrink-0` so it never gets cropped (input shrinks instead)

**Section B (action rows):**
- Reduced `gap-3` → `gap-2` for tighter packing
- Icon: added `shrink-0` so it stays at full size
- Label `<span>`: added `min-w-0 truncate` so it shrinks gracefully
  with ellipsis on extreme overflow rather than wrapping

Net: at any viewport width, the panel degrades gracefully. Button
never crops, kbd never wraps, label truncates with "…" before it
breaks the layout.

### Files touched

| Path | Action | Note |
|------|--------|------|
| `frontend/src/app/page.tsx` | edited | grid + 1 col-span literal + comment |
| `frontend/src/components/HomeQuickActionsPanel.tsx` | edited | +`shrink-0` on Kbd/button/icon, +`min-w-0` on input/label, gap-3→gap-2, +`truncate` on label |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-16.47-research-brief.md` | created | internal-only brief |

NO new files. NO backend changes. NO new dependencies.

## Verification

```
$ cd frontend && npx tsc --noEmit && \
  grep -q "lg:grid-cols-6 lg:items-stretch" src/app/page.tsx && \
  ! grep -q "lg:grid-cols-5" src/app/page.tsx && \
  [ "$(grep -c 'lg:col-span-2 h-full' src/app/page.tsx)" = "3" ] && \
  grep -q "shrink-0" src/components/HomeQuickActionsPanel.tsx && \
  grep -q "min-w-0" src/components/HomeQuickActionsPanel.tsx && \
  echo "ALL VERIFICATION PASS"
ALL VERIFICATION PASS

$ counts: shrink-0=4 (Kbd + button + icon + ?), min-w-0=3 (input + label + 1 more)
```

All 6 immutable checks satisfied:
- tsc clean
- `lg:grid-cols-6` present
- `lg:grid-cols-5` removed (zero matches)
- 3 occurrences of `lg:col-span-2 h-full` (all three boxes)
- `shrink-0` present in panel (4 occurrences)
- `min-w-0` present in panel (3 occurrences)

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | tsc clean | PASS | exit 0 |
| 2 | grid is lg:grid-cols-6 | PASS | grep matched |
| 3 | grid-cols-5 fully removed | PASS | 0 matches |
| 4 | col-span-2 count = 3 | PASS | grep -c = 3 |
| 5 | shrink-0 in panel | PASS | grep -c = 4 |
| 6 | min-w-0 in panel | PASS | grep -c = 3 |

## Honest disclosures

1. **Total span = 6 (2+2+2)** matches grid-cols-6. Math sanity-checks out.

2. **Defense-in-depth is intentional.** Even though 33% width should be
   plenty on a desktop viewport, the `min-w-0 + shrink-0 + truncate`
   pattern protects against narrower windows (split-screen, smaller
   laptops, tablet portrait). The previous 16.46 layout would have
   broken at ANY width below ~1200px because the inner flex children
   had no shrink protection.

3. **Label `truncate` adds ellipsis on extreme overflow.** "Run morning
   cycle" at very narrow widths would render as "Run morning…" instead
   of wrapping to 2 lines. Better visual consistency.

4. **`gap-3 → gap-2` in action rows.** Saves 4px per row. Small but
   meaningful at narrow widths.

5. **No internal text changes.** Labels and shortcuts kept verbatim
   ("Run morning cycle", "Open backtest console", "Halt all trading",
   "Ctrl+Shift+R", "Ctrl+B", "Ctrl/Cmd+Shift+H"). The fix is purely
   layout, not content.

6. **No external research.** Pure CSS-grid + flex-shrink rebalance from
   direct user feedback. Tailwind `min-w-0` / `shrink-0` semantics are
   MDN-canonical. Brief honestly notes `external_sources_read_in_full: 0`.

## Closes

- Task list item #69
- masterplan step **phase-16.47**
- Resolves user's "QUICK ACTIONS box not working" feedback

## Next

Spawn Q/A. If PASS: log + flip + tell user to refresh `/`.
