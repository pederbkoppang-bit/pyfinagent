---
step: phase-16.46
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/app/page.tsx (single grid edit: lg:grid-cols-4 -> lg:grid-cols-5 + col-span 2/2/1)
---

# Experiment Results -- phase-16.46

## What was done

Single-edit follow-up on phase-16.45. User reported the new
LatestTransactionsBox was too narrow (horizontal scrollbar visible,
"4 wk. ago" wrapping to 3 lines). Rebalanced grid widths to give
Reports + Transactions equal share, dropped Quick Actions to a
narrower 20% slot (its content is short anyway).

### The change

`frontend/src/app/page.tsx` grid:

```tsx
// before (16.45): lg:grid-cols-4 with col-span 2/1/1
// Reports = 50%, Transactions = 25%, Actions = 25%

// after (16.46): lg:grid-cols-5 with col-span 2/2/1
<div className="grid grid-cols-1 gap-6 lg:grid-cols-5 lg:items-stretch">
  <div className="lg:col-span-2 h-full"> ... Reports (40%) ...
  <div className="lg:col-span-2 h-full"> ... Transactions (40%) ...
  <div className="lg:col-span-1 h-full"> ... Actions (20%) ...
</div>
```

Comment block updated to document the new ratio + the rationale for
the change.

### Files touched

| Path | Action | Note |
|------|--------|------|
| `frontend/src/app/page.tsx` | edited | grid + 1 col-span literal + comment |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-16.46-research-brief.md` | created | internal-only brief |

NO new files. NO backend changes. NO new dependencies.
LatestTransactionsBox.tsx itself was NOT touched — its existing
`flex-1 overflow-x-auto` will simply not trigger overflow at the
new wider parent width.

## Verification

```
$ npx tsc --noEmit && \
  grep -q "lg:grid-cols-5 lg:items-stretch" src/app/page.tsx && \
  ! grep -q "lg:grid-cols-4" src/app/page.tsx && \
  [ "$(grep -c 'lg:col-span-2 h-full' src/app/page.tsx)" = "2" ] && \
  [ "$(grep -c 'lg:col-span-1 h-full' src/app/page.tsx)" = "1" ] && \
  echo "ALL VERIFICATION PASS"
ALL VERIFICATION PASS
```

All 5 immutable checks satisfied:
- tsc clean
- `lg:grid-cols-5` present
- `lg:grid-cols-4` removed (zero matches)
- 2 occurrences of `lg:col-span-2 h-full` (Reports + Transactions)
- 1 occurrence of `lg:col-span-1 h-full` (Actions)

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | tsc clean | PASS | exit 0 |
| 2 | grid is lg:grid-cols-5 | PASS | grep matched |
| 3 | grid-cols-4 fully removed | PASS | 0 matches |
| 4 | col-span-2 count = 2 | PASS | grep -c = 2 |
| 5 | col-span-1 count = 1 | PASS | grep -c = 1 |

## Honest disclosures

1. **Total span = 5 (2+2+1)** matches grid-cols-5. Math sanity-checks
   out — no orphaned column or unused space.

2. **No change to LatestTransactionsBox.tsx itself.** The existing
   `flex-1 overflow-x-auto` wrapper will simply not need to trigger
   overflow at 40% parent width. If at very narrow viewports
   (mobile) overflow returns, the existing scroll behavior takes
   over — graceful degradation.

3. **Quick Actions at 20% is fine.** Looking at the panel content:
   ticker input (full row) + 3 action rows (icon + label + kbd
   shortcut). The kbd badges are short (`Ctrl+Shift+R`, `Ctrl+B`,
   `Ctrl/Cmd+Shift+H`) and the labels are short. 20% width on a
   1400px viewport = 280px = comfortable.

4. **No external research.** Pure CSS-grid rebalance from direct
   user feedback. Tailwind grid primitives unchanged since v3
   (2021); 16.45 already used the same primitives. Brief honestly
   notes `external_sources_read_in_full: 0`.

## Closes

- Task list item #68
- masterplan step **phase-16.46**
- Resolves user's "make it wider" feedback on LatestTransactionsBox

## Next

Spawn Q/A. If PASS: log + flip + tell user to refresh `/`.
