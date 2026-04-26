---
step: phase-16.47
title: Fix Quick Actions overflow -- widen column + shrink-protect internal layout
cycle_date: 2026-04-26
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/src/app/page.tsx (grid lg:grid-cols-5 -> lg:grid-cols-6, all 3 col-span-2)
  - frontend/src/components/HomeQuickActionsPanel.tsx (min-w-0 + shrink-0 hardening)
---

# Sprint Contract -- phase-16.47

## User feedback (verbatim)

"that is not wokring look at QUICK ACTIONS box"

Visible bugs from screenshot:
1. Analyze button cropped on right edge
2. Action labels wrapping to 2 lines ("Run morning cycle", "Halt all trading")

## Concrete plan

### Fix 1: page.tsx — equal-thirds grid

```tsx
// before (16.46): lg:grid-cols-5 with 2/2/1
// Reports 40% | Transactions 40% | Actions 20% (TOO NARROW)

// after (16.47): lg:grid-cols-6 with 2/2/2
<div className="grid grid-cols-1 gap-6 lg:grid-cols-6 lg:items-stretch">
  <div className="lg:col-span-2 h-full"> ... Reports (33%) ...
  <div className="lg:col-span-2 h-full"> ... Transactions (33%) ...
  <div className="lg:col-span-2 h-full"> ... Actions (33%) ...
```

### Fix 2: HomeQuickActionsPanel.tsx — defense-in-depth layout hardening

**Section A (ticker input + Analyze button):**
- Wrap input in a `<div className="flex-1 min-w-0">` so flex-1 shrinks
  cleanly below content min-width
- Add `shrink-0` to Analyze button so it never gets cropped (input
  shrinks instead)

**Section B (action rows):**
- `<button className="... gap-2 ...">` — reduce gap from 3 to 2 to give
  label more room
- Label `<span className="flex-1 min-w-0 text-sm text-slate-200 truncate">`
  — `min-w-0` allows shrink, `truncate` adds ellipsis on extreme overflow
- `<Kbd>` already wraps in whitespace-nowrap-friendly `<kbd>` element;
  add explicit `whitespace-nowrap` + `shrink-0` to Kbd helper for safety

## Success Criteria (verbatim, immutable)

```
cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
npx tsc --noEmit && \
grep -q "lg:grid-cols-6 lg:items-stretch" src/app/page.tsx && \
! grep -q "lg:grid-cols-5" src/app/page.tsx && \
[ "$(grep -c 'lg:col-span-2 h-full' src/app/page.tsx)" = "3" ] && \
grep -q "shrink-0" src/components/HomeQuickActionsPanel.tsx && \
grep -q "min-w-0" src/components/HomeQuickActionsPanel.tsx
```

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. Grid is `lg:grid-cols-6` with 3 col-span-2 children (total span = 6).
3. HomeQuickActionsPanel.tsx has shrink-0 on Analyze button + Kbd; min-w-0 on input wrapper + label span.
4. tsc clean.
5. No backend changes; no other components touched.
6. The fix is defense-in-depth (column-width + internal-shrink) so degradation is graceful at narrower viewports.
