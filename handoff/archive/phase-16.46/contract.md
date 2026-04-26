---
step: phase-16.46
title: Rebalance home grid widths -- Latest Transactions wider, Quick Actions narrower
cycle_date: 2026-04-26
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/src/app/page.tsx (grid lg:grid-cols-4 -> lg:grid-cols-5 + col-span 2/2/1)
---

# Sprint Contract -- phase-16.46

## User feedback (verbatim)

"i dont want to scroll in the box there is more room in RECENT REPORTS
make this one smaller so LATEST TRANSACTIONS could be wider"

## Concrete plan

Single edit in `frontend/src/app/page.tsx`:

```tsx
// before (16.45)
<div className="grid grid-cols-1 gap-6 lg:grid-cols-4 lg:items-stretch">
  <div className="lg:col-span-2 h-full"> ... Reports ...
  <div className="lg:col-span-1 h-full"> ... Transactions ...
  <div className="lg:col-span-1 h-full"> ... Actions ...

// after (16.46): grid-cols-5 with 2/2/1 spans
<div className="grid grid-cols-1 gap-6 lg:grid-cols-5 lg:items-stretch">
  <div className="lg:col-span-2 h-full"> ... Reports (40%) ...
  <div className="lg:col-span-2 h-full"> ... Transactions (40%) ...
  <div className="lg:col-span-1 h-full"> ... Actions (20%) ...
```

Reports stays at col-span-2 (now 40% instead of 50%). Transactions
gets bumped to col-span-2 (40% — same width as Reports, so the table
breathes). Actions reduces to col-span-1 (20% — fine for a short
input + 3-row list).

## Success Criteria (verbatim, immutable)

```
cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
npx tsc --noEmit && \
grep -q "lg:grid-cols-5 lg:items-stretch" src/app/page.tsx && \
! grep -q "lg:grid-cols-4" src/app/page.tsx && \
[ "$(grep -c 'lg:col-span-2 h-full' src/app/page.tsx)" = "2" ] && \
[ "$(grep -c 'lg:col-span-1 h-full' src/app/page.tsx)" = "1" ]
```

(Two col-span-2's = Reports + Transactions; one col-span-1 = Actions.)

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. Grid uses `lg:grid-cols-5` (NOT 4), col-span 2/2/1 distribution.
3. tsc clean.
4. No backend changes; no other frontend file touched.
5. Total span count = 5 (2+2+1) matches grid-cols-5.
