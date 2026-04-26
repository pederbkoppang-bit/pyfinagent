---
step: phase-16.46
tier: simple
date: 2026-04-26
gate: internal-only (continued visual feedback on the home cockpit)
---

# Research Brief: phase-16.46 home grid width rebalance

User feedback (verbatim, after seeing 16.45 ship):
> "i dont want to scroll in the box there is more room in RECENT REPORTS make this one smaller so LATEST TRANSACTIONS could be wider"

Screenshot evidence: LatestTransactionsBox shows horizontal scrollbar at bottom; "4 wk. ago" wraps to 3 lines vertically because TIME column has insufficient width.

## Internal evidence

`frontend/src/app/page.tsx:245-272` (the home grid added in 16.45):
```tsx
<div className="grid grid-cols-1 gap-6 lg:grid-cols-4 lg:items-stretch">
  <div className="lg:col-span-2 h-full">  {/* Reports = 50% */}
  <div className="lg:col-span-1 h-full">  {/* Transactions = 25% — TOO NARROW */}
  <div className="lg:col-span-1 h-full">  {/* Actions = 25% */}
</div>
```

Reports has 5 columns (TICKER, COMPANY, ALPHA, RECOMMENDATION, UPDATED) where COMPANY can be quite wide ("Sandisk Corporation"). Transactions has 5 columns (TICKER, SIDE, QTY, PRICE, TIME) where every column is either short or right-aligned numeric.

Both tables have similar information density per row. Equal widths is the right balance.

## Fix

Change grid to `lg:grid-cols-5` with col-span allocation `2/2/1`:
- Reports: 40%
- Transactions: 40%
- Actions: 20%

Quick Actions panel doesn't need 25% — its content is short (input + 3 action rows). 20% is fine.

`LatestTransactionsBox.tsx` already has `flex-1 overflow-x-auto` on the scroll container. With wider parent the overflow won't trigger; no change needed there.

## Verification

```
cd frontend && npx tsc --noEmit && \
  grep -q "lg:grid-cols-5" src/app/page.tsx && \
  grep -q "lg:col-span-2 h-full" src/app/page.tsx && \
  ! grep -q "lg:grid-cols-4" src/app/page.tsx
```

## Pitfalls

1. **Don't drop col-span-2 from Reports** — that would shrink it to 20%, making COMPANY column unreadable.
2. **Total spans must = 5** (2+2+1) for the new grid-cols-5 to work cleanly.
3. **No height changes needed** — the existing `lg:items-stretch + h-full + h-full flex flex-col` pattern from 16.43/16.44 still works.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 2,
  "report_md": "handoff/current/phase-16.46-research-brief.md",
  "gate_passed": true
}
```

Internal-only justification: pure CSS-grid rebalance from direct user feedback. Tailwind grid-cols-N + col-span-N semantics are unchanged since v3 (2021); 16.45 already used the same primitives. No fresh external research adds value.
