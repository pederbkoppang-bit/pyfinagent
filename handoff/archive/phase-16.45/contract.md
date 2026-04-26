---
step: phase-16.45
title: Latest Transactions box between Recent Reports and Quick Actions
cycle_date: 2026-04-26
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/src/components/LatestTransactionsBox.tsx (new, mirrors RecentReportsTable pattern)
  - frontend/src/app/page.tsx (grid 3 -> 4 cols, add fetcher + state + new component slot)
  - frontend/src/lib/icons.ts (add Robot identity re-export)
---

# Sprint Contract -- phase-16.45

## Research-gate summary

`handoff/current/phase-16.45-research-brief.md`. tier=simple, 6 in-full,
16 URLs, recency scan present, gate_passed=true. 10 internal files
inspected.

**Critical confirmations:**
- `GET /api/paper-trading/trades?limit=5` exists at backend/api/paper_trading.py:176
- `getPaperTrades(limit)` already in frontend/src/lib/api.ts:280
- `PaperTrade` interface already in frontend/src/lib/types.ts:582-594
- `formatRelativeTime` reusable from frontend/src/lib/formatRelativeTime.ts (16.42)
- BUY/SELL pill pattern at frontend/src/app/paper-trading/page.tsx:650-659

No backend work needed. Pure frontend composition.

## Concrete plan

### 1. New component `frontend/src/components/LatestTransactionsBox.tsx` (~140 LOC)

Mirror `RecentReportsTable.tsx` structure verbatim:
- "use client"
- Props: `{ trades: PaperTrade[]; loaded: boolean; loadError: string | null }`
- Outer wrapper: `h-full flex flex-col rounded-xl border border-navy-700 bg-navy-800/40`
- Header: "LATEST TRANSACTIONS" + "View all →" link to `/paper-trading`
- 5 columns: TICKER (mono bold) | SIDE (BUY emerald / SELL rose pill) | QTY (right-aligned) | PRICE (right-aligned `$X.XX`) | TIME (relative via formatRelativeTime)
- Loading state: 5 skeleton rows
- Empty state: NavPaperTrading icon + "No trades yet"
- Error state: rose banner
- Row click → `router.push("/paper-trading")` (the trades sub-page)

### 2. `frontend/src/app/page.tsx` updates

Add fetcher to existing Promise.allSettled batch:
- Import `getPaperTrades` from `@/lib/api` and `PaperTrade` type
- Add `[trades, setTrades]` + `[tradesError, setTradesError]` state (loaded shared with existing `loaded`)
- Append `getPaperTrades(5)` to the Promise.allSettled call at line 73-94
- Handle result: `setTrades(reps.value.trades ?? [])` (note: response is `{trades, count}`, not bare array)
- Add `LatestTransactionsBox` import

Change grid (line 245):
```tsx
// before
<div className="grid grid-cols-1 gap-6 lg:grid-cols-3 lg:items-stretch">
  <div className="lg:col-span-2 h-full">
    <RecentReportsTable ... />
  </div>
  <div className="lg:col-span-1 h-full">
    <HomeQuickActionsPanel ... />
  </div>
</div>

// after
<div className="grid grid-cols-1 gap-6 lg:grid-cols-4 lg:items-stretch">
  <div className="lg:col-span-2 h-full">
    <RecentReportsTable ... />
  </div>
  <div className="lg:col-span-1 h-full">
    <LatestTransactionsBox trades={trades} loaded={loaded} loadError={tradesError} />
  </div>
  <div className="lg:col-span-1 h-full">
    <HomeQuickActionsPanel ... />
  </div>
</div>
```

### 3. `frontend/src/lib/icons.ts` add Robot identity re-export

Already has `Robot as NavPaperTrading` (line 14). Add `Robot as Robot` to the identity-re-export block (line 172+) so `LatestTransactionsBox` can import it as `Robot` for the empty-state icon — OR just import `NavPaperTrading` as the icon to avoid the icons.ts edit. **Decision: use `NavPaperTrading` directly to keep diff small; don't touch icons.ts.**

(Drops the icons.ts edit from the deliverable list — will not be touched.)

## Success Criteria (verbatim, immutable)

```
cd /Users/ford/.openclaw/workspace/pyfinagent && \
test -f frontend/src/components/LatestTransactionsBox.tsx && \
grep -q "LatestTransactionsBox" frontend/src/app/page.tsx && \
grep -q "lg:grid-cols-4" frontend/src/app/page.tsx && \
grep -q "getPaperTrades" frontend/src/app/page.tsx && \
(cd frontend && npx tsc --noEmit) && \
curl -s "http://localhost:8000/api/paper-trading/trades?limit=5" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'trades' in d and isinstance(d['trades'], list); print('shape ok, count:', len(d['trades']))"
```

Plus:
- `anti_hardcoding_gate`: 0 occurrences of mock ticker names like "AAPL"/"NVDA"/"MSFT"/etc. as literal sample data in LatestTransactionsBox.tsx
- `no_backend_changes`: 0 files touched in `backend/`
- `lint_clean`: 0 phosphor warnings
- `pattern_match_RecentReportsTable`: outer wrapper has `h-full flex flex-col`, header has "View all →" link, loading state has 5 skeleton rows

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. Backend GET /api/paper-trading/trades?limit=5 returns `{trades: [], count: 0}` shape (confirms wiring is real).
3. `LatestTransactionsBox` consumes `trades` prop (does NOT fetch internally; fetch happens in page.tsx).
4. Grid is `lg:grid-cols-4` with col-spans 2/1/1 (not 4 equal columns).
5. New box pattern matches RecentReportsTable exactly (h-full flex flex-col + skeleton + empty + error states + View all link).
6. BUY/SELL pill colors: emerald for BUY, rose for SELL (mirrors paper-trading/page.tsx pattern).
7. Anti-hardcoding gate clean.
8. No backend changes; no edits outside the 2 frontend files (LatestTransactionsBox new + page.tsx edited).
9. tsc + lint clean.
