---
step: phase-16.42
title: Home redesign -- Recent Reports table + Quick Actions panel (no hardcoded data)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/src/lib/formatRelativeTime.ts (new, ~25 lines)
  - frontend/src/components/RecentReportsTable.tsx (new, ~140 lines)
  - frontend/src/components/HomeQuickActionsPanel.tsx (new, ~150 lines)
  - frontend/src/app/page.tsx (replace lines 184-276 with two-column layout)
---

# Sprint Contract -- phase-16.42

## Research-gate summary

`handoff/current/phase-16.42-research-brief.md`. tier=moderate, 7 in-full,
17 URLs, recency scan present, gate_passed=true. 11 internal files
inspected.

## User intent (verbatim)

Replace the current vertically-stacked "Recent Reports" + "Quick Actions"
sections on the authenticated home with the target screenshot's
two-column layout. **STRICT: no hardcoded data — all data from backend.**

## Concrete plan

### File 1: `frontend/src/lib/formatRelativeTime.ts` (new, ~25 LOC)

`Intl.RelativeTimeFormat`-based pure helper. Renders "12m ago",
"2h ago", "5h ago", "1d ago", "2d ago" matching screenshot style.
Stdlib only. Client-callable; no SSR concerns since component is
"use client".

### File 2: `frontend/src/components/RecentReportsTable.tsx` (new, ~140 LOC)

Props:
```ts
{ reports: ReportSummary[]; loaded: boolean; loadError: string | null }
```

Columns: TICKER (mono) | COMPANY (with `—` fallback for null company_name)
| ALPHA (`final_score?.toFixed(2)`, color-coded ≥8 emerald / ≥6.5 sky /
≥4.5 amber / else rose) | RECOMMENDATION (pill via `recColor`) | UPDATED
(`formatRelativeTime(analysis_date)`).

States: loading (5 skeleton rows), empty (icon + "No reports yet"),
error (rose banner).

Row click → `router.push("/reports?ticker=...")`. Keyboard:
`tabIndex={0}` + Enter/Space.

Header: "RECENT REPORTS" label + "View all →" link to `/reports`.

### File 3: `frontend/src/components/HomeQuickActionsPanel.tsx` (new, ~150 LOC)

Props:
```ts
{ ticker: string; onTickerChange: (t: string) => void; onAnalyze: () => void }
```

Section A — ticker input + Analyze button (mirrors current page.tsx logic).

Section B — three rows, each: `<Icon> Label <kbd>Shortcut</kbd>`:
1. **Run morning cycle** [Ctrl+Shift+R] → `triggerPaperTradingCycle()`
2. **Open backtest console** [Ctrl+B] → `router.push("/backtest")`
3. **Halt all trading** [Ctrl/Cmd+Shift+H] → calls
   `postPaperKillSwitchAction("FLATTEN_ALL")` then `("PAUSE")`
   (same sequence as `KillSwitchShortcut.tsx:18-21`)

Keyboard shortcuts wired via `useEffect` + window listeners.
**MUST NOT re-implement Ctrl+Shift+H** — `KillSwitchShortcut` already
owns that listener; the Halt button just shares the helper logic
(spawn its own halt() that mirrors KillSwitchShortcut's). To avoid
double-fire on Ctrl+Shift+H: only register listeners for
Ctrl+Shift+R + Ctrl+B in this panel; the Halt button is click-only,
the global Ctrl+Shift+H listener stays in KillSwitchShortcut.

Each action: shows pending / success / error inline (rose banner on
error, sky pulse while pending).

### File 4: `frontend/src/app/page.tsx` (replace lines 184-276)

Remove the old Recent Reports block (184-230) AND the old Quick Actions
block (232-276). Replace with single two-column grid:

```tsx
<div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
  <div className="lg:col-span-2">
    <RecentReportsTable reports={reports} loaded={loaded} loadError={loadError} />
  </div>
  <div className="lg:col-span-1">
    <HomeQuickActionsPanel
      ticker={ticker}
      onTickerChange={setTicker}
      onAnalyze={() => { if (ticker.trim()) router.push(`/signals?ticker=${encodeURIComponent(ticker.trim())}`); }}
    />
  </div>
</div>
```

Existing state (`reports`, `loaded`, `loadError`, `ticker`, `setTicker`,
`router`) is already in scope at page.tsx:62-69.

The `recColor` helper at page.tsx:32-39 moves into `RecentReportsTable.tsx`
(local copy; trivial 8-line function, not worth a shared util this cycle).

## Success Criteria (verbatim, immutable)

```
test -f frontend/src/lib/formatRelativeTime.ts && \
test -f frontend/src/components/RecentReportsTable.tsx && \
test -f frontend/src/components/HomeQuickActionsPanel.tsx && \
grep -q "RecentReportsTable" frontend/src/app/page.tsx && \
grep -q "HomeQuickActionsPanel" frontend/src/app/page.tsx && \
! grep -E "(NVIDIA Corporation|Apple Inc\.|Microsoft|Tesla, Inc\.|Intel Corporation)" frontend/src/components/RecentReportsTable.tsx frontend/src/components/HomeQuickActionsPanel.tsx && \
cd frontend && npx tsc --noEmit && npm run lint 2>&1 | grep -c '@phosphor-icons/react' | grep -q '^0$'
```

The `! grep` clause is the **anti-hardcoding gate**: if any of the
target screenshot's specific company names appear in the new
component source, the verification FAILS.

Plus:
- `no_hardcoded_alpha_values`: grep verifies no literal "7.42", "6.81",
  "5.12", "3.74", "2.11" in the new components
- `tsc_clean`: `npx tsc --noEmit` exits 0
- `lint_clean`: zero phosphor warnings + zero new errors
- `live_data_smoke`: `curl http://localhost:8000/api/reports/?limit=5`
  returns valid `[ReportSummary]` JSON (proves the data path works)

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. **Anti-hardcoding gate:** zero occurrences of the target-screenshot
   sample data in the new component sources (NVIDIA Corporation, Apple
   Inc., etc., the alpha values 7.42/6.81/etc.).
3. `RecentReportsTable` consumes the `reports` prop (does NOT fetch
   internally — fetch happens in page.tsx already).
4. `HomeQuickActionsPanel`'s "Halt all trading" button calls BOTH
   `FLATTEN_ALL` AND `PAUSE` (per pitfall #3 from research brief).
5. No new Ctrl+Shift+H listener registered (only Ctrl+Shift+R + Ctrl+B).
6. Loading + empty + error states present in `RecentReportsTable`.
7. `company_name` null-fallback (per pitfall #5).
8. `lg:grid-cols-3` (mobile-stacked, per pitfall #8).
9. Page.tsx still imports `Sidebar`, `OpsStatusBar`, `KillSwitchShortcut`,
   `RedLineMonitor` — top-of-page structure preserved.
10. tsc + lint clean.
11. Live `curl /api/reports/?limit=5` returns valid JSON shape.
