---
step: phase-16.49
title: UX audit pass B -- high-risk pages (backtest/reports/agents/paper-trading)
cycle_date: 2026-04-26
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/src/app/reports/page.tsx (CRITICAL: two-zone shell + tab-bar pin + loading skeleton + scrollbar-thin)
  - frontend/src/app/backtest/page.tsx (error banners moved to scrollable zone + 4x scrollbar-thin)
  - frontend/src/app/paper-trading/page.tsx (2x scrollbar-thin)
---

# Sprint Contract -- phase-16.49

## Research-gate summary

`handoff/current/phase-16.49-research-brief.md`. tier=simple/internal-only,
gate_passed=true. 5 files audited (~3712 LOC total). agents/page.tsx
(728 LOC) + paper-trading/learnings/page.tsx (22 LOC) both COMPLIANT.

## Violations found (11 across 3 files)

| Sev | File:line | Violation | Fix |
|-----|-----------|-----------|-----|
| **CRITICAL** | `reports/page.tsx:223` | Single-zone shell; tab bar scrolls off-screen | Wrap in canonical two-zone (same fix as 16.48 signals/performance) |
| LOW | `reports/page.tsx:253` | Loading state is plain text | Replace with `<PageSkeleton />` |
| LOW | `reports/page.tsx:482` | `overflow-x-auto` missing scrollbar-thin | Add class |
| MED | `backtest/page.tsx:683-709` | Error banners in `flex-shrink-0` fixed-header zone | Move to scrollable zone |
| LOW | `backtest/page.tsx:209` | RunSelector dropdown `overflow-y-auto` missing scrollbar-thin | Add class |
| LOW | `backtest/page.tsx:1055` | Strategy-vs-Baselines table `overflow-x-auto` missing | Add class |
| LOW | `backtest/page.tsx:1118` | Walk-Forward Windows table missing | Add class |
| LOW | `backtest/page.tsx:1301` | Trade list table missing | Add class |
| LOW | `paper-trading/page.tsx:533` | Positions table `overflow-x-auto` missing | Add class |
| LOW | `paper-trading/page.tsx:617` | Trades table `overflow-x-auto` missing | Add class |

DEFERRED: backtest RunSelector relocation (Fix 5 in research brief) —
researcher recommended deferring unless visually confirmed.

## Concrete plan

### Fix 1: reports/page.tsx two-zone shell (CRITICAL)

Mirror the 16.48 signals/performance pattern:
- L223: `<main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">` ->
  `<main className="flex flex-1 flex-col overflow-hidden">`
- Wrap header + tab bar (L224-245) in `<div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">`
- Wrap content (L246+) in `<div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">`
- Close both new wrappers before the existing `</main>`

### Fix 2: reports/page.tsx loading state

L253: `{loading && <p className="text-slate-400">Loading reports...</p>}` ->
PageSkeleton import + `{loading && <PageSkeleton />}`

### Fix 3: reports/page.tsx + backtest + paper-trading scrollbar-thin (7 occurrences)

Add `scrollbar-thin` after each `overflow-x-auto` or `overflow-y-auto` flagged.

### Fix 4: backtest/page.tsx error banners

Cut error banner JSX (L683-709) from inside `flex-shrink-0` fixed-header
div. Paste at top of scrollable zone (just after the opening of L790).

## Success Criteria (verbatim, immutable)

```
cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
npx tsc --noEmit && \
grep -q "flex flex-1 flex-col overflow-hidden" src/app/reports/page.tsx && \
grep -q "PageSkeleton" src/app/reports/page.tsx && \
[ "$(grep -c 'overflow-x-auto scrollbar-thin' src/app/backtest/page.tsx)" -ge "3" ] && \
[ "$(grep -c 'overflow-x-auto scrollbar-thin' src/app/paper-trading/page.tsx)" -ge "2" ] && \
[ "$(grep -c 'overflow-x-auto scrollbar-thin' src/app/reports/page.tsx)" -ge "1" ] && \
npm run lint 2>&1 | grep -c '@phosphor-icons/react' | grep -q '^0$'
```

Plus:
- `tsc_clean`: exit 0
- `lint_clean`: 0 phosphor warnings, no NEW errors
- `no_backend_changes`: only the 3 frontend pages + handoff/* rolling
- `agents_unchanged`: agents/page.tsx + paper-trading/learnings/page.tsx untouched (already compliant)

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. reports two-zone shell complete (header pinned, content scrolls).
3. PageSkeleton imported + used in reports loading guard.
4. 6+ new scrollbar-thin instances across the 3 files.
5. backtest error banners now render inside the scrollable zone, not in the fixed-header div.
6. agents/page.tsx + paper-trading/learnings/page.tsx UNCHANGED.
7. tsc + lint clean.
8. No backend changes.
