---
step: phase-16.48
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/app/login/page.tsx (min-h-screen -> h-screen overflow-hidden)
  - frontend/src/app/signals/page.tsx (canonical two-zone shell)
  - frontend/src/app/performance/page.tsx (two-zone shell + scrollbar-thin + loading/empty states)
  - frontend/src/components/AlphaLeaderboard.tsx (scrollbar-thin)
---

# Experiment Results -- phase-16.48

UX audit pass A: low-risk pages + 5 components against frontend.md +
frontend-layout.md. Settings page (1243 LOC) was the canonical reference;
not modified.

## Violations fixed (6 across 4 files)

| Sev | File:line | Fix |
|-----|-----------|-----|
| HIGH | `login/page.tsx:35` | `min-h-screen` -> `h-screen ... overflow-hidden` |
| MED | `signals/page.tsx:88-90` | Single-zone -> canonical two-zone (header pinned, content scrolls) |
| MED | `performance/page.tsx:41-43` | Same two-zone fix |
| LOW | `performance/page.tsx:179` | `overflow-x-auto` -> `overflow-x-auto scrollbar-thin` |
| LOW | `performance/page.tsx` cost-history | Added loading spinner + empty state ("No cost history yet") |
| LOW | `AlphaLeaderboard.tsx:190` | Same scrollbar-thin add |

## Files NOT touched (clean per audit)

`/Users/ford/.openclaw/workspace/pyfinagent/frontend/src/app/sovereign/page.tsx`,
`/Users/ford/.openclaw/workspace/pyfinagent/frontend/src/app/page.tsx` (home; recently overhauled in 16.42-16.47),
`/Users/ford/.openclaw/workspace/pyfinagent/frontend/src/components/Sidebar.tsx`,
`/Users/ford/.openclaw/workspace/pyfinagent/frontend/src/components/OpsStatusBar.tsx`,
`/Users/ford/.openclaw/workspace/pyfinagent/frontend/src/components/RedLineMonitor.tsx`,
`/Users/ford/.openclaw/workspace/pyfinagent/frontend/src/components/StrategyDetail.tsx`.

## Verification

```
$ npx tsc --noEmit && \
  ! grep -q "min-h-screen" src/app/login/page.tsx && \
  grep -q "flex flex-1 flex-col overflow-hidden" src/app/signals/page.tsx && \
  grep -q "flex flex-1 flex-col overflow-hidden" src/app/performance/page.tsx && \
  npm run lint 2>&1 | grep -c '@phosphor-icons/react' | grep -q '^0$'
[exit 0 -- ALL PASS]

$ overflow-x-auto scrollbar-thin count: 1 in performance + 1 in AlphaLeaderboard = 2 (matches expected)
$ phosphor lint: 0
```

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | login no longer has min-h-screen | PASS |
| 2 | signals two-zone shell | PASS |
| 3 | performance two-zone shell | PASS |
| 4 | scrollbar-thin on the 2 noted overflow-x-auto containers | PASS |
| 5 | tsc clean | PASS |
| 6 | lint phosphor count = 0 | PASS |

## Honest disclosures

1. **login centering preserved.** `flex items-center justify-center` still works because `h-screen` provides a definite height; `overflow-hidden` is the new addition per rule.

2. **signals + performance two-zone restructure** wraps the existing JSX in `<div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">` after pulling the header out into `<div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">`. The existing page content (BentoCards, tables, etc.) is unchanged inside the scrollable zone.

3. **performance cost-history loading + empty state** added. Previously the section silently disappeared when `costHistory.length === 0`; now it renders a spinner during loading and an icon + "No cost history yet" empty state with explanation.

4. **No backend changes.** No new dependencies. No file deletions.

5. **34 pre-existing react-hooks lint warnings unchanged.**

## Closes

Task list item #70. Masterplan step phase-16.48.

## Next

Spawn Q/A.
