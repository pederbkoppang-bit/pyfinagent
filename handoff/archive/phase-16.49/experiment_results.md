---
step: phase-16.49
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/app/reports/page.tsx (CRITICAL: two-zone shell + tab-bar pin + PageSkeleton + scrollbar-thin)
  - frontend/src/app/backtest/page.tsx (error banners moved + 4 scrollbar-thin)
  - frontend/src/app/paper-trading/page.tsx (2 scrollbar-thin)
---

# Experiment Results -- phase-16.49

UX audit pass B: 4 highest-complexity pages (~3712 LOC across 5 files
including learnings sub-tab) audited against frontend.md +
frontend-layout.md.

## Findings vs fixes

`agents/page.tsx` (728 LOC) and `paper-trading/learnings/page.tsx`
(22 LOC) — both already COMPLIANT, untouched.

11 violations across 3 files, all fixed:

| Sev | File:line | Fix |
|-----|-----------|-----|
| CRITICAL | `reports/page.tsx:223` | Single-zone shell -> canonical two-zone (header+tab-bar pinned, content scrolls) |
| LOW | `reports/page.tsx:253` | Plain-text loading -> `<PageSkeleton />` |
| LOW | `reports/page.tsx:486` | +scrollbar-thin |
| MED | `backtest/page.tsx:683-709` | Error banners moved from fixed-header zone to scrollable zone |
| LOW | `backtest/page.tsx:209,1055,1118,1301` | +scrollbar-thin (4 sites) |
| LOW | `paper-trading/page.tsx:533,617` | +scrollbar-thin (positions + trades tables) |

DEFERRED per research recommendation: backtest RunSelector relocation
from fixed-header (V5; layout-sensitive, defer unless visually
confirmed).

## Verification

```
$ npx tsc --noEmit
[exit 0]

$ grep -q "flex flex-1 flex-col overflow-hidden" src/app/reports/page.tsx && echo "reports shell OK"
reports shell OK

$ grep -q "PageSkeleton" src/app/reports/page.tsx
[OK]

$ grep -c "overflow-x-auto scrollbar-thin" src/app/backtest/page.tsx
3
$ grep -c "overflow-x-auto scrollbar-thin" src/app/paper-trading/page.tsx
2
$ grep -c "overflow-x-auto scrollbar-thin" src/app/reports/page.tsx
1

$ npm run lint 2>&1 | grep -c '@phosphor-icons/react'
0

ALL VERIFICATION PASS
```

backtest also gained an `overflow-y-auto scrollbar-thin` on the
RunSelector dropdown (line 209), bringing total scrollbar-thin
additions in backtest to 4 (3 overflow-x + 1 overflow-y).

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | reports two-zone shell | PASS |
| 2 | reports PageSkeleton import + use | PASS |
| 3 | 6+ new scrollbar-thin instances across 3 files | PASS (3+2+1=6) |
| 4 | backtest error banners now in scrollable zone | PASS (cut from 683-709, pasted after scrollable-zone open at 790) |
| 5 | agents/page.tsx + paper-trading/learnings unchanged | PASS |
| 6 | tsc clean | PASS |
| 7 | lint phosphor count = 0 | PASS |
| 8 | no backend changes | PASS |

## Honest disclosures

1. **reports two-zone restructure** wraps all existing JSX without
   touching content. Header + tab bar moved into `flex-shrink-0`
   pinned zone; content body wrapped in `flex-1 overflow-y-auto
   scrollbar-thin px-6 py-6 md:px-8`. Closing `</div>` added before
   `</main>`.

2. **PageSkeleton import** added alongside existing Sidebar/BentoCard
   imports. The plain-text "Loading reports..." replaced with
   `<PageSkeleton />` from the existing Skeleton.tsx component
   library.

3. **backtest error banner relocation** was a clean cut+paste of
   lines 683-709 from inside the fixed-header div. Placeholder
   comment left at the original location documenting the move.

4. **RunSelector relocation deferred** per research-brief
   recommendation. The component was flagged MED but layout-sensitive
   — moving Tier-4 controls to the scrollable zone could create
   regressions if it was intentionally placed above the tab bar for
   global access. Visual check needed.

5. **paper-trading scrollbar-thin** added to both positions (line 533)
   and trades (line 617) table wrappers.

6. **34 pre-existing react-hooks lint warnings unchanged.**

7. **No backend changes; no settings page changes.**

## Closes

Task list item #71. Masterplan step phase-16.49.

## Next

Spawn Q/A.
