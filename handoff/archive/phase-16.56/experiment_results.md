---
step: phase-16.56
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/app/sovereign/page.tsx (grid proportions: 3+2 -> 2+3, swapped col-span)
  - frontend/src/components/AlphaLeaderboard.tsx (cell padding: px-3 -> px-2.5, all 8 occurrences)
---

# Experiment Results -- phase-16.56

## What was done

Swapped the sovereign two-hero grid proportions so AlphaLeaderboard now
gets 60% of the row width (was 40%) and RedLineMonitor gets 40% (was
60%). Also tightened AlphaLeaderboard table cell horizontal padding
from `px-3` to `px-2.5` for ~14px additional table width margin. All 7
columns should now fit without horizontal scroll.

## Deliverables

### `frontend/src/app/sovereign/page.tsx` (grid swap)

```tsx
- <div className="lg:col-span-3">  ... RedLineMonitor ... </div>
- <div className="lg:col-span-2 h-full"> ... AlphaLeaderboard ... </div>
+ <div className="lg:col-span-2">  ... RedLineMonitor ... </div>
+ <div className="lg:col-span-3 h-full"> ... AlphaLeaderboard ... </div>
```

`lg:grid-cols-5` parent unchanged. `h-full` on AlphaLeaderboard wrapper
preserved (16.55 carry-over so the card still stretches to row height).

### `frontend/src/components/AlphaLeaderboard.tsx` (cell padding)

8 occurrences replaced via Edit replace_all:

```tsx
- className="... px-3 py-2.5 ..."
+ className="... px-2.5 py-2.5 ..."
```

(7 `<td>` cells + 1 `<th>` header rule.)

## Verification (verbatim, immutable from masterplan)

```
$ cd frontend && npx tsc --noEmit
(exit 0; no output)

$ cd frontend && npm run lint
0 errors, 34 warnings (all pre-existing in unmodified files)
```

## Files touched

| Path | Action | Note |
|------|--------|------|
| `frontend/src/app/sovereign/page.tsx` | edit (L140 + L148) | grid: 3+2 -> 2+3 (Alpha gets 60% width) |
| `frontend/src/components/AlphaLeaderboard.tsx` | replace_all (8) | cell padding px-3 -> px-2.5 |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-16.56-research-brief.md` | created (internal-only) | -- |

NO RedLineMonitor changes (height stays h-64 from 16.55).

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Grid swapped: AlphaLeaderboard now `lg:col-span-3` (60%) | PASS |
| 2 | RedLineMonitor now `lg:col-span-2` (40%) | PASS |
| 3 | Cell padding tightened on all 7 columns + header | PASS (8 occurrences via replace_all) |
| 4 | tsc + lint clean | PASS |
| 5 | Operator visual: no horizontal scroll on AlphaLeaderboard | DEFERRED (operator will refresh) |

## Honest disclosures

1. **RedLineMonitor narrower** -- 40% instead of 60%. The chart is mostly flat NAV data so should remain readable. If the chart looks cramped or the x-axis labels overlap, a follow-up could rotate labels or tighten the chart's right margin.

2. **Cell padding shrunk modestly** (1px per side per column = ~14px total). Visual density should be similar.

3. **No column hiding or truncation** -- all 7 columns remain visible at full content width.

4. **No regression risk on homepage** -- AlphaLeaderboard not used on homepage.

5. **Cycle-2 not needed.** First-pass clean.

## Closes

Net-new task #85 (UAT-16.56). Adds new step phase-16.56 to masterplan.

## Next

Spawn Q/A.
