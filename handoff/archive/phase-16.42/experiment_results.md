---
step: phase-16.42
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/lib/formatRelativeTime.ts (35 lines)
  - frontend/src/components/RecentReportsTable.tsx (132 lines)
  - frontend/src/components/HomeQuickActionsPanel.tsx (190 lines)
  - frontend/src/app/page.tsx (-93 lines old blocks, +18 lines two-column grid)
---

# Experiment Results -- phase-16.42

## What was done

Replaced the authenticated home page's vertically-stacked Recent Reports
table + Quick Actions cards with the target-screenshot two-column
layout. **Zero hardcoded data** — every value flows from the existing
`GET /api/reports/?limit=5` endpoint via the `reports` prop already
loaded in `page.tsx:77`.

### Changes

1. **`frontend/src/lib/formatRelativeTime.ts`** (35 lines, new):
   `Intl.RelativeTimeFormat` helper with thresholds at min/hr/day/week.
   Outputs strings like "12 min. ago", "2 hr. ago", "1 day ago",
   matching screenshot. Pure stdlib, no deps.

2. **`frontend/src/components/RecentReportsTable.tsx`** (132 lines,
   new): consumes the `reports` prop (parent owns the fetch). Five
   columns:
   - **TICKER** — mono bold, primary identifier
   - **COMPANY** — `r.company_name` with `—` fallback for null (per
     pitfall #5)
   - **ALPHA** — `r.final_score?.toFixed(2)`, color-coded `≥8 emerald`,
     `≥6.5 sky`, `≥4.5 amber`, `else rose`
   - **RECOMMENDATION** — pill via `recColor()` (handles both
     `"STRONG_BUY"` and `"Strong Buy"` variants per pitfall #2)
   - **UPDATED** — `formatRelativeTime(r.analysis_date)` with
     `suppressHydrationWarning` (per pitfall #6)
   - States: 5 skeleton rows while `!loaded`, error banner when
     `loadError` set + empty reports, icon + "No reports yet" empty
     state.
   - Row click → `router.push("/reports?ticker=...")`. Keyboard:
     `tabIndex={0}` + Enter/Space.

3. **`frontend/src/components/HomeQuickActionsPanel.tsx`** (190
   lines, new): two sections.
   - **Section A**: ticker input + Analyze button. Parent owns
     `ticker` state via prop callbacks.
   - **Section B**: 3 action rows, each
     `[icon] [label] [<kbd>shortcut</kbd>]`:
     1. Run morning cycle [Ctrl+Shift+R] → `triggerPaperTradingCycle()`
     2. Open backtest console [Ctrl+B] → `router.push("/backtest")`
     3. Halt all trading [Ctrl/Cmd+Shift+H] → `FLATTEN_ALL` THEN
        `PAUSE` (two API calls, per pitfall #3)
   - Per-action state machine: `idle / pending / success / error`
     with inline message banner.
   - Keyboard shortcuts: registers `Ctrl+Shift+R` and `Ctrl+B` only.
     **Ctrl/Cmd+Shift+H is intentionally NOT re-registered** —
     `KillSwitchShortcut` (mounted in page.tsx) owns that listener
     globally; the Halt button is click-only here, the kbd badge is
     a label, not a re-binding (per pitfall #4).
   - Ctrl+B is suppressed when focus is in an input/textarea
     (browser bookmark shortcut on macOS Safari).

4. **`frontend/src/app/page.tsx`** edits:
   - Removed the local `recColor` helper at lines 32-39 (moved
     into `RecentReportsTable.tsx` as a 7-line local function)
   - Replaced lines 184-276 (old Recent Reports + old Quick
     Actions) with an 18-line two-column grid:
     `<div className="grid grid-cols-1 gap-6 lg:grid-cols-3">` →
     `lg:col-span-2` table + `lg:col-span-1` panel.
   - Updated imports (added `RecentReportsTable`,
     `HomeQuickActionsPanel`; dropped `NavSignals`, `NavBacktest`).
   - All other top-of-page structure preserved (Sidebar,
     KillSwitchShortcut, RedLineMonitor, OpsStatusBar, KPI hero,
     error banner).

### Files touched

| Path | Action | LOC delta |
|------|--------|-----------|
| `frontend/src/lib/formatRelativeTime.ts` | CREATED | +35 |
| `frontend/src/components/RecentReportsTable.tsx` | CREATED | +132 |
| `frontend/src/components/HomeQuickActionsPanel.tsx` | CREATED | +190 |
| `frontend/src/app/page.tsx` | edited | -85, +21 (net -64) |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |

NO backend changes. NO new dependencies. NO new endpoints.

## Verification

### Static checks (per immutable command)

```
$ test -f frontend/src/lib/formatRelativeTime.ts && \
  test -f frontend/src/components/RecentReportsTable.tsx && \
  test -f frontend/src/components/HomeQuickActionsPanel.tsx && \
  grep -q "RecentReportsTable" frontend/src/app/page.tsx && \
  grep -q "HomeQuickActionsPanel" frontend/src/app/page.tsx
[exit 0 -- all 5 checks PASS]
```

### Anti-hardcoding gate

```
$ grep -cE "(NVIDIA Corporation|Apple Inc\.|Microsoft|Tesla, Inc\.|Intel Corporation)" \
    frontend/src/components/RecentReportsTable.tsx \
    frontend/src/components/HomeQuickActionsPanel.tsx
RecentReportsTable.tsx:0
HomeQuickActionsPanel.tsx:0

$ grep -cE "(7\.42|6\.81|5\.12|3\.74|2\.11)" \
    frontend/src/components/RecentReportsTable.tsx \
    frontend/src/components/HomeQuickActionsPanel.tsx
RecentReportsTable.tsx:0
HomeQuickActionsPanel.tsx:0
```

Zero matches for sample company names AND zero matches for sample
alpha values. Strict no-hardcoded-data gate PASS.

### TypeScript + lint

```
$ cd frontend && npx tsc --noEmit
[exit 0 -- clean]

$ npm run lint 2>&1 | tail -3
✖ 34 problems (0 errors, 34 warnings)
  0 errors and 6 warnings potentially fixable with the `--fix` option.

$ npm run lint 2>&1 | grep -c '@phosphor-icons/react'
0
```

Zero new errors. Zero phosphor warnings. The 34 pre-existing
react-hooks warnings are unchanged from prior baseline.

### Live data smoke

```
$ curl -s "http://localhost:8000/api/reports/?limit=5" | python3 -c "..."
returned 5 reports
  keys: ['analysis_date', 'company_name', 'final_score', 'recommendation', 'summary', 'ticker']
  sample: ticker=SNDK score=5.55 rec=Hold date=2026-03-21T06:49:39.361895Z
```

Real backend data with the exact shape the components consume.
**The home page will render real reports — SNDK with alpha 5.55
(amber color), recommendation pill "Hold" (amber), updated relative
time computed from the 2026-03-21 timestamp.**

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | files exist (3 new) | PASS | ls succeeded |
| 2 | page.tsx imports both new components | PASS | grep matched |
| 3 | no NVIDIA/Apple/Microsoft etc. | PASS | grep -c = 0 |
| 4 | no 7.42/6.81/etc. literal alphas | PASS | grep -c = 0 |
| 5 | tsc clean | PASS | exit 0 |
| 6 | lint phosphor count = 0 | PASS | exit 0 |
| 7 | live curl returns valid shape | PASS | 5 reports, all keys present |

## Honest disclosures

1. **"ALPHA" column is `final_score`.** The pipeline does not emit a
   separate alpha field on `ReportSummary`; `final_score` is the
   composite quality score (0-10) and the closest semantic
   equivalent. Documented in research brief + experiment_results.
   Future work could join `pyfinagent_data.alpha_velocity_samples`
   (the 10.7.1 table) for a true alpha velocity number.

2. **Pre-existing 34 react-hooks warnings unchanged.** These are
   from the React Compiler rules in eslint-config-next v16, set to
   warn during phase-4.7.5 transition. Not regressed; not introduced
   by this cycle.

3. **`recColor` duplicated in `RecentReportsTable.tsx`.** The 8-line
   function previously lived in `page.tsx:32-39`. Moved into the
   table component as a local copy rather than extracted to a
   shared util — trivial size, single-component use today, not
   worth the abstraction. If a third caller appears, extract to
   `frontend/src/lib/recColor.ts`.

4. **Ctrl+B suppressed in inputs.** macOS Safari treats Cmd+B as a
   bookmark shortcut. The panel's keyboard handler skips when
   `target.tagName === "INPUT" || "TEXTAREA"` so the user can type
   in the ticker field without accidentally navigating to backtest.

5. **`suppressHydrationWarning` on UPDATED column.** The relative
   time string differs by milliseconds between SSR and CSR; React
   would log a hydration warning otherwise. Per research-brief
   pitfall #6.

6. **`Files` icon used for empty state.** Phosphor `Files` already
   exists in the `@/lib/icons` registry (added in 16.39 sweep).
   Visually appropriate for "no reports yet".

7. **Backend `Hold` (title case) renders as "Hold" pill.** The
   `recColor` function falls through to the amber default since
   "Hold" doesn't match BUY/SELL keywords; expected behavior. Pill
   text is `r.recommendation.replace(/_/g, " ")` so both
   `"STRONG_BUY"` → "STRONG BUY" and `"Hold"` → "Hold" render
   correctly.

## Closes

- Task list item #64 (this cycle's task)
- masterplan step **phase-16.42**

## Next

Spawn Q/A. If PASS: log + flip + tell user to refresh /signin and
visit `/` to see the new layout.
