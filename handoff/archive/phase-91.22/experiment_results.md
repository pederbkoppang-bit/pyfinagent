# Experiment Results -- phase-91.22
Step: Recharts chart tooltips illegible -- itemStyle never set

## What was built/changed
1. New shared constant `frontend/src/lib/chart-tooltip-style.ts::CHART_TOOLTIP_ITEM_STYLE = { color: "#e2e8f0" }`.
2. Added `itemStyle={CHART_TOOLTIP_ITEM_STYLE}` to all ~15 default-content `<Tooltip>` instances across 10 files (the research's corrected, wider scope):
   - `frontend/src/components/SectorDashboard.tsx` (1)
   - `frontend/src/components/OptimizerInsights.tsx` (2)
   - `frontend/src/components/PaperReconciliationChart.tsx` (1)
   - `frontend/src/components/TransformerForecastPanel.tsx` (1)
   - `frontend/src/components/RedLineMonitor.tsx` (1)
   - `frontend/src/components/StrategyDetail.tsx` (1)
   - `frontend/src/components/StockChart.tsx` (2)
   - `frontend/src/app/backtest/page.tsx` (3)
   - `frontend/src/app/paper-trading/nav/page.tsx` (1)
   - `frontend/src/app/reports/page.tsx` (2)
3. Removed the 3 dead `contentStyle.color` no-ops (they never reached item rows -- confirmed a no-op by the research; left in place they'd misrepresent as load-bearing):
   - `RedLineMonitor.tsx`, `StrategyDetail.tsx`, `TransformerForecastPanel.tsx`
4. Deliberately NOT touched: the 6 custom-`content` tooltip files (`itemStyle` is inert there), `labelStyle` anywhere, series/legend colors, recharts version.

## File list
- `frontend/src/lib/chart-tooltip-style.ts` (new)
- `frontend/src/components/SectorDashboard.tsx`
- `frontend/src/components/OptimizerInsights.tsx`
- `frontend/src/components/PaperReconciliationChart.tsx`
- `frontend/src/components/TransformerForecastPanel.tsx`
- `frontend/src/components/RedLineMonitor.tsx`
- `frontend/src/components/StrategyDetail.tsx`
- `frontend/src/components/StockChart.tsx`
- `frontend/src/app/backtest/page.tsx`
- `frontend/src/app/paper-trading/nav/page.tsx`
- `frontend/src/app/reports/page.tsx`

## Verbatim verification command output
```
$ grep -c 'itemStyle' frontend/src/components/SectorDashboard.tsx frontend/src/components/OptimizerInsights.tsx frontend/src/components/PaperReconciliationChart.tsx frontend/src/components/TransformerForecastPanel.tsx frontend/src/components/RedLineMonitor.tsx frontend/src/components/StrategyDetail.tsx frontend/src/components/StockChart.tsx
frontend/src/components/SectorDashboard.tsx:1
frontend/src/components/OptimizerInsights.tsx:2
frontend/src/components/PaperReconciliationChart.tsx:1
frontend/src/components/TransformerForecastPanel.tsx:1
frontend/src/components/RedLineMonitor.tsx:1
frontend/src/components/StrategyDetail.tsx:1
frontend/src/components/StockChart.tsx:2
```
Non-zero for every named file, matching the immutable criteria. Confirmed the same for the 3
additional route files the research found (backtest/page.tsx: 3, paper-trading/nav/page.tsx: 1,
reports/page.tsx: 2).

```
$ npx tsc --noEmit -p tsconfig.json
(no output -- compiles clean)
```

## Live captures (Playwright, real authenticated NextAuth session)
1. `handoff/current/captures_91.22/sector_rotation_tooltip_fixed.png` -- the exact chart from the
   operator's original screenshot (Signals page, Sector deep-dive, MRVL). Hover on the Healthcare
   bar now shows both "Healthcare" and "Return : 16.9%" clearly legible against the dark tooltip
   background -- the value row was the one reported illegible.
2. `handoff/current/captures_91.22/stockchart_tooltip_spotcheck.png` -- spot-check per the
   immutable criteria. MRVL Price Chart tooltip shows "Mar 5", "Volume : 42.6M", "Close : $75.62",
   "SMA 50 : $81.46" all legible.
3. `handoff/current/captures_91.22/redlinemonitor_tooltip_spotcheck.png` -- second spot-check.
   Homepage Red Line Monitor tooltip shows "2026-08-04" and "nav : 23803.94" both legible.

All three captures verified via Next.js Fast Refresh, no restart needed for this frontend-only change.

## Artifact shape
- Code diff: 1 new shared-constant file, 10 files edited (15 itemStyle additions + 3 dead-code removals), 0 files with custom content touched.
- Live evidence: 3 Playwright screenshots across 3 different components, matching the immutable criteria's requirement of the primary reported instance (Sector Rotation) plus 2 spot-checks (StockChart, RedLineMonitor).
- Scope was corrected upward during GENERATE per the research's finding (16 files total vs. the 7 originally named) -- disclosed in the contract, not silently expanded.

## Follow-up items queued (per the research's explicit disclosure of what's out of scope)
1. Contrast audit of the 6 custom-`content` tooltip files (`MfeMaeScatter.tsx`, `BudgetDashboard.tsx`, `ComputeCostBreakdown.tsx`, `OptimizerProgressChart.tsx`, `PerfProgressChart.tsx`, `SharpeHistoryChart.tsx`) -- itemStyle is inert there; each needs its own render-function fix.
2. WCAG SC 1.4.13 (hoverable/dismissible/persistent tooltip content) -- a distinct accessibility question from contrast, not actioned here.
Both to be filed as new masterplan steps after this round of fixes completes.
