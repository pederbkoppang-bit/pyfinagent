# Live Check -- phase-91.22
Step: Recharts chart tooltips illegible -- itemStyle never set

## Required evidence (per masterplan step 91.22's `verification.live_check`)
"Playwright screenshot of the Sector Rotation chart tooltip on hover, post-fix, showing legible
value text, plus one additional chart's tooltip as a spot-check"

## Evidence

All three captures taken via Playwright MCP behind the real, authenticated NextAuth session,
URL-confirmed target pages (not `/login` redirects).

### 1. Primary: Sector Rotation tooltip (the exact chart from the operator's original report)
`handoff/current/captures_91.22/sector_rotation_tooltip_fixed.png`

Navigated to `/signals`, fetched signals for MRVL, expanded "Sector + Macro deep dive", hovered the
Healthcare bar in the Sector Rotation (3M Returns) chart. Tooltip shows both "Healthcare" (label)
and "Return : 16.9%" (value) clearly legible against the dark `#0f172a` background -- this is the
exact defect the operator screenshotted, now fixed.

### 2. Spot-check 1: StockChart (MRVL Price Chart)
`handoff/current/captures_91.22/stockchart_tooltip_spotcheck.png`

Navigated to a real MRVL report, hovered the price chart. Tooltip shows "Mar 5", "Volume : 42.6M",
"Close : $75.62", "SMA 50 : $81.46" all legible.

### 3. Spot-check 2: RedLineMonitor (homepage NAV chart)
`handoff/current/captures_91.22/redlinemonitor_tooltip_spotcheck.png`

Navigated to the homepage, hovered the Red Line Monitor chart. Tooltip shows "2026-08-04" and
"nav : 23803.94" both legible. This component is also one of the 3 that had a dead
`contentStyle.color` no-op removed as part of this fix.

## Verification command re-run
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
Non-zero for every named file. The same command additionally confirmed non-zero on the 3 extra
route files the research found (`app/backtest/page.tsx`: 3, `app/paper-trading/nav/page.tsx`: 1,
`app/reports/page.tsx`: 2), which are in scope per the criterion's "any others found by the same
grep sweep during GENERATE" clause.
