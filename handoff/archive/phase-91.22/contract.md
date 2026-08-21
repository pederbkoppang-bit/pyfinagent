# Sprint Contract -- phase-91.22
Step: Recharts chart tooltips illegible -- itemStyle never set, defaults to black text on dark backgrounds

## Research Gate
researcher (tier=simple) gate_passed=true.
Brief: `handoff/current/research_brief_91.22.md`.
- 7 external sources read in full (5 W3C normative/technique + 2 official Recharts docs/wiki).
- Recency scan performed: confirmed this repo runs recharts 2.15.4 (not v3), so v3-era changes don't apply; WCAG 2.2 thresholds unchanged 2024-2026.
- Key findings (major scope correction vs. my original filing):
  - **True scope is 16 files / 20 Tooltip instances, not the 7 I originally named.** 9 additional files were found: `app/backtest/page.tsx`, `app/paper-trading/nav/page.tsx`, `app/reports/page.tsx`, plus `MfeMaeScatter.tsx`, `BudgetDashboard.tsx`, `ComputeCostBreakdown.tsx`, `OptimizerProgressChart.tsx`, `PerfProgressChart.tsx`, `SharpeHistoryChart.tsx`.
  - **Two populations.** 10 files / ~15 Tooltip instances render Recharts' `DefaultTooltipContent` (addressable with `itemStyle`) -- `SectorDashboard.tsx`, `OptimizerInsights.tsx` (x2), `PaperReconciliationChart.tsx`, `TransformerForecastPanel.tsx`, `RedLineMonitor.tsx`, `StrategyDetail.tsx`, `StockChart.tsx` (x2), `app/backtest/page.tsx` (x3), `app/paper-trading/nav/page.tsx`, `app/reports/page.tsx` (x2). The other 6 files supply a custom `content` prop, so `DefaultTooltipContent` never runs and an `itemStyle` fix there is a **guaranteed no-op** -- explicitly out of scope for this step (their contrast needs its own separate audit).
  - **Root cause**: Recharts' `DefaultTooltipContent.js:58` sets item-row color to `entry.color || '#000'` when `itemStyle` is absent. Measured (G18 formula): `#000` on `#0f172a` = 1.18:1, vs. the WCAG 2.2 SC 1.4.3 floor of 4.5:1.
  - **`contentStyle.color` is a no-op for item rows** (it lands on the wrapper `<div>`; the item text is an inline style on the `<li>`, and inline beats inheritance). Three files already tried this exact wrong fix: `RedLineMonitor.tsx:194`, `StrategyDetail.tsx:87`, `TransformerForecastPanel.tsx:122` -- their dead `color` keys are removed as part of this fix, not left as misleading residue.
  - **The label is not broken** -- `labelStyle` has no default color and inherits the app's ambient `#e2e8f0` (14.48:1). No labelStyle changes needed.
  - **A per-series `entry.color` fallback is background-coupled and re-breakable**: this repo uses two tooltip backgrounds (`#0f172a` everywhere else, `#1e293b` on `app/backtest/page.tsx:962`), and `#ef4444`/`#f43f5e`/`#a855f7` pass contrast on the first and FAIL on the second. A uniform, explicit `itemStyle` color (not tied to series color) is the only variant invariant to this. Chose `#e2e8f0` (14.48:1 / 11.87:1 on both backgrounds), matching `design-tokens.ts`'s `text.primary` contrast tier -- exported as a shared constant (`frontend/src/lib/chart-tooltip-style.ts::CHART_TOOLTIP_ITEM_STYLE`) so all ~15 sites use one definition, not 15 copies (per my own immutable criteria's explicit preference).
  - Two out-of-scope items disclosed by the research, NOT folded into this step: (a) contrast audit of the 6 custom-`content` tooltips (separate mechanism, separate fix); (b) WCAG SC 1.4.13 (hover/focus content dismissibility) is a distinct exposure class from contrast, flagged but not actioned here.

## Hypothesis
Adding `itemStyle={CHART_TOOLTIP_ITEM_STYLE}` to all 10 files / ~15 Tooltip instances that render Recharts' default tooltip content (and removing the 3 dead `contentStyle.color` no-ops) makes every default-rendered tooltip's item text meet WCAG 2.2 AA contrast (>=4.5:1) against both tooltip backgrounds in use, without needing to touch the 6 custom-content tooltips (a different, out-of-scope population) or any series/legend coloring.

## Success Criteria (immutable)
```
grep -c 'itemStyle' frontend/src/components/SectorDashboard.tsx frontend/src/components/OptimizerInsights.tsx frontend/src/components/PaperReconciliationChart.tsx frontend/src/components/TransformerForecastPanel.tsx frontend/src/components/RedLineMonitor.tsx frontend/src/components/StrategyDetail.tsx frontend/src/components/StockChart.tsx
```
Plus sub-criteria (copied verbatim from `.claude/masterplan.json` phase-91 step 91.22):
- every Recharts `<Tooltip>` in the 9 identified components (and any others found by the same grep sweep during GENERATE) sets an explicit itemStyle with a readable color (e.g. #e2e8f0 / text-slate-100, matching the project's contrast tokens), ideally via one shared constant/style object rather than 9 copies
- the command above returns a non-zero itemStyle count for every listed file after the fix
- a live Playwright screenshot of the Sector Rotation tooltip (hover state) shows both the label and the value clearly legible against the dark background
- spot-check at least 2 other affected charts (e.g. StockChart, RedLineMonitor) post-fix to confirm the shared fix pattern was actually applied everywhere, not just to the one reported instance

## Plan (PRE-commit; will NOT diverge in Generate)
1. Add shared constant `CHART_TOOLTIP_ITEM_STYLE = { color: "#e2e8f0" }` in `frontend/src/lib/chart-tooltip-style.ts` (done during research write-up).
2. Add `itemStyle={CHART_TOOLTIP_ITEM_STYLE}` to all ~15 default-content Tooltip instances across the 10 files listed above (the 7 immutable-criteria-named files plus the 3 the research found: `app/backtest/page.tsx`, `app/paper-trading/nav/page.tsx`, `app/reports/page.tsx` -- included because the immutable criteria explicitly say "and any others found by the same grep sweep during GENERATE").
3. Remove the 3 dead `contentStyle.color` no-ops (`RedLineMonitor.tsx:194`, `StrategyDetail.tsx:87`, `TransformerForecastPanel.tsx:122`).
4. Run the immutable grep command across the 7 named files, confirm non-zero everywhere.
5. Capture a live Playwright screenshot of the Sector Rotation tooltip (hover), plus spot-checks on StockChart and RedLineMonitor per the immutable criteria.
6. Do NOT touch the 6 custom-`content` files, `labelStyle` anywhere, series/legend colors, or the recharts version.

## Scope honesty / out-of-scope
- The 6 custom-`content` tooltip files (`MfeMaeScatter.tsx`, `BudgetDashboard.tsx`, `ComputeCostBreakdown.tsx`, `OptimizerProgressChart.tsx`, `PerfProgressChart.tsx`, `SharpeHistoryChart.tsx`) are NOT touched -- an `itemStyle` prop is inert there. Their contrast needs a separate audit of their own custom render functions; queued as a new step.
- WCAG SC 1.4.13 (hoverable/dismissible/persistent tooltip content) is a distinct accessibility question from contrast; not actioned here.
- No recharts version change (stays on 2.15.4).

## References
- Research brief: `handoff/current/research_brief_91.22.md`
- Filed from: `.claude/masterplan.json` phase-91 step 91.22 (originally 86.148, renumbered during the same-day phase-91 split)
