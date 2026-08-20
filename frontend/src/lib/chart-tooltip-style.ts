/**
 * phase-91.22: shared Recharts <Tooltip> item-row style. Recharts'
 * DefaultTooltipContent falls back to `entry.color || '#000'` for item text
 * when `itemStyle` is not set (installed recharts 2.15.4,
 * DefaultTooltipContent.js:58) -- 1.18:1 on this app's #0f172a tooltip
 * background, far under the WCAG 2.2 SC 1.4.3 text floor of 4.5:1. A
 * per-series `entry.color` fallback is also background-coupled: colors that
 * pass on #0f172a (most components) can fail on #1e293b
 * (app/backtest/page.tsx), so a fixed, uniform color is the only variant
 * that's invariant to which tooltip background a given chart uses.
 *
 * #e2e8f0 measures 14.48:1 on #0f172a and 11.87:1 on #1e293b (both AA/AAA),
 * matching design-tokens.ts's text.primary contrast tier.
 */
export const CHART_TOOLTIP_ITEM_STYLE = { color: "#e2e8f0" } as const;
