---
name: tooltip-contrast-91-22
description: Recharts tooltip contrast (step 91.22) -- contentStyle.color is a NO-OP for item rows and 3 sites already rely on it; the label is NOT the bug; 6 of 16 files bypass the default so an itemStyle fix is vacuous there
metadata:
  type: project
---

Step 91.22 (Recharts `<Tooltip>` contrast on dark backgrounds). Five findings that
are NOT derivable from a quick read of the call sites.

**Fact 1 -- `contentStyle.color` never reaches the item rows, and the codebase
already bets on it.** `contentStyle` lands on the wrapper `<div>`;
`DefaultTooltipContent.js:58` puts `color: entry.color || '#000'` **inline on the
`<li>`**, and an inline colour beats inheritance. `RedLineMonitor.tsx:194`,
`StrategyDetail.tsx:87`, `TransformerForecastPanel.tsx:122` all set
`color:"#e2e8f0"` in `contentStyle` -- the intent is in the code, in the wrong
prop, doing nothing for the values users read. Only `itemStyle` reaches them.

**Fact 2 -- the label is NOT the bug (plausible hypothesis, refuted).**
`finalLabelStyle = { margin: 0, ...labelStyle }` has **no default colour**, so the
label inherits `#e2e8f0` from `globals.css:14` (`body`) = 14.48:1. The 6 tooltips
with no `labelStyle` are already compliant. Do not spend fix budget there. This
also means a visual spot-check of the LABEL "confirms" a fix that never landed --
which is how the three Fact-1 near-misses survived.

**Fact 3 -- the compliance verdict flips on the BACKGROUND, not the series.** Two
tooltip backgrounds are live: `#0f172a` (most) and `#1e293b`
(`app/backtest/page.tsx:962`). `#ef4444` 4.74 -> 3.89, `#f43f5e` 4.86 -> 3.98,
`#a855f7` 4.51 -> 3.70. Any per-series-colour fix is silently re-breakable by a
`contentStyle` edit. A uniform `itemStyle.color` is the only background-invariant
variant.

**Fact 4 -- root cause is a WCAG TIER mismatch, not a palette accident.** SC
1.4.11 holds "each line in a graph" to 3:1; SC 1.4.3 holds text to 4.5:1. Recharts
reuses the series stroke (graphic-tier) as tooltip text (text-tier). `#64748b` at
3.75:1 on `#0f172a` is a legal line colour and an illegal text colour at once.

**Fact 5 -- the denominator is 16 files, not the 7 the caller named**, and 6 of
them supply a custom `content` (MfeMaeScatter, BudgetDashboard,
ComputeCostBreakdown, OptimizerProgressChart, PerfProgressChart,
SharpeHistoryChart) so `DefaultTooltipContent` never runs -- an `itemStyle` prop
there is **inert**, a textbook vacuous guard. Addressable population is 10 files /
14 instances. Unnamed-but-affected: `app/backtest/page.tsx`,
`app/paper-trading/nav/page.tsx`, `app/reports/page.tsx`.

**Version trap:** `package.json` declares `recharts: ^2.12.0`, installed is
**2.15.4** (2.x). v3's "Tooltip now correctly adds color styles for Scatter" is a
2.x DEFECT disclosure read in reverse -- on 2.x a Scatter payload may carry no
`entry.color`, so it resolves to `#000` (1.18:1). One Scatter tooltip exists:
`OptimizerInsights.tsx:164`.

**Why:** filed during the 91.22 research gate; every anchor verified against the
installed build, and every ratio computed with the W3C G18 formula rather than a
contrast-checker site.

**How to apply:** if a later step proposes an `itemStyle` sweep, check it excludes
the 6 custom-`content` files, deletes the 3 dead `contentStyle.color` keys, and
asserts the **computed ratio**, not the presence of the prop
([[feedback_assert_the_property_not_a_proxy]]).
