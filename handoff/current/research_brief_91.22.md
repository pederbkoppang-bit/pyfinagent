# Research Brief -- step 91.22

**Topic:** Recharts `<Tooltip>` `itemStyle` / `labelStyle` / `contentStyle` default
theming behavior on dark backgrounds; WCAG 2.2 contrast requirements applied to
chart tooltip text.

**Tier:** simple (caller-specified). **Audit-class:** NO (`coverage.dry` reported
for information only; not required for this step). **Accessed:** 2026-08-20.

```json
{
  "brief_status": "COMPLETE",
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 19,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 22,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 4,
    "dry": false
  },
  "gate_passed": true
}
```

---

## Headline

**No file in this repo sets `itemStyle` on any Recharts `<Tooltip>` -- 0 of 16
files, 0 of 20 tooltip instances.** Item rows therefore fall through to the
library default, which is literally `color: entry.color || '#000'`
(`DefaultTooltipContent.js:58`). Measured with the G18 formula, `#000` on this
project's tooltip background `#0f172a` is **1.18:1** against a WCAG 2.2 AA floor
of **4.5:1**. Where `entry.color` *is* populated it is the **series stroke/fill**
-- a colour chosen to satisfy the 3:1 *graphical-object* bar of SC 1.4.11, then
reused as *text*, which is held to 4.5:1 by SC 1.4.3. Three call sites already
tried to fix this and put `color` in `contentStyle`, where it is a **no-op for
item rows**.

---

## Search queries run (three-variant discipline, `.claude/rules/research-gate.md`)

| Variant | Query |
|---|---|
| Year-less canonical | `Recharts Tooltip itemStyle labelStyle contentStyle default styling dark background` |
| Current-year frontier | `WCAG 2.2 contrast minimum 4.5:1 chart tooltip text 2026` |
| Last-2-year window | `recharts v3 tooltip default styles accessibility 2025 changes` |

---

## Read in full (7; >=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|-----|----------|------|-------------|----------------------|
| 1 | https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum | 2026-08-20 | official W3C normative (tier 2) | WebFetch, full | "The visual presentation of text and images of text has a contrast ratio of at least 4.5:1". Large-scale = "at least 18 point or 14 point bold" ~= "18.5px and 24px" -> 3:1. "If no background color is specified, then white is assumed." |
| 2 | https://www.w3.org/WAI/WCAG22/Understanding/content-on-hover-or-focus | 2026-08-20 | official W3C normative | WebFetch, full | SC 1.4.13: Dismissible / Hoverable / Persistent. "Custom tooltips, sub-menus, and other nonmodal popups that display on hover and focus are examples of additional content covered by this criterion." **Carries no contrast clause** -- contrast on a tooltip comes from 1.4.3, not 1.4.13. |
| 3 | https://recharts.github.io/en-US/api/Tooltip/ | 2026-08-20 | official library docs | WebFetch, full | `itemStyle` default `{}`, "CSS styles of individual items inside the tooltip, a `<li>` element." `labelStyle` default `{}`, "CSS styles of the tooltip title." `contentStyle` default `{}`, "CSS styles to be applied to the wrapper `div` element." `wrapperStyle` carries the **same** description -- a documentation collision (see Pitfalls). |
| 4 | https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast | 2026-08-20 | official W3C normative | WebFetch, full | SC 1.4.11 3:1 for UI components + graphical objects; "the important parts of a more complex diagram such as each line in a graph". "the computed values should not be rounded (e.g. 2.999:1 would not meet the 3:1 threshold)". **"The document does not directly address tooltips or their background/border boundaries as a distinct category."** |
| 5 | https://github.com/recharts/recharts/wiki/Recharts-and-accessibility | 2026-08-20 | official project wiki | WebFetch, full | "Recharts' charts have a prop `accessibilityLayer`. This is `false` by default in 2.x, and `true` by default in 3.0 and later." Arrow-key (not Tab) navigation. **No content at all on colour contrast or theming** -- the library does not claim contrast-safe defaults. |
| 6 | https://www.w3.org/WAI/WCAG22/Techniques/general/G18 | 2026-08-20 | official W3C technique | WebFetch, full | Relative luminance `L = 0.2126*R + 0.7152*G + 0.0722*B` with the sRGB linearisation; contrast `(L1 + 0.05) / (L2 + 0.05)`. This is the formula used for every ratio in this brief. |
| 7 | https://github.com/recharts/recharts/wiki/3.0-migration-guide | 2026-08-20 | official project wiki | WebFetch, full | Tooltip gains `portal` + `axisId`; `TooltipProps` -> `TooltipContentProps`; `accessibilityLayer` flips to true. **"The guide contains no specific information about changes to `itemStyle`, `labelStyle`, `contentStyle`, `wrapperStyle`, or default colors for tooltips."** |

## Identified but snippet-only (19; context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://github.com/recharts/recharts/blob/2.x/src/component/Tooltip.tsx | source | Superseded: read the *installed* 2.15.4 build directly (stronger evidence than a branch tip). |
| https://github.com/recharts/recharts/issues/1377 | issue | "DefaultTooltipContent is overriding wrapperStyle" -- corroborates the prop collision; anecdotal tier. |
| https://github.com/recharts/recharts/issues/5181 | issue | "Better documentation of tooltips" -- confirms docs are the known weak point. |
| https://github.com/recharts/recharts/issues/663 | issue | "[Question]: Styling of Tooltip" -- community tier. |
| https://github.com/recharts/recharts/discussions/5984 | discussion | v3.0.0 release thread; migration guide read instead. |
| https://github.com/recharts/recharts/releases | changelog | Version confirmed locally from `node_modules`. |
| https://newreleases.io/project/github/recharts/recharts/release/v3.0.0 | mirror | Third-party mirror of a primary source already read. |
| https://www.w3.org/TR/WCAG22/ | normative spec | The Understanding docs (1,2,4) carry the same normative text plus rationale. |
| https://webaim.org/resources/contrastchecker/ | tool | Interactive tool; ratios computed locally via G18 instead. |
| https://getwcag.com/en/contrast-checker | tool | Same. |
| https://www.makethingsaccessible.com/guides/contrast-requirements-for-wcag-2-2-level-aa/ | practitioner | Secondary summary of sources 1/4. |
| https://testparty.ai/blog/wcag-contrast-ratio-guide-2025 | blog | Secondary; 2025 recency signal only. |
| https://www.allaccessible.org/blog/color-contrast-accessibility-wcag-guide-2025 | blog | Secondary; 2025 recency signal only. |
| https://callingallminds.com/resources/wcag/1.4.3-contrast-minimum | blog | Restates source 1. |
| https://marcom.wwu.edu/accessibility/guide/ensure-text-and-controls-have-enough-color-contrast | .edu guide | Restates source 1. |
| https://medium.com/design-bootcamp/a-plain-language-guide-to-wcag-contrast-requirements-6fe18569d5df | blog | Community tier; hierarchy rank 5. |
| https://www.shadcndesign.com/blog/shadcn-ui-charts | industry blog | Recharts-3 theming; not applicable on 2.15.4. |
| https://developer.semrush.com/intergalactic/data-display/scatterplot-chart/scatterplot-chart-changelog | vendor changelog | Unrelated chart library. |
| https://app.studyraid.com/en/read/11352/354995/tooltip-customization | tutorial | Community tier; no normative value. |

**URL total: 26 unique (7 read in full + 19 snippet-only).** Counted by row, de-duplicated;
G18 appears once (in the read-in-full set) and is not double-counted.

---

## Recency scan (2024-2026) -- performed

Searched the 2024-2026 window (`recharts v3 tooltip default styles accessibility
2025 changes`, plus the 2026-scoped WCAG query). **Three findings, none of which
supersedes the canonical sources, and one of which is a trap:**

1. **Recharts v3.0.0 shipped 2025-06-23** and flips `accessibilityLayer` to
   `true` by default (source 5). **Not applicable here:** this repo runs
   **2.15.4** (`frontend/package.json` declares `recharts: ^2.12.0`; installed
   build reports 2.15.4). Do not plan against v3 semantics.
2. **v3 "correctly adds color styles to Tooltip by default" for Scatter charts**
   (v3 release notes, snippet). Read in reverse this is a **2.x defect
   disclosure**: on 2.15.4 a Scatter payload may not carry `entry.color`, so
   `entry.color || '#000'` resolves to **black**. This repo has exactly one
   Scatter tooltip: `OptimizerInsights.tsx:164`.
3. **WCAG 2.2 is stable** -- W3C Recommendation, and the 4.5:1 / 3:1 thresholds
   are unchanged from 2.1. No 2024-2026 change alters the requirement. APCA
   (the perceptual replacement) remains WCAG-3 draft work and is **not**
   normative; do not gate on it.

---

## Key findings

1. **`itemStyle` is the only prop that reaches item text; the default is black.**
   `finalItemStyle = { display:'block', paddingTop:4, paddingBottom:4, color:
   entry.color || '#000', ...itemStyle }` -- `DefaultTooltipContent.js:53-58`.
   Author styles spread *last*, so they win, **but only for keys supplied**.
   Omitting `color` leaves `entry.color || '#000'` in force.
   (Source: installed recharts 2.15.4; docs corroborate defaults, source 3.)

2. **`contentStyle.color` does NOT reach item rows -- and three sites rely on it.**
   `contentStyle` lands on the wrapper `<div>` (`:103`); `finalItemStyle` is an
   **inline style on the `<li>`**, and an explicit inline `color` beats
   inheritance. `RedLineMonitor.tsx:194`, `StrategyDetail.tsx:87` and
   `TransformerForecastPanel.tsx:122` all set `color: "#e2e8f0"` inside
   `contentStyle`. It is a **no-op for the values the user actually reads**.
   This is the single highest-value finding: the intent is already in the code,
   in the wrong prop.

3. **The label is NOT the bug -- a plausible hypothesis, refuted.**
   `finalLabelStyle = { margin: 0, ...labelStyle }` (`:107-108`) has **no default
   colour**, so the label inherits. Ambient colour on this app is `#e2e8f0`
   (`frontend/src/app/globals.css:14`, on `body`), which is **14.48:1** on
   `#0f172a`. So the 6 tooltips with no `labelStyle` still render a compliant
   label. Do not spend the fix budget on `labelStyle`.

4. **Measured ratios (G18 formula, source 6), computed locally:**

   | Foreground | on `#0f172a` | on `#1e293b` | AA text 4.5:1 |
   |---|---|---|---|
   | `#000000` (Recharts default fallback) | **1.18:1** | **1.44:1** | FAIL / FAIL |
   | `#e2e8f0` (ambient + some labelStyle) | 14.48:1 | 11.87:1 | PASS / PASS |
   | `#94a3b8` (other labelStyle) | 6.96:1 | 5.71:1 | PASS / PASS |
   | `#64748b` (Scatter fill, `OptimizerInsights.tsx:164`) | **3.75:1** | **3.07:1** | FAIL / FAIL |
   | `#ef4444` (`StockChart.tsx:286`) | 4.74:1 | **3.89:1** | PASS / **FAIL** |
   | `#f43f5e` (`RedLineMonitor.tsx:205`) | 4.86:1 | **3.98:1** | PASS / **FAIL** |
   | `#a855f7` (`StockChart.tsx:274`) | 4.51:1 | **3.70:1** | PASS / **FAIL** |
   | `#22c55e`, `#38bdf8`, `#f59e0b`, `#10b981`, `#0ea5e9`, `#f97316` | 6.4-8.3:1 | 5.2-6.8:1 | PASS / PASS |

5. **The compliance verdict flips on the background, not the series.** This repo
   uses **two** tooltip backgrounds -- `#0f172a` (most components) and `#1e293b`
   (`app/backtest/page.tsx:962`). `#ef4444`, `#f43f5e` and `#a855f7` pass on the
   first and fail on the second. Any fix that keeps per-series colouring is
   therefore **re-breakable by a background edit**, silently.

6. **The 3:1-vs-4.5:1 tier mismatch is the root cause, not a palette accident.**
   SC 1.4.11 holds "each line in a graph" to 3:1 (source 4); SC 1.4.3 holds text
   to 4.5:1 (source 1). `#64748b` at 3.75:1 on `#0f172a` is a **legitimate line
   colour and an illegal text colour simultaneously**. Recharts' default reuses
   the graphic-tier colour for the text-tier role, so a palette that is correct
   for the chart is, by construction, not guaranteed correct for the tooltip.

7. **Tooltip box border/background is out of 1.4.11 scope.** Source 4 states the
   spec "does not directly address tooltips or their background/border
   boundaries as a distinct category". `border: 1px solid #1e293b` on `#0f172a`
   is 1.22:1, but **do not file that as a violation** -- it is a container
   boundary, not a UI component nor an information-bearing graphical object.

8. **SC 1.4.13 is a separate, unmeasured exposure.** Recharts 2.x tooltips are
   hover-triggered and vanish on pointer-out; source 2's *Hoverable* and
   *Dismissible* requirements are plausibly unmet. **Out of scope for 91.22**
   (which is a contrast step) -- flag for a follow-up step, do not silently widen.

---

## Internal code inventory

### A. The library default -- `frontend/node_modules/recharts` @ **2.15.4**

| File:line | Evidence |
|---|---|
| `es6/component/DefaultTooltipContent.js:58` | `color: entry.color \|\| '#000'` -- the only colour an item row gets absent `itemStyle`. |
| `es6/component/DefaultTooltipContent.js:103-104` | `backgroundColor: '#fff', border: '1px solid #ccc'` -- the light default every chart here overrides. |
| `es6/component/DefaultTooltipContent.js:107-108` | `finalLabelStyle = { margin: 0, ...labelStyle }` -- **no default colour**; label inherits. |
| `es6/component/DefaultTooltipContent.js:48` | `listStyle = { padding: 0, margin: 0 }` on the `<ul>`; not colour-bearing. |
| `es6/component/Tooltip.js:100,109,110,125` | `defaultProps`: `contentStyle:{}`, `itemStyle:{}`, `labelStyle:{}`, `wrapperStyle:{}` -- all four empty, confirming source 3. |
| `frontend/src/app/globals.css:14` | `color: #e2e8f0` on `body` -- the ambient colour the tooltip label inherits. |
| `frontend/src/app/layout.tsx:20-21` | `<html className="dark ...">` / `<body className="font-sans antialiased">` -- dark theme is app-wide. |

### B. Call-site census -- 16 files, 20 `<Tooltip>` instances, `itemStyle` count = **0**

The caller's scope named 7 component files. The true denominator is **16 files**:
13 under `frontend/src/components/`, plus **3 route files the caller did not
name**. Reporting the larger set.

| File | `<Tooltip>` at | `contentStyle` | `labelStyle` | `itemStyle` | Custom `content` | Status |
|---|---|---|---|---|---|---|
| `frontend/src/components/SectorDashboard.tsx` | :121 | :122 (`#0f172a`) | :127 `#e2e8f0` | **none** | no | EXPOSED |
| `frontend/src/components/OptimizerInsights.tsx` | :157, :233 | :158, :234 (`#0f172a`) | none | **none** | no | EXPOSED -- Scatter at :164 (`#64748b`), see finding 2 of recency scan |
| `frontend/src/components/PaperReconciliationChart.tsx` | :144 | :145 (`#0f172a`) | :150 `#94a3b8` | **none** | no | EXPOSED |
| `frontend/src/components/TransformerForecastPanel.tsx` | :118 | :119 (`#0f172a`, **`color` at :122**) | none | **none** | no | EXPOSED -- colour in wrong prop |
| `frontend/src/components/RedLineMonitor.tsx` | :190 | :191 (`#0f172a`, **`color` at :194**) | none | **none** | no | EXPOSED -- colour in wrong prop; series `#f43f5e` at :205 |
| `frontend/src/components/StrategyDetail.tsx` | :83 | :84 (`#0f172a`, **`color` at :87**) | none | **none** | no | EXPOSED -- colour in wrong prop |
| `frontend/src/components/StockChart.tsx` | :227, :314 | :228 (`#0f172a`), :315 | :234 `#94a3b8` (1st only) | **none** | no | EXPOSED -- :314 has no `labelStyle`; series `#a855f7` :274, `#ef4444` :286 |
| `frontend/src/app/backtest/page.tsx` *(unnamed by caller)* | :961, :1375, :1414 | :962 (**`#1e293b`**), :1376, :1415 | :963 (1st only) | **none** | no | EXPOSED -- the only `#1e293b` background; see finding 5 |
| `frontend/src/app/paper-trading/nav/page.tsx` *(unnamed)* | :89 | :90 | :95 `#94a3b8` | **none** | no | EXPOSED -- series `#64748b` at :110 |
| `frontend/src/app/reports/page.tsx` *(unnamed)* | :463, :486 | :464, :487 (`#0f172a`) | none | **none** | no | EXPOSED |
| `frontend/src/components/MfeMaeScatter.tsx` | :184 | n/a | n/a | **none** | :186 inline | BYPASSES default |
| `frontend/src/components/BudgetDashboard.tsx` | :263 | n/a | n/a | **none** | :264 inline | BYPASSES default |
| `frontend/src/components/ComputeCostBreakdown.tsx` | :182 | n/a | n/a | **none** | `<CostTooltip />` | BYPASSES default |
| `frontend/src/components/OptimizerProgressChart.tsx` | :225 | n/a | n/a | **none** | `<CustomTooltip />` | BYPASSES default |
| `frontend/src/components/PerfProgressChart.tsx` | :197 | n/a | n/a | **none** | `<CustomTooltip />` | BYPASSES default |
| `frontend/src/components/SharpeHistoryChart.tsx` | :378 | n/a | n/a | **none** | `<CustomTooltip />` | BYPASSES default |

**Two populations, and this is the scoping decision for the contract.** 6 files
supply a custom `content`, so `DefaultTooltipContent` never runs and an
`itemStyle` fix is a **guaranteed no-op** there -- their contrast must be audited
separately in their own JSX. **10 files / 14 `<Tooltip>` instances** render the
default content and are the addressable population.

---

## Consensus vs debate (external)

**Consensus:** the 4.5:1 text / 3:1 large-text and non-text thresholds are
uncontested across sources 1, 4, 6 and every secondary hit; tooltips are
explicitly in scope for SC 1.4.13 (source 2); Recharts publishes empty `{}`
defaults for all four style props (source 3, corroborated by `Tooltip.js:100-125`).

**Debate / gap, not conflict:** no source -- including Recharts' own
accessibility wiki (source 5) -- states any contrast guarantee for tooltip item
text. Source 5 is *silent* on colour entirely. The library's position is
implicitly "theming is the consumer's problem", and the v3 migration guide
(source 7) confirms nothing changed there. So there is no upstream fix to wait
for; this must be fixed in application code.

**Unresolved:** whether a tooltip's own border must clear 3:1 under SC 1.4.11.
Source 4 declines to address it. Treated as out of scope (finding 7) rather than
resolved in either direction.

---

## Pitfalls (from literature + source reading)

1. **`contentStyle` vs `wrapperStyle` are documented identically** (source 3:
   both say "CSS styles to be applied to the wrapper `div` element"), and issue
   #1377 reports `DefaultTooltipContent` overriding `wrapperStyle`. They are
   *different* divs. Do not swap one for the other during the fix.
2. **`contentStyle.color` looks like it works** because the label inherits it --
   so a visual spot-check of the label "confirms" a fix that never reached the
   item rows. This is exactly how the three existing near-misses survived.
3. **Rounding is forbidden** (source 4): 4.499:1 does not meet 4.5:1. `#a855f7`
   at 4.51:1 has ~0.01 of headroom -- treat it as failing in practice.
4. **Do not gate on APCA.** WCAG 3 is draft; 2.2 is the normative target.
5. **A per-series-colour fix is background-coupled** (finding 5) and will
   silently regress on a `contentStyle` edit. A uniform `itemStyle` colour is
   the only variant that is invariant to the palette.

---

## Application to pyfinagent

- **The minimal correct change is `itemStyle={{ color: <token> }}`** on the 14
  exposed `<Tooltip>` instances. `#e2e8f0` gives 14.48:1 / 11.87:1 and `#94a3b8`
  gives 6.96:1 / 5.71:1 -- both clear AA on **both** backgrounds in use, so the
  fix survives finding 5.
- **Design trade-off Main must decide in PLAN, not something research can settle:**
  a uniform `itemStyle.color` removes the per-series colour key from tooltip
  rows (users currently match a row to a line by colour). Options: (a) uniform
  light item text + rely on the `<Legend>`; (b) keep `entry.color` but restrict
  the series palette to the 6 colours measured >=5.2:1 on both backgrounds.
  (b) is fragile per pitfall 5 and finding 5.
- **Delete the dead `color` keys** at `RedLineMonitor.tsx:194`,
  `StrategyDetail.tsx:87`, `TransformerForecastPanel.tsx:122` when adding
  `itemStyle`, or the codebase keeps a prop that reads as load-bearing and is not.
- **Widen the scope to 16 files, not 7.** `app/backtest/page.tsx`,
  `app/paper-trading/nav/page.tsx` and `app/reports/page.tsx` are affected and
  were not in the caller's list; `app/backtest/page.tsx:962` carries the only
  `#1e293b` background and is the worst case.
- **Exclude the 6 custom-`content` files from an `itemStyle` fix** -- an
  `itemStyle` prop there is inert and would create a vacuous guard.
- **A guard must assert the rendered `<li>` colour, not the presence of the prop.**
  Per project memory `feedback_assert_the_property_not_a_proxy`: grepping for
  `itemStyle=` would have passed on the three `contentStyle.color` near-misses'
  intent. Compute the ratio from the two colours, or assert the computed style.
- **`recharts: ^2.12.0`** in `frontend/package.json` resolves to 2.15.4. If a v3
  upgrade is ever bundled into this step it changes `TooltipProps` ->
  `TooltipContentProps` and flips `accessibilityLayer` -- keep them separate steps.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**7**: 5 W3C
      normative/technique + 2 official Recharts docs/wiki; hierarchy tiers 1-2, no
      community-tier source counted)
- [x] 10+ unique URLs total incl. snippet-only (**26**)
- [x] Recency scan (last 2 years) performed + reported (3 findings, one a trap)
- [x] Full pages read, not abstracts, for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (16 files / 20 tooltip
      instances -- **9 more files than the caller's scope named**)
- [x] Contradictions / consensus noted (incl. the unresolved 1.4.11 border question)
- [x] All claims cited per-claim
- [ ] **Gap, disclosed:** contrast was computed from source colours via the G18
      formula, **not** captured from the running app via Playwright MCP. Values are
      authored hex, so computed == rendered unless a CSS rule overrides the inline
      style -- none found, but this was not verified live. A live capture belongs
      in the step's `live_check`, not in this gate.
- [ ] **Gap, disclosed:** the 6 custom-`content` tooltips were inventoried but
      their internal contrast was not measured (out of scope for the `itemStyle`
      question; needs its own pass).

---

## Status log (write-first)

- [x] Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full.
- [x] Internal inventory (16 files, 20 sites) with file:line anchors.
- [x] Library default proven from the *installed* 2.15.4 build.
- [x] 7 external sources read in full via WebFetch.
- [x] Contrast ratios computed locally with the G18 formula.
- [x] Recency scan (2024-2026) written up.
- [x] Envelope flipped to COMPLETE as the final act.
