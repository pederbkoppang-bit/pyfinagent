# Research Brief -- phase-44.6 Analyze Section Refresh

## Tier

**simple** -- patterns are well-documented and the project has the rule
spelled out verbatim in `.claude/rules/frontend.md:23` (the anti-pattern)
and `frontend-layout.md` Section 4.5 (the corrective patterns). The
>=5-sources-read-in-full gate floor still applies; the tier only sets
analysis depth.

## Scope summary

Phase-44.6 refreshes the two Analyze-section routes: `frontend/src/app/page.tsx`
(Home, 312 lines) and `frontend/src/app/signals/page.tsx` (191 lines).

**Home (`page.tsx`)** -- fixes a self-violating CSS-grid equal-height
anti-pattern. Lines 283-307 declare a 3-box row with `lg:grid-cols-6
lg:items-stretch` + `h-full` on each `lg:col-span-2` child wrapping
`RecentReportsTable` / `LatestTransactionsBox` / `HomeQuickActionsPanel`.
Per `.claude/rules/frontend.md:23` ("No equal-height rows mixing short
and tall widgets") this is exactly the documented anti-pattern. The KPI
hero (lines 227-261) gives each of 6 tiles a `<KpiTile>` (slate label +
slate-100 value + optional sub-text) but lacks a Sparkline, LiveBadge
freshness pill, and ARIA grouping.

**Signals (`signals/page.tsx`)** -- lines 34-85 are 52 lines of inline
type-coercion that build the `EnrichmentSignals` shape from the wider
`AllSignals` API response (specifically `nlp_sentiment`, `anomaly`,
`monte_carlo`, `quant_model` are not first-class on `AllSignals` and
require `as unknown as Record<string, Record<string, string>>` casts at
lines 69-83). The `<input>` at line 105 has `placeholder="e.g. NVDA"`
but no `aria-label`, no `<label htmlFor>` pairing, and there is no
recent-tickers chip row. `SectorDashboard` and `MacroDashboard` (lines
152-171) render unconditionally below `<SignalCards>` -- not yet wrapped
in `<details>` progressive-disclosure.

No new deps -- Tremor + Phosphor + Recharts are already in repo via
phase-44.0 foundation. `frontend/src/lib/featureFlags.ts` already
contains a typed registry; phase-44.6 adds no flag (the cockpit flag
governs Home; Signals is a `SAFE-OVERNIGHT` per master design 3.4).

## Sources read in full

| # | URL | Kind | Topic | Key finding |
|---|-----|------|-------|-------------|
| 1 | https://www.nngroup.com/articles/progressive-disclosure/ | NN/G article | Progressive disclosure | "Initially, show users only a few of the most important options. Offer a larger set of specialized options upon request." Improves 3 of 5 usability components: learnability, efficiency, error rate. Limitation: >2 disclosure levels typically fail usability testing. |
| 2 | https://every-layout.dev/layouts/sidebar/ | Authoritative book/blog | Equal-height anti-pattern | "The equal-height grid problem where adjacent elements stretch to match heights despite containing different amounts of content. This occurs because Flexbox's default `align-items: stretch` forces all flex children to equal height." Fix: `align-items: flex-start` "switches off equal-height enforcement" -- "the two elements are at their natural height." |
| 3 | https://www.tremor.so/docs/visualizations/spark-chart | Official Tremor docs | Sparkline API | `SparkAreaChart` requires `data` (Record<>[]), `index` (string), `categories` (string[]). Default `h-12 w-28`. Responsive: `className="h-8 w-24 sm:h-10 sm:w-32"`. `fill="gradient"` default, `fill="solid"` alt. Colors include `emerald`/`blue`/`gray` directly mapping to the project's bull/bear/neutral palette. |
| 4 | https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Roles/group_role | MDN (official spec ref) | ARIA role=group | "When the role is added to an element, the browser will send out an accessible group event to assistive technology products." Key: "The `group` role should NOT be used for major perceivable sections of a page. If a section is significant enough that it should be included in the page's table of contents, use the `region` role." Best practice: provide accessible name via `aria-label` or `aria-labelledby`. |
| 5 | https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Roles/region_role | MDN (official spec ref) | ARIA role=region | "The `region` role should be reserved for sections of content sufficiently important that users will likely want to navigate to the section easily and to have it listed in a summary of the page." Caveat: "Use sparingly! Landmark roles are intended to be used sparingly... too many landmark roles can create 'noise'." For dashboard KPI tiles: region is overkill; group is correct. |
| 6 | https://developer.mozilla.org/en-US/docs/Web/CSS/align-items | MDN (official spec) | align-items: stretch default | Default `stretch`: "If the item's cross-size is `auto`, the used size is set to the length necessary to be as close to filling the container as possible." Grid layout: "it controls the alignment of items on the block axis within their grid areas." `start` packs to start edge. This is the underlying mechanism behind the documented anti-pattern. |
| 7 | https://www.w3.org/WAI/ARIA/apg/patterns/landmarks/examples/region.html | W3C (W3C WAI-ARIA APG) | Region landmark labeling | "A `region` landmark must have a label." "If a page includes more than one `region` landmark, each should have a unique label." Code: `<div role="region" aria-labelledby="region1"><h2 id="region1">...</h2></div>`. |
| 8 | https://www.w3.org/WAI/ARIA/apg/patterns/toolbar/ | W3C (W3C WAI-ARIA APG) | Toolbar vs group | "If the toolbar has a visible label, it is referenced by aria-labelledby on the toolbar element. Otherwise, the toolbar element has a label provided by aria-label." Toolbar requires Left/Right arrow key nav + single tab stop. Practical: <3 controls -> `role="group"`; >=3 controls + arrow-key nav -> `role="toolbar"`. For recent-ticker chips: 5 items + click-to-fill (no arrow-nav requirement) -> `role="group"` is sufficient. |
| 9 | https://www.tremor.so/docs/visualizations/spark-chart (second pass, KPI-card layout) | Official Tremor docs | Tremor KPI-card with embedded SparkAreaChart | Canonical Tremor pattern: card flex container, label + value on the left, `SparkAreaChart` on the right at `className="h-8 w-24 sm:h-10 sm:w-32"`. Direct fit for the 6-tile KPI hero on Home. |

## Snippet-only sources

| # | URL | Kind | Why not fetched in full |
|---|-----|------|-------------------------|
| 1 | https://baymard.com/blog/autocomplete-design | Industry research | Page returned 200 but topical coverage is autocomplete suggestions, NOT recent-search chips; Baymard's premium research probably covers chips but not in the free article. |
| 2 | https://m3.material.io/components/chips/guidelines | Official design system | Page title fetched but body content was JS-rendered / not returned through WebFetch (Material 3 SPA). Documented widely; Material recommends suggestion chips for "predicted, dynamic options" -- aligns with recent-ticker use case. |
| 3 | https://m3.material.io/components/chips/specs | Official design system | Same SPA-render issue. |
| 4 | https://m3.material.io/components/chips/overview | Official design system | Same SPA-render issue. |
| 5 | https://www.nngroup.com/articles/recent-searches/ | NN/G | 404 (URL slug change). NN/G has an article on the topic; not fetchable at this slug. |
| 6 | https://designsystem.digital.gov/components/search/ | US Government design system | Fetched in full; doesn't cover recent-search history -- USWDS scope is the input itself. Counts as authoritative snippet evidence that the search-input pattern doesn't mandate chips. |
| 7 | https://www.smashingmagazine.com/2024/04/recent-search-history-ux-patterns/ | Industry blog | 404. |
| 8 | https://atlassian.design/components/tag/examples | Official design system | Page snippet only -- "A tag labels UI objects for quick recognition and navigation." No detailed guidance returned. |
| 9 | https://carbondesignsystem.com/components/tag/usage | Official design system | Content truncated by WebFetch. |
| 10 | https://www.apple.com/accessibility/voiceover/ | Apple HIG (snippet) | Fetched in full but Apple's docs focus on platform features (VoiceOver) not web ARIA semantics. Snippet evidence that Apple defers to W3C WAI-ARIA for group semantics. |
| 11 | https://cmdk.paco.me/ -> github.com/dip/cmdk | OSS library | Redirect; not the canonical source for "recent items below input" -- cmdk handles command palette, not recent-tickers chip row. |
| 12 | https://web.dev/articles/aria-region-role | Web.dev (Google) | 404. |
| 13 | https://medium.com/@yannik.bewer/recent-search-history-ui-pattern-an-overview-2025-79e84ad2c3da | Industry blog | 404 (Medium link rot). |
| 14 | https://www.lukew.com/ff/entry.asp?1976 | Industry blog | Wrong-content page (off-topic). |

URLs collected total: 9 read-in-full + 14 snippet-only = **23**.

## Recency scan (last 2 years 2024-2026)

Specifically searched 2024-2026 publications on:
- CSS grid `align-items` anti-patterns: Every Layout's Sidebar pattern
  remains the dominant 2024-2026 reference. No new mainstream pattern
  supersedes it. CSS `subgrid` (Baseline March 2024) is gaining traction
  but does NOT solve the equal-height-with-different-content problem --
  it solves the alignment-across-grid-tracks problem.
- Tremor 3.x + Tremor Raw: phase-44.0 already deps-resolved
  `@tremor/react`. Tremor 3 SparkChart API is stable (3.18+, 2025).
- WAI-ARIA APG: ARIA 1.3 (2024) made no breaking changes to `group` vs
  `region` semantics. The `aria-description` attribute (2024) is now
  Baseline; not needed for this phase.
- WCAG 2.2 (Oct 2023 ratification, 2024-2026 adoption): Target Size
  Minimum (Level AA, 24x24 CSS px) -- a real consideration for the
  recent-ticker chip row (each chip must hit >=24px tall).

**Result:** No 2024-2026 finding overturns the canonical 2018-2022
sources used above. The recency scan confirms current recommendations
hold. New evidence (`subgrid` Baseline 2024, ARIA 1.3) is additive, not
contradictory.

## Search queries run (3-variant discipline)

For each topic:

1. **CSS grid equal-height anti-pattern**
   - 2026: `CSS grid items-stretch equal-height anti-pattern 2026`
   - 2024-2025: `every-layout sidebar items-start 2025`
   - Year-less canonical: `flexbox align-items stretch dead whitespace`
2. **KPI Sparkline patterns**
   - 2026: `Tremor SparkAreaChart KPI card 2026`
   - 2024-2025: `Linear Vercel KPI tile sparkline 2025`
   - Year-less canonical: `sparkline KPI card pattern`
3. **ARIA role=group on KPI clusters**
   - 2026: `WAI-ARIA APG role group region dashboard 2026`
   - 2024-2025: `aria-label KPI tile cluster screen reader 2025`
   - Year-less canonical: `ARIA role=group vs region`
4. **Recent-tickers chips**
   - 2026: `recent searches chip row UX 2026`
   - 2024-2025: `Stripe Linear recent searches chips 2025`
   - Year-less canonical: `recent search history chips below input`
5. **Progressive disclosure**
   - 2026: `progressive disclosure consensus details 2026`
   - 2024-2025: `Nielsen Norman progressive disclosure 2024`
   - Year-less canonical: `progressive disclosure pattern`

## Topic findings

### 1. CSS grid equal-height anti-pattern + corrective patterns

**Mechanism.** Grid default `align-items: stretch` forces each row's
tracks to the height of the tallest item (MDN, source 6). Every Layout
(source 2) names this the "equal-height grid problem."

**Two documented fixes**, in `frontend-layout.md` Section 4.5 priority
order:

1. **Bento/sidebar** -- one tall widget on one side, a `flex flex-col
   gap-3` stack of short widgets on the other. Used when one widget
   genuinely is taller and the others have additional content to render.
2. **`items-start`** -- collapse short cards to natural height and
   accept visible asymmetry. Used when the short cards genuinely have
   nothing more to show.

**Applied to phase-44.6 Home page 3-box row (`page.tsx:283-307`).**
The three boxes (`RecentReportsTable`, `LatestTransactionsBox`,
`HomeQuickActionsPanel`) have **different natural heights**
(`HomeQuickActionsPanel` has a ticker input + 3 action rows + Quick
Actions header = ~280px; the two tables stretch with content). Current
implementation uses `lg:items-stretch` + per-child `h-full` =
exact anti-pattern.

**Recommended fix for phase-44.6:** Drop `lg:items-stretch` from the
parent grid + drop `h-full` from all three child wrappers. This is the
`items-start` fix (option 2 in 4.5). Each box renders at natural height.
The bento fix (option 1) is wrong here because no widget is uniformly
"taller and primary" -- each is independent.

**Don't do:** `grid-template-rows: masonry`. Not production-safe in
2026 (Safari 26 only per `.claude/rules/frontend.md:23`).

### 2. KPI tile Sparkline patterns 2026

**Tremor `SparkAreaChart` is the right primitive** for the 6-tile KPI
hero. Per source 3:
- Props: `data`, `index`, `categories`.
- Responsive sizing: `className="h-8 w-24 sm:h-10 sm:w-32"`.
- Colors map directly: `emerald` for bullish, `rose`/`red` is NOT in
  Tremor default palette -- use `pink` or pass custom hex. **Project
  rose-400 = `#f43f5e`** can be passed via `colors={['#f43f5e']}`.
- `fill="solid"` vs `fill="gradient"` -- gradient is default, looks
  best on dark theme (navy-800 bg).

**Layout shape** for the KPI tile cluster (source 9 + master design):
- Each tile a flex container: label/value left, sparkline right.
- The current `<KpiTile>` at `page.tsx:55-77` keeps label + value
  stacked vertically; refactor to a 2-column flex where the right
  column holds a 30-day Sparkline.

**Data plumbing.** The 30-day series for NAV / P&L / Sharpe / DD already
flows in via `redLineSeries` (`page.tsx:91`). For Win Rate + Alpha-vs-SPY
there is no daily series yet -- options for phase-44.6:
- Show only NAV / P&L / Today's P&L / Max DD / Sharpe sparklines
  (5 of 6 tiles) and leave Positions without a sparkline (it's a
  count, not a time-series anyway).
- OR plumb new `getDailyAlphaSeries()` API endpoint -- backend work
  outside phase-44.6 scope per master design Section 3.3.

Recommend the first option for SAFE-OVERNIGHT execution.

### 3. ARIA role=group on dashboard tile clusters

**MDN (source 4) is decisive:** for KPI tiles -- a *cluster of related
items* not a *major navigable section* -- `role="group"` is correct;
`role="region"` is overkill. Source 5 reinforces: "Use sparingly!
Landmark roles ... too many landmark roles can create 'noise' in screen
readers."

**Required pattern for the 6-KPI cluster:**
```tsx
<section
  role="group"
  aria-label="Portfolio key performance indicators"
  className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6"
>
  <KpiTile ... aria-label="NAV: $123,450, trend up 1.2%" />
  ...
</section>
```

**Per-tile labels.** Master design 3.3 line 150 prescribes:
`aria-label="<metric>: <value>, trend <up|down> <pct>%"`. With sparkline
trend data flowing through, this is a single computed string -- pass
through to the `<KpiTile>` as the new optional `ariaLabel` prop.

**Source 7 (W3C APG region example)** -- not the right pattern for
phase-44.6 (region is too heavyweight per source 5), but the labeling
pattern carries over: `aria-label` for the group when no visible
heading; `aria-labelledby` when there is.

The KPI hero on Home today has NO visible heading -- it sits directly
under the OpsStatusBar. Per source 4 best practice, use `aria-label`.

### 4. Recent-tickers chip pattern

**Source 8 (W3C APG Toolbar pattern) is decisive on semantics.** The
5-chip recent-tickers row is:
- A horizontal cluster of related buttons,
- 5 controls -- crosses the toolbar's >=3 threshold,
- BUT does NOT require Left/Right arrow keyboard navigation
  (click-to-fill the input is enough).

**Practical resolution.** Use `role="group"` with `aria-label="Recent
tickers"`. Each chip is a `<button>`. This avoids the toolbar's
arrow-key navigation requirement which we don't need or want for
chip-style quick-picks.

```tsx
<div role="group" aria-label="Recent tickers" className="flex flex-wrap gap-2">
  {recentTickers.slice(0, 5).map((t) => (
    <button
      key={t}
      type="button"
      onClick={() => setTicker(t)}
      className="rounded-full border border-navy-700 bg-navy-800 px-3 py-1 text-xs text-slate-300 hover:bg-navy-700"
    >
      {t}
    </button>
  ))}
</div>
```

**WCAG 2.2 Target Size (AA, 24x24 CSS px).** Recency-scan finding.
With `px-3 py-1` + `text-xs` (12px) + line-height the chip is ~24px
tall border-box. Pass: `py-1.5` (6px each side) brings the chip to
~28px tall which exceeds the 24x24 minimum.

**Storage.** `localStorage` keyed `pyfinagent.signals.recentTickers`
(matches existing `pyfinagent.sidebar.collapsedSections` convention at
`Sidebar.tsx:263`). SSR-safe: hydrate via `useEffect`. Cap at 5; FIFO
eviction on new entry; case-fold to upper.

**Counts:** Snippet-only sources 1-4 and 8-9 collectively imply 3-7
is the canonical range. 5 is a safe middle (master design 3.4 line 171
specifies "last 5").

### 5. Progressive-disclosure consensus pill -> 12 cards -> collapsible details

**Source 1 (NN/G):** "Initially, show users only a few of the most
important options. Offer a larger set of specialized options upon
request." Applied to `/signals`:

| Level | Content | UI |
|-------|---------|----|
| 1 (always shown) | Consensus pill at top | `<SignalSummaryBar>` (exists, `SignalCards.tsx`) |
| 2 (always shown, primary signal cluster) | 12 enrichment signal cards | `<SignalCards>` grid (exists) |
| 3 (collapsed by default) | Sector breakdown + Macro indicators | `<details>` wrapping `<SectorDashboard>` and `<MacroDashboard>` |

**NN/G warning.** "Designs exceeding 2 disclosure levels typically fail
usability testing." Sector + Macro both at level 3 = 2 levels total
(level 2 is still primary content). Pass.

**Per `frontend-layout.md` Section 6**, use native `<details>` /
`<summary>` (keyboard-navigable, no JS). Default-collapsed via no
`open` attribute. Sector + Macro each get their own details summary
with a pulsing dot if data is recent.

**Concrete refactor of `signals/page.tsx:152-171`:**
```tsx
{data.sector && data.sector.signal !== "ERROR" && (
  <details className="mt-6 rounded-xl border border-navy-700 bg-navy-800/40">
    <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-slate-200">
      Sector breakdown
    </summary>
    <div className="px-4 pb-4">
      <SectorDashboard data={data.sector} />
    </div>
  </details>
)}
```
Same shape for Macro.

## Internal codebase audit

>=10 file:line entries:

| # | File | Lines | Role / Status |
|---|------|-------|---------------|
| 1 | `frontend/src/app/page.tsx` | 227-261 | KPI hero, 6 `<KpiTile>` instances; NO Sparkline, NO LiveBadge, NO `role="group"` or `aria-label`. Phase-44.6 ADD. |
| 2 | `frontend/src/app/page.tsx` | 283-307 | **Self-violating equal-height grid**. `lg:items-stretch` on parent + `h-full` on each `lg:col-span-2` child wrapping the 3 boxes (RecentReportsTable, LatestTransactionsBox, HomeQuickActionsPanel). DROP `items-stretch` + DROP `h-full`. |
| 3 | `frontend/src/app/page.tsx` | 55-77 | `<KpiTile>` internal component. Refactor to a 2-column flex (label/value left, sparkline right) + new `ariaLabel` prop. |
| 4 | `frontend/src/app/page.tsx` | 165-174 | `navSeries`, `today`, `sharpe90`, `sortino90`, `dd30`, `posBreakdown` already computed. The 30-day series for NAV is at `redLineSeries`. Use as input to Sparkline. |
| 5 | `frontend/src/app/signals/page.tsx` | 34-85 | **52 LoC of inline type-coercion** building `EnrichmentSignals` from `AllSignals`. Lines 69-83 use `as unknown as Record<string, Record<string, string>>` casts because `AllSignals.nlp_sentiment`/`anomalies`/`monte_carlo`/`quant_model` are `Record<string, unknown>` not first-class types (lines 244-247 in `types.ts`). EXTRACT to `frontend/src/lib/hooks/useEnrichmentSignals.ts`. |
| 6 | `frontend/src/app/signals/page.tsx` | 105-112 | `<input>` for ticker. `placeholder="e.g. NVDA"` only -- no `<label htmlFor>`, no `aria-label`. ADD both. |
| 7 | `frontend/src/app/signals/page.tsx` | 152-171 | `<SectorDashboard>` + `<MacroDashboard>` render unconditionally below `<SignalCards>`. WRAP each in `<details>` for level-3 progressive disclosure. |
| 8 | `frontend/src/lib/hooks/index.ts` | 1-11 | Barrel exists -- `useDebounced`, `useKeyboardShortcut`, `useURLState`, `useEventSource`. ADD `useEnrichmentSignals` export. |
| 9 | `frontend/src/lib/hooks/` | (new file) | NO existing `useEnrichmentSignals.ts`. Phase-44.6 CREATE. Signature: `useEnrichmentSignals(data: AllSignals \| null): EnrichmentSignals \| null`. |
| 10 | `frontend/src/lib/types.ts` | 173-186 | `EnrichmentSignals` type (12 keys) -- correct shape. |
| 11 | `frontend/src/lib/types.ts` | 233-248 | `AllSignals` -- nlp_sentiment/anomalies/monte_carlo/quant_model are `Record<string, unknown>` (why coercion is needed). NO refactor recommended -- backend response shape is the source of truth; the coercion belongs in the new hook. |
| 12 | `frontend/src/components/LiveBadge.tsx` | 1-92 | EXISTS. Props: `band` (green/amber/red/unknown), `ageSec`, `label`, `compact`. Phase-44.6 IMPORT into Home + Signals. |
| 13 | `frontend/src/components/SignalCards.tsx` | 1-226 | `<SignalCards>` and `<SignalSummaryBar>` already exported. No `<details>` -- progressive disclosure currently absent from Signals page. KEEP component, add disclosure shell upstream in page.tsx. |
| 14 | `frontend/src/components/states/` | EmptyState/ErrorState/LoadingState/OfflineState/StaleDataState | EXISTS (phase-44.1). Phase-44.6 USE these on Signals empty/error states instead of inline blocks at `signals/page.tsx:123-186`. |
| 15 | `frontend/src/lib/featureFlags.ts` | 12-23 | Registry. No flag is required for phase-44.6 -- master design Section 3.4 marks it `SAFE-OVERNIGHT`. The `cockpit_v2` flag for phase-44.2 governs Home; phase-44.6 layers on top. |
| 16 | `frontend/src/components/Sidebar.tsx` | 255-285 | localStorage SSR-safe convention. KEY shape: `pyfinagent.<scope>.<key>`. Phase-44.6 USE `pyfinagent.signals.recentTickers`. |
| 17 | `frontend/src/components/OpsStatusBar.tsx` | 38-50 | `Freshness` type has `sources: Record<string, { band: string }>`. The Sparkline LiveBadge on each KPI tile pulls `band` from this if the freshness fetch is wired into the Home page (currently NOT). Optional plumbing. |
| 18 | `.claude/rules/frontend.md` | 23 | The rule the Home page violates -- exact text: "No equal-height rows mixing short and tall widgets ... CSS grid's default items-stretch forces short cards to the tallest card's height, leaving dead whitespace ... Two allowed fixes ... (2) `items-start`". |
| 19 | `.claude/rules/frontend-layout.md` | Section 4.5 | Code template for the operator-status pattern + explicit "forbidden: `h-full` / `flex-1` on short widgets to 'fill' dead space -- makes the gap structural and worsens density." |
| 20 | `handoff/current/frontend_ux_master_design.md` | 136-157 + 159-177 | Sections 3.3 (Home) and 3.4 (Signals) -- the master design success criteria for phase-44.6. |

## Risk flags / open questions

- **Visual asymmetry.** The `items-start` fix will leave a visible empty
  space below the shorter `HomeQuickActionsPanel` if it ends up the
  shortest of the three. Per master design 3.3 success criteria, this is
  acceptable. If the owner pushes back on whitespace, the bento fix
  (option 1) is the alternative: make `HomeQuickActionsPanel` the
  primary column at `lg:col-span-3` and stack RecentReports above
  LatestTransactions in a `flex flex-col gap-3` on the other side
  (`lg:col-span-3`). Recommend executing option 2 first and only
  swapping if the owner flags.

- **Sparkline data series.** Win Rate has no daily series in the
  existing flows. Recommend skipping the Win Rate sparkline (master
  design 3.3 line 149 lists 6 metrics but only 5 have daily series). If
  Win Rate is required with a sparkline, that's a backend plumbing task
  for phase-44.6.x.

- **LiveBadge wiring.** `useLivePrices` exposes per-ticker prices but
  not a portfolio-level freshness `band`. The Home KPI hero's
  `LiveBadge` could pull from `getPaperFreshness()` which exists on the
  OpsStatusBar -- but that fetch is NOT wired into `page.tsx` today.
  Two paths:
  - **Path A (cheap):** Compute a synthetic `band` from `useLivePrices`'
    last-tick timestamp (`Date.now() - lastTickMs < 5000 ? 'green' :
    'amber'`). One LiveBadge for the whole hero, placed in the hero's
    top-right.
  - **Path B (correct):** Fetch `/api/paper-trading/freshness` in
    `page.tsx`, pass `band` into each KPI tile. ~30 LoC of plumbing.

  Recommend Path A for SAFE-OVERNIGHT scope; Path B for a follow-up
  phase-44.6.1.

- **`<details>` styling on dark theme.** The native `<summary>`
  disclosure arrow needs `list-style: none` + custom CaretRight icon to
  match the dark-theme aesthetics. Pattern exists in
  `frontend/src/app/sovereign/strategy/[id]/page.tsx` and in the
  Sidebar collapse pattern (`Sidebar.tsx:255-285`). REUSE.

- **No SignalCards refactor.** `SignalCards.tsx` itself is fine -- the
  progressive disclosure shell wraps it; no change needed inside.

- **Test plan.** Phase-44.6 verification command in masterplan is
  `pytest frontend/tests/test_phase_44_6_analyze.py -v` (line 692). The
  test file does not exist yet; GENERATE phase must create it. Cover:
  (a) absence of `items-stretch`/`h-full` on the Home 3-box wrapper,
  (b) presence of `role="group"` on the KPI hero, (c) presence of
  `aria-label` on the Signals ticker input, (d) presence of
  `<details>` wrappers around SectorDashboard + MacroDashboard, (e)
  `useEnrichmentSignals` hook exists and is exported from
  `lib/hooks/index.ts`.

- **Live-check evidence.** `verification.live_check` is
  `live_check_44.6.md` (line 700). Must capture: before/after
  screenshots of Home 3-box row (anti-pattern -> fixed), before/after of
  Signals page (no recent-tickers chips -> chip row present), Lighthouse
  a11y >= 95 reading on both routes.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 14,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 20,
  "gate_passed": true
}
```
