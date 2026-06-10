# Research brief -- phase-44.2 cockpit refactor

## Tier
moderate

## Scope summary

phase-44.2 refactors `/paper-trading` (1284 LoC monolith, 6 tabs) into a
sub-routed cockpit using the foundation components shipped in cycle 62
(`DataTable`, `LiveBadge`, `SectorBarList`, `AgentRationaleDrawer`). Of
the 13 success criteria in `frontend_ux_master_design.md` Section 3.7,
this session executes 7 code-side criteria: criterion 2 (sub-route
migration `app/paper-trading/{positions,trades,nav,reality-gap,exit-quality}/page.tsx`),
criterion 3 (link-based tablist with `role="tablist"` + per-tab
`role="tab"` + `aria-selected`), criteria 4-5 (DataTable wiring for
positions + trades with TanStack v8 sort/filter), criterion 6
(AgentRationaleDrawer dual-source -- already opens on Trades row click;
add identical drawer for Positions row click), criterion 7 (LiveBadge
per Positions row driven by `useLivePrices` age), and criterion 8
(Tremor BarList sector-concentration card via the existing
`SectorBarList` foundation). The 6 operator-gated criteria are deferred
this cycle: criterion 1 (Manage tab removal needs `operator_approval`
because it changes 3 years of operator habit), criteria 11-13 (Playwright
+ Lighthouse + 5-question UAT runs in operator's browser session). The
Manage tab and the existing `learnings` sub-route remain untouched.

## Sources read in full (>=5)

| # | URL | Tier | What it confirms / shows | How it shapes the contract |
|---|-----|------|-------------------------|----------------------------|
| 1 | https://nextjs.org/docs/app/api-reference/file-conventions/route-groups | 2 (official Next.js docs, v16.2.6 dated 2026-05-19) | `(folderName)` route groups organize files without changing URL; can host multiple sibling routes that share a layout; multiple root layouts trigger full page reloads on cross-group navigation; conflicting URLs cause errors | Recommend a STANDARD nested-route pattern `app/paper-trading/{positions,trades,nav,reality-gap,exit-quality}/page.tsx` with a shared `app/paper-trading/layout.tsx`. NO route groups needed -- they help when "opting specific segments into sharing a layout while keeping others out"; here all 5 share the same layout. NO parallel routes needed -- they are for "simultaneously rendering" multiple slots (dashboards/feeds), not for tabbed exclusive-selection UI. |
| 2 | https://nextjs.org/docs/app/api-reference/file-conventions/parallel-routes | 2 (official Next.js docs, v16.2.6 dated 2026-05-19) | Parallel routes use `@slot` syntax; slots are NOT route segments and do NOT change the URL; require `default.js` for unmatched slots; "Tab Groups" section shows a layout inside the slot for tab nav | Confirms parallel routes are OVERKILL for 44.2. Single-slot exclusive-selection tabs map cleanly to standard nested routes -- the Tab Groups example in parallel-routes docs uses parallel routes only because they want each slot to retain its own state across other nav; we don't. Avoid the complexity of `default.js` files and the soft-vs-hard-navigation footgun. |
| 3 | https://www.w3.org/WAI/ARIA/apg/patterns/tabs/ + https://www.w3.org/WAI/ARIA/apg/patterns/tabs/examples/tabs-manual/ | 2 (W3C WAI APG, current) | Container has `role="tablist"`; each tab has `role="tab"` + `aria-selected="true|false"`; tabpanels have `role="tabpanel"` + `aria-labelledby`; roving tabindex (`tabindex="0"` on active, `-1` on others); manual activation is recommended -- requires Space/Enter to activate after arrow-key focus | The 5 cockpit tabs ARE navigation tabs (each is a URL) -- use `<Link>` not `<button>`, but STILL apply `role="tab"` + `aria-selected` (NOT `aria-current="page"` -- see source #4). Use MANUAL activation (Space/Enter triggers the Link, matching browser-link semantics) -- automatic activation would change URL on every arrow press which is hostile. Implement roving tabindex on the 5 tabs. |
| 4 | https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Attributes/aria-current (via DigitalA11Y + a11y-collective summaries) | 2 (MDN, current) + 3 (a11y-collective.com) | "Authors should not use the aria-current attribute as a substitute for aria-selected in widgets where aria-selected has the same meaning, for example, in a tablist." `aria-selected` = "what you've chosen for future action"; `aria-current="page"` = "where you are right now (the page you're viewing)" | DECIDED: even though each cockpit tab is a route Link, use `aria-selected` (because the tab bar IS a tablist semantically) NOT `aria-current="page"`. This differs from the SIDEBAR pattern (`Sidebar.tsx:125` correctly uses `aria-current="page"` because that nav is NOT a tablist). Both attributes can coexist on the same Link when wrapped in a tablist context -- but `aria-selected` is the load-bearing one for tab semantics. |
| 5 | https://www.nngroup.com/articles/tabs-used-right/ | 3 (Nielsen Norman Group, founding-tier UX authority, year-less canonical) | "When users don't need to simultaneously see information presented under different tabs" tabs are appropriate. Fewer tabs are better -- "when the number of tabs overflows the tab list, the tab bar often becomes a carousel" hurting discoverability. Consistency is a heuristic -- removing familiar tabs disorients users. In-page vs navigation tabs MUST NOT mix. | Supports the Manage-tab-removal DEFERRAL: removing a 3-year-old tab IS a consistency violation per NN/G, so it correctly belongs behind `operator_approval` and a drawer-or-link offramp. Supports moving the 5 remaining tabs to nav-tabs (sub-routes) -- they were already a navigation-tab pattern in disguise (no simultaneous comparison need across positions/trades/nav-chart). Use icon + label both because tab list crowds at narrow viewports (current page truncates labels per master_design Section 3.7). |
| 6 | https://www.nngroup.com/articles/progressive-disclosure/ | 3 (NN/G, founding-tier, year-less canonical) | Rarely used settings belong in secondary areas. Worries that hiding harms understanding are "groundless" -- people understand systems BETTER when features are prioritized. >2 disclosure levels suffer poor usability -- drawers beat nested hierarchies. | Once 44.2's operator approves Manage removal in a future cycle, the right destination is a `<Drawer/>` mounted from the page header (one disclosure level), NOT a sub-route nor a /settings link (two levels). This brief defers BOTH the removal AND the drawer; 44.2 leaves Manage as-is. |
| 7 | https://tanstack.com/table/v8/docs/guide/virtualization + https://tanstack.com/table/v8/docs/guide/features | 2 (official TanStack docs) | TanStack Table does NOT ship virtualization built-in -- pair with `@tanstack/react-virtual` for >1000 rows. Mid-size dashboards (10K-50K rows) need virtualization. 1000-row threshold is the predictable wall; 50,000 rows + 5 columns optimized = 4.8 MB memory, 18ms init. Common params `estimateSize: 48px` + `overscan: 10`. | DECIDED: 44.2's positions table (~20-200 rows) and trades table (~50-500 rows) are FAR below the 1000-row virtualization threshold. NO virtualization needed -- the existing `DataTable.tsx` foundation (no virtualization) is correct. Adding virtualization would be premature optimization. Recommend documenting the threshold in `DataTable.tsx` so future consumers know when to wire up `@tanstack/react-virtual`. |
| 8 | https://github.com/tremorlabs/tremor/blob/main/src/components/BarList/BarList.tsx (BarList source) + https://www.tremor.so/docs/visualizations/barlist | 2 (vendor source + docs) | BarList type `Bar<T> = T & { key?: string; href?: string; value: number; name: string }` -- NO `color` prop in the type. Component uses hardcoded `bg-blue-200 dark:bg-blue-900` styling for ALL bars. Per-item color customization is NOT supported. | RISK FLAG for existing `SectorBarList.tsx:54-61`: the `color: colorFor(...)` field is currently stripped at the type boundary (`as unknown as ...`) and visually IGNORED by Tremor -- all bars render blue regardless of the amber/red threshold logic. The comment at SectorBarList.tsx:54 ("Tremor BarList accepts color via per-item `color` field") is FACTUALLY WRONG. Two options for 44.2: (A) pin and assume future Tremor support; (B) replace BarList internals with a Tailwind-only horizontal-bar grid that respects per-item color tokens. Owner should choose; B is the durable fix. |
| 9 | https://ui.shadcn.com/docs/components/radix/tabs + https://www.answeroverflow.com/m/1399973431134388324 | 2 (shadcn docs, the most-installed React UI lib) + 4 (TanStack support thread) | Radix UI Tabs primitive: "Tabs should not be used for page navigation -- use Tab Nav instead." Tab Nav has equivalent styles but is built for navigation. Practical pattern with router: wrap `<Link>` in `<TabsTrigger asChild>` with `activeProps` setting `data-state="active"`. | Confirms: do NOT use the Radix Tabs primitive for 44.2 (no Radix in pyfinagent yet, so this isn't a regression). Hand-roll the link-based tablist using the source #3 ARIA pattern -- it's ~30 LoC in the new `app/paper-trading/layout.tsx`. Consistent with Sidebar which is hand-rolled. |
| 10 | https://medium.com/@ashwinrishipj/building-a-high-performance-virtualized-table-with-tanstack-react-table-ced0bffb79b5 + https://tanstack.com/table/v8/docs/framework/react/examples/virtualized-rows + https://github.com/TanStack/table/pull/5927 | 3 (industry blog) + 2 (official example + perf PR) | Practical implementations use `estimateSize: 50` + `overscan: 10`. Recent v8 perf PR cuts memory 28x on 50K rows. Server-side ops + virtualization needed at 10K+ rows. ColumnMeta module-augmentation lets you put `align: 'right' | 'left' | 'center'` + `className` on every column for numeric tables. | Recommend: in 44.2's positions + trades tables, declare a `ColumnMeta` augmentation file `frontend/src/lib/tanstack-meta.d.ts` with `align` + `className` fields. Add a TS-level helper in `DataTable.tsx` to apply `meta.className` to `<th>` + `<td>`. Numeric columns (price, qty, P&L, market value) right-align. Currently the foundation `DataTable.tsx` does NOT support per-column alignment -- audit gap to close in 44.2. |

## Snippet-only sources

| URL | Snippet | Why noted |
|-----|---------|-----------|
| https://www.contentful.com/blog/tanstack-table-react-table/ | "Complete guide to TanStack Table" overview | Background only; doesn't add over source #7 |
| https://strapi.io/blog/table-in-react-performance-guide | "Table performance hits predictable walls at 1,000 and 10,000+ rows" | Quantitative confirmation of the 1000-row threshold; corroborates source #7 |
| https://www.contentful.com/blog/tanstack-table-react-table/ | "URL-based state for sharing filters" | Aligns with master_design's `useURLState` foundation slot for future 44.X work |
| https://refine.dev/blog/building-react-admin-dashboard-with-tremor/ | "Tremor Guide: Step-by-Step React Dashboard Setup" | Background on Tremor adoption shape; doesn't fix the per-item color gap |
| https://github.com/TanStack/table/discussions/4097 + #4439 + #5319 | Module-augmentation pattern for `ColumnMeta` | Confirms source #10's approach is the community-canonical pattern -- multiple discussions, accepted answers |
| https://ariakit.org/examples/tab-next-router | Ariakit Tab + Next.js Router example | Alternative library reference; pyfinagent doesn't use Ariakit so noted but not adopted |
| https://www.eleken.co/blog-posts/tabs-ux | "Use a tab bar when a few tasks dominate daily use, use a drawer for secondary or infrequent items" | Supports the drawer-for-Manage decision (deferred this cycle) |
| https://github.com/vercel/next.js/issues/49569 | Parallel-route + nested route SSR NEXT_NOT_FOUND bug | Risk reference if anyone proposes parallel routes for tabs -- adds friction |
| https://www.digitala11y.com/aria-current-state/ | Plain-language `aria-current` reference | Supplements source #4; supports the aria-selected vs aria-current decision |
| https://app.studyraid.com/en/read/11919/379820/tab-and-sidebar-components | Tab + sidebar composition references | Snippet only; conceptual |
| https://www.uxpin.com/studio/blog/what-is-progressive-disclosure/ | Progressive-disclosure 2026 overview | Background; deeper material in source #6 |
| https://www.thewcag.com/examples/navigation | WCAG 2.2 navigation pattern reference | Confirms WCAG 2.2 keeps the `aria-selected` vs `aria-current` distinction; no relevant 2026 change |
| https://www.gitnexa.com/blogs/saas-dashboard-ux-patterns | 2026 SaaS pattern overview ("sidebar + drawer for secondary") | Background on drawer-for-settings trend; supports deferred Manage->drawer decision |

## Recency scan (last 2 years 2024-2026)

**Result: 3 new findings + 1 confirmed non-change.**

1. **TanStack Table v8 memory PR (#5927, merged Q1 2026)** -- 28x memory reduction on 50K rows. Does NOT change the 1000-row virtualization threshold for 44.2's use case (20-500 rows), but means even larger trades-history tables in future phases stay viable. Source: source #10.
2. **Next.js 16.2 (May 2026)** -- the documentation snapshot fetched 2026-05-19 still describes route groups + parallel routes with the SAME conventions used since Next.js 13. The "Tab Groups" example in parallel-routes docs has been STABLE -- no API breakage. Source: source #2.
3. **EU WCAG 2.2 AA mandate (Jun 2025 enforcement)** -- the `aria-selected` vs `aria-current` distinction is unchanged but `2.5.8 Target Size (Minimum)` is now legally enforced in the EU. Sidebar already complies (`min-h-[24px]` at Sidebar.tsx:129); cockpit tabs must match. Source: master_design Section 2.5 + source #4 cross-ref.
4. **No new finding** -- the Radix UI explicit guidance "Tabs should not be used for page navigation -- use Tab Nav instead" predates 2024 and remains current in shadcn/Radix docs (source #9). The recommended hand-rolled link-based tablist using ARIA primitives has not been superseded.

## Search queries run (3-variant discipline)

Per `.claude/rules/research-gate.md`, three query variants per topic:

| Topic | Current-year (2026) | Last-2-year (2024-2025) | Year-less canonical |
|-------|--------------------|-----------------------|---------------------|
| TanStack Table v8 | "TanStack Table v8 best practices financial dashboard 2026" | "TanStack Table v8 column meta numeric right-align className 2025" | "TanStack Table v8 row virtualization threshold performance financial data" |
| Tremor BarList | "Tremor BarList dark theme color thresholds dashboard 2026" | n/a (vendor lib, version-pinned in repo) | "Tremor BarList color prop per item array example" |
| Next.js App Router sub-routes | "Next.js 15 App Router tabbed page sub-routes route groups parallel routes 2026" | "Next.js App Router page.tsx parent layout.tsx tabs URL pattern best practice 2025 2026" | "tabs vs sub-routes URL deep linking trading dashboard 2026" |
| ARIA tablist + Link | "WAI-ARIA tabs pattern Link navigation aria-current page tablist" + "shadcn ui tabs Link aria-selected route" | "aria-current page navigation links vs aria-selected tabs WCAG 2.2" | (W3C APG canonical pages directly fetched) |
| Drawer-vs-tab UX | "drawer vs tab settings dashboard UX Stripe Linear pattern 2026" | (NN/G articles -- evergreen, year-less) | "tabs UX progressive disclosure Nielsen Norman" + "tabs information architecture sub-page versus inline switching" |

## Topic findings

### 1. TanStack Table v8 best practices for financial dashboards

**Key insight:** TanStack v8 is headless and ships sorting, filtering, and column features in the core, but NOT virtualization. The 1000-row threshold is where unvirtualized tables hit "predictable walls" (source #10). Positions (20-200 rows) and trades (50-500 rows) are FAR below that threshold, so the existing `DataTable.tsx` foundation (no virtualization) is correct as-is.

**Recommended approach for 44.2:**

- Reuse `frontend/src/components/DataTable.tsx` as-is for positions + trades wiring (source #1 of internal audit). No new dependencies.
- ADD a `frontend/src/lib/tanstack-meta.d.ts` module-augmentation file declaring `ColumnMeta` with `align?: 'left' | 'right' | 'center'` and `className?: string` properties (source #10).
- ADD support in `DataTable.tsx` to apply `column.columnDef.meta?.className` to both `<th>` and `<td>` (currently both cells use a hardcoded `px-3 py-2` class; add the meta className alongside).
- Numeric columns (qty, entry, current, market value, P&L, stop loss, days held, fee) get `meta: { align: 'right' }` -- visual scanability per Tufte data-ink + Cleveland-McGill position-encoding principle (cited in `frontend-layout.md` Section 9).
- Define column factories `positionsColumns(tickerMeta, livePrices)` and `tradesColumns(tickerMeta)` in new files `frontend/src/app/paper-trading/positions/columns.tsx` and `.../trades/columns.tsx` -- co-located so each route owns its own column logic.
- Wire `globalFilterPlaceholder="Filter tickers..."` for both tables. Positions row click opens AgentRationaleDrawer via the new `position.trade_id` lookup (see topic 4 + risk flag P-1 below). Trades row click already works in the legacy page (line 920ish) -- preserve that behavior.

Source citations: sources #7, #10, snippet #5.

### 2. Tremor BarList for sector-concentration visualization

**Key insight:** The existing `SectorBarList.tsx` foundation is misled about Tremor's per-item color support. The Tremor BarList type definition (source #8) does NOT include a `color` field; the component renders all bars in `bg-blue-200 dark:bg-blue-900` regardless of what gets passed in. The `as unknown as` cast at `SectorBarList.tsx:71-72` silently strips the color field, and the amber-at-5pp / red-at-or-over color logic at `SectorBarList.tsx:29-33` currently has NO visual effect.

**Recommended approach for 44.2 (with risk flag):**

This is a foundation-component defect that 44.2 should NOT silently inherit. Two options for the owner:

- **Option A (minimal, 44.2 scope):** Keep `SectorBarList.tsx` as-is; bars render uniform blue. The cap-line and threshold text on `SectorBarList.tsx:66-70` still convey the cap visually. Wire `SectorBarList` into the positions sub-route right column (next to the positions DataTable). Mark Option B as a follow-up issue.
- **Option B (proper fix, may bleed past 44.2):** Rewrite `SectorBarList.tsx` internals as a Tailwind-only horizontal-bar grid that respects per-item color tokens. Drops `BarList` and `Card` from `@tremor/react`. ~40 LoC. Requires a unit-test update at `SectorBarList.test.tsx:30-55` to assert on the rendered color classes.

Owner should choose A vs B in the contract. Recommend B because the foundation comment claims a feature that doesn't exist, and a uniform-blue chart undermines the criticality signal the UX-DoD criterion 8 promises.

Source citations: sources #2 (Tremor official docs), #8 (BarList source). Confirmed live by fetching the GitHub source on 2026-05-25.

### 3. Next.js 15 App Router sub-route migration patterns for tabbed pages

**Key insight:** Standard nested routes are the right primitive for 44.2 -- NOT route groups, NOT parallel routes. Route groups (`(folderName)`) help when sibling routes should share a layout while OTHER siblings don't (source #1); all 5 cockpit tabs share the same layout, so the parens are noise. Parallel routes (`@slot`) are for SIMULTANEOUS rendering of multiple slots in a dashboard -- they introduce `default.js` overhead and a hard-vs-soft-navigation footgun (source #2). For exclusive-selection tabbed UI where each tab is its own URL, standard nested folders with a shared `layout.tsx` are the documented Next.js convention.

**Recommended approach for 44.2:**

- Create `frontend/src/app/paper-trading/layout.tsx` containing the FIXED HEADER ZONE (page shell + Tier 1 page header + Tier 4 OpsStatusBar + SummaryHero + Tier 5 tab bar) per `frontend-layout.md` Sections 1, 3, 5.
- Create 5 sibling `page.tsx` files under `app/paper-trading/{positions,trades,nav,reality-gap,exit-quality}/page.tsx`. Each owns its own tab content (Tier 6).
- The Manage tab tracks the page shell at `app/paper-trading/manage/page.tsx` -- DEFERRED removal but route-migrated for consistency (operator approval gates the Sidebar entry exposure and the Tab Bar entry; the page can exist without nav entries).
- The existing `app/paper-trading/page.tsx` (1284 LoC monolith) becomes the index that redirects to `/paper-trading/positions` -- `import { redirect } from "next/navigation"; export default function Page() { redirect("/paper-trading/positions"); }`. This preserves the home-link from Sidebar (`Sidebar.tsx:43`).
- Each sub-route's `page.tsx` is a focused 50-200 LoC client component owning ONLY that tab's data fetching + rendering. The shared data fetched once (status, portfolio, positions, trades, snapshots, perf, livePrices, liveNav, tickerMeta) hoists to the shared layout via React Context OR (preferred) each sub-route fetches its own. PREFER per-route fetch because (a) sub-routes can render `loading.tsx` independently, (b) tab switching only refetches what changes, (c) avoids the prop-drilling tower.
- No `loading.tsx` or `error.tsx` per-sub-route in 44.2's scope -- the foundation `LoadingState` + `ErrorBanner` components from the master_design Section 2.2 are not yet built (Section 2.2 is part of 44.1, NOT 44.2).
- Sidebar (`Sidebar.tsx:42-50`) keeps the single "Paper Trading" entry pointing at `/paper-trading` (which now redirects to /positions). Per master_design's roadmap, sub-route nav entries can be added in a later UX pass; the 5-tab tablist on `app/paper-trading/layout.tsx` provides the primary navigation surface.

Source citations: sources #1, #2.

### 4. ARIA tablist patterns linking to URLs

**Key insight:** Even when each tab is a route `<Link>`, the tab BAR is still semantically a `tablist`. The W3C APG (source #3) requires `role="tablist"` on the container, `role="tab"` on each link, `aria-selected="true|false"`, roving tabindex, and arrow-key navigation. Crucially, `aria-current="page"` is NOT a substitute for `aria-selected` in tablist contexts (source #4). The Radix UI / shadcn-ui guidance explicitly says "Tabs should not be used for page navigation -- use Tab Nav instead" (source #9), but that recommends a different STYLED COMPONENT, not a different ARIA pattern. The ARIA primitives are the same.

**Recommended approach for 44.2:**

Hand-roll the link-based tablist in `app/paper-trading/layout.tsx`:

```tsx
// Pseudocode -- contract should reference this shape
const TABS = [
  { href: "/paper-trading/positions", label: "Positions", icon: TabPositions, badge: positionsCount },
  { href: "/paper-trading/trades",    label: "Trades",    icon: TabTrades,    badge: tradesCount },
  { href: "/paper-trading/nav",       label: "NAV Chart", icon: TabNavChart },
  { href: "/paper-trading/reality-gap", label: "Reality gap", icon: TabRealityGap },
  { href: "/paper-trading/exit-quality", label: "Exit quality", icon: TabExitQuality },
];
const pathname = usePathname();
return (
  <div role="tablist" aria-label="Paper trading sections" className="flex gap-1 rounded-lg bg-navy-800/50 p-1">
    {TABS.map((t, i) => {
      const isActive = pathname === t.href || pathname.startsWith(t.href + "/");
      return (
        <Link
          key={t.href}
          href={t.href}
          role="tab"
          aria-selected={isActive}
          aria-controls={`panel-${t.href.split("/").pop()}`}
          tabIndex={isActive ? 0 : -1}
          onKeyDown={(e) => /* ArrowLeft / ArrowRight / Home / End */}
          className={clsx("flex-1 rounded-md px-4 py-2 text-sm font-medium min-h-[24px]", isActive ? "..." : "...")}
        >
          <t.icon size={16} weight={isActive ? "fill" : "regular"} />
          {t.label}
          {t.badge != null && ` (${t.badge})`}
        </Link>
      );
    })}
  </div>
);
```

Each sub-route's `page.tsx` wraps its content in `<div role="tabpanel" id="panel-positions" aria-labelledby="..." tabIndex={0}>`. Keyboard arrow handler implements roving tabindex (source #3). DO NOT add `aria-current="page"` -- the tab bar uses `aria-selected`. The Sidebar (a non-tablist navigation) correctly uses `aria-current="page"` at `Sidebar.tsx:125` and that stays unchanged.

Source citations: sources #3, #4, #9.

### 5. Drawer-vs-tab UX research for operator dashboards

**Key insight:** NN/G's progressive-disclosure article (source #6) backs the Manage-tab-to-drawer migration as a textbook case: "rarely used settings belong in secondary areas" and worries that hiding harms understanding are "groundless." Stripe, Linear, and Vercel all use this pattern (snippet #13 + master_design Section 3.7). However, NN/G's tabs-used-right article (source #5) also warns that "consistency" -- the operator's mental model of where Manage lives -- is a usability heuristic. Pulling a 3-year tab from under operator fingers IS a violation unless the new location is discoverable AND has clear information scent.

**Recommended approach for 44.2 (defer the removal):**

Manage tab REMAINS in 44.2 scope. The contract should:

- Keep the Manage tab WIRED IN the new `layout.tsx` tablist (6 tabs total, not 5).
- Migrate the existing Manage tab content to `app/paper-trading/manage/page.tsx` as a verbatim copy of the current monolith's manage section (page.tsx:1040-1284). This is a refactor, not a UX change.
- Mark a `MANAGE_REMOVAL_DEFERRED` comment in the layout file referencing the operator-approval gate.
- The drawer-based replacement lands in a later phase (master_design Section 3.7 lists it as `OWNER-APPROVAL-REQUIRED`).

This preserves operator mental model AND advances the structural refactor. Once the operator approves removal in a follow-up cycle, the change is `delete app/paper-trading/manage/` + drop one entry from the TABS array + add a `<Drawer/>` trigger to the page header.

Source citations: sources #5, #6, snippets #7, #13.

## Internal codebase audit

The audit covers EVERY file the 44.2 refactor touches or depends on. ASCII anchors are file:line.

| # | File | Lines | Role | What it implies for 44.2 |
|---|------|-------|------|--------------------------|
| 1 | `frontend/src/app/paper-trading/page.tsx` | 1-1284 | The monolith. 27 hooks + 9 helper components + 1 default export | Split into `app/paper-trading/layout.tsx` (shell + tab bar + shared header/status bar) + 5+1 sub-route `page.tsx`. The 9 helper components (Dollar, PnlBadge, SummaryHero, etc.) hoist out to `frontend/src/components/paper-trading/` -- co-locate where they belong. |
| 2 | `frontend/src/app/paper-trading/page.tsx` | 391-403 | Tab definitions: 6 tabs (positions, trades, chart, reality-gap, exit-quality, manage) | Migrate verbatim into `layout.tsx` as the link-based tablist. ID slugs in URL: `positions / trades / nav / reality-gap / exit-quality / manage` -- note `chart` -> `nav` to match master_design Section 3.7's URL list. |
| 3 | `frontend/src/app/paper-trading/page.tsx` | 407-456 | 27 useState/useEffect/useMemo hooks for page-wide state (status, portfolio, positions, trades, snapshots, perf, livePrices, liveNav, tickerMeta, manageSettings, depositAmount, etc.) | DECISION POINT: hoist into Context provider at layout level, OR distribute per sub-route. Recommend DISTRIBUTE -- each sub-route owns its data needs. positions+trades subroutes both need positions + livePrices + tickerMeta; share via a lean Context (e.g. `PaperTradingDataContext`). Manage sub-route owns manageSettings + depositAmount independently. |
| 4 | `frontend/src/app/paper-trading/page.tsx` | 708-728 | Existing tab bar (button-based, no ARIA roles) | Delete; replaced by link-based tablist in `layout.tsx`. |
| 5 | `frontend/src/app/paper-trading/page.tsx` | 783-889 | Positions tab content (raw `<table>` markup, 107 LoC) | Replace with `<DataTable columns={positionsColumns} data={positions} ... onRowClick={(pos) => setRationaleTradeId(pos.last_trade_id)} />`. Live price + age stay in the cell renderers. Move to `app/paper-trading/positions/page.tsx`. |
| 6 | `frontend/src/app/paper-trading/page.tsx` | 892-967 | Trades tab content (raw `<table>` markup, 76 LoC) | Replace with `<DataTable columns={tradesColumns} data={trades} ... onRowClick={(t) => setRationaleTradeId(t.trade_id)} />`. Move to `app/paper-trading/trades/page.tsx`. |
| 7 | `frontend/src/app/paper-trading/page.tsx` | 969-1027 | NAV Chart tab content | Move verbatim to `app/paper-trading/nav/page.tsx`. |
| 8 | `frontend/src/app/paper-trading/page.tsx` | 1028-1037 | Reality-gap + exit-quality tab content | Move to `app/paper-trading/reality-gap/page.tsx` + `.../exit-quality/page.tsx`. |
| 9 | `frontend/src/app/paper-trading/page.tsx` | 1040-1284 | Manage tab content (~245 LoC of deposit + settings forms) | Move verbatim to `app/paper-trading/manage/page.tsx`. NO removal in 44.2. |
| 10 | `frontend/src/app/paper-trading/page.tsx` | 460-490 | `refresh()` callback fetching status/portfolio/trades/snapshots/perf in parallel | Hoists to layout if Context approach chosen; OR each sub-route fetches its own subset. Either way, preserve the `Promise.all` parallelism (frontend.md "Parallel fetches"). |
| 11 | `frontend/src/components/DataTable.tsx` | 1-150 | Foundation component cycle 62 | GAP: no per-column alignment support. Add `column.columnDef.meta?.className` application at lines 94 (`<th>`) and 138 (`<td>`). Add `frontend/src/lib/tanstack-meta.d.ts` ColumnMeta module augmentation. |
| 12 | `frontend/src/components/LiveBadge.tsx` | 1-91 | Foundation component cycle 62 | Ready for use. Compact mode (line 66-75) is for in-table cells: positions DataTable wires `<LiveBadge band={live ? "green" : "red"} ageSec={live?.age_sec} compact />` per row. |
| 13 | `frontend/src/components/SectorBarList.tsx` | 1-78 | Foundation component cycle 62 | RISK FLAG: `color` field on lines 54-61 is a no-op -- Tremor BarList doesn't render per-item colors (source #8). Option A: ship as-is with uniform blue bars; Option B: rewrite internals as Tailwind grid. Owner choice. |
| 14 | `frontend/src/components/AgentRationaleDrawer.tsx` | 1-253 | Existing drawer, opens on `tradeId` | Reused as-is. Positions sub-route needs `pos.last_trade_id` (or equivalent) on each position row -- audit `PaperPosition` type in `frontend/src/lib/types.ts` to confirm the field exists, OR derive it from a position->latest-trade lookup. |
| 15 | `frontend/src/components/Sidebar.tsx` | 22-64 + 119-139 | Nav entries; `aria-current="page"` at line 125 (CORRECT for nav, not tablist) | Unchanged in 44.2. The "Paper Trading" link at line 43 still points at `/paper-trading` (which now redirects). The "Learnings" link at line 44 is unaffected (already a sub-route). |
| 16 | `frontend/src/lib/icons.ts` | 195-200 | Tab icon exports (TabPositions etc.) | Reused verbatim in new `layout.tsx`. |
| 17 | `frontend/src/lib/useLivePrices.ts` | 1-50 | Polls /api/paper-trading/live-prices every 60s | Stays unchanged. Used by positions sub-route (NOT trades -- trades are historical). |
| 18 | `frontend/src/lib/useLiveNav.ts` | 1-40 | Derived NAV + total-P&L pct | Stays unchanged. Used by the shared layout (SummaryHero hero) and by both home + paper-trading. |
| 19 | `frontend/src/lib/useTickerMeta.ts` | 1-50 (file exists) | Polls company name + sector per ticker | Stays unchanged. Both positions + trades sub-routes need it. |
| 20 | `frontend/src/app/paper-trading/learnings/page.tsx` | 1-53 | Existing sub-route precedent | CONFIRMED PATTERN: it uses the same page-shell (`Sidebar` + `<main className="flex flex-1 flex-col overflow-hidden">` + scrollable content zone) per `frontend-layout.md` Section 1. The new sub-routes follow the SAME pattern, but with the shell hoisted to `layout.tsx` (preferred) so the tab bar stays pinned across switches. The learnings page does NOT use the tab bar (it's a separate destination); leave it alone in 44.2. |
| 21 | `.claude/rules/frontend.md` | full | Conventions: 30s API timeout, Phosphor icons, scrollbar-thin, BentoCard pattern, color codes, error states never `.catch(() => null)` on all, no emojis | Cockpit refactor MUST preserve every rule. Specifically: polling failure limits (frontend.md), `scrollbar-thin` on the DataTable's `overflow-x-auto` wrappers (DataTable.tsx:75 already correct), Phosphor-only icons. |
| 22 | `.claude/rules/frontend-layout.md` | Sections 1, 3, 5, 7, 8 | Page-shell two-zone flex; 6-tier hierarchy; pill-style tabs; table conventions; empty/loading/error states | New `layout.tsx` implements Sections 1 + 3 + 5 verbatim. Each sub-route's `page.tsx` is the "scrollable content zone" content. Status bar (OpsStatusBar) + SummaryHero stay in the layout (master_design 3.7 line 770-780 specifies this is the always-visible shell). |
| 23 | `handoff/current/frontend_ux_master_design.md` | Section 3.7, lines 216-242 + Section C row 5 + Section verification UX-1 through UX-13 | Master design intent for 44.2 | All 13 criteria mapped. This brief reaffirms which 7 land in 44.2 code work and which 6 are owner-gated. |
| 24 | `frontend/src/components/DataTable.test.tsx` + `LiveBadge.test.tsx` + `SectorBarList.test.tsx` | exist; covers cycle-62 foundation | Test scaffolding exists | 44.2 must add `frontend/tests/test_phase_44_2_cockpit.py` (assert sub-route HTML structure) OR equivalent Playwright spec for the routing acceptance criteria. Playwright spec file `frontend/playwright/cockpit.spec.ts` is in the masterplan but does NOT exist yet -- this is operator-gated 11/12/13 work. |
| 25 | `frontend/package.json` | line 23 | `@tremor/react: ^3.18.7` | Tremor BarList per-item color limitation is a feature-gap of THIS pinned version. Option B's rewrite obviates the dep boundary. |

## Risk flags / open questions

| # | Risk | Detail | Mitigation in 44.2 contract |
|---|------|--------|----------------------------|
| P-1 | `last_trade_id` field on `PaperPosition`? | The positions table row-click drawer wiring (criterion 6) needs a stable mapping from a position to its decision rationale. If `PaperPosition` lacks `last_trade_id`, must derive from `trades.find(t => t.ticker === pos.ticker)` -- but a single position may have multiple BUY rows; need to pick the most recent. | Q/A and the implementor must audit `frontend/src/lib/types.ts::PaperPosition` and `backend/api/paper_trading.py` for `position_id -> trade_id` mapping. If missing, fall back to a derivation helper in `frontend/src/lib/paper-trading-utils.ts`. |
| P-2 | Tremor BarList per-item color is a no-op | The foundation `SectorBarList.tsx` comments at line 54 claim per-item color works; it does not (source #8). Currently all bars render uniform blue, undermining UX-DoD criterion 8's "amber within 5pp, red at/over" signal. | Owner picks Option A (ship as-is) or Option B (rewrite internals). Brief recommends B because criterion 8's value proposition relies on the color signal. |
| P-3 | Hot-reload restart hazard | Per `feedback_npm_install_requires_launchctl_kickstart.md`: after any `npm install` in `frontend/`, `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` is required. 44.2 doesn't install packages (Tremor + TanStack already pinned) but if Option B for P-2 lands, it might remove `@tremor/react` -- consult before doing so. | Contract should NOT touch `package.json` unless P-2 Option B is approved. If approved, kickstart launchd watchdog post-install. |
| P-4 | Operator habit -- tab order | The current tab order (Positions, Trades, NAV Chart, Reality gap, Exit quality, Manage) is preserved. Per NN/G (source #5) "consistency is a usability heuristic" -- do NOT reorder. | Contract specifies the exact TABS array order matching `page.tsx:394-401`. |
| P-5 | Manage tab's deep nested state | The Manage tab uses `manageDirty: Partial<FullSettings>` to track unsaved settings deltas, with save/load fetchers. Moving it to its own sub-route means the unsaved-state warning ("you have unsaved changes") may not fire if the operator clicks to another tab -- the tab is now a link navigation, not a state toggle. | Add a `beforeunload` / Next.js `useRouter().events`-style guard in `app/paper-trading/manage/page.tsx` OR document the limitation. Brief recommends documenting and revisiting in the Manage-drawer phase (when the issue dissolves -- a drawer doesn't have route-loss semantics). |
| P-6 | URL deep-link from the existing /paper-trading URL | Anyone who has bookmarked `/paper-trading` and was on (say) the Trades tab loses their context on refresh, since the old `useState<TabId>("positions")` always defaulted to positions. The new sub-route URL preserves the tab on refresh -- a USABILITY WIN -- but any external links to `/paper-trading?tab=trades` (if they exist) will land on `/paper-trading/positions` after the index redirect. | Audit external linking surfaces -- there are none likely; the old tabs were not URL-deeplinkable in any case. Document this as an intentional UX improvement. |
| P-7 | Loading flash on tab-switch | Each sub-route fetches its own data on mount. If the layout-level shared fetch (status, portfolio, positions, trades, snapshots, perf) doesn't run, every tab switch could trigger a refetch flash. | Hoist the shared "low-cardinality, all-tabs-need" fetches (status, portfolio basics) to the layout. Per-tab heavy fetches (trades history, snapshots) stay in their sub-route. `useLivePrices` and `useTickerMeta` live in the layout because both positions and trades sub-routes need them. |
| P-8 | DataTable lacks per-column alignment in foundation | Lines 94 (`<th>`) and 138 (`<td>`) hardcode `px-3 py-2 text-left` -- ignores `meta.className`. Numeric financial data (P&L, market value, qty) should right-align per Tufte / Cleveland-McGill (source #10 cross-ref). | 44.2 contract must include the `DataTable.tsx` audit-gap fix: add `meta` className application + the `tanstack-meta.d.ts` module augmentation. This is foundational and benefits 44.4 + 44.5 too. |

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 13,
  "urls_collected": 23,
  "recency_scan_performed": true,
  "internal_files_inspected": 25,
  "gate_passed": true
}
```
