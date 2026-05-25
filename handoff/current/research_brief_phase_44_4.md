# Research Brief -- phase-44.4 Reports + Performance refresh

**Tier:** simple-moderate
**Date:** 2026-05-25
**Researcher:** Layer-3 researcher subagent
**Scope:** Reports section refresh + Performance page enhancements
**Caller:** Main (phase-44.4)

## Scope summary

phase-44.4 has two targets in one cycle:

1. **Reports page** (`frontend/src/app/reports/page.tsx`, 604 LoC):
   - Migrate manual `searchParams.get("tab"/"ticker")` to the
     `useURLState` foundation hook (cycle 16) for bidirectional
     URL <-> state sync.
   - Convert Compare wizard to a Drawer overlay (mirror
     `AgentRationaleDrawer` pattern).
   - Reports history table -> `DataTable` foundation with a
     sparkline column (30d score).
   - Replace inline empty-state `<p>` blocks with the
     `EmptyState` foundation component.
2. **Performance page** (`frontend/src/app/performance/page.tsx`,
   267 LoC):
   - Tremor `AreaChart` above cost history table (cumulative cost).
   - Sparkline next to win rate (30d trend).
   - Per-pillar performance bars from `SynthesisReport` data.
   - `TimeRangeSelector` (7d/30d/90d/all) - segmented control.
3. **Both pages:** ARIA `tablist` role on tab bars; Lighthouse a11y
   >=95.

The pattern foundation is mostly in-repo (cycles 16 / 23.1.7 /
44.0 / 44.2 / 64). External research is sanity-check for
accessibility primitives, Tremor cell-renderer patterns, and the
sparkline column shape.

## Sources read in full (>=5 required for gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://nextjs.org/docs/app/api-reference/functions/use-search-params | 2026-05-25 | Official doc | WebFetch (full) | "useSearchParams is a Client Component hook that lets you read the current URL's query string. useSearchParams returns a read-only version of the URLSearchParams interface." Recommends wrapping in `<Suspense>` boundary; update via `useRouter()` + `router.push(pathname + '?' + createQueryString(...))`. Doc version 16.2.6, lastUpdated 2026-05-19. |
| https://www.w3.org/WAI/ARIA/apg/patterns/tabs/ | 2026-05-25 | Official W3C spec | WebFetch (full) | Required roles: `tablist` (container), `tab` (each tab), `tabpanel`. Required ARIA: `aria-controls` on tab pointing to its tabpanel, `aria-selected` true on active / false on others, `aria-labelledby` on tabpanel pointing to tab, `aria-orientation="vertical"` only if vertical. Keyboard: Left/Right arrows wrap; Home/End move to first/last; Space/Enter activates in manual-activation mode. |
| https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html | 2026-05-25 | Official W3C spec | WebFetch (full) | "The size of the target for pointer inputs is at least 24 by 24 CSS pixels." 5 exceptions verbatim: Spacing (24px diameter circle test), Equivalent (other control on same page), Inline (text link in sentence), User Agent Control, Essential. Spatially-positioned controls (radio groups, segmented) are "considered one target." Last updated October 1 2025. |
| https://npm.tremor.so/docs/visualizations/area-chart | 2026-05-25 | Vendor official doc | WebFetch (full) | AreaChart props: `data`, `categories`, `index` (required); `colors`, `valueFormatter`, `showLegend` (def true), `showXAxis` (def true), `showYAxis` (def true), `yAxisWidth` (def 56), `showGradient` (def true), `stack: boolean` (default false), `curveType` (def "linear"), `connectNulls`, `customTooltip: React.ComponentType` with `{active, payload, label}` access, `minValue`/`maxValue`. |
| https://www.tremor.so/docs/visualizations/spark-chart | 2026-05-25 | Vendor official doc | WebFetch (full) | SparkAreaChart / SparkLineChart / SparkBarChart -- 3 compact variants. Required: `data`, `index`, `categories`. Optional: `colors` (default Blue, emerald, violet, amber, gray, cyan, pink, lime, fuchsia), `fill: 'gradient'|'solid'|'none'`, `type: 'default'|'stacked'|'percent'`, `autoMinValue`, `minValue`, `maxValue`, `connectNulls`. Requires `recharts` (already installed). Dark theme via Tailwind `dark:` prefixes automatic. |
| https://uxmovement.com/buttons/why-segmented-buttons-are-better-filters-than-dropdowns/ | 2026-05-25 | UX practitioner | WebFetch (full) | "Segmented buttons allow users to see every sorting option available." Click-efficiency: segmented = 1 click; dropdown = 2 clicks + scroll. Sweet spot: up to 5 options; examples show 5 and 8 successfully. Tradeoff: "Dropdown menus save space... Segmented buttons give users higher visibility but limits space for options." |
| https://github.com/borisyankov/react-sparklines | 2026-05-25 | OSS library | WebFetch (full) | 2.9k stars, 195 forks, "no releases have been published" -- historical project. API: `<Sparklines data={[...]} limit={5} width={100} height={20} margin={5}>` with child `SparklinesLine`, `SparklinesBars`, `SparklinesSpots`, `SparklinesReferenceLine`, `SparklinesNormalBand`. **Verdict:** Tremor SparkAreaChart (already in deps as `@tremor/react@^3.18.7`) is the better fit -- no new dep. |

## Snippet-only sources (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://nextjs.org/docs/app/api-reference/functions/use-router | Official doc | Pattern already in `useURLState.ts` (cycle 44.1); `router.replace(url, {scroll: false})` confirmed. |
| https://www.tremor.so/docs/visualizations/area-chart | Vendor doc | Both .so and npm.tremor.so docs cover the same AreaChart; npm.tremor.so fetched in full. |
| https://blog.logrocket.com/url-state-usesearchparams/ | Blog | Reinforces shareable-link argument; LogRocket synthesis below. |
| https://blog.logrocket.com/build-accessible-modal-focus-trap-react/ | Blog | Practical focus-trap-react patterns; existing `AgentRationaleDrawer` already implements `role="dialog"` + `aria-modal="true"` + Esc close. |
| https://www.radix-ui.com/primitives/docs/components/dialog | Library doc | Authoritative WAI-ARIA Dialog pattern. We don't use Radix -- bespoke Drawer at `AgentRationaleDrawer.tsx`. |
| https://developer.chrome.com/docs/lighthouse/accessibility/scoring | Vendor doc | Lighthouse accessibility score is a weighted average; "Each accessibility audit is pass or fail." Confirms >=95 is achievable with proper ARIA on tabs + tables. |
| https://design.gitlab.com/components/segmented-control/ | Design system | GitLab Pajamas segmented-control reference; informs pattern. |
| https://www.eleken.co/blog-posts/time-picker-ux | UX blog | Time-picker UX patterns 2025; segmented for fixed intervals. |
| https://tanstack.com/table/v8/docs/guide/cells | Official doc | "Cells are the smallest unit of measurement in a table." TanStack v8 `cell` accepts a React component returning JSX -- our SparkAreaChart fits directly. |
| https://alfy.blog/2025/10/31/your-url-is-your-state.html | Blog | "Your URL Is Your State" -- 2025-10-31 case for URL-as-source-of-truth. |
| https://mobbin.com/glossary/segmented-control | UX reference | Segmented control design variants gallery. |
| https://state-in-url.dev/llms.txt | Library | Alternative to bespoke `useURLState` -- we already have ours; not needed. |
| https://www.shadcn.io/blocks/tables-sparkline | Code block | Confirms sparkline-in-cell is a 2025 standard pattern. |
| https://blocks.tremor.so/blocks/spark-charts | Vendor blocks | Concrete Tremor SparkChart-in-table examples. |
| https://dev.to/gnykka/how-to-create-a-sparkline-component-in-react-4e1 | Blog | Zero-dep SVG sparkline pattern -- alternative if Tremor SparkAreaChart proves too heavy. |
| https://mui.com/material-ui/react-modal/ | Library doc | Material UI Modal pattern; not used. |
| https://nextjs.org/docs/app/getting-started/linking-and-navigating | Official doc | Pattern reference for `Link` / `useRouter` -- already captured via useSearchParams fetch. |

## Recency scan (last 2 years: 2024-2026)

**Searched for 2024-2026 literature on:** Next.js 15 useSearchParams,
Tremor AreaChart/SparkChart, TanStack v8 cell renderer, WCAG 2.2
target-size, WAI-ARIA tabs pattern, sparkline-in-cell, segmented
control vs dropdown UX.

**Findings:**

1. **Next.js useSearchParams (v13.0.0 introduced, doc v16.2.6
   lastUpdated 2026-05-19):** No breaking changes since cycle 44.1
   foundation. `useSearchParams` is stable and the
   `useRouter() + router.replace({scroll: false})` pattern in
   `useURLState.ts` matches the doc-recommended approach. The 2026
   doc explicitly recommends the Suspense-boundary wrap that
   `reports/page.tsx:87` already implements via `<Suspense>` around
   `<ReportsContent />`.
2. **WCAG 2.2 Success Criterion 2.5.8 (Target Size Minimum) --
   Recommendation:** Published Oct 5 2023, last updated Oct 1 2025;
   no rule changes in 2024-2026. The 24x24 CSS pixel rule plus 5
   exceptions are stable. Tab buttons in `reports/page.tsx:233-248`
   currently render with `px-4 py-2 text-sm` which yields >=32x32
   actual tap target (passes). EmptyState component already
   complies (`min-h-[24px] min-w-[24px]` in
   `frontend/src/components/states/EmptyState.tsx:53`).
3. **WAI-ARIA Tabs Pattern -- 2026:** Document copyright 2026 but
   no substantive pattern changes from 2023. Manual-activation mode
   recommended when tabpanel content has heavy load (true for
   Compare tab -- it fetches per-ticker price data). Auto-activation
   acceptable for History tab (already-loaded data).
4. **Tremor 2025:** `@tremor/react@^3.18.7` is current in
   `frontend/package.json:23`. SparkAreaChart got `autoMinValue`,
   `minValue`, `maxValue` props in a recent update (per changelog).
   Dark theme: Tremor applies `dark:` Tailwind classes automatically;
   colors default array includes 9 palette entries.
5. **TanStack Table v8 (cycle 44.0 foundation):** Custom features
   guide unchanged since 2024; the `meta.className` + `meta.align`
   extensions added in cycles 62-63 are still TanStack-idiomatic.
   The `cell: ({ row }) => <SparkAreaChart .../>` pattern in
   `positions-columns.tsx:65-84` is the documented approach.
6. **Sparkline-in-table-cell (shadcn.io blocks, 2025):** Confirmed
   as 2025 standard pattern. shadcn/ui + Tailwind explicitly ships
   a "tables-sparkline" block (https://www.shadcn.io/blocks/tables-sparkline)
   demonstrating inline mini trend charts with percentage change
   indicators.
7. **No relevant new findings that supersede** the existing
   in-repo foundations. Everything required for phase-44.4 is
   already in-repo via cycles 16 + 23.1.7 + 44.0 + 44.1 + 44.2 +
   64.

## Search queries run

Composed the 3-variant set per `.claude/rules/research-gate.md`:

| # | Type | Query | Hits used |
|---|------|-------|-----------|
| 1 | Current-year | "Next.js 15 useSearchParams useRouter URL state synchronization deep linking 2026" | nextjs.org official doc (full) + LogRocket blog |
| 2 | Current-year | "Tremor AreaChart cumulative chart cell renderer dark theme 2025" | npm.tremor.so doc (full) |
| 3 | Current-year | "Tremor SparkAreaChart inline mini chart table cell 2025" | tremor.so spark-chart doc (full) |
| 4 | Current-year | "TanStack table v8 sparkline cell renderer SVG inline mini chart 2025" | tanstack.com cell guide (snippet) |
| 5 | Current-year | "WAI-ARIA tablist role tabs accessibility 2025 implementation pattern" | w3.org tabs pattern (full) |
| 6 | Current-year | "WCAG 2.2 target size minimum 24x24 pixels segmented control radio group 2024" | w3.org target-size-minimum (full) |
| 7 | Last-2-year | "drawer overlay component React accessibility focus trap modal dialog patterns 2025" | radix-ui.com snippet, logrocket snippet |
| 8 | Last-2-year | "segmented control vs dropdown filter time range UX best practice 2025" | uxmovement.com (full) |
| 9 | Year-less canonical | "URL state hooks shareable links React patterns" | alfy.blog 2025-10-31 snippet (year-less surfaced canonical entry) |
| 10 | Year-less canonical | "drawer dialog Radix UI accessibility ARIA pattern" | radix-ui.com snippet |
| 11 | Year-less canonical | "React sparkline component zero dependencies inline SVG path table cell pattern 2025" | github.com/borisyankov/react-sparklines (full) |
| 12 | Recency | "Lighthouse accessibility audit table tabs ARIA score 95 2026" | developer.chrome.com snippet |

## Topic findings

### Topic 1: URL deep-linking patterns 2026

**Consensus** (Next.js doc + Alfy 2025-10-31 + LogRocket + repo
cycle 16):

The pattern is unambiguous and stable. `useSearchParams()` returns
a **read-only** `URLSearchParams`-like object; mutations happen via
`useRouter().push(url, opts)` or `.replace(url, opts)` where `url`
is built by spreading the current params into a fresh
`URLSearchParams(searchParams.toString())`, mutating, and emitting
`pathname + "?" + sp.toString()`. The `{scroll: false}` option is
**mandatory** for in-page state sync to avoid scroll jumps.

The pyfinagent `useURLState` hook
(`frontend/src/lib/hooks/useURLState.ts:62-76`) implements exactly
this pattern with one additional safety: the default-value sentinel
where `value === defaultValue` removes the param from the URL
entirely (keeps shareable URLs compact). The hook also keeps state
in sync if URL changes externally (back/forward nav) via a
`useEffect` watching `params` (lines 55-60).

**For phase-44.4, replace** these 3 sites in `reports/page.tsx`:
- Line 95: `const initialTab = (searchParams.get("tab") as Tab) ?? "history"` -> `const [activeTab, setActiveTab] = useURLState<Tab>("tab", "history", { parser: (r) => (r === "compare" ? "compare" : "history") })`
- Line 103: `const [filter, setFilter] = useState(searchParams.get("ticker") ?? "")` -> `const [filter, setFilter] = useURLState<string>("ticker", "")`
- Line 107: `const [selected, setSelected] = useState<Set<string>>(new Set())` -> consider `useURLState<string[]>("compare", [], { parser: (r) => r.split(","), serializer: (v) => v.length ? v.join(",") : null })` for compare deep-linking. (Tier 2 stretch; not on critical path.)

**Pitfall:** the existing `useState(searchParams.get(...))` reads
once at mount but does NOT update if URL changes via browser
back/forward (operator returns to `/reports?tab=compare` from
bookmark, tab stays "history"). `useURLState` fixes this via the
useEffect at lines 55-60.

### Topic 2: Tremor AreaChart for cumulative cost history

**Consensus** (Tremor official doc + repo cycle 63):

Tremor `AreaChart` is the right primitive for cumulative cost
history. The API is direct:

```tsx
<AreaChart
  data={cumulativeData}
  index="analysis_date"
  categories={["cumulative_cost_usd"]}
  colors={["amber"]}  // matches existing amber theme on /performance
  valueFormatter={(v) => `$${v.toFixed(2)}`}
  showLegend={false}
  showGradient
  className="h-64"
/>
```

The transformation from `costHistory: CostHistoryEntry[]` to a
cumulative series is a `useMemo`:

```ts
const cumulativeData = useMemo(() => {
  let sum = 0;
  return [...costHistory]
    .sort((a, b) => a.analysis_date.localeCompare(b.analysis_date))
    .map((r) => {
      sum += r.total_cost_usd ?? 0;
      return { analysis_date: r.analysis_date, cumulative_cost_usd: sum };
    });
}, [costHistory]);
```

**Cycle 63 cross-check (Tremor hardcoded-blue):** Per
existing memory + master design doc Section 3.6, Tremor charts
respect a hardcoded blue tone in some props -- the operator's
override is via the explicit `colors={["amber"]}` array. This is
verified working in `ComputeCostBreakdown.tsx` (already in repo)
which uses Tremor charts with explicit per-series color overrides.

**Dark theme:** Tremor 3.18.7 applies `dark:` Tailwind classes
automatically per the theming doc. The chart will inherit the
existing `bg-navy-800` BentoCard background; no additional theming
needed.

**Pitfall:** Tremor `AreaChart` requires `data` rows to be flat
objects with the `index` field as a string (date string is fine).
Date-formatted x-axis labels: Tremor handles auto-formatting when
the `index` value is parseable as a date. The cost-history
endpoint already returns `analysis_date` as ISO date strings.

### Topic 3: DataTable sparkline column patterns

**Consensus** (Tremor SparkAreaChart + TanStack v8 cell guide +
repo cycle 63):

The repo already has `@tremor/react@^3.18.7` and `@tanstack/react-table@^8.21.3`
installed (frontend/package.json:22-23). The pattern is:

```tsx
{
  id: "sparkline_30d",
  accessorFn: (row) => row.final_score,  // for sortability
  header: "30d Trend",
  cell: ({ row }) => {
    const series = sparklineDataByTicker[row.original.ticker] ?? [];
    if (series.length < 2) return <span className="text-slate-500">--</span>;
    return (
      <SparkAreaChart
        data={series}
        index="date"
        categories={["score"]}
        colors={["sky"]}
        fill="gradient"
        className="h-8 w-24"
      />
    );
  },
  meta: { align: "right", className: "tabular-nums" },
}
```

**`sparklineDataByTicker`** is built by grouping `reports` by
ticker and sorting by analysis_date ascending, extracting the
last 30 entries. This is a `useMemo` over the existing
`ReportSummary[]` array -- NO new backend endpoint required.

**Cell-renderer pitfall (TanStack v8):** the `cell` function
receives `{row, getValue, table}` -- `row.original` is the
typed row data. Use `row.original.ticker` (not `getValue("ticker")`)
when accessing fields the column was NOT defined to access. This
is the pattern used in `positions-columns.tsx:71`.

**Sparkline-column sort:** define `accessorFn` to return the
latest score (or numeric delta vs first score) so the column is
sortable by trend direction. Don't return the chart JSX from
`accessorFn` -- TanStack will try to sort by it as a value.

**Bundle:** `SparkAreaChart` is part of `@tremor/react` (already
installed); no new dep, no bundle delta beyond what cycle 63
already added.

### Topic 4: EmptyState component patterns 2026

**Repo precedent** (cycle 16 foundation, EmptyState.tsx):

The component already exists at
`frontend/src/components/states/EmptyState.tsx:29-61`. It accepts:

```ts
{
  icon?: Icon;          // defaults to MagnifyingGlass
  title: string;
  description?: string;
  action?: { label: string; onClick: () => void };
  className?: string;
}
```

WCAG 2.2 compliance is built-in: `min-h-[24px] min-w-[24px]` on
the action button (line 53). `role="status"` on the wrapper (line
38) ensures screen-reader announcement.

**For phase-44.4, replace** 2 sites in `reports/page.tsx`:

- Line 292-294 (History tab no-results inline `<p>`):
  ```tsx
  <EmptyState
    icon={Files}
    title={filter ? `No reports found for "${filter}".` : "No reports found yet."}
    description={filter ? "Try clearing the filter or running an analysis for this ticker." : "Reports appear here after the first analysis runs."}
    action={filter ? { label: "Clear filter", onClick: () => setFilter("") } : undefined}
  />
  ```
- Line 334 (Compare tab no-reports inline `<p>`):
  ```tsx
  <EmptyState
    icon={ArrowsLeftRight}
    title="No reports to compare"
    description="Compare requires at least 2 completed analyses."
  />
  ```

And 1 site in `performance/page.tsx`:
- Lines 149-157 (cost history empty block) -> wrap in `<EmptyState icon={TabCost} title="No cost history yet" description="Costs appear here after the first analysis runs" />`

**When to inline vs use foundation:** master design Section 3.18
calls out ~10 inline empty blocks needing canonicalization. Per
project convention, the foundation component is the default;
inline `<p>` is allowed only for transient/one-off micro-empty
cases (e.g., "No bull case." inside a debate accordion at
`AgentRationaleDrawer.tsx:231` -- a sub-list inside an already-
labeled section). The two `<p>` sites in /reports both qualify
as full-page empty states -> foundation required.

### Topic 5: TimeRangeSelector pattern (7d/30d/90d/all)

**Consensus** (uxmovement.com + GitLab Pajamas + repo
RedLineMonitor precedent):

For exactly 4 options (7d, 30d, 90d, all), segmented control beats
dropdown decisively:

| Dimension | Segmented (4 buttons) | Dropdown |
|-----------|----------------------|----------|
| Visibility | "see every sorting option available" (UX Movement) | options hidden until clicked |
| Clicks | 1 (target -> click) | 2 + scroll |
| Cognitive load | low (all options pre-attentive) | higher (memory of choices) |
| Space | ~280px (4 x ~70px buttons) | ~100px |
| WCAG 2.2 target size | 24x24 trivially | dropdown handle 24x24 trivially |

**Verdict:** segmented for 7d/30d/90d/all.

**WAI-ARIA pattern for segmented:** a `role="radiogroup"` with
`aria-label="Time range"` containing 4 buttons each with
`role="radio"` + `aria-checked={isSelected}`. Alternative:
`role="tablist"` if the segmented control conceptually switches
between views. For pyfinagent: the segmented control filters the
displayed data window (does not switch panel content), so
radiogroup is semantically correct.

**Existing RedLineMonitor precedent:** `RedLineMonitor.tsx` already
has a 7d/30d/90d window selector (per master design Section 4.6).
The new `TimeRangeSelector` foundation must mirror that visual
treatment (sky-400 active, slate-400 inactive) and extend with
"all" as a 4th option.

**Component shape** (proposed):

```tsx
type TimeRange = "7d" | "30d" | "90d" | "all";

interface TimeRangeSelectorProps {
  value: TimeRange;
  onChange: (next: TimeRange) => void;
  options?: TimeRange[];  // default ["7d","30d","90d","all"]
  ariaLabel?: string;
}
```

Each button: `min-h-[32px]` to clear WCAG 2.2 trivially;
`role="radio"`, `aria-checked={value === opt}`; group wrapper
`role="radiogroup" aria-label={ariaLabel ?? "Time range"}`.

Keyboard nav handled by browser default (Tab between buttons +
Space/Enter activates) -- no custom Arrow-key handler needed
because each button is independently tabbable.

## Internal codebase audit

| File | Lines | Role | Status |
|---|---|---|---|
| `frontend/src/app/reports/page.tsx` | 604 | Reports page | TARGET: tab + filter URL state migration; Compare wizard -> Drawer; history -> DataTable; empty -> EmptyState. Existing TABS array at lines 80-83 (no role attrs). |
| `frontend/src/app/reports/page.tsx:87` | 1 | Suspense wrap | KEEP -- already matches Next.js doc recommendation for useSearchParams. |
| `frontend/src/app/reports/page.tsx:94-104` | 11 | Manual URL reading | REPLACE -- migrate to `useURLState` for `tab` (line 95) and `ticker` (line 103). |
| `frontend/src/app/reports/page.tsx:233-248` | 16 | Tab bar render | ADD aria: `role="tablist" aria-label="Reports views"` on container; `role="tab" aria-selected={activeTab === tab.id} aria-controls={`panel-${tab.id}`} id={`tab-${tab.id}`}` on each button. Content blocks at lines 260+332 get `role="tabpanel" aria-labelledby={`tab-${tab.id}`} id={`panel-${tab.id}`}`. |
| `frontend/src/app/reports/page.tsx:292-294` | 3 | Inline empty `<p>` | REPLACE -> `<EmptyState>` foundation. |
| `frontend/src/app/reports/page.tsx:297-327` | 31 | History list (BentoCard per row) | REPLACE -> `<DataTable>` with columns: Ticker, Date, Score, Recommendation, **Sparkline (30d)**, Expand. |
| `frontend/src/app/reports/page.tsx:332-381` | 50 | Compare select wizard | REPLACE -> open in Drawer overlay; mirror `AgentRationaleDrawer` shape. |
| `frontend/src/app/performance/page.tsx` | 267 | Performance page | TARGET: AreaChart above cost table; Sparkline next to win rate; per-pillar bars; TimeRangeSelector. |
| `frontend/src/app/performance/page.tsx:82-138` | 57 | KPI grid | ADD sparkline next to Win Rate at line 87-88 (use SparkAreaChart on 30d outcome data). Add `aria-label` per metric card. |
| `frontend/src/app/performance/page.tsx:149-157` | 9 | Inline cost-history empty | REPLACE -> `<EmptyState>`. |
| `frontend/src/app/performance/page.tsx:158-260` | 103 | Cost history section | ADD Tremor `<AreaChart>` above table (cumulative cost over time). Table itself stays for now (manual; cycle 44.5 will migrate to DataTable). |
| `frontend/src/app/performance/page.tsx:57-63` | 7 | Evaluate button | ADD `aria-busy={evaluating}`. |
| `frontend/src/lib/hooks/useURLState.ts:29-79` | 51 | URL state hook | KEEP -- foundation in place; handles default-value sentinel + back/forward sync. |
| `frontend/src/components/DataTable.tsx:39-169` | 131 | TanStack v8 wrapper | KEEP -- has `meta.className` + `meta.align` support (cycles 62-63); has ARIA `aria-sort` per header. |
| `frontend/src/components/states/EmptyState.tsx:29-61` | 33 | Empty foundation | KEEP -- WCAG-compliant min-target-size already in place. |
| `frontend/src/components/AgentRationaleDrawer.tsx:42-148` | 107 | Existing drawer pattern | MIRROR -- copy this shape for Compare drawer. Has `role="dialog" aria-modal="true"`, Esc close, click-outside close, focus trap via stopPropagation pattern. |
| `frontend/src/components/paper-trading/positions-columns.tsx:17-166` | 150 | Column factory pattern | MIRROR -- this is the canonical `ColumnDef<TData, unknown>[]` factory shape; use it for `reportsColumns(sparklineDataByTicker)`. |
| `frontend/src/lib/types.ts:67-89` | 23 | `SynthesisReport` shape | KEEP -- has `scoring_matrix` (per-pillar) + `final_weighted_score` for per-pillar bars. |
| `frontend/src/lib/types.ts:116-123` | 8 | `ReportSummary` shape | KEEP -- has `ticker`, `analysis_date`, `final_score`, `recommendation` for DataTable rows. |
| `frontend/src/lib/types.ts:125-132` | 8 | `PerformanceStats` shape | KEEP -- has `win_rate`, `wins`, `losses`, `avg_return`, `benchmark_beat_rate`. NOTE: no per-day breakdown -- 30d sparkline needs an additional fetch or backend support (see Risk flags). |
| `frontend/src/lib/types.ts:495-503` | 9 | `CostHistoryEntry` shape | KEEP -- has `analysis_date`, `total_cost_usd`, `total_tokens` for cumulative chart. |
| `frontend/src/components/states/` | 6 files | States library | KEEP -- `EmptyState`, `ErrorState`, `LoadingState`, `OfflineState`, `StaleDataState`, `index.ts` (re-exports). |
| `frontend/package.json:22-23` | 2 | Deps | `@tanstack/react-table@^8.21.3`, `@tremor/react@^3.18.7` -- both already installed. |
| `frontend/package.json:38` | 1 | Recharts dep | `recharts@^2.12.0` -- transitive Tremor dep; phase-44.4 keeps Recharts for the existing Compare-tab radar/bar/line (DON'T re-wire those to Tremor in this cycle; out of scope). |
| `handoff/current/frontend_ux_master_design.md:178-196` | 19 | Section 3.5 (Reports) | Source of truth for success criteria. |
| `handoff/current/frontend_ux_master_design.md:198-214` | 17 | Section 3.6 (Performance) | Source of truth for success criteria. |

## Risk flags

1. **Sparkline 30-d data source for Win Rate:** `PerformanceStats`
   (`types.ts:125`) is a single aggregate (win_rate, wins, losses)
   with NO per-day breakdown. The phase-44.4 sparkline next to
   Win Rate REQUIRES a 30-day rolling-win-rate series. Options:
   - **Option A (no backend change):** derive from existing
     outcome_tracking BigQuery table via a new
     `/api/performance/win-rate-30d` endpoint (new endpoint = scope
     creep; flag for owner).
   - **Option B (frontend-only stub):** render the sparkline only
     when the data is available; otherwise show a `--` placeholder.
     Acceptable for phase-44.4 if backend work is out of scope.
   - **Option C (use evaluations data):** the existing
     `evaluateOutcomes()` flow stores per-recommendation outcomes;
     a frontend-derived 30d trend is possible if the endpoint
     returns a list. Verify before implementing.
   **Recommendation:** Option B for the cycle; defer backend work
   to a follow-up. The master-design spec at line 208 says
   "30d trend" but doesn't dictate the data source.
2. **Sparkline column 30-d score data for Reports:** Similar issue
   for the DataTable sparkline column showing per-ticker 30-day
   score history. The current `ReportSummary[]` array IS the
   history -- group by ticker, sort by analysis_date, take last
   30 entries. **No backend change needed.** Build the map in a
   `useMemo` from the existing `reports` state.
3. **Tremor + Recharts coexistence:** the Reports page currently
   uses Recharts (LineChart, BarChart, RadarChart at lines 414-477).
   Adding `@tremor/react` SparkAreaChart on the same page is fine
   -- Tremor uses Recharts internally, so they share a peer-dep.
   Watch for bundle-size CI guard (cycle 44.0 added bundle budget?
   verify before commit).
4. **Tab `aria-controls` ID stability:** the migration to ARIA
   tablist needs stable IDs across React re-renders.
   `id={\`tab-${tab.id}\`}` and `id={\`panel-${tab.id}\`}` derived
   from the static `TABS` array IDs ("history", "compare") are
   stable -- no risk.
5. **Compare wizard URL deep-linking:** if compare-mode IDs are
   pushed to the URL (e.g., `?compare=AAPL|2026-05-25,NVDA|2026-05-24`),
   the operator can share a comparison link. This is a stretch
   feature for cycle -- the spec only requires `tab` + `ticker`
   sync. Keep `selected: Set<string>` as in-memory state for
   phase-44.4; flag a follow-up.
6. **Drawer focus trap:** `AgentRationaleDrawer.tsx` does NOT
   implement true focus-trapping (no `inert` on background, no
   Tab-cycle inside drawer). It works in practice because the
   stopPropagation + click-outside-to-close + Esc-via-onClose
   pattern is "good enough" -- but a Lighthouse a11y >=95 audit
   may flag this. Two paths:
   - Accept the existing AgentRationaleDrawer pattern (current
     repo standard; consistent with master design Section 3.7
     which lists drawer migration WITHOUT focus-trap as the goal).
   - Add `inert` attribute to the page main when drawer is open
     (modern browser-native; no library needed). Recommend the
     latter as a single 1-line addition.
7. **Performance page state migration:** the page currently has
   minimal state (lines 13-17). Adding `TimeRangeSelector` adds
   `timeRange: TimeRange` state; should this also be URL-synced?
   Per consistency with Reports page tab/filter sync, **yes** --
   use `useURLState<TimeRange>("range", "30d")`.

## Application to phase-44.4

**Concrete edit list with file:line anchors:**

### Reports page

| Where | What | Source |
|-------|------|--------|
| `reports/page.tsx:95` | `const initialTab = ...` -> `const [activeTab, setActiveTab] = useURLState<Tab>("tab", "history", { parser: ... })` | useURLState doc, repo cycle 44.1 |
| `reports/page.tsx:103` | `useState(searchParams.get(...))` -> `useURLState<string>("ticker", "")` | useURLState doc |
| `reports/page.tsx:233` | `<div className="mb-6 flex gap-1 ...">` -> add `role="tablist" aria-label="Reports views"` | W3C tabs pattern |
| `reports/page.tsx:235-248` | per `<button>`: add `role="tab"`, `aria-selected={activeTab === tab.id}`, `aria-controls={`panel-${tab.id}`}`, `id={`tab-${tab.id}`}` | W3C tabs pattern |
| `reports/page.tsx:260` + `:332` | wrap each tab content `<>` in `<div role="tabpanel" id={...} aria-labelledby={...}>` | W3C tabs pattern |
| `reports/page.tsx:292-294` | inline empty `<p>` -> `<EmptyState>` foundation | repo cycle 16 |
| `reports/page.tsx:297-327` | BentoCard list -> `<DataTable>` with `reportsColumns(sparklineDataByTicker)` | repo cycles 44.0 / 44.2 |
| `reports/page.tsx:332-381` (Compare phase 1 / select wizard) | -> Drawer component (NEW: `CompareWizardDrawer.tsx`) | repo cycle 23.1.7 |

### Performance page

| Where | What | Source |
|-------|------|--------|
| `performance/page.tsx:46` (fixed-header zone) | add `<TimeRangeSelector value={range} onChange={setRange}>` | UX Movement segmented |
| `performance/page.tsx:60-63` | `<button>` Evaluate: add `aria-busy={evaluating}` | WCAG 2.2 |
| `performance/page.tsx:84-94` (Win Rate BentoCard) | add `<SparkAreaChart>` next to the value (30d win-rate trend if data available; placeholder if not) | repo cycle 64 |
| `performance/page.tsx:138` (after KPI grid, before Cost section) | add Per-pillar performance bars from `SynthesisReport.scoring_matrix` aggregate | repo precedent |
| `performance/page.tsx:149-157` | inline empty -> `<EmptyState>` | repo cycle 16 |
| `performance/page.tsx:158` (before cost-history `<h3>`) | add Tremor `<AreaChart>` (cumulative cost) wrapped in BentoCard | Tremor doc |

### New foundation component

| New file | Purpose | Anchor |
|----------|---------|--------|
| `frontend/src/components/TimeRangeSelector.tsx` | 4-option segmented control (radiogroup) | master design Section 3.6 line 211 |
| `frontend/src/components/CompareWizardDrawer.tsx` | Compare-flow drawer; mirrors AgentRationaleDrawer shape | master design Section 3.5 line 189 |
| `frontend/src/components/reports-columns.tsx` (or inline in page) | DataTable column factory (`positionsColumns` shape) | repo cycle 44.2 |

## Research Gate Checklist

Hard blockers (gate_passed=false if any unchecked):
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
      (7 sources: Next.js doc, W3C tabs APG, W3C target-size,
      Tremor AreaChart doc, Tremor SparkChart doc, UX Movement
      segmented, react-sparklines repo)
- [x] 10+ unique URLs total (incl. snippet-only): 7 full + 17 snippet
      = 24 URLs
- [x] Recency scan (last 2 years) performed + reported (7 findings
      across 5 source families, all current-stable)
- [x] Full papers / pages read (not abstracts)
- [x] file:line anchors for every internal claim (24 file:line refs
      in the audit table)

Soft checks:
- [x] Internal exploration covered every relevant module (reports
      page, performance page, useURLState hook, DataTable, EmptyState,
      AgentRationaleDrawer, positions-columns, types.ts)
- [x] Contradictions / consensus noted (no contradictions; all
      five topics have clear consensus from authoritative sources
      and repo precedent)
- [x] All claims cited per-claim (file:line for repo, URL for
      external)

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 17,
  "urls_collected": 24,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "gate_passed": true
}
```
