# Research Brief: phase-44.2 Donut Replacement (Tremor -> Inline SVG)

**Tier:** simple
**Date:** 2026-05-26
**Researcher:** researcher subagent
**Status:** COMPLETE

## Problem statement

Cycle 69 shipped `frontend/src/components/PortfolioAllocationDonut.tsx`
using Tremor's `<DonutChart>` and added the `@tremor/**` content path
to `tailwind.config.js` to satisfy JIT scanning. Operator confirmed
(2026-05-26) two defects remain:

1. **Slices render uncolored** -- the donut ring shows uniform dark
   navy; only the custom legend dots colored via the JIT-safe
   `DOT_BG_CLASS` map (`PortfolioAllocationDonut.tsx:52-69`) carry the
   palette. The `colors` prop passed to `<DonutChart>`
   (`PortfolioAllocationDonut.tsx:118`) does NOT actually color slices.
2. **Hover tooltip escapes the card** -- the Tremor default tooltip
   renders white-on-dark, positioned below the chart, breaking the
   navy/slate aesthetic and overflowing the card border.

This mirrors **cycle-63 SectorBarList Option B** decision pattern:
Tremor's chart internals route through a third-party theme/CSS chain
we don't fully control, and chasing the customization knob costs more
than just owning the rendering. The right move is to replace
`<DonutChart>` with a Tailwind + inline-SVG donut + a hand-rolled
tooltip we fully control -- consistent with the
`MiniSpark`/`LogEventRateSpark` precedents already in the codebase.

## Search queries run (3-variant discipline)

1. Current frontier: `"SVG donut chart stroke-dasharray multiple slices implementation 2026"`
2. Last-2-year window: `"chart tooltip" "react" implementation pattern 2024`
3. Year-less canonical: `"SVG donut chart React component implementation"`,
   `"WCAG 2.2 SC 1.4.13 Content on Hover or Focus tooltip pattern"`,
   `"SVG accessibility role img aria-label title element WAI-ARIA Graphics Module"`,
   `"Tailwind CSS SVG stroke fill attribute classes stroke-blue-500 JIT"`,
   `""data visualization" "donut chart" alternatives "stacked bar" "treemap" allocation"`

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|---------------------|
| https://heyoka.medium.com/scratch-made-svg-donut-pie-charts-in-html5-2c587e935d72 | 2026-05-26 | blog (Mark Caron / Medium) | WebFetch full | "r = 100/(2pi) = 15.91549430918952. To start at 12:00 use offset=25. Subsequent slices: offset = 100 - (sum of all previous segments) + 25." Canonical stroke-dasharray formula. |
| https://www.w3.org/WAI/WCAG22/Understanding/content-on-hover-or-focus.html | 2026-05-26 | official spec (W3C WCAG 2.2 AA) | WebFetch full | "Dismissible: a mechanism is available to dismiss the additional content without moving pointer hover or keyboard focus... Hoverable... Persistent." Three-leg requirement for any hover-revealed content. |
| https://tailwindcss.com/docs/stroke | 2026-05-26 | official docs (Tailwind) | WebFetch full | "stroke-blue-500 -> stroke: var(--color-blue-500)" -- yes, `className="stroke-blue-500"` colors an SVG `<circle>` stroke; arbitrary values `stroke-[#243c5a]` work; CSS custom props via `stroke-(--my-stroke-color)`. |
| https://medium.com/@theAngularGuy/how-to-create-an-interactive-donut-chart-using-svg-107cbf0b5b6 | 2026-05-26 | blog (Mustapha AOUAS / Medium) | WebFetch full | "Using `<path>` elements (rather than `stroke-dasharray`) enables interactivity because each donut slice using a `<path>` creates individual elements that respond to hover events." Arc command syntax: `A rx ry x-rotation large-arc-flag sweep-flag x y`. |
| https://css-tricks.com/building-a-donut-chart-with-vue-and-svg/ | 2026-05-26 | blog (CSS-Tricks) | WebFetch full | Stroke-dashoffset = circumference - (percentage * circumference). `angleOffset: -90` to start at 12 o'clock. `rotate(degrees, cx, cy)` transform per slice. "Subtract 2 from circumference for visual spacing." |
| https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Roles/img_role | 2026-05-26 | official docs (MDN ARIA) | WebFetch full | "Set role='img' on the outer `<svg>` and give it a label. This will cause screen readers to consider it as a single entity. All Descendants Become Presentational." Provides exact donut example with stroke-dasharray. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://dev.to/mustapha/how-to-create-an-interactive-svg-donut-chart-using-angular-19eo | blog | Angular-specific; React/Vue articles covered the algorithm |
| https://github.com/cedricdelpoux/react-svg-donut-chart | library README | Add-a-dep alternative; project policy is no new deps without owner approval |
| https://www.npmjs.com/package/react-donut-chart | npm package page | Same -- snippet-only as reference for "what dep we are NOT adding" |
| https://codepen.io/hue94/pen/NgMGWO | codepen | Mark-up only; the formula source covers the math |
| https://www.wcag.com/authors/1-4-13-content-on-hover-or-focus/ | guideline summary | W3C primary source covered in full |
| https://www.unimelb.edu.au/accessibility/techniques/accessible-svgs | university accessibility guide | Returned 403; MDN role=img source covers the same ground |
| https://www.w3.org/WAI/standards-guidelines/act/rules/7d6734/proposed/ | ACT rule | Conformance test; less relevant for implementation guidance |
| https://www.swizec.com/blog/tooltips-tooltips-are-not-so-easy/ | blog | Snippet sufficient; the pattern (state + pointer-events:none) covered |
| https://www.npmjs.com/package/react-svg-tooltip | library README | Add-a-dep alternative; not used |
| https://mui.com/x/react-charts/tooltip/ | library docs | MUI X is a separate ecosystem; not a candidate |
| https://www.grafana.com/developers/saga/components/tooltip/ | design system | Grafana-specific; pattern is the same |
| https://github.com/carbon-design-system/carbon-charts/issues/1301 | bug report | Documents the "missing ARIA labels on donut hover tooltip" failure mode -- evidence of the W3C requirement |
| https://inforiver.com/insights/11-pie-chart-alternatives-and-when-to-use-them/ | industry blog | Alternative chart types -- snippet sufficient |
| https://www.tableau.com/blog/5-unusual-alternatives-pie-charts | industry blog | Same |

## Recency scan (last 2 years; 2024-2026)

Searched: `"chart tooltip" "react" implementation pattern 2024`,
`"SVG donut chart stroke-dasharray multiple slices implementation 2026"`,
`"WCAG 2.2 SC 1.4.13 ... 2025"`,
`"data visualization" "donut chart" alternatives ... allocation 2025"`.

**Finding: no new techniques in the 2024-2026 window that supersede
the canonical sources above.** The stroke-dasharray formula
(2015-2018 era), the WCAG 2.1/2.2 hover/focus rules (2018/2023), and
the role=img + aria-label pattern have all been stable through 2026.
The 2024 pattern surveys for React chart tooltips (MUI X, Recharts +
shadcn, AG Charts) confirm the state-management + `pointer-events:
none` + position-relative-to-mouse pattern is still canonical -- no
new framework primitive replaced it. Grafana's 2025 design-system
docs explicitly recommend "ARIA tags when building tooltips" --
consistent with the W3C rule.

The 2024-2026 snippet on Carbon Charts donut tooltips (GitHub issue
#1301, still open) is a useful adversarial data point: it
**documents** that even mature chart libraries ship donut hover
tooltips that fail WCAG SC 1.4.13 by missing aria labels on the
hover popup. Avoid this trap.

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/src/components/PortfolioAllocationDonut.tsx` | 1-161 | Current Tremor-based impl with broken slice colors + escaping tooltip | TO REPLACE |
| `frontend/src/components/PortfolioAllocationDonut.test.tsx` | 1-109 | 7 vitest cases covering empty state, legend, percentages, ARIA, sort, totalNav fallback | KEEP -- public API (props + DOM contract) must survive |
| `frontend/src/components/SectorBarList.tsx` | 1-149 | Cycle-63 Option B precedent: Tremor BarList -> Tailwind grid; uses `BAR_CLASS` / `VALUE_TEXT_CLASS` JIT-safe class maps | REFERENCE PRECEDENT |
| `frontend/src/components/cron/LogEventRateSpark.tsx` | 1-93 | Cycle-66 inline-SVG MiniSpark precedent; uses `polygon` + `polyline` + inline `stroke="#38bdf8"`; renders nothing when no data; `role="region"` + `aria-label` | REFERENCE PRECEDENT |
| `frontend/src/app/page.tsx` | 63-101 (MiniSpark) | KPI-tile sparkline; viewBox=`0 0 80 24`; `preserveAspectRatio="none"`; inline `stroke` hex (`#34d399` / `#fb7185`); `aria-hidden="true"` because tile wrapper carries the label | REFERENCE PRECEDENT |
| `frontend/tailwind.config.js` | 22-44 | navy.500..900 palette; default Tailwind colors (blue, amber, indigo, ...) work because Tailwind ships them | REFERENCE |
| `frontend/src/components/paper-trading/positions-columns.tsx` | (file exists; no inline MiniSparkSVG found) | -- | N/A |
| `.claude/rules/frontend.md` | (section "Dark-mode + readability") | Mandates: navy/slate palette, no light-mode fallbacks, JIT-safe class maps, third-party CSS path requirement, visual verification mandatory, WCAG-AAA contrast targets | NORMATIVE |

The `LogEventRateSpark` and `MiniSpark` precedents establish the
pattern this brief recommends: inline-SVG, hex-string strokes (not
`stroke-blue-500`), `polyline`/`polygon` for paths, `viewBox` +
`preserveAspectRatio="none"`, `role="region"` on a wrapper div +
`aria-hidden="true"` on the SVG.

## Key findings

### F1: stroke-dasharray on 100-circumference circle is the right primitive for a small allocation donut

The classical formula is:
- `r = 100 / (2*PI)` so that `circumference = 2*PI*r = 100`. Now each
  slice's `stroke-dasharray="<pct> <100-pct>"` directly encodes its
  share without further math (Source: Caron / heyoka 2018-2025 era).
- One `<circle>` element per slice, all centered at the same
  `(cx, cy)` with the same `r`, stacked. Each slice is shown as the
  visible portion of its `stroke-dasharray`.
- `stroke-dashoffset` rotates each slice: first slice gets offset=25
  to start at 12 o'clock; subsequent slice offset =
  `100 - (sum of all previous segment percentages) + 25`.

**Why this primitive over `<path>` arcs?** For a small (h-32 w-32 =
128px) donut with no per-slice interactivity beyond hover-tooltip,
stroke-dasharray is simpler, the math fits on one line per slice,
and Tailwind's `stroke-*` utilities cover the colors. The
AOUAS/Angular-Guy article notes path-based wins on interactivity --
but we can attach hover to the individual `<circle>` elements just
fine (each is its own DOM node).

**Adopted approach:** stroke-dasharray with one `<circle>` per slice
(see implementation outline below).

### F2: Tailwind stroke-* utilities work but inline `stroke="#hex"` is the more conservative path

`className="stroke-blue-500"` works and resolves to
`stroke: var(--color-blue-500)` (Source: Tailwind docs 2026). But:

1. The DOT_BG_CLASS pattern already maps a color-token name
   (`"blue"`, `"amber"`, ...) to a `bg-*-500` literal. The
   `stroke-*-500` set must ALSO be a JIT-safe literal map -- so
   adding `STROKE_CLASS` would double the maintenance.
2. The MiniSpark + LogEventRateSpark precedents use inline
   `stroke="#38bdf8"` / `stroke="#34d399"` hex strings. This is
   consistent and skips the JIT-class concern entirely.

**Adopted approach:** add a third map alongside SECTOR_COLOR_MAP +
DOT_BG_CLASS:

```ts
const SLICE_STROKE_CLASS: Record<string, string> = {
  blue: "stroke-blue-500",
  amber: "stroke-amber-500",
  indigo: "stroke-indigo-500",
  emerald: "stroke-emerald-500",
  fuchsia: "stroke-fuchsia-500",
  lime: "stroke-lime-500",
  orange: "stroke-orange-500",
  yellow: "stroke-yellow-500",
  cyan: "stroke-cyan-500",
  violet: "stroke-violet-500",
  rose: "stroke-rose-500",
  slate: "stroke-slate-500",
  pink: "stroke-pink-500",
  teal: "stroke-teal-500",
  sky: "stroke-sky-500",
  purple: "stroke-purple-500",
};
```

This is the same JIT-safe-literal pattern that DOT_BG_CLASS uses.
Tailwind compiles every entry because they're all literal strings.
Each `<circle>` gets `className={SLICE_STROKE_CLASS[colors[i]]}`.

### F3: WCAG SC 1.4.13 requires dismissible + hoverable + persistent for chart-slice tooltips

Per W3C 2026 reading:

- **Dismissible:** ESC key closes the tooltip.
- **Hoverable:** moving the pointer onto the tooltip itself does not
  dismiss it; the tooltip can be read carefully.
- **Persistent:** stays visible until the user removes the trigger
  (moves pointer off slice + off tooltip) or presses ESC.

**Common failure (F95):** content shown on hover cannot itself be
hovered. For a chart-slice tooltip positioned next to the slice,
this is easy to violate if you set `pointer-events: none` on the
tooltip wrapper (the user's pointer just goes back over the slice
and the tooltip vanishes).

**Adopted approach:**
- Tooltip is a positioned `<div>` inside the chart container
  (NOT a portal). Position is computed from the hovered slice's
  index (e.g. anchor on the slice midpoint translated to chart
  coords), not from the mouse cursor -- this is more stable than
  cursor-follow and avoids the "leave-and-re-enter flicker"
  problem.
- Tooltip is NOT given `pointer-events: none`; user can hover
  onto it.
- Hover state cleared on mouseleave of BOTH the slice AND the
  tooltip (i.e. of the chart container as a whole, or with
  matched delays).
- ESC keydown listener on the container clears the hover state.

### F4: SVG accessibility -- `role="img"` + `aria-label` is right for a small allocation donut; `graphics-symbol` / `graphics-document` are over-engineered here

Per MDN role=img + W3C ARIA Graphics Module (Caron's example,
unimelb 403'd but MDN covers it):

- `role="img"` on the outer `<svg>` makes screen readers treat the
  whole donut as ONE unit and read its `aria-label`. All children
  are flagged presentation-only. Best fit for a small allocation
  donut where the *legend list* (already rendered separately in the
  existing `<ul>`) provides the actual data access.
- `role="graphics-symbol"` would expose each slice as a sub-element
  -- not useful here because the legend already does this.
- `role="graphics-document"` is for complex multi-component
  figures (e.g. annotated time-series with axes + labels + legend
  all inside the SVG). Overkill for this donut.

**Adopted approach:**
- Outer SVG: `role="img"` + `aria-label="Allocation chart: {top
  slice name} {top pct}%, {second slice name} {second pct}%, ..."`
- Container `<div>`: `role="region"` + `aria-label={title}` (kept
  from current impl + matches `LogEventRateSpark` precedent).
- Legend list: a `<ul>` (kept; this is the screen-reader-friendly
  data path).
- No `<title>` element inside SVG (would be redundant with
  aria-label; also `<title>` inside SVG produces a browser native
  tooltip that conflicts with our custom hover tooltip).

### F5: Hover-highlight UX -- show tooltip + slightly increase stroke opacity, no slice darkening

Survey of Stripe/Linear/Grafana 12 + the data-vis literature:

- **Darken non-active slices:** common in pie/donut libraries
  (Tremor included), but on a small (h-32) donut the visual change
  is muddy.
- **Outline active slice:** works on flat-fill paths; less elegant
  on stroke-dasharray rings.
- **Increase active stroke opacity from 0.85 -> 1.0:** subtle but
  matches the existing `bg-emerald-500/80` band-fill convention in
  `SectorBarList.tsx:46-48`. Implementable with one className
  conditional and no per-slice geometry change.
- **Show tooltip only, no slice visual change:** the simplest path;
  Grafana 12's compact mode does this.

**Adopted approach:** tooltip-only on hover (option 4). The slice
visual stays static; the tooltip carries all the affordance. Aligns
with the MiniSpark precedent which has no hover state at all.

### F6: Donut as chart-type -- defensible for 5-12 sector slices

The "donut is bad" literature (Tableau, Inforiver, Tufte) attacks
the use of pie/donut for 20+ categories or where rank/variance is
the question being asked. **Our use case is different:** 6-12
sectors + cash, where the question is "at a glance, what is the
allocation?" and the legend list carries the precise rank data
side-by-side. The center label carrying total NAV is a useful
data-ink-ratio gain.

The strongest alternatives are:
- **Horizontal stacked bar** (one wide bar, slices end-to-end). Same
  pre-attentive length-encoding as `SectorBarList` -- but
  `SectorBarList` already exists and is the per-sector cap
  visualization. Showing the SAME data as a stacked bar would
  duplicate effort.
- **Treemap.** Better for 15+ categories or hierarchical data.
  Overkill here.

**Adopted approach:** keep the donut shape; the alternatives are
worse for this specific job (Source: Datylon 2025 chart-type guide;
Caron, Few 2006).

## Consensus vs debate (external)

| Question | Consensus | Debate |
|----------|-----------|--------|
| stroke-dasharray vs `<path>` for donut slices | stroke-dasharray simpler for non-interactive small donuts; `<path>` better for per-slice click + animation | none -- both are valid; trade-off is interactivity vs simplicity |
| Tailwind stroke-* utilities vs inline `stroke=` | Both work | Light debate: the JIT-safe-literal map is more maintainable than inline hex if many palettes; inline hex is simpler if just 1-2 colors |
| WCAG SC 1.4.13 dismissible + hoverable + persistent | Required for AA conformance | None -- this is a hard W3C rule |
| `role="img"` on outer SVG | Recommended pattern (MDN, W3C) | None |
| Donut vs alternatives for allocation | Mixed -- "use sparingly" but acceptable for <=10 categories with legend | Tableau / Tufte / Few argue against; Caron / Datylon argue for in this constrained case |

## Pitfalls (from literature)

1. **F95 (WCAG):** `pointer-events: none` on the tooltip wrapper
   makes the tooltip un-hoverable. Don't do this on the tooltip.
   (Source: W3C SC 1.4.13.)
2. **Missing aria-label on hover popup** -- Carbon Charts donut
   tooltip issue #1301 documents this failure mode in a shipping
   product. Our custom tooltip should set `role="tooltip"` + carry
   the data textually so screen readers read it.
3. **Tailwind JIT misses template-string class names.** If we ever
   need to render a runtime-chosen color, the class string MUST be
   a literal in a Record<>, not interpolated. (Source: cycle-68 fix
   on this exact codebase + Tailwind docs 2026.)
4. **stroke-dashoffset moves counter-clockwise** -- the formula
   in F1 already accounts for this, but it's easy to get the sign
   wrong when extending. The 12-o'clock-start adjustment of `+25`
   on every offset is the gotcha. (Source: Caron 2018-2025.)
5. **Tooltip flicker on slice edges.** If hover state is cleared by
   `mouseleave` on the slice and re-set by `mouseenter` on the
   tooltip, there's a 1-frame gap where the tooltip vanishes. Fix:
   parent the tooltip in the same container as the slices and
   compute "hovered" from a single mouseenter/leave on the chart
   container. (Source: Swizec "Tooltips are not so easy" blog.)
6. **SVG `<title>` child element produces a native browser tooltip
   that clashes with custom tooltip.** Don't add `<title>` if you
   have aria-label on the SVG + a custom hover popup. (Source: MDN
   role=img.)

## Application to pyfinagent (mapping external findings to file:line)

Concrete implementation outline for `PortfolioAllocationDonut.tsx`:

### Outer container (unchanged)
- Keep the existing `containerClass` + `role="region"` +
  `aria-label={title}` pattern from
  `PortfolioAllocationDonut.tsx:96, 108`.
- Keep empty-state path at lines 98-105.
- Keep the legend `<ul>` at lines 136-157 (the legend is the screen-
  reader path; the SVG carries the visual only).

### Replace `<DonutChart>` (lines 114-135) with inline SVG

```tsx
const SIZE = 128;             // matches h-32 w-32 footprint
const STROKE_W = 18;          // ring thickness
const RADIUS = 100 / (2 * Math.PI);  // ~15.915, so circumference = 100
const CX = 50;
const CY = 50;
const VB = 100;               // viewBox 0 0 100 100

// Compute per-slice cumulative offsets in useMemo:
type SliceWithGeom = {
  name: string;
  value: number;
  pct: number;
  color: string;
  dasharray: string;   // "<pct> <100-pct>"
  dashoffset: number;  // 25 for first, then 100 - sum(prev) + 25
};

const sliceGeom: SliceWithGeom[] = useMemo(() => {
  let runningSum = 0;
  return data.map((s, i) => {
    const pct = (s.value / totalValue) * 100;
    const offset = 100 - runningSum + 25;
    runningSum += pct;
    return {
      name: s.name,
      value: s.value,
      pct,
      color: colors[i],
      dasharray: `${pct} ${100 - pct}`,
      dashoffset: offset,
    };
  });
}, [data, colors, totalValue]);
```

### SVG markup

```tsx
const [hoverIdx, setHoverIdx] = useState<number | null>(null);

// ESC handler for WCAG 1.4.13 dismissibility
useEffect(() => {
  if (hoverIdx === null) return;
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") setHoverIdx(null);
  };
  window.addEventListener("keydown", onKey);
  return () => window.removeEventListener("keydown", onKey);
}, [hoverIdx]);

const ariaSummary = sliceGeom
  .map((s) => `${s.name} ${s.pct.toFixed(1)}%`)
  .join(", ");

<div className="relative" onMouseLeave={() => setHoverIdx(null)}>
  <svg
    viewBox={`0 0 ${VB} ${VB}`}
    className="h-32 w-32"
    role="img"
    aria-label={`Allocation chart: ${ariaSummary}`}
  >
    {/* Background ring (navy for empty contrast) */}
    <circle
      cx={CX} cy={CY} r={RADIUS}
      fill="none"
      strokeWidth={STROKE_W / 2}
      className="stroke-navy-700"
    />
    {sliceGeom.map((s, i) => (
      <circle
        key={s.name}
        cx={CX} cy={CY} r={RADIUS}
        fill="none"
        strokeWidth={STROKE_W / 2}
        strokeDasharray={s.dasharray}
        strokeDashoffset={s.dashoffset}
        className={`${SLICE_STROKE_CLASS[s.color] ?? "stroke-slate-500"}
          cursor-pointer transition-opacity
          ${hoverIdx !== null && hoverIdx !== i ? "opacity-60" : "opacity-100"}`}
        transform={`rotate(-90 ${CX} ${CY})`}
        onMouseEnter={() => setHoverIdx(i)}
        onFocus={() => setHoverIdx(i)}
        onBlur={() => setHoverIdx(null)}
        tabIndex={0}
        aria-label={`${s.name}: ${s.pct.toFixed(1)}% of NAV`}
      />
    ))}
    {/* Center NAV label */}
    {navForCenter > 0 && (
      <text
        x={CX} y={CY}
        textAnchor="middle"
        dominantBaseline="central"
        className="fill-slate-100 text-[6px] font-mono"
      >
        ${navForCenter.toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </text>
    )}
  </svg>

  {/* Hover tooltip -- WCAG 1.4.13 hoverable + persistent */}
  {hoverIdx !== null && (
    <div
      role="tooltip"
      className="absolute left-1/2 top-full mt-2 -translate-x-1/2
                 z-10 whitespace-nowrap rounded-lg
                 border border-navy-600 bg-navy-900/95 shadow-lg
                 px-2.5 py-1.5 text-xs text-slate-100"
    >
      <span className={`inline-block w-2 h-2 rounded-full mr-1.5 align-middle
                        ${DOT_BG_CLASS[sliceGeom[hoverIdx].color] ?? "bg-slate-500"}`}
            aria-hidden="true" />
      <span className="font-medium">{sliceGeom[hoverIdx].name}</span>
      <span className="ml-2 font-mono tabular-nums text-slate-300">
        ${sliceGeom[hoverIdx].value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        {" "}({sliceGeom[hoverIdx].pct.toFixed(1)}%)
      </span>
    </div>
  )}
</div>
```

### Key parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| viewBox | `0 0 100 100` | Standard coord system; matches Caron formula |
| RADIUS | `100 / (2 * PI)` | Makes circumference = 100 so each slice's stroke-dasharray is its % directly |
| STROKE_W | `18 / 2 = 9` SVG units | At h-32 (128px) display, viewBox 100 -> 1 unit = 1.28px, so 9 units = 11.5px ring -- visually balanced for a 128px donut |
| transform | `rotate(-90 50 50)` | Starts first slice at 12 o'clock (Caron / Vue article) |
| Background ring | `stroke-navy-700` | Gives the empty space a visible darker ring so the donut shape reads even with 1 slice |
| Hover dim | `opacity-60` on non-active | Subtle highlight; standard data-vis pattern |
| Tooltip position | `absolute left-1/2 top-full mt-2 -translate-x-1/2` | Below the donut, centered; stays inside the card container (no portal -> no card-border escape) |
| Tooltip z-index | `z-10` | Above the donut + legend; below page-level modals |

### Hover mechanism: state + mouseleave on container

- `hoverIdx: number | null` state, set on each `<circle>` mouseenter
  and on focus (keyboard nav).
- Cleared on `onMouseLeave` of the **container** `<div>`, NOT the
  `<circle>`. This means the user can move the pointer from the
  slice INTO the tooltip without flicker (the tooltip is inside the
  container too).
- ESC keydown listener clears state -> dismissibility leg.
- Tooltip does NOT have `pointer-events: none` -> hoverable leg
  (F95 avoidance).
- Tooltip stays until pointer leaves container OR ESC -> persistent
  leg.

### Removed code

- `import { DonutChart } from "@tremor/react";` (line 12)
- Tremor dependency check: search-grep `@tremor/react` after this
  cycle to see if any OTHER component still uses it. If not, the
  `@tremor/**` content-path line in `tailwind.config.js:19` and the
  `@tremor/react` package itself become removable (separate PR;
  not blocking this cycle).

### Tests (`PortfolioAllocationDonut.test.tsx`)

All 7 existing test cases survive untouched because they test the
container DOM contract (title, subtitle, legend list,
percentages, role+aria-label, sort, totalNav fallback). Add two
new cases:
- "renders one `<circle>` per slice + one background ring" (asserts
  `svg circle` count = data.length + 1).
- "tooltip not present when hoverIdx is null" -> no element with
  `role="tooltip"` in initial render.

A third visual-verification case (hover -> tooltip appears) belongs
in Playwright not Vitest (operator visual verification mandated by
`frontend.md:5` -- unit tests can't catch slice color rendering).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
      (6 sources: heyoka/Caron, W3C SC 1.4.13, Tailwind docs,
      AOUAS/Angular-Guy, CSS-Tricks/Vue, MDN role=img)
- [x] 10+ unique URLs total (20 URLs incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
      (PortfolioAllocationDonut.tsx + .test.tsx, SectorBarList.tsx,
      LogEventRateSpark.tsx, page.tsx MiniSpark, tailwind.config.js,
      frontend.md, frontend-layout.md)
- [x] Contradictions / consensus noted (F6 donut-vs-alternatives)
- [x] All claims cited per-claim (URLs in tables + section refs)

---

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 14,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```
