---
paths:
  - "frontend/src/app/**"
  - "frontend/src/components/**"
---

# Frontend Layout Blueprint

Research-backed layout rules for every page in PyFinAgent. Derived from Tufte, Few, Shneiderman, Cleveland & McGill, Bach et al., and peer project analysis (QuantConnect, FreqUI, Grafana, OpenClaw). See `UX-AGENTS.md` for component-level specs and design tokens.

---

## 1. Page Shell (mandatory)

Every page MUST use this exact outer structure — no exceptions:

```tsx
<div className="flex h-screen overflow-hidden">
  <Sidebar />
  <main className="flex flex-1 flex-col overflow-hidden">
    {/* Fixed header zone */}
    <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
      {/* Tiers 1-5: header, banners, tabs — never scroll */}
    </div>
    {/* Scrollable content zone */}
    <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
      {/* Tier 6: tab content — scrolls independently */}
    </div>
  </main>
</div>
```

### Shell rules (non-negotiable)

| Rule | Why |
|------|-----|
| Outer div: `h-screen overflow-hidden` | Constrains layout to viewport height. **Never use `min-h-screen`** — it lets the parent grow, pushing the sidebar off-screen. |
| Sidebar: `h-screen flex-shrink-0` | Sidebar is viewport-locked. It never scrolls with page content. |
| `<main>`: `flex flex-1 flex-col overflow-hidden` | Main is a flex column with two zones: fixed header + scrollable content. |
| Fixed header: `flex-shrink-0` | Title, action buttons, and tab bar stay pinned. They never scroll. |
| Scrollable content: `flex-1 overflow-y-auto scrollbar-thin` | Only tab content scrolls. |
| `scrollbar-thin` on scrollable zone | Custom scrollbar styling as defined in `globals.css`. Never use browser defaults. |

**Why this matters:** When using `min-h-screen`, the outer div grows with content, making the sidebar scroll with the page. Using `h-screen overflow-hidden` locks everything to the viewport. The sidebar stays fixed, the header stays fixed, and only content scrolls. This matches the OpenClaw control UI pattern.

---

## 2. Sidebar (matches OpenClaw pattern)

The sidebar has three zones with independent scroll behavior:

```
┌─────────────────────────┐
│ Fixed header             │  ← Logo + app name, never scrolls
│ (flex-shrink-0)          │
├─────────────────────────┤
│ Scrollable nav           │  ← Nav sections with collapsible groups
│ (flex-1 overflow-y-auto) │     Uses CaretDown toggle per section
│ (scrollbar-thin)         │     Click section header to collapse/expand
├─────────────────────────┤
│ Fixed footer             │  ← Settings + user info, never scrolls
│ (flex-shrink-0)          │     Separated by border-t
└─────────────────────────┘
```

### Sidebar rules
- **Fixed header:** Logo + "PyFinAgent" title. `flex-shrink-0`, no scroll.
- **Nav sections:** Collapsible groups (Analyze, Reports, Trading). Each has a `CaretDown` toggle icon that rotates `-rotate-90` when collapsed. State managed via `collapsedSections` record.
- **Fixed footer:** Settings link + user avatar + auth buttons + version. `flex-shrink-0`, separated by `border-t border-navy-700`.
- **Never** let the sidebar scroll with main page content. The sidebar is `h-screen` and viewport-locked.
- **Icons:** Phosphor icons only. **No emoji anywhere in the UI.**

---

## 3. Page Anatomy (top-to-bottom order)

Content inside `<main>` follows a strict 6-tier vertical hierarchy, split across two zones:

### Fixed header zone (Tiers 1-5) — `flex-shrink-0`
```
┌─────────────────────────────────────────────┐
│ Tier 1: Page Header                         │  ← always visible, never scrolls
│   title + subtitle + optional action buttons│
├─────────────────────────────────────────────┤
│ Tier 5: Tab Bar                             │  ← if page has tabs, never scrolls
│   pill-style, Phosphor icons                │
└─────────────────────────────────────────────┘
```

### Scrollable content zone (Tier 6) — `flex-1 overflow-y-auto`
```
┌─────────────────────────────────────────────┐
│ Tier 2: Status Banners                      │  ← conditional (error/success)
│   dismissible, rose/emerald borders         │
├─────────────────────────────────────────────┤
│ Tier 3: Progress Panels                     │  ← collapsible <details>
│   for long-running background tasks         │
├─────────────────────────────────────────────┤
│ Tier 4: Global Controls                     │  ← filters, run selectors
│   things that affect ALL tabs               │
├─────────────────────────────────────────────┤
│ Tier 6: Tab Content                         │  ← tab-scoped data
│   metrics, grids, tables, charts            │
└─────────────────────────────────────────────┘
```

**Rule:** Page header (Tier 1) and tab bar (Tier 5) are ALWAYS in the fixed zone — they never scroll off-screen. Status banners, progress panels, and tab content go in the scrollable zone. This ensures the user always knows what page they're on and can switch tabs without scrolling up. *(OpenClaw pattern; Shneiderman "Overview first"; Bach et al. screen-fit over overflow.)*

**Rule:** Only globally relevant content lives above the tab bar (Tiers 1-4). Tab-specific metrics, cards, and tables go INSIDE their tab (Tier 6). This prevents viewport waste and ensures tab content starts near the top. If a Tier 4 control is only relevant to a subset of tabs, conditionally hide it based on `tab` state.

### Page header patterns

Standard header (no actions):
```tsx
<div className="mb-8">
  <h2 className="text-2xl font-bold text-slate-100">Page Title</h2>
  <p className="text-sm text-slate-500">Descriptive subtitle</p>
</div>
```

Header with action buttons:
```tsx
<div className="mb-6 flex items-center justify-between">
  <div>
    <h2 className="text-2xl font-bold text-slate-100">Page Title</h2>
    <p className="text-sm text-slate-500">Subtitle</p>
  </div>
  <div className="flex items-center gap-2">
    {/* action buttons */}
  </div>
</div>
```

---

## 4. Metric Grids

### Hero metrics (up to 6 KPIs)

```tsx
<div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
  <Metric label="Sharpe" value="1.42" color="text-emerald-400" />
  {/* ... */}
</div>
```

### Summary cards (4 KPIs)

```tsx
<div className="mb-8 grid grid-cols-2 gap-4 md:grid-cols-4">
  <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-5">
    <p className="text-xs font-medium uppercase tracking-wider text-slate-500">LABEL</p>
    <p className="mt-1 text-2xl font-bold text-slate-100">VALUE</p>
  </div>
</div>
```

### Card anatomy tokens

| Element | Classes |
|---------|---------|
| Card container | `rounded-xl border border-navy-700 bg-navy-800/60 p-5` |
| Label | `text-xs font-medium uppercase tracking-wider text-slate-500` |
| Value | `text-2xl font-bold text-slate-100` |
| Sub-text | `text-xs text-slate-500` |
| Positive value | `text-emerald-400` |
| Negative value | `text-rose-400` |

**Rule:** Show each metric ONCE prominently — the "KPI hero zone" (Few 2006). Use delta columns in comparison tables, never repeated absolute values. If a metric appears in a hero grid, don't repeat it in a standalone card elsewhere on the same tab. *(Tufte data-ink ratio; Shneiderman Rule 8 "reduce short-term memory load".)*

---

## 5. Tab Bar

### Pill-style tabs (standard) — inside fixed header zone

```tsx
<div className="flex gap-1 rounded-lg bg-navy-800/60 p-1">
  {TABS.map((t) => (
    <button
      key={t.id}
      onClick={() => setTab(t.id)}
      className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
        tab === t.id
          ? "bg-sky-500/10 text-sky-400"
          : "text-slate-400 hover:text-slate-200"
      }`}
    >
      <t.icon size={16} weight={tab === t.id ? "fill" : "regular"} />
      {t.label}
    </button>
  ))}
</div>
```

**Rule:** Tab bar lives in the fixed header zone (`flex-shrink-0`). It never scrolls off-screen. Do NOT use `sticky top-0` hacks — use the proper two-zone flex layout instead.

### Tab definition pattern

```tsx
const TABS: { id: TabId; label: string; icon: Icon }[] = [
  { id: "results", label: "Results", icon: Table },
  { id: "equity",  label: "Equity Curve", icon: ChartLineUp },
];
```

Optional badges for count/status: append `badge?: string | number | null` to the type.

### Tab content rendering

```tsx
{tab === "results" && (
  <div className="space-y-6">
    {/* All results-tab content here — metrics, tables, charts */}
  </div>
)}
```

**Rule:** Each tab answers a different analytical question (Optuna/PyFolio pattern). Don't duplicate content between tabs. If two tabs show the same metric, one of them is wrong. *(PyFolio tear sheets: 7 functions, zero metric overlap.)*

---

## 6. Collapsible Sections

Use native HTML `<details>`/`<summary>` — accessible, keyboard-navigable, no React state needed.

### When to use
- Progress panels for long-running background tasks
- Error tracebacks
- Supplementary reference content users rarely need

### Pattern

```tsx
<details open={isActive} className="mb-4 rounded-xl border border-slate-700/60 bg-[#080f1e]">
  <summary className="flex cursor-pointer items-center gap-2 px-4 py-3 text-sm">
    <span className={`h-2 w-2 rounded-full ${isActive ? "bg-sky-400 animate-pulse" : "bg-emerald-500"}`} />
    <span className="font-medium text-slate-200">Section Title</span>
    <span className="ml-auto font-mono text-xs text-slate-500">compact info</span>
  </summary>
  <div className="px-4 pb-4">
    {/* Full expanded content */}
  </div>
</details>
```

### Summary line anatomy (~40px collapsed height)
- Status dot: pulsing `bg-sky-400` (active), static `bg-emerald-500` (done), `bg-slate-600` (idle)
- Title: `font-medium text-slate-200`
- Trailing info: `font-mono text-xs text-slate-500` (step count, elapsed time, etc.)

**Rule:** Default-collapsed saves viewport. User can always expand to see full detail. Use `open={condition}` to auto-expand for active states. *(Grafana collapsible row groups; FreqUI compact progress bar; Bach et al. screenfit tradeoffs.)*

---

## 7. Content Blocks

### BentoCard (grouped content)

```tsx
<BentoCard>
  <h3 className="mb-4 text-lg font-semibold text-slate-300">Section Title</h3>
  {/* content */}
</BentoCard>
```

Use BentoCard to group related content (a table + its header, a chart + its legend). Don't wrap a single metric in a BentoCard.

### Tables

```tsx
<div className="overflow-hidden rounded-xl border border-navy-700">
  <table className="w-full text-left text-sm">
    <thead className="border-b border-navy-700 bg-navy-800/80">
      <tr>
        <th className="px-4 py-3 font-medium text-slate-400">Column</th>
      </tr>
    </thead>
    <tbody className="divide-y divide-navy-700/50">
      <tr className="transition-colors hover:bg-navy-700/40">
        <td className="px-4 py-3 font-mono text-slate-200">Value</td>
      </tr>
    </tbody>
  </table>
</div>
```

### Charts

- Always inside a BentoCard wrapper
- Use Recharts (`ResponsiveContainer` at 100% width)
- Provide an empty-state placeholder when data is missing (see Section 8)

**Rule:** Tables for lookup, charts for patterns (Few 2012). Both are needed, but each answers a different question — never duplicate between them. Use `overflow-x-auto` for wide tables on mobile.

---

## 8. Empty States & Loading

### Empty state (no data yet)

```tsx
<div className="flex flex-col items-center justify-center py-24 text-center">
  <SomePhosphorIcon size={48} weight="duotone" className="text-slate-600" />
  <p className="mt-4 text-lg text-slate-400">Guidance text</p>
  <p className="mt-1 text-sm text-slate-600">Additional context</p>
</div>
```

### Loading state

```tsx
{loading && <PageSkeleton />}
```

Or inline:
```tsx
<div className="flex items-center gap-3 py-12 text-slate-400">
  <div className="h-5 w-5 animate-spin rounded-full border-2 border-sky-500 border-t-transparent" />
  Loading {context}...
</div>
```

### Error banner

```tsx
{error && (
  <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-950/30 p-3">
    <p className="text-sm text-rose-300">{error}</p>
  </div>
)}
```

Add `<details>` for tracebacks. Add a dismiss button for user-clearable errors.

**Rule:** Never show blank space. Every conditional render (`{data && ...}`) must have a corresponding empty state for when data is null/empty. Chart components must render a placeholder when they have insufficient data, never `return null`. *(W&B "always show informative default"; Shneiderman "overview first".)*

---

## 9. Information Hierarchy Principles

These are non-negotiable design constraints, not suggestions. Cite the source when explaining a layout decision in code review.

| Principle | Source | Enforcement |
|-----------|--------|-------------|
| Overview first, zoom and filter, then details on demand | Shneiderman 1996 | Hero metrics → tab selection → detail tables |
| Maximize data-ink ratio | Tufte 1983 | Every pixel carries new information; remove redundant encodings |
| Show once, compare in context | Few 2006, Bloomberg | Hero grid = single source of truth; baselines show deltas only |
| Reduce short-term memory load | Shneiderman Rule 8 | Never show the same metric 3× on one page |
| Position >> Length >> Angle >> Area | Cleveland & McGill 1984 | Scatter/bar charts over pie/gauge/donut |
| Tables for lookup, charts for patterns | Few 2012 | Both needed; each answers a different question |
| One visualization per question | Optuna, PyFolio | No overlapping chart purposes between tabs |
| Pre-attentive attributes for status | NNG, Cleveland & McGill | Color-coded status (green/red/amber/gray) processed in <250ms |
| Screen-fit over overflow for analytic dashboards | Bach et al. 2022 | Tabs preferred; avoid scrolling that hides comparisons |
| Consistent micro-interactions | Material Design 3, Apple HIG | Same scrollbar, same hover states, same transitions everywhere |
| No emoji in UI | PyFinAgent convention | Use Phosphor icons (`@phosphor-icons/react`) exclusively |
| Fixed navigation elements | OpenClaw pattern | Sidebar, page header, and tab bar never scroll off-screen |

---

## New Page Template

Copy-paste skeleton for creating a new page:

```tsx
"use client";

import { useState } from "react";
import { Sidebar } from "@/components/Sidebar";

export default function NewPage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* ── Fixed header zone ── */}
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          {/* Tier 1: Header */}
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-slate-100">Page Title</h2>
            <p className="text-sm text-slate-500">Descriptive subtitle</p>
          </div>

          {/* Tier 5: Tab bar (if needed) */}
        </div>

        {/* ── Scrollable content zone ── */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
          {/* Tier 2: Status banners (conditional) */}

          {/* Tier 3: Collapsible progress (conditional) */}

          {/* Tier 4: Global controls (conditional) */}

          {/* Tier 6: Tab content */}
        </div>
      </main>
    </div>
  );
}
```
