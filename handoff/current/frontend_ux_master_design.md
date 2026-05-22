# pyfinagent -- Frontend UX/AI Master Design

**Authored:** 2026-05-22, phase-44.0 super-planning pass.
**Author:** Main (Claude Opus 4.7, this Claude Code session).
**Source of truth for findings:** `handoff/current/research_brief.md` (775 lines; researcher subagent `a63a24f40308f8093`, effort `deep`/max, 10 external sources read in full, gate_passed: true, 84 internal files inspected).
**Purpose:** PLAN -- the next session(s) execute. NOT implementation. NO frontend code changes outside `.claude/masterplan.json` + this file + companion handoff files.

---

## 1 -- State of the Frontend Union (one paragraph)

The pyfinagent frontend has a **solid shell** (Next.js 15 + React 19 + TypeScript 5.6 + Tailwind + Phosphor + Recharts, 256px sidebar with 4 sections + 12 routes, BentoCard + OpsStatusBar canonicalized, `useLiveNav` + `useLivePrices` + `useTickerMeta` live-data foundations), but **four 2026-frontier gaps dominate** the trajectory from "shipped" to "top-notch": (1) **zero Cmd-K command palette** across 15 routes -- table-stakes 2026 per Linear / Vercel / Stripe / cmdk-by-Vercel; (2) **WCAG 2.2 gap** with single-digit `aria-*` per page + OpsStatusBar inline buttons failing the 24x24 CSS px target-size rule (EU AAA mandate 2026); (3) **`/agents` is a flat live-log not a LangSmith-style trace tree** even though backend `MASEvent.event_type` already emits the needed span types (classify / plan / delegate / tool_call / tool_result / thinking / synthesize / loop_check / quality_gate / citation / complete / error); (4) **`/settings` Cost tab duplicates `/paper-trading` Manage tab** -- DRY violation that has caused operator confusion across 3 audit cycles. **Three pages account for ~3300 of the project's frontend LoC** and need route-topology cleanup BEFORE any visual upgrade lands: `/paper-trading` (1284 LoC, cockpit), `/backtest` (1594 LoC, 7 tabs across 4 domains), `/settings` (1410 LoC, dupes paper-trading Manage). **Component dedup ratio:** 58 components -> 36 KEEP, 4 REFACTOR, 4 DEPRECATE-MERGE, 1 EXTRACT-to-states-lib, **14 NEW** (CommandPalette / TraceTree / DataTable / Sparkline / Drawer / EmptyState / ErrorBanner / LiveBadge / TimeRangeSelector / Breadcrumb / KeyboardShortcuts modal / useEventSource / useURLState / useDebounced). **The gap from good-enough to top-notch is 4-5 medium phases, not a rewrite.**

---

## 2 -- Foundation Layer (cross-cutting before any page work)

These foundations cross-cut every later step. Build first; everything else compounds.

### 2.1 -- Design tokens + semantic color system

**Current:** Tailwind tokens like `navy-700`, `slate-400` are mid-2024. **Per Apple HIG 2026 + artofstyleframe 2026** (Section B.10): semantic tokens (`color-bg-surface`, `color-fg-muted`, `color-accent`, `color-feedback-danger`) via CSS custom properties allow future-proofing without rewriting every component. Light-mode is OUT-OF-SCOPE (project is dark-only per `globals.css`), but the semantic-token layer should be in place so adding light-mode later is a config swap, not a refactor.

**Files:** new `frontend/src/styles/tokens.css`, edit `frontend/tailwind.config.ts`.

### 2.2 -- States library

**Current:** Loading / empty / error states inline across 10+ files (`research_brief Section A.18` -- "EmptyState: ~10 inline empty blocks"; "ErrorBanner: ~12 inline rose blocks"). **Per `frontend-layout.md` §8 + saasui.design 2026 progressive disclosure** (Section B.1): extract to `frontend/src/components/states/`.

**Components to extract:** `<LoadingState/>`, `<EmptyState/>`, `<ErrorState/>` (with retry + curl-hint pattern from phase-25.B12), `<OfflineState/>`, `<StaleDataState/>`. `Skeleton.tsx` (50 lines, currently top-level) promotes into the lib.

### 2.3 -- Reusable hooks library

**Per Section A.2 brief:** No `useEventSource` (SSE inline in `/agents/page.tsx:187-232`), no `useDebouncedSearch`, no `useURLState`. **Per cmdk + Linear + TanStack patterns** (Sections B.5, B.6): these are 2026 baseline.

**Hooks to add:**
- `useEventSource<T>(url)` -- shared SSE consumer with reconnection + heartbeat tracking
- `useURLState<T>(key, default)` -- syncs state to `?search=` URL params (deep-linking)
- `useDebounced<T>(value, ms)` -- standard debounce for search inputs
- `useKeyboardShortcut(combo, handler)` -- generalizes `KillSwitchShortcut.tsx`

**Files:** new `frontend/src/lib/hooks/`.

### 2.4 -- Command Palette (Cmd-K) -- the single biggest 2026 gap

**Per Section H.1 of brief:** "the single most impactful change". **Per `cmdk` (Pacos/Vercel) + uxpatterns.dev** (Section B.6): keyboard-first; minimal ARIA; grouped result organization. 3 KB headless lib from Vercel.

**Component:** `<CommandPalette/>` mounted in `frontend/src/app/layout.tsx`. Initial command set:
- **Navigate** (15 routes): "Go to Home", "Go to Paper Trading", etc.
- **Actions** (~30): "Analyze ticker {input}", "Set kill-switch state", "Start backtest", "View latest run", "Stop optimizer", "Force-divest position", "Set max positions"
- **Recent** (5): last analyzed tickers, last viewed reports
- **Shortcuts** (1): "Open keyboard shortcuts modal"

**Trigger:** `Cmd+K` / `Ctrl+K` globally. Linear / Stripe / Vercel convention.

### 2.5 -- WCAG 2.2 AA compliance baseline

**Per Section B.7:** EU AAA mandate by 2026.

**Cross-cutting checks** to land BEFORE page-by-page work:
- Skip-to-main-content link in root layout (`<a href="#main" className="sr-only focus:not-sr-only">`)
- `<main id="main">` wrapper convention
- `focus-visible:ring-2 focus-visible:ring-sky-400` on EVERY interactive element (audit + fix gaps)
- OpsStatusBar inline buttons bumped to `min-h-[24px] min-w-[24px]` (currently `px-2 py-0.5` = ~20-22px) -- target-size violation per WCAG 2.2 2.5.8

### 2.6 -- Sidebar refresh

**Per Section A.1:** `aria-` count = 0; collapse state not persisted (`useState({})` at line 249); no mobile collapse / hamburger; no Cmd-K trigger; no tooltip on health dot; footer crowded.

**Per Linear + artofstyleframe 2026 (Section B.1):** "Sidebar. Full stop. Expanded width should be 256px (16rem), collapsing to 64px with icon tooltips."

**Changes:**
- Persist collapse state in `localStorage` (key: `pyfinagent.sidebar.collapsedSections`)
- Mobile (<768px): hamburger toggle, slide-over from left
- Add Cmd-K trigger button in header / footer ("Press Cmd+K" hint)
- ARIA: `role="navigation"`, `aria-label="Primary"`, each `<Link>` gets `aria-current="page"` when active
- Backend health dot: `<Tooltip>` showing "last checked: 12s ago" + last status
- Footer: collapse 3 buttons (Settings, Account, Logout) into a user-menu popover -- Stripe / Linear pattern

### 2.7 -- New reusable component primitives (14 total per brief Section D)

| New component | Replaces / consolidates | Why |
|---|---|---|
| `<CommandPalette/>` | -- new | Cmd-K table-stakes (B.6) |
| `<TraceTree/>` | /agents Live Stream tab | LangSmith parity (B.2) |
| `<DataTable/>` (TanStack v8 wrapper) | 4+ raw `<table>` blocks | Sort/filter/virtualize/resize (B.5) |
| `<Sparkline/>` (Tremor wrapper) | 0 (new) | KPI tile trend lines (B.9) |
| `<Drawer/>` (generic) | AgentRationaleDrawer (specialize) | Manage + settings inline edit |
| `<EmptyState/>` | ~10 inline empty blocks | Codify pattern from `frontend-layout.md` §8 |
| `<ErrorBanner/>` | ~12 inline rose blocks | Codify retry + curl-hint pattern |
| `<LiveBadge/>` | live-price age in paper-trading; OpsStatusBar Last/Next; observability bands | Standardize "live (3s)" / "stale" pills |
| `<TimeRangeSelector/>` | RedLineMonitor 7d/30d/90d -> extend to /observability, /performance, /cron | Common control |
| `<Breadcrumb/>` | strategy detail back-link | Standardize |
| `<KeyboardShortcuts/>` modal | -- new | Discoverable shortcut list |
| `useEventSource()` | inline EventSource in /agents | Reusable SSE foundation |
| `useURLState()` | tab + filter + selected state | Deep-linking across /reports, /backtest, /paper-trading |
| `useDebounced()` | search inputs (settings, reports) | Standard pattern |

---

## 3 -- Per-Page Expanded Plans

Each page section covers: current setup with file:line (from research_brief Section A), gaps vs 2026 standard, specific patterns to apply (citing research_brief Section B), per-page success criteria (mechanically verifiable), effort + risk class.

### 3.1 -- Sidebar (`Sidebar.tsx`, 383 lines) -- foundation cross-cut

**Current:** 4 collapsible sections (Analyze / Reports / Trading / System) at `Sidebar.tsx:22-64`; 12 routes; 30s `healthCheck()` poll feeding dot color at `Sidebar.tsx:256-266`; collapse state in non-persisted `useState({})` at line 249; `aria-` count = 0; no mobile collapse; no Cmd-K trigger.

**Gaps:** see Section A.1 brief.

**Patterns:** Linear sidebar 256px (B.1); Stripe + Vercel user-menu popover footer; WCAG 2.2 nav landmark + skip link (B.7); cmdk Cmd+K trigger (B.6).

**Success criteria:**
- `Sidebar.tsx` persists section-collapse state via `localStorage` (key `pyfinagent.sidebar.collapsedSections`); reload preserves state
- Mobile (`<768px`): hamburger toggle replaces fixed 256px; slide-over from left
- `role="navigation"` + `aria-label="Primary"`; every `<Link>` gets `aria-current="page"` when active; focus-visible ring on all
- Backend health dot has a tooltip showing last-check timestamp + status
- Footer 3 buttons collapse into user-menu popover
- Cmd-K trigger visible in sidebar; pressing it opens the palette

**Effort:** Moderate (1-2 cycles). **Risk class:** NEEDS-LIVE-VERIFY (real-browser Playwright).

### 3.2 -- Lib refresh (`api.ts`, `types.ts`, `icons.ts`, `useLiveNav.ts`, `kpiMetrics.ts`, ...)

**Current:** Section A.2 brief -- 8 files; api.ts (746 lines, single source of truth); kpiMetrics.ts has `@deprecated sharpe()`; no `useEventSource` / `useDebounced` / `useURLState`.

**Patterns:** SSE 2026 dominant transport (B.4); cmdk hook patterns (B.6); URL-state via `useSearchParams` (B.5).

**Success criteria:**
- New `frontend/src/lib/hooks/` directory with `useEventSource.ts`, `useURLState.ts`, `useDebounced.ts`, `useKeyboardShortcut.ts`
- `kpiMetrics.ts::sharpe` removed (was `@deprecated`)
- `api.ts` adds SSE wrapper functions for future stream endpoints (no-op until backend ships them)

**Effort:** Simple (1 cycle). **Risk class:** SAFE-OVERNIGHT.

### 3.3 -- Page 1: `/` (Home, 312 lines, `frontend/src/app/page.tsx`)

**Current** (Section A.3):
- 10 `useState`; `useLivePrices` + `useLiveNav`; Promise.allSettled across 5 endpoints
- **Painful UX issue:** 3 side-by-side h-full boxes (RecentReportsTable + LatestTransactionsBox + HomeQuickActionsPanel) at `lg:col-span-2` are **forced equal-height** via `h-full` -- exact anti-pattern that `frontend.md:23` warns against. **Self-violating rule.**
- A11y: 1 `aria-attribute` total; no `aria-label` on KPI tiles; no `role` on hero section.

**Gaps:** See Section C row 1 -- self-violating h-full + no Cmd-K.

**Patterns:** Tremor `SparkLineChart` for NAV / P&L / Sharpe tiles (B.9); `LiveBadge` freshness pill on each KPI tile via existing `useLivePrices` age data; Cmd-K palette (B.6); WCAG aria-label on every tile (B.7).

**Success criteria:**
- Self-violating `h-full` removed from 3-box row at `page.tsx:~250` (replace with `grid-rows-[auto]` or per-box natural-height)
- KPI hero: 6 tiles each with `<Sparkline/>` showing 30-day trend (NAV / Total P&L / Today's P&L / Win Rate / Sharpe / Alpha vs SPY)
- Each KPI tile has `aria-label="<metric>: <value>, trend <up|down> <pct>%"` + `role="group"` for screen readers
- LiveBadge on each tile shows "live (3s)" / "stale (60s)" pulled from `useLivePrices` `lastUpdatedAt` field
- Cmd-K palette opens with Cmd+K from anywhere on the page; route navigation + "Analyze ticker {input}" commands work
- Lighthouse a11y >= 95 on cockpit
- LCP <= 2.0s on cold load
- 375px viewport: no horizontal scroll

**Effort:** Moderate (2 cycles). **Risk class:** NEEDS-LIVE-VERIFY.

### 3.4 -- Page 2: `/signals` (191 lines, `frontend/src/app/signals/page.tsx`)

**Current** (Section A.4):
- 4 useState; `getAllSignals(ticker)` single call
- **Painful UX issue:** 50 LoC of nested type-coercion at `signals/page.tsx:34-85` -- belongs in a `useEnrichmentSignals(ticker)` hook
- A11y: 1 aria-attribute; `<input>` has no `aria-label`

**Patterns:** Hook extraction; Cmd-K integration ("Analyze NVDA"); progressive disclosure pattern (B.1) -- consensus pill -> 12 cards -> sector + macro collapsible.

**Success criteria:**
- Move `signals/page.tsx:34-85` type-coercion into new `useEnrichmentSignals(ticker)` hook in `frontend/src/lib/hooks/`
- `<input>` gains `aria-label="Ticker symbol"` + `<label htmlFor=>` pairing
- Recently-fetched tickers as chips below input (last 5; Linear "recent" pattern)
- Consensus pill at top, 12 cards middle, sector + macro as `<details>` (collapsed by default)
- Cmd-K integration: typing "/signals NVDA" from anywhere navigates here pre-loaded
- Lighthouse a11y >= 95

**Effort:** Simple (1 cycle). **Risk class:** SAFE-OVERNIGHT.

### 3.5 -- Page 3: `/reports` (604 lines, `frontend/src/app/reports/page.tsx`)

**Current** (Section A.5):
- 11 `useState` + 2 `useMemo`; tabs (`history` / `compare`); 3 Recharts (Line / Bar / Radar)
- **Painful UX issue:** Compare wizard has 3 phases (select -> startCompare -> render) with no breadcrumb / progress; **`?ticker=NVDA` URL param only seeds the filter -- changing tab doesn't update the URL, deep-linking broken**
- A11y: 0 aria-attributes
- Empty state weak: plain `<p>` "No reports found yet."

**Patterns:** Compare-wizard as modal (B.1 progressive disclosure -- not in-place state switching); `useURLState()` for `tab`/`ticker`/`selected[]` (B.5); TanStack Table v8 for history list (B.5); sparkline column 30-day score per ticker (B.9).

**Success criteria:**
- Compare wizard moves to `<Drawer/>` overlay; "Back to selection" becomes Drawer's `<button>` Esc-handler
- URL deep-linking: `useURLState()` syncs `tab`, `ticker`, `selected` to `?` params; shareable links work
- History list migrates to `<DataTable/>` (TanStack v8) with sortable / virtualized rows + sparkline column for 30d score history
- New `<EmptyState/>` replaces "No reports found yet." with Phosphor icon + guidance + "Get started" link
- A11y: all checkboxes have `aria-label`; "Compare" button has `aria-disabled` when no selection; tab bar has `role="tablist"` + per-tab `role="tab"` + `aria-selected`
- Lighthouse a11y >= 95

**Effort:** Moderate (2 cycles). **Risk class:** NEEDS-LIVE-VERIFY.

### 3.6 -- Page 4: `/performance` (267 lines, `frontend/src/app/performance/page.tsx`)

**Current** (Section A.6):
- 4 useState; `getPerformanceStats`, `getCostHistory(50)`
- **Painful UX issue:** No time-series visualization -- 3 metric cards but no trend. Cost history is a 50-row `<table>` -- hard to see whether cost-per-analysis is rising.

**Patterns:** Tremor `AreaChart` for cumulative cost (B.9); sparkline next to Win Rate (B.9); per-pillar performance bars (data exists in `SynthesisReport`).

**Success criteria:**
- Tremor `AreaChart` (filled) above the cost-history table showing cumulative cost over time
- `<Sparkline/>` next to the `52%` Win Rate number showing 30d trend
- Per-pillar performance bar chart (data already in `SynthesisReport`)
- `aria-label` on every metric card + `aria-busy` on Evaluate button during mutation
- Time-range selector (7d/30d/90d/all) via shared `<TimeRangeSelector/>`
- Lighthouse a11y >= 95

**Effort:** Simple (1 cycle). **Risk class:** SAFE-OVERNIGHT.

### 3.7 -- Page 5: `/paper-trading` (1284 lines, `frontend/src/app/paper-trading/page.tsx`) -- THE COCKPIT

**Current** (Section A.7):
- 19 `useState` + 3 refs + 4 `useMemo`; **densest state surface of any page**
- 6 tabs: Positions / Trades / NAV Chart / Reality gap / Exit quality / **Manage**
- 12 API endpoints
- **Painful UX issue:** Manage tab has 10+ paper-specific knobs that ALSO appear in `/settings` Cost tab -- DRY violation flagged across 3 audit cycles
- A11y: Drawer has `role="dialog"` + `aria-modal` (good); **tab bar has no `role="tablist"` / `role="tab"` (gap)**; tables have no `<caption>` / `aria-label`
- Mobile: tab labels crop below 600px

**Gaps:** See Section C row 5 -- Manage duplicates /settings + Manage tab is sub-page-as-tab.

**Patterns:** Drawer for Manage (B.1 progressive disclosure -- settings as overlay, not tab); route-split tabs to sub-routes (B.5); `<DataTable/>` for both positions + trades (B.5); LangSmith-style `AgentRationaleDrawer` already exists -- expose on more rows; SSE for live-prices on liquid tickers (B.4).

**Success criteria:**
- **Manage tab REMOVED** from /paper-trading; opens as `<Drawer/>` instead; settings live solely in /settings (or vice-versa -- pick one source of truth, the other becomes a link)
- Tabs migrate to sub-routes: `/paper-trading/positions`, `/paper-trading/trades`, `/paper-trading/nav`, `/paper-trading/reality-gap`, `/paper-trading/exit-quality`. URL deep-linking works
- Tab bar gains `role="tablist"` + per-tab `role="tab"` + `aria-selected="true|false"` + `aria-controls=<panel-id>`
- Positions + trades migrate to `<DataTable/>` (TanStack v8): sortable columns, user-resizable, virtualized for >100 rows
- Positions table row-click opens `<AgentRationaleDrawer/>` (already exists); ADD same drawer for trades rows
- LiveBadge component on each position row: "live (3s)" / "stale (60s)"
- Tremor `BarList` for sector concentration (right column)
- LCP <= 2.0s; 5 north-star questions answerable in <= 5s in real browser (Playwright timed)
- 375px viewport: no horizontal scroll; tab labels truncate to icon + tooltip
- Lighthouse a11y >= 95

**Effort:** Large (3 cycles -- split into sub-steps if needed). **Risk class:** NEEDS-LIVE-VERIFY + OWNER-APPROVAL-REQUIRED for Manage tab removal (operator behavior change).

### 3.8 -- Page 6: `/paper-trading/learnings` (53 lines, `frontend/src/app/paper-trading/learnings/page.tsx`)

**Current** (Section A.8):
- 53 LoC shell; `getPaperLearnings(30)`; `VirtualFundLearnings` child
- **Painful UX issue:** No fixed-header zone, no page title. Violates `frontend-layout.md` canonical shell.

**Patterns:** Merge as a tab on `/paper-trading` (data is paper-scoped); `<TimeRangeSelector/>` for windowDays.

**Success criteria:**
- Decision: merge into `/paper-trading/learnings` as a sub-route (NOT a tab -- already at sub-route depth)
- Add `<Breadcrumb/>` "Paper Trading > Learnings"
- Add proper page header with title + `<TimeRangeSelector/>` (currently hardcoded windowDays=30)
- Closes DoD-6 (learn-loop alive in production) visually

**Effort:** Simple (1 cycle). **Risk class:** SAFE-OVERNIGHT.

### 3.9 -- Page 7: `/backtest` (1594 lines, `frontend/src/app/backtest/page.tsx`) -- THE OPERATOR'S WORKBENCH

**Current** (Section A.9):
- 1594 LoC -- **largest page**; 16 useState + 2 polling loops + nested `RunSelector`
- 7 tabs across 4 distinct domains: Overview / Results / Equity / Features / Optimizer / **Harness** / **Budget**
- 17 API endpoints
- **Painful UX issue:** 7 tabs too many; Budget + Harness don't belong here (squatting because no route hosts them); `RunSelector` is ~140 LoC of mountain-of-state
- A11y: 0 on tabs; modal-style `<RunSelector>` lacks `role`
- Mobile: NOT mobile-friendly (7 tabs in single row + complex tables)

**Patterns:** Route-split into `/backtest`, `/harness`, `/budget` (B.1 strategic minimalism); cmdk for run-switching (B.6); `<DataTable/>` for trade-list + experiments (B.5); SSE for optimizer progress (B.4).

**Success criteria:**
- **Route-split:** `/backtest` keeps Overview / Results / Equity / Features / Optimizer (5 tabs); `Harness` tab promoted to `/harness`; `Budget` tab promoted to `/budget`. Sidebar updated.
- `RunSelector` replaced with cmdk-driven palette ("Switch run", "Compare runs", "Load latest")
- Trade-list table migrates to `<DataTable/>` (TanStack v8 with sort/filter/pagination); current `tradePage` / `tradeSort` / `tradeSearch` state extracted into TanStack
- Polling (2s) replaced with SSE on `/api/backtest/stream` (NEW backend endpoint required -- see Section 5 Backend Coupling)
- A11y: tab bar `role="tablist"` + per-tab `role="tab"`; RunSelector `role="listbox"` + arrow keys
- Mobile: tabs reduce to icon-only below 600px with tooltip; trade-list horizontal scroll explicit (`overflow-x-auto`)
- LCP <= 2.5s (Optimizer charts are heavy)
- Lighthouse a11y >= 95

**Effort:** Large (3 cycles -- one per: route-split, RunSelector->cmdk, DataTable+SSE). **Risk class:** NEEDS-LIVE-VERIFY + OWNER-APPROVAL-REQUIRED for route-split (operator URL/bookmark change).

### 3.10 -- Page 8: `/sovereign` (188 lines, `frontend/src/app/sovereign/page.tsx`)

**Current** (Section A.10):
- 6 useState; two-hero (RedLine + Leaderboard) + ComputeCost
- **Painful UX issue:** Cockpit-vs-sovereign split unclear; RedLine appears on BOTH `/` and `/sovereign`. Operator never knows where to look.

**Patterns:** Color-group sync (B.3 Bloomberg) -- selecting a ticker/strategy in one place highlights it across panels; RedLine event annotations as timeline pills underneath (not just Recharts ReferenceDots).

**Success criteria:**
- Decision: keep RedLine on /sovereign as the primary surface; **REMOVE RedLine from /** -- replace with a higher-level "trading control plane" call-to-action card pointing to /sovereign
- Per-strategy click-through wired (already exists via `/sovereign/strategy/[id]`)
- RedLine event annotations: add timeline pills row below the chart showing event types + click-to-jump
- `aria-label` on each leaderboard row + cost-breakdown bar
- Lighthouse a11y >= 95

**Effort:** Moderate (1-2 cycles). **Risk class:** OWNER-APPROVAL-REQUIRED (removing RedLine from / changes operator habit).

### 3.11 -- Page 9: `/sovereign/strategy/[id]` (87 lines, `frontend/src/app/sovereign/strategy/[id]/page.tsx`)

**Current** (Section A.11):
- 87 LoC shell; `getSovereignStrategy(id)`
- **Painful UX issue:** Inconsistent back-button style (custom `<Link>` with `CaretLeft` vs plain text button elsewhere)

**Patterns:** Standardize on `<Breadcrumb/>` component (Section D row 10).

**Success criteria:**
- Replace custom back-link with `<Breadcrumb/>` -- "Sovereign > Strategy {name}"
- Loading state upgrades from plain `<p>Loading...</p>` to `<LoadingState/>` from states-lib
- A11y: Breadcrumb has `aria-label="Breadcrumb"` + `aria-current="page"` on last item
- Lighthouse a11y >= 95

**Effort:** Simple (0.5 cycle). **Risk class:** SAFE-OVERNIGHT.

### 3.12 -- Page 10: `/agents` (728 lines, `frontend/src/app/agents/page.tsx`) -- MAS LIVE OBSERVABILITY

**Current** (Section A.12):
- 728 LoC; 4 tabs: Live Stream / Run History / Agent Map / OpenClaw
- **The only SSE-driven page in the app** (`agents/page.tsx:187-232`)
- **Painful UX issue:** Live Stream is unbounded log scroll -- no filter ("show only `error` events"), no severity color; Agent Map tab is **hand-coded SVG with hardcoded x/y coordinates** at `agents/page.tsx:610-650` -- duplicates `/agent-map` route which uses React Flow

**Patterns:** LangSmith-style trace tree (B.2 + B.8) -- group SSE events by `run_id` AND nest tool_call -> tool_result hierarchically; severity filters; side-by-side run comparison (LangSmith killer feature); annotation queue; replace hand-coded SVG with React Flow (merge with `/agent-map`).

**Backend already supports it:** `MASEvent.event_type` field emits classify / plan / delegate / tool_call / tool_result / thinking / synthesize / loop_check / quality_gate / citation / complete / error -- categorization already exists.

**Success criteria:**
- Live Stream tab refactored to `<TraceTree/>` component:
  - SSE events grouped by `run_id` (parent rows expand into children)
  - tool_call -> tool_result nested hierarchically
  - Severity filter pills (error / warning / info) above the tree
  - Each event has a `<button>` to "Mark for review" (annotation queue persisted to BQ)
- Side-by-side compare: select 2 run_ids -> open `<Drawer/>` with diff highlighting
- Agent Map tab REMOVED from /agents; users redirected to `/agent-map` (single source of truth via React Flow)
- `useEventSource()` shared hook replaces inline EventSource
- A11y: connection state pill has `aria-live="polite"`; each trace node has `role="treeitem"` + `aria-expanded` + `aria-level`
- Lighthouse a11y >= 95

**Effort:** Moderate (2 cycles -- one for TraceTree, one for compare + annotation). **Risk class:** NEEDS-LIVE-VERIFY.

### 3.13 -- Page 11: `/agent-map` (34 lines, `frontend/src/app/agent-map/page.tsx`)

**Current** (Section A.13):
- 34 LoC shell; `AgentMap` child uses React Flow with dagre layout
- **Painful UX issue:** Duplicates `/agents` Agent Map tab (hand-coded SVG)

**Patterns:** Merge -- /agents Agent Map tab DELETED; /agent-map becomes the single topology view (Section 3.12 dep).

**Success criteria:**
- `/agent-map` retained; /agents Agent Map tab REMOVED (per 3.12)
- Add page header with title + last-updated timestamp
- Per-agent click opens drawer with recent calls (from `/api/mas/events?agent_id={id}`)
- `aria-label` on each node + edge

**Effort:** Simple (1 cycle). **Risk class:** SAFE-OVERNIGHT.

### 3.14 -- Page 12: `/cron` (449 lines, `frontend/src/app/cron/page.tsx`)

**Current** (Section A.14):
- 449 LoC; 2 tabs: Jobs / Logs
- 5s polling; stops after 5 failures (phase-23.3.5 hardening)
- **Painful UX issue:** Logs viewer is plain `<pre>` -- no search, no level-color, no follow-tail toggle, no permalink. "Operator's pager 2014" pattern.

**Patterns:** Grafana-Loki facet search (B.5 inspired); per-source 7d sparkline (B.9); level-color severity (B.2 -- analogous to LangSmith trace severity); follow / pause toggle.

**Success criteria:**
- Log search box with facet pills (level: ERROR / WARN / INFO); filter applies client-side first, server-side if >1000 lines
- JSON lines syntax-color (basic regex highlighting)
- Sparkline above the log showing event-rate per minute (Tremor `SparkChart`)
- "Follow" / "Pause" toggle (default: follow newest)
- Permalink to specific line (`#L1234` URL fragment scrolls + highlights)
- Compact density toggle (32-line / 16-line view)
- A11y: log lines have `role="log"` + `aria-live="polite"`; pause state has `aria-pressed`
- Lighthouse a11y >= 95

**Effort:** Moderate (2 cycles). **Risk class:** SAFE-OVERNIGHT.

### 3.15 -- Page 13: `/observability` (169 lines, `frontend/src/app/observability/page.tsx`)

**Current** (Section A.15):
- 169 LoC; 30s polling; freshness table only
- **Painful UX issue:** No timeline view of staleness. Tells me what's stale NOW but not "yesterday's BQ ingest lag was 3.2 min vs today's 8.4 min."

**Patterns:** Per-source 7d sparkline (B.9 Tremor); "next ingest in {dt}" countdown if SLA amber/red; cross-link to /cron `Logs` filtered to that source.

**Success criteria:**
- Per-source row gains a 7d `<Sparkline/>` column showing lag-over-time
- "Next ingest in {dt}" countdown when band is amber/red
- Cross-link: clicking a source row opens /cron prefiltered to that source's log
- 0 Unknown bands across all source rows (closes DoD-5)
- `<TimeRangeSelector/>` (7d/30d) controls sparkline window
- A11y: each freshness band pill has `aria-label="Source {source}: {band} ({age} ago)"`
- Lighthouse a11y >= 95

**Effort:** Simple (1 cycle). **Risk class:** NEEDS-LIVE-VERIFY.

### 3.16 -- Page 14: `/settings` (1410 lines, `frontend/src/app/settings/page.tsx`)

**Current** (Section A.16):
- **1410 LoC, 2nd-largest page**; 3 tabs: Models & Analysis / Cost & Weights / Performance
- 11 useState + 2 useMemo; inline ModelPicker (~130 LoC custom dropdown)
- **Painful UX issue:** /paper-trading Manage tab DUPLICATES ~50% of these settings -- DRY violation flagged across 3 audit cycles

**Patterns:** DRY merge with /paper-trading Manage (B.1 strategic minimalism); search box at top filtering all settings (Linear cmd-K-inline-edit pattern); Cmd-K inline-edit ("Set max positions to 15") (B.6).

**Success criteria:**
- **/paper-trading Manage tab REMOVED** (per 3.7); settings live solely here
- New search box at top filtering all setting rows ("paper_default_stop_loss_pct" matches scroll-to-row)
- New tabbed sections (subdivided per master_roadmap_to_production.md DoD):
  - **Models & Analysis** (existing, refactored)
  - **Cost & Weights** (existing, refactored)
  - **Risk** (NEW: daily_loss_pct / trailing_dd_pct / sector_cap_pct sliders + audit-log row appended for every change)
  - **LLM Route** (NEW: standard / deep-think tier dropdowns + live test-call button)
  - **Cycle** (NEW: manual run-now + cycle-budget override + kill-switch state/history + auto-resume toggle)
  - **Audit Log** (NEW: append-only operator-action log)
  - **Feature Flags** (NEW: visible toggle for every flag in `featureFlags.ts`)
- ModelPicker custom dropdown gains `role="listbox"` + arrow-key nav + `aria-activedescendant`
- Destructive actions (flatten-all, force-divest) require typed confirmation ("FLATTEN" / "FORCE-DIVEST")
- Cmd-K integration: "Set paper_max_positions to 15" inline-edit via palette
- A11y: every input has `<label htmlFor=>` (audit existing 1410 LoC); `aria-describedby` for help text
- Lighthouse a11y >= 95

**Effort:** Large (3 cycles -- one per: DRY merge, new tabs, ModelPicker a11y). **Risk class:** OWNER-APPROVAL-REQUIRED for destructive-action wiring.

### 3.17 -- Page 15: `/login` (109 lines, `frontend/src/app/login/page.tsx`)

**Current** (Section A.17):
- 109 LoC; NextAuth Google + Passkey/WebAuthn
- **Painful UX issue:** No `<form>` -- buttons with onClick handlers; Enter key does nothing; no passkey explainer

**Patterns:** Form wrap + Enter-key submit; helper text for passkey ("Passkey is a more secure passwordless sign-in -- learn more"); WCAG Accessible Authentication 3.3.8 (no cognitive function test -- already satisfied via NextAuth).

**Success criteria:**
- Wrap buttons in `<form onSubmit={...}>` so Enter triggers Google sign-in (the default action)
- Helper text + tooltip explaining Passkey (with link to a docs page)
- `aria-label` on each button; `aria-busy` during signIn
- Clear error state if email not in `ALLOWED_EMAILS` (currently logs generic "An error occurred")
- Focus-visible rings on all interactive elements
- Lighthouse a11y >= 95

**Effort:** Simple (0.5 cycle). **Risk class:** SAFE-OVERNIGHT.

### 3.18 -- Components audit (58 entries from Section A.18)

**Verdict tally:** 36 KEEP, 4 REFACTOR, 4 DEPRECATE-MERGE, 1 EXTRACT, 14 NEW.

**Immediate action (no operator question -- per Section H.8):**
- **DELETE** `OptimizerInsights.tsx` (explicitly marked deprecated in `backtest/page.tsx:74` comment)
- **VERIFY ZERO CALLERS** then DELETE: `KillSwitchPanel.tsx`, `CycleHealthStrip.tsx` (absorbed by OpsStatusBar)
- **REMOVE** `kpiMetrics.ts::sharpe` function (explicitly `@deprecated` in favor of backend value)
- **EXTRACT** `Skeleton.tsx` from top-level into `frontend/src/components/states/` alongside the new EmptyState/ErrorBanner/LoadingState

**REFACTOR:**
- `Sidebar.tsx` (per Section 3.1)
- `BudgetDashboard.tsx` -- moves out of /backtest into new /budget route (per 3.9)
- `HarnessDashboard.tsx` -- moves out of /backtest into new /harness route (per 3.9)
- `CostDashboard.tsx` -- merge with `BudgetDashboard` (overlap)

**DEPRECATE-MERGE candidates flagged but pending owner-call:**
- `DecisionTraceView.tsx` + `DebateView.tsx` + `GlassBoxCards.tsx` + `AgentRationaleDrawer.tsx` -- 4 components for the "show me the AI's reasoning" use-case. Section A.18 keeps all 4 KEEP but the planner should consider consolidating into a single `<DecisionTrailDrawer/>` (LangSmith-trace-tree shape). **Owner question:** consolidate or keep specialized?

**Effort:** Moderate (2 cycles -- one for housekeeping deletions, one for refactors). **Risk class:** SAFE-OVERNIGHT for deletions; OWNER-APPROVAL-REQUIRED for the 4-into-1 consolidation question.

---

## 4 -- Risk Classification per Step

| Step | Risk class | Owner-approval reason |
|---|---|---|
| 44.0 (this) | SAFE-OVERNIGHT | Plan-only |
| 44.1 Foundation: tokens + states-lib + hooks + Cmd-K + WCAG baseline + Sidebar | NEEDS-LIVE-VERIFY | -- |
| 44.2 Cockpit (/paper-trading + route-split + Manage->Drawer) | OWNER-APPROVAL-REQUIRED | Manage tab removal changes operator habit |
| 44.3 Decision Trail Drawer consolidation (4-into-1?) | OWNER-APPROVAL-REQUIRED | Component consolidation question |
| 44.4 Reports section (/reports + /performance) | NEEDS-LIVE-VERIFY | -- |
| 44.5 Trading non-paper (/backtest split + /sovereign + detail + learnings) | OWNER-APPROVAL-REQUIRED | /backtest route-split changes bookmark URLs |
| 44.6 Analyze section (/ + /signals) | NEEDS-LIVE-VERIFY | -- |
| 44.7 System section (/agents trace-tree + /agent-map merge + /cron + /observability) | NEEDS-LIVE-VERIFY | -- |
| 44.8 Settings + Login (DRY merge + new tabs + login form) | OWNER-APPROVAL-REQUIRED | Settings consolidation; destructive-action wiring |
| 44.9 Mobile + a11y + states polish | SAFE-OVERNIGHT | -- |
| 44.10 SSE everywhere (requires backend stream endpoints) | OWNER-APPROVAL-REQUIRED + BACKEND COUPLING | Adds backend stream endpoints |

**Rollup:** 1 SAFE-OVERNIGHT (this planning step), 5 NEEDS-LIVE-VERIFY, 5 OWNER-APPROVAL-REQUIRED, 0 HIGH-BLAST.

---

## 5 -- Backend Coupling Notes

Section H.7 of research_brief warns: SSE replacements for polling require backend stream endpoints.

**Frontend steps that depend on new backend endpoints:**
- 44.10 (SSE everywhere) needs: `/api/paper-trading/stream`, `/api/backtest/stream`, `/api/observability/stream`
- 44.7 /agents enhancements (annotation queue): needs `/api/mas/annotations` (POST + GET)
- 44.8 Settings audit-log: needs `/api/settings/audit-log` (GET)
- 44.8 LLM-route live test-call: needs `/api/llm/test-call` (POST -- returns whatever route resolves to)

**Strategy:** EACH new backend endpoint = additive GET (or single POST). NO mutations beyond what's already exposed. Pair with backend phase coordination -- 44.10 cannot land until backend ships streams.

---

## 6 -- Definition of UX/UI Done (Production-Grade Frontend)

12 concrete measurable criteria. Each MUST PASS for the UX/UI gate.

| # | Criterion | Measurement | Status today |
|---|---|---|---|
| **UX-1** | **Cmd-K palette works from every route** | Real-browser test: Cmd+K opens palette on each of the 15 routes; "Go to {route}" navigates correctly | FAIL (zero coverage) |
| **UX-2** | **WCAG 2.2 AA compliance across all 15 routes** | Lighthouse a11y >= 95 on every route; `axe-core` zero-violations on every route | FAIL (single-digit aria per page; target-size violations on OpsStatusBar) |
| **UX-3** | **Mobile-friendly at 375px** | `python scripts/qa/responsive_check.py 375 768 1280 1920` exits 0 across all 15 routes | FAIL (Sidebar fixed 256px; /backtest 7 tabs crop) |
| **UX-4** | **No DRY-violation duplicate settings** | grep: paper-specific setting keys appear in EXACTLY ONE page (/settings) | FAIL (/paper-trading Manage tab duplicates ~10 keys) |
| **UX-5** | **TraceTree replaces flat-log in /agents** | /agents Live Stream tab renders `run_id`-grouped tree; severity filters work; side-by-side compare works | FAIL (today flat log) |
| **UX-6** | **TanStack DataTable across 4+ tables** | grep: `<DataTable` appears in /paper-trading, /reports, /backtest, /cron; raw `<table>` blocks reduced by >= 4 | FAIL (today 0) |
| **UX-7** | **URL deep-linking works** | Real-browser: copy URL with `?tab=trades&ticker=NVDA&selected=AAPL,MSFT`; paste in new tab; state restored | FAIL (URL only seeds filter on /reports; tab + selected lost) |
| **UX-8** | **States library used everywhere** | grep: no inline `animate-pulse` / "Loading..." / inline rose banner; all use `<LoadingState/>` / `<EmptyState/>` / `<ErrorBanner/>` | FAIL (today: ~10+ inline empty, ~12+ inline error) |
| **UX-9** | **Sparklines on KPI tiles** | / Home + /paper-trading cockpit: every numeric KPI tile shows a 30d trend sparkline | FAIL (today 0) |
| **UX-10** | **Live updates within 1s on cockpit** | Real-browser: NAV ticker visibly updates within 1s of backend write | FAIL (today 30s polling) -- depends on backend stream endpoint |
| **UX-11** | **Keyboard nav covers full app** | `?` opens shortcut overlay; all primary actions reachable via shortcut; Tab traverses page in document order | FAIL (only `KillSwitchShortcut` exists) |
| **UX-12** | **Lighthouse perf score >= 90 on cockpit** | LCP <= 2.0s; TBT <= 200ms; cockpit `/paper-trading` route | UNKNOWN (no current benchmark) |

**Today (2026-05-22) UX-DoD pass rate:** 0 of 12.
**To reach top-notch:** close 12 criteria via phases 44.1-44.10.

---

## 7 -- JSON-Ready Masterplan Inserts

Copy-paste-ready blocks for `.claude/masterplan.json`. Schema reference: existing phase-44 entries (planned to be inserted alongside the phase-35-43 backend roadmap from cycle 10). All steps inserted with `status: in-progress` per `feedback_masterplan_status_flip_order`; the next-session executor flips to `done` AFTER work + harness_log append.

```json
    {
      "id": "phase-44",
      "name": "Production-Grade Frontend UX/AI overhaul (top-notch, complete view, all 15 pages)",
      "status": "in-progress",
      "depends_on": [],
      "gate": null,
      "steps": [
        {
          "id": "44.0",
          "name": "Super-planning: deep research + per-page expanded master design",
          "status": "in-progress",
          "harness_required": true,
          "priority": "P1",
          "depends_on_step": null,
          "audit_basis": "User /goal 2026-05-22: deep research codebase + top production-ready UX/AI 2026 patterns, expand each frontend design step with findings. Researcher subagent a63a24f40308f8093 (deep/max) produced research_brief.md 775 lines, 10 external sources read in full, 84 internal files inspected. 33 OPEN findings across 15 pages + 4 cross-cutting (Cmd-K, WCAG 2.2, trace-tree, settings DRY).",
          "verification": {
            "command": "test -f handoff/current/frontend_ux_master_design.md && test -f handoff/current/research_brief.md && test -f handoff/current/contract.md && grep -qE 'PASS|FAIL' handoff/current/live_check_44.0.md",
            "success_criteria": [
              "frontend_ux_master_design_md_exists_with_per_page_plan_for_15_pages_plus_sidebar_plus_lib_plus_components",
              "every_page_has_at_least_3_success_criteria_citing_research_brief_section_B",
              "12_DoD_criteria_in_section_6",
              "JSON_inserts_valid_for_phase_44_with_status_in_progress_NOT_done",
              "no_frontend_code_changes_git_diff_stat_frontend_src_empty"
            ],
            "live_check": "live_check_44.0.md carries Q/A PASS verdict + coverage-grep evidence + 12-row DoD baseline."
          },
          "retry_count": 0,
          "max_retries": 3
        },
        {
          "id": "44.1",
          "name": "Foundation: design tokens + states lib + hooks + Cmd-K + WCAG baseline + Sidebar refresh",
          "status": "pending",
          "harness_required": true,
          "priority": "P1",
          "depends_on_step": "44.0",
          "audit_basis": "Per master_design Section 2: foundations cross-cut every later step. Per research_brief Section H.1: Cmd-K is single most impactful change. Per Section B.7: WCAG 2.2 EU AAA mandate 2026.",
          "verification": {
            "command": "pytest frontend/tests/test_phase_44_1_foundation.py -v && grep -q 'CommandPalette' frontend/src/app/layout.tsx && grep -q 'states/' frontend/src/components/index.ts && test -f handoff/current/live_check_44.1.md",
            "success_criteria": [
              "frontend_src_components_states_directory_exists_with_5_components",
              "frontend_src_lib_hooks_directory_exists_with_4_hooks",
              "CommandPalette_mounted_in_root_layout_cmd_k_opens_it",
              "WCAG_2_2_baseline_skip_link_and_24x24_target_size_audit_passes",
              "Sidebar_collapse_state_persists_in_localStorage",
              "Sidebar_mobile_hamburger_works_at_375px",
              "Lighthouse_a11y_at_least_95_on_three_sampled_pages"
            ],
            "live_check": "live_check_44.1.md captures Lighthouse + axe-core results + Playwright Cmd-K trace."
          },
          "retry_count": 0,
          "max_retries": 3
        },
        {
          "id": "44.2",
          "name": "Cockpit (/paper-trading + route-split + Manage->Drawer + TanStack tables + Sparklines + BarList)",
          "status": "pending",
          "harness_required": true,
          "priority": "P1",
          "depends_on_step": "44.1",
          "audit_basis": "Per master_design Section 3.7: 1284 LoC cockpit is the primary surface; Manage tab duplicates /settings (DRY violation across 3 audit cycles); positions+trades raw <table>; tab bar a11y gap; 6 tabs cross 4 domains.",
          "verification": {
            "command": "pytest frontend/tests/test_phase_44_2_cockpit.py -v && playwright test frontend/playwright/cockpit.spec.ts && test -f handoff/current/live_check_44.2.md",
            "success_criteria": [
              "paper_trading_Manage_tab_removed_opens_as_Drawer_instead",
              "tabs_migrated_to_sub_routes_positions_trades_nav_reality_gap_exit_quality",
              "tab_bar_has_role_tablist_and_per_tab_role_tab_aria_selected",
              "positions_table_uses_DataTable_TanStack_v8_with_sort_filter_virtualize",
              "trades_table_uses_DataTable_TanStack_v8",
              "AgentRationaleDrawer_opens_from_both_positions_and_trades_rows",
              "LiveBadge_on_each_position_row_shows_live_or_stale",
              "Tremor_BarList_for_sector_concentration_right_column",
              "five_north_star_questions_answerable_in_5_seconds_real_browser",
              "LCP_under_2_seconds_cold_load",
              "no_horizontal_scroll_at_375px",
              "Lighthouse_a11y_at_least_95"
            ],
            "live_check": "live_check_44.2.md captures Playwright trace of 5-question answerability + Lighthouse + screenshots at 4 breakpoints."
          },
          "retry_count": 0,
          "max_retries": 3
        },
        {
          "id": "44.3",
          "name": "Decision Trail Drawer consolidation (4-into-1: DecisionTraceView+DebateView+GlassBoxCards+AgentRationaleDrawer)",
          "status": "pending",
          "harness_required": true,
          "priority": "P2",
          "depends_on_step": "44.1",
          "audit_basis": "Per master_design Section 3.18: 4 components for 'show me the AI reasoning' use-case can consolidate into one <DecisionTrailDrawer/>. Owner-approval needed before deletion.",
          "verification": {
            "command": "pytest frontend/tests/test_phase_44_3_decision_drawer.py -v && test -f handoff/current/live_check_44.3.md",
            "success_criteria": [
              "DecisionTrailDrawer_component_exists_in_frontend_src_components",
              "drawer_renders_enrichment_debate_synthesis_critic_risk_judge_trade_stages_in_order",
              "risk_judge_stage_quotes_portfolio_sector_exposure_verbatim_from_llm_call_log",
              "drawer_opens_in_under_200ms_from_positions_table_click",
              "4_deprecated_components_deleted_after_owner_approval"
            ],
            "live_check": "live_check_44.3.md captures the drawer open-timing + verbatim portfolio_sector_exposure quote + Playwright trace."
          },
          "retry_count": 0,
          "max_retries": 3
        },
        {
          "id": "44.4",
          "name": "Reports section refresh (/reports + /performance: URL deep-linking + DataTable + Tremor AreaChart)",
          "status": "pending",
          "harness_required": true,
          "priority": "P2",
          "depends_on_step": "44.1",
          "audit_basis": "Per master_design Section 3.5 + 3.6: /reports has 604 LoC with broken URL deep-linking, weak compare-flow UX; /performance has no time-series visualization.",
          "verification": {
            "command": "pytest frontend/tests/test_phase_44_4_reports.py -v && playwright test frontend/playwright/reports-deep-link.spec.ts && test -f handoff/current/live_check_44.4.md",
            "success_criteria": [
              "reports_useURLState_syncs_tab_ticker_selected_to_url_params",
              "reports_compare_wizard_uses_Drawer_overlay",
              "reports_history_uses_DataTable_with_sparkline_column",
              "performance_AreaChart_above_cost_history_table",
              "performance_sparkline_next_to_win_rate",
              "performance_TimeRangeSelector_7d_30d_90d_all",
              "Lighthouse_a11y_at_least_95_on_both_pages"
            ],
            "live_check": "live_check_44.4.md captures URL deep-link Playwright trace + screenshots."
          },
          "retry_count": 0,
          "max_retries": 3
        },
        {
          "id": "44.5",
          "name": "Trading non-paper refresh (/backtest route-split + /sovereign + /sovereign/strategy + /paper-trading/learnings)",
          "status": "pending",
          "harness_required": true,
          "priority": "P2",
          "depends_on_step": "44.1",
          "audit_basis": "Per master_design Section 3.9: /backtest is 1594 LoC with 7 tabs across 4 domains; Budget + Harness don't belong; RunSelector is mountain-of-state. Section 3.10: RedLine duplicates on / and /sovereign.",
          "verification": {
            "command": "pytest frontend/tests/test_phase_44_5_trading.py -v && test -f handoff/current/live_check_44.5.md",
            "success_criteria": [
              "backtest_route_split_into_backtest_harness_budget_with_sidebar_update",
              "backtest_RunSelector_replaced_with_cmdk_palette",
              "backtest_trade_list_uses_DataTable_TanStack_v8",
              "sovereign_RedLine_removed_from_home_kept_only_on_sovereign",
              "sovereign_RedLine_event_annotations_timeline_pills_below_chart",
              "strategy_detail_uses_Breadcrumb_component",
              "paper_trading_learnings_has_proper_header_and_TimeRangeSelector",
              "Lighthouse_a11y_at_least_95_on_all_four_pages"
            ],
            "live_check": "live_check_44.5.md captures the route-split + Playwright bookmark migration + screenshots."
          },
          "retry_count": 0,
          "max_retries": 3
        },
        {
          "id": "44.6",
          "name": "Analyze section refresh (/ Home self-violating h-full fix + Sparklines + LiveBadge; /signals hook extraction)",
          "status": "pending",
          "harness_required": true,
          "priority": "P2",
          "depends_on_step": "44.1",
          "audit_basis": "Per master_design Section 3.3: home page has self-violating h-full anti-pattern; KPI tiles missing sparklines + aria-labels. Section 3.4: /signals has 50 LoC of type-coercion that belongs in a hook.",
          "verification": {
            "command": "pytest frontend/tests/test_phase_44_6_analyze.py -v && test -f handoff/current/live_check_44.6.md",
            "success_criteria": [
              "home_3box_row_h_full_anti_pattern_removed",
              "home_6_KPI_tiles_have_Sparkline_LiveBadge_aria_label",
              "signals_useEnrichmentSignals_hook_replaces_inline_type_coercion",
              "signals_recent_tickers_chips_below_input",
              "Lighthouse_a11y_at_least_95_on_both_pages"
            ],
            "live_check": "live_check_44.6.md captures the home + signals refactor + screenshots."
          },
          "retry_count": 0,
          "max_retries": 3
        },
        {
          "id": "44.7",
          "name": "System section refresh (/agents TraceTree + side-by-side compare + annotation; /agent-map merge; /cron facet search; /observability sparkline)",
          "status": "pending",
          "harness_required": true,
          "priority": "P1",
          "depends_on_step": "44.1",
          "audit_basis": "Per master_design Section 3.12-3.15: /agents Live Stream is flat log (should be LangSmith trace tree per B.2); Agent Map tab duplicates /agent-map React Flow; /cron logs viewer is plain <pre> with no search; /observability has no timeline view.",
          "verification": {
            "command": "pytest frontend/tests/test_phase_44_7_system.py -v && playwright test frontend/playwright/trace-tree.spec.ts && test -f handoff/current/live_check_44.7.md",
            "success_criteria": [
              "agents_Live_Stream_uses_TraceTree_grouped_by_run_id",
              "agents_side_by_side_compare_via_Drawer_with_diff_highlighting",
              "agents_annotation_queue_persists_to_BQ",
              "agents_Agent_Map_tab_removed_users_redirected_to_agent_map_route",
              "agent_map_page_gains_header_and_per_agent_drawer",
              "cron_logs_facet_search_with_level_pills",
              "cron_logs_sparkline_above_log_event_rate_per_minute",
              "cron_logs_follow_pause_toggle_and_permalink_to_line",
              "observability_per_source_7d_sparkline_column",
              "observability_TimeRangeSelector_7d_30d",
              "observability_zero_unknown_bands",
              "Lighthouse_a11y_at_least_95_on_all_four_pages"
            ],
            "live_check": "live_check_44.7.md captures TraceTree + cron facet-search + observability sparkline screenshots + Playwright traces."
          },
          "retry_count": 0,
          "max_retries": 3
        },
        {
          "id": "44.8",
          "name": "Settings + Login (DRY merge + new tabs Risk/LLMRoute/Cycle/AuditLog/FeatureFlags; Login form wrap)",
          "status": "pending",
          "harness_required": true,
          "priority": "P1",
          "depends_on_step": "44.2",
          "audit_basis": "Per master_design Section 3.16: /settings (1410 LoC) duplicates /paper-trading Manage tab. Section 3.17: /login has no <form>, Enter does nothing.",
          "verification": {
            "command": "pytest frontend/tests/test_phase_44_8_settings_login.py -v && test -f handoff/current/live_check_44.8.md",
            "success_criteria": [
              "paper_trading_Manage_tab_removed_settings_live_solely_in_settings_page",
              "settings_has_7_tabs_models_cost_risk_llmroute_cycle_auditlog_featureflags",
              "settings_search_box_filters_all_rows_scroll_to_match",
              "settings_ModelPicker_has_role_listbox_arrow_key_nav",
              "settings_destructive_actions_require_typed_confirmation",
              "settings_audit_log_row_appended_for_every_operator_action",
              "settings_LLM_route_test_call_button_returns_banner_string_under_5s",
              "login_buttons_wrapped_in_form_enter_triggers_google_signin",
              "login_passkey_helper_text_and_tooltip",
              "Lighthouse_a11y_at_least_95_on_both_pages"
            ],
            "live_check": "live_check_44.8.md captures the DRY merge + audit-log + login Playwright trace."
          },
          "retry_count": 0,
          "max_retries": 3
        },
        {
          "id": "44.9",
          "name": "Mobile + a11y + states-library global polish pass",
          "status": "pending",
          "harness_required": true,
          "priority": "P2",
          "depends_on_step": "44.2",
          "audit_basis": "Per master_design Section 6 UX-2 + UX-3 + UX-8: WCAG 2.2 AA across all 15 routes; 375px responsive; states library replaces all inline spinners/empties/errors.",
          "verification": {
            "command": "python scripts/qa/responsive_check.py 375 768 1280 1920 && python scripts/qa/axe_core_runner.py && test -f handoff/current/live_check_44.9.md",
            "success_criteria": [
              "lighthouse_a11y_at_least_95_on_every_15_routes",
              "axe_core_zero_violations_on_every_15_routes",
              "responsive_check_passes_at_375_768_1280_1920_on_every_route",
              "no_inline_animate_pulse_or_loading_text_grep_returns_empty",
              "no_inline_rose_banner_grep_returns_empty",
              "keyboard_shortcuts_overlay_lists_all_shortcuts",
              "all_KPI_tiles_have_aria_label",
              "all_buttons_have_aria_label_or_text_content",
              "all_tab_bars_have_role_tablist"
            ],
            "live_check": "live_check_44.9.md captures Lighthouse + axe-core + responsive results across all 15 routes."
          },
          "retry_count": 0,
          "max_retries": 3
        },
        {
          "id": "44.10",
          "name": "SSE live-updates everywhere (cockpit + /sovereign + /observability + /cron + /backtest optimizer)",
          "status": "pending",
          "harness_required": true,
          "priority": "P2",
          "depends_on_step": "44.2",
          "audit_basis": "Per master_design Section 5 (Backend Coupling) + research_brief Section B.4: SSE is dominant transport 2026 for LLM streaming; replacing 30s polling needs backend stream endpoints.",
          "verification": {
            "command": "pytest frontend/tests/test_phase_44_10_sse.py -v && playwright test frontend/playwright/live-updates.spec.ts && test -f handoff/current/live_check_44.10.md",
            "success_criteria": [
              "backend_api_paper_trading_stream_endpoint_exists_and_emits_one_event_per_second",
              "frontend_useEventSource_hook_replaces_polling_on_cockpit",
              "NAV_ticker_updates_within_1s_of_backend_write_real_browser_test",
              "cycle_status_pill_flips_without_page_refresh",
              "network_panel_shows_one_EventSource_connection_not_N_polls",
              "graceful_degrade_to_5s_polling_when_EventSource_fails_3x",
              "observability_freshness_bands_update_without_F5"
            ],
            "live_check": "live_check_44.10.md captures the SSE Playwright trace + network panel screenshot showing single EventSource."
          },
          "retry_count": 0,
          "max_retries": 3
        }
      ]
    }
```

---

## 8 -- Execute-Prompt Skeleton

For the next session(s) that walks this phase-44 roadmap:

```
# Execute phase-44 frontend UX overhaul -- one step per goal-cycle

Walk through .claude/masterplan.json phase-44 steps 44.1 -> 44.10 in dependency order
(44.1 is the foundation; 44.2/.4/.5/.6/.7/.8/.9/.10 mostly parallelizable after 44.1).

For each step:

  1. Read its audit_basis + success_criteria + live_check from masterplan.json.
  2. Re-read .claude/rules/frontend.md + .claude/rules/frontend-layout.md.
  3. Spawn researcher ONLY if the step adds NEW external dependencies; SKIP
     researcher for code-only steps applying patterns already in
     handoff/archive/phase-44.0/research_brief.md.
  4. Write handoff/current/contract.md with VERBATIM immutable criteria from
     the masterplan.json verification block + per-page success criteria from
     this master_design doc.
  5. GENERATE frontend code per success_criteria. Per-step rules:
     - ASCII-only loggers (no emojis anywhere)
     - Phosphor Icons only (no @phosphor-icons/react direct imports)
     - Recharts only for charts; Tremor as additive layer
     - Per `feedback_no_emojis`: zero emojis in UI/code/files
     - Mount new top-level features behind feature flags in featureFlags.ts
  6. Run the verification.command -- it MUST exit 0.
  7. Run Playwright tests at 4 breakpoints (375/768/1280/1920); screenshot
     each.
  8. Write the live_check file named in verification.live_check.
  9. Spawn qa subagent ONCE. The Q/A subagent MUST consume the
     screenshots, not just the unit-test output.
  10. Append the cycle block to handoff/harness_log.md FIRST.
  11. Flip the step status to `done` in .claude/masterplan.json LAST (the
      auto-commit hook will fire and push with prefix `phase-44.X:`).

UX-DoD gate: phase-44 closes only AFTER all 12 UX-DoD criteria in Section 6
of master_design pass. Owner must approve any step marked
OWNER-APPROVAL-REQUIRED before its destructive actions land.

Hard guardrails (verbatim from phase-44.0 contract):
- feedback_masterplan_status_flip_order: never write status `done` on a step
  in the initial insert -- start `pending`/`in-progress`, flip after log.
- feedback_log_last: harness_log.md append BEFORE status flip.
- feedback_qa_harness_compliance_first: every qa prompt starts with the
  5-item protocol audit before the technical content.
- feedback_no_emojis: zero emojis anywhere.
- feedback_research_gate: researcher spawn ONLY when adding new external
  patterns not already in this master_design doc.
- No new deps in `frontend/package.json` without explicit citation from
  research_brief Section B.

Per-step effort budget: simple <= 1 cycle, moderate <= 2 cycles, complex
<= 3 cycles. Anything over 3 cycles on a single step is a circuit-breaker
trip: `status: blocked` + STOP + escalate to operator.

If a step is OWNER-APPROVAL-REQUIRED (44.2 Manage-tab removal, 44.3
4-into-1 consolidation, 44.5 /backtest URL change, 44.8 destructive
actions, 44.10 backend coupling), record operator's typed approval in
handoff/current/operator_approval_44.X.md BEFORE landing the change.
```

---

## 9 -- Appendix: Coverage of all surfaces

Cross-check that every surface from research_brief Section A appears in
this master_design doc:

| Surface | Master_design Section | Phase step |
|---|---|---|
| Sidebar.tsx | 3.1 | 44.1 |
| Lib files (api/types/icons/useLiveNav/kpiMetrics/motion/format/hooks) | 3.2 | 44.1 |
| Page 1: / (Home) | 3.3 | 44.6 |
| Page 2: /signals | 3.4 | 44.6 |
| Page 3: /reports | 3.5 | 44.4 |
| Page 4: /performance | 3.6 | 44.4 |
| Page 5: /paper-trading | 3.7 | 44.2 |
| Page 6: /paper-trading/learnings | 3.8 | 44.5 |
| Page 7: /backtest | 3.9 | 44.5 |
| Page 8: /sovereign | 3.10 | 44.5 |
| Page 9: /sovereign/strategy/[id] | 3.11 | 44.5 |
| Page 10: /agents | 3.12 | 44.7 |
| Page 11: /agent-map | 3.13 | 44.7 |
| Page 12: /cron | 3.14 | 44.7 |
| Page 13: /observability | 3.15 | 44.7 |
| Page 14: /settings | 3.16 | 44.8 |
| Page 15: /login | 3.17 | 44.8 |
| Components audit (58 entries) | 3.18 | 44.1 (housekeeping) + per-step |
| Foundation: tokens / states-lib / hooks / Cmd-K / WCAG / Sidebar | 2.1-2.7 | 44.1 |
| Decision Trail consolidation | 3.18 + 44.3 | 44.3 |
| SSE live-updates | 2.3 + 5 | 44.10 |
| Mobile + a11y + states polish | 2.5 + 6 UX-2/3/8 | 44.9 |

**No silent drops.** Every surface from the brief is mapped to a phase step.

---

## End of master design

Total open findings consolidated: **33 issues + 14 missing components** across **15 pages + Sidebar + 8 lib files + 58 components**.
Total proposed roadmap steps: **11** (44.0 planning + 44.1 foundation + 44.2 cockpit + 44.3 decision-drawer + 44.4 reports + 44.5 trading-non-paper + 44.6 analyze + 44.7 system + 44.8 settings/login + 44.9 mobile/a11y/states + 44.10 SSE).
Today's UX-DoD pass rate: **0 of 12**.
Definition of top-notch: 12 UX-DoD criteria PASS + owner approval recorded for OWNER-APPROVAL-REQUIRED steps.

Closing the gap from good-enough to top-notch = **4-5 medium phases** (44.1 + 44.2 + 44.7 + 44.9, optionally 44.10) per Section H.5 of the research brief.
