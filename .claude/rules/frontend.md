---
paths:
  - "frontend/**"
---

# Frontend — Next.js 15 + React 19 Conventions

## Stack
- Next.js 15 App Router, React 19, TypeScript 5.6 strict
- Tailwind CSS + Geist font, Phosphor Icons (not emoji), Recharts for charts
- NextAuth.js v5 (Google SSO + Passkey/WebAuthn), Prisma (SQLite auth DB)

## Architecture
- Pages in `src/app/` (10 routes), components in `src/components/`, shared lib in `src/lib/`
- API client: `src/lib/api.ts` — Bearer token auth, 401 → redirect to `/login`, 30s AbortController timeout on all requests
- Types: `src/lib/types.ts` — all TypeScript interfaces for backend responses
- Icons: `src/lib/icons.ts` — Phosphor icon aliases (never use emoji in UI)

## Dark-mode + readability (cycle-69 lessons, MANDATORY for any visual work)

This project is dark-mode-only. The `tailwind.config.js` sets `darkMode: "selector"` and `<html>` carries the `dark` class via `app/layout.tsx`. Three rules are non-negotiable for any component that ships pixels:

1. **Use the project's navy + slate palette, NOT Tailwind's default zinc palette.** Cards use `bg-navy-800/70` + `border-navy-700`. Body text is `text-slate-100` (bright) / `text-slate-200` (default cell) / `text-slate-300` (secondary) / `text-slate-400` (tertiary) / `text-slate-500` (dim). Never use `text-zinc-200`, `bg-zinc-900`, `border-zinc-800` etc. -- they're a slightly different palette and will read as "off" in the cockpit. (Cycle 66 SectorBarList + cycle 67 DataTable + cycle 68 root-cause fix all chased this gap.)

2. **NEVER write light-mode `bg-white`/`text-zinc-700` fallbacks `class="bg-white dark:bg-navy-800/70"` as a base default.** Tailwind's CSS resolution order is deterministic by stylesheet position, NOT className-string order, so a consumer-passed `bg-navy-800/70` can't reliably defeat the `bg-white` base. Either (a) write only the dark token (project is dark-only) or (b) accept the consumer can't override.

3. **Tailwind JIT-safe class strings.** Tailwind v3 JIT scans source for LITERAL class strings; it does NOT compile classes built via template-string concatenation like `` `bg-${color}-500` ``. If you need a runtime-chosen color, use a static lookup map that lists every possible literal class:
   ```ts
   const DOT_BG: Record<string, string> = {
     blue: "bg-blue-500",
     amber: "bg-amber-500",
     // ... every color you might pass
   };
   const dotClass = DOT_BG[color] ?? "bg-slate-500";
   ```
   See `frontend/src/components/PortfolioAllocationDonut.tsx::DOT_BG_CLASS` for the canonical pattern.

4. **Third-party visualization libs need their node_modules path in `tailwind.config.js::content`.** Tremor in particular ships class strings INSIDE its package; without `"./node_modules/@tremor/**/*.{js,ts,jsx,tsx}"` in `content`, chart slices render uncolored. This bit cycle 69. If you add a new visualization lib (charts, gauges, sparklines), add its node_modules path to `content` in the same commit.

5. **Visual verification is mandatory for any chart or color-coded UI.** Unit tests + greps cannot see what the operator sees. After shipping a chart or color-coded view, you MUST either (a) open the dev server in a browser and probe, or (b) explicitly mark the work as "visual verification pending operator review". Q/A returning PASS on unit tests + grep is necessary but not sufficient for visual correctness.

6. **Contrast targets (WCAG 2.2 AAA on dark navy):**
   - Primary text on `bg-navy-800/70` -- use `text-slate-100` (>= 13:1) or `text-slate-200` (>= 12:1).
   - Secondary text -- use `text-slate-300` (>= 10:1). Avoid `text-slate-400` for primary readable text.
   - Table headers -- `uppercase tracking-wider text-xs text-slate-200` (12.6:1).
   - Dim/tertiary -- `text-slate-400` (>= 7:1) is acceptable for chrome (e.g. fee, days held) but NOT for risk-relevant numbers (stop loss, P&L).
   - Hover row -- `hover:bg-navy-700/40` (subtle elevation, preserves text contrast). Never `hover:bg-zinc-50` / `hover:bg-white` (washes out text).

## Conventions
- **Glass Box**: Every agent I/O must be visible. Debate shows bull/bear arguments. Bias flags surfaced prominently.
- **Scrollbar styling**: All scrollable containers (sidebars, panels, lists) must use the `scrollbar-thin` class for a consistent custom scrollbar, as defined in `globals.css`. Never rely on browser default scrollbars.
- **BentoCard pattern**: Cards use `bg-white dark:bg-zinc-900 rounded-2xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-6`
- **No equal-height rows mixing short and tall widgets**: When a row's widgets have very different natural heights (e.g. a 400px checklist next to a 100px status banner), do NOT use a plain `grid-cols-N` row — CSS grid's default `items-stretch` forces short cards to the tallest card's height, leaving dead whitespace (anti-pattern: Every Layout, Tailwind Bento, Grafana 12, QuantConnect Cloud, Robinhood Legend, FreqUI). Two allowed fixes, in order: (1) **bento/sidebar** — one tall widget on one side, a `flex flex-col gap-3` stack of short widgets on the other (pull in additional content like the KPI hero so both columns meaningfully fill); (2) **`items-start`** — collapse short cards to their natural height and accept visible asymmetry when the short cards genuinely have nothing more to show. See `frontend-layout.md` §4.5 for the code template. Never use `grid-template-rows: masonry` (not production-safe in 2026) or add `h-full`/`flex-1` to short widgets just to fill dead space.
- **Operator status = ONE dense bar, not stacked cards**: When a dashboard page has 3+ pieces of live operator status (go-live gate, kill switch, cycle health, scheduler, etc.), render them as a **single horizontal status bar** with labeled segments, ~48-60px tall, with inline action buttons — NOT as independent card widgets. This is the Stripe / Linear / Vercel / Grafana 12 / QuantConnect pattern (see `frontend-layout.md` §4.5 and `frontend/src/components/OpsStatusBar.tsx` for the canonical implementation). Full widget details belong behind progressive disclosure (click-to-open drawer or a dedicated "System" sub-tab), never at full card size on the main view. The KPI hero renders as a full-width 6-column row below the status bar. Do NOT use bento / grid-cols-N to house status widgets — bento is for "one tall chart + secondary cards", which is a different problem (Few 2006 single-screen density; Tufte data-ink ratio).
- **Loading states**: Use `Skeleton.tsx` components (SkeletonPulse, SkeletonCard, SkeletonGrid, PageSkeleton)
- **Error states**: Never use `.catch(() => null)` on ALL calls in a group. If all calls in a `Promise.all` fail, surface an error banner (rose-900 border, rose-950/50 bg) with retry button. Individual `.catch(() => null)` is OK for optional/graceful-degradation calls only. Pattern: track whether ALL primary calls returned null / rejected, then `setError(...)`.
- **Polling failure limits**: Polling loops (setInterval) must count consecutive failures and stop after 5 with an error message. Never poll forever on a dead backend.
- **Color coding**: green=bullish, red=bearish, amber=neutral, gray=error/unavailable
- **Parallel fetches**: Use `Promise.all()` for independent API calls, never sequential `await` chains
- **Lightweight polling**: Poll status-only endpoints while tasks are running, full refresh on completion
- **Run-scoped data**: While optimizer is running, pass `run_id` from status to `getOptimizerExperiments()` for live updates. When idle, use `run_index` (0=latest) via `optRunIndex` state + run selector dropdown. `getOptimizerRuns()` returns per-baseline summaries for the dropdown.
- **Step indicators**: For long-running background tasks, show pulsing step indicator (sky dot + step name + detail) below status/metric cards
- **Mutual exclusion**: Backtest and optimizer cannot run concurrently. Buttons are disabled with tooltip when the other process is active. Progress panel shows "(via Optimizer)" when `engine_source === "optimizer"`. Both processes use the same Walk-Forward Progress panel.
- **"One Truth" progress**: The Walk-Forward banner is the single command center for ALL running state. When optimizer runs, the banner `<summary>` shows inline metric pills (Iter, Sharpe, DSR, kept, discarded) + step subtitle + single Stop button. The Optimizer tab hides its Stop button + Metric cards + step indicator while running (shows "see progress above" notice). Only chart + experiment table remain in tab. Never add duplicate stop/progress controls.
- **Backtest persistence**: Previous backtest results auto-load on page mount. Run history selector (dropdown above tabs) lets users switch between saved runs. `OptimizerInsights.tsx` provides 5-section visualization (Data Scope Gantt, Slice Plots, Param Importance, Feature Stability, Decision Log) in a dedicated Insights tab.
- **Trade visibility**: Results tab includes Trade Statistics (3-col bento: Performance/Extremes/Cost Impact) and Trade List (sortable paginated table, 25/page, green/red rows). Types: `BacktestRoundTrip` (12 fields), `TradeStatistics` (23 fields) in `types.ts`. State: `tradePage`, `tradeSort` for client-side pagination/sorting.

## Auth Flow
- `src/middleware.ts` protects all routes except `/login` and `/api/auth/*`
- `auth.config.ts` = Edge-compatible (used by middleware), `auth.ts` = full config (PrismaAdapter + WebAuthn)
- Session refetched every 15 minutes via `AuthProvider.tsx`

## Layout Blueprint
See `frontend-layout.md` (this directory) for the 6-tier page anatomy, metric grid patterns, tab conventions, collapsible sections, empty states, and research-backed information hierarchy principles. Applies to `frontend/src/app/**` and `frontend/src/components/**`.

## UX Reference
See `UX-AGENTS.md` for full component specs, design tokens, and icon conventions.
