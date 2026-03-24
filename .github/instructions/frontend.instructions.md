---
applyTo: "frontend/**"
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

## Conventions
- **Glass Box**: Every agent I/O must be visible. Debate shows bull/bear arguments. Bias flags surfaced prominently.
- **Scrollbar styling**: All scrollable containers (sidebars, panels, lists) must use the `scrollbar-thin` class for a consistent custom scrollbar, as defined in `globals.css`. Never rely on browser default scrollbars.
- **BentoCard pattern**: Cards use `bg-white dark:bg-zinc-900 rounded-2xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-6`
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
See `frontend-layout.instructions.md` for the 6-tier page anatomy, metric grid patterns, tab conventions, collapsible sections, empty states, and research-backed information hierarchy principles. Applies to `frontend/src/app/**` and `frontend/src/components/**`.

## UX Reference
See `UX-AGENTS.md` for full component specs, design tokens, and icon conventions.
