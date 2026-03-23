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
- API client: `src/lib/api.ts` — Bearer token auth, 401 → redirect to `/login`
- Types: `src/lib/types.ts` — all TypeScript interfaces for backend responses
- Icons: `src/lib/icons.ts` — Phosphor icon aliases (never use emoji in UI)

## Conventions
- **Glass Box**: Every agent I/O must be visible. Debate shows bull/bear arguments. Bias flags surfaced prominently.
- **BentoCard pattern**: Cards use `bg-white dark:bg-zinc-900 rounded-2xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-6`
- **Loading states**: Use `Skeleton.tsx` components (SkeletonPulse, SkeletonCard, SkeletonGrid, PageSkeleton)
- **Color coding**: green=bullish, red=bearish, amber=neutral, gray=error/unavailable
- **Parallel fetches**: Use `Promise.all()` for independent API calls, never sequential `await` chains
- **Lightweight polling**: Poll status-only endpoints while tasks are running, full refresh on completion
- **Run-scoped data**: Pass `run_id` from status endpoints to data-fetching endpoints (e.g., optimizer experiments) to show only current run's results
- **Step indicators**: For long-running background tasks, show pulsing step indicator (sky dot + step name + detail) below status/metric cards
- **Mutual exclusion**: Backtest and optimizer cannot run concurrently. Buttons are disabled with tooltip when the other process is active. Progress panel shows "(via Optimizer)" when `engine_source === "optimizer"`. Both processes use the same Walk-Forward Progress panel.
- **Backtest persistence**: Previous backtest results auto-load on page mount. Run history selector (dropdown above tabs) lets users switch between saved runs. `OptimizerInsights.tsx` provides 5-section visualization (Data Scope Gantt, Slice Plots, Param Importance, Feature Stability, Decision Log) in a dedicated Insights tab.
- **Trade visibility**: Results tab includes Trade Statistics (3-col bento: Performance/Extremes/Cost Impact) and Trade List (sortable paginated table, 25/page, green/red rows). Types: `BacktestRoundTrip` (12 fields), `TradeStatistics` (23 fields) in `types.ts`. State: `tradePage`, `tradeSort` for client-side pagination/sorting.

## Auth Flow
- `src/middleware.ts` protects all routes except `/login` and `/api/auth/*`
- `auth.config.ts` = Edge-compatible (used by middleware), `auth.ts` = full config (PrismaAdapter + WebAuthn)
- Session refetched every 15 minutes via `AuthProvider.tsx`

## UX Reference
See `UX-AGENTS.md` for full component specs, design tokens, and icon conventions.
