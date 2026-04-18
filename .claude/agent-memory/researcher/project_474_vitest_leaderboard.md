---
name: phase-4.7.4 vitest setup and leaderboard UI conventions
description: Vitest setup for Next.js 15 App Router, --filter semantics, React 19 deps, DSR/PBO thresholds, fake-timers polling pattern. Researched for AutoresearchLeaderboard component.
type: project
---

Researched April 2026 for phase-4.7.4 AutoresearchLeaderboard component + vitest test.

**Why:** Masterplan verification command uses `npm run test -- --filter=AutoresearchLeaderboard`. Need to confirm this works with vitest, and understand DSR/PBO display conventions for the leaderboard UI.

**Key facts:**

- Vitest is the idiomatic choice for Next.js 15 App Router (official docs updated 2026-04-15). Jest requires babel config; vitest uses @vitejs/plugin-react natively. Next.js has an official `--example with-vitest` template.
- `--filter` is NOT a named vitest CLI flag. The positional argument `vitest <pattern>` filters FILES by path inclusion (no regex). So `npm run test -- --filter=AutoresearchLeaderboard` will NOT work as written. The positional form `npm run test -- AutoresearchLeaderboard` (no `--filter=`) DOES filter files by name substring. Naming the file `AutoresearchLeaderboard.test.tsx` makes `vitest AutoresearchLeaderboard` match. The `test` script in package.json must be `"test": "vitest run"` (or `"vitest"`).
- Minimum deps (official Next.js docs): `vitest @vitejs/plugin-react jsdom @testing-library/react @testing-library/dom vite-tsconfig-paths` as devDependencies. Add `@testing-library/jest-dom` for `.toBeInTheDocument()`.
- React 19 + vitest known issue: `act()` warnings appear with some RTL versions. Use `@testing-library/react@^16` which ships React 19 support. No `@types/react-dom` needed separately with React 19.
- DSR threshold (Bailey & Lopez de Prado): DSR >= 0.95 = strong evidence against noise. DSR 0.95+ → green; 0.8-0.95 → amber; <0.8 → red. The 95% confidence level is the standard pass bar (confirmed by Balaena Quant and SSRN paper).
- PBO threshold: PBO > 0.5 is the conventional veto threshold (strategy ranked highest in-sample is more likely to underperform median out-of-sample than not). Colour convention: PBO > 0.5 → red badge; 0.3-0.5 → amber; <0.3 → green. No single authoritative source mandates this cutoff numerically but PBO ~0.47 is described as "not reliably predictive" in literature.
- Polling pattern: `setInterval` at 5000ms (5s) in `useEffect` with `clearInterval` on cleanup. Convention is 5s for live leaderboard. Vitest fake timers: `vi.useFakeTimers()` in `beforeEach`, `vi.advanceTimersByTime(5000)` to trigger, `vi.useRealTimers()` in `afterEach`. Test asserts fetch was called once after 5s advance. Test also asserts interval <= 10s by checking `setInterval` was called with second arg <= 10000.

**How to apply:** Wire `"test": "vitest run"` in package.json (not watch mode, so CI passes). Name the file `AutoresearchLeaderboard.test.tsx`. Use `npm run test -- AutoresearchLeaderboard` to match masterplan semantics (drop `--filter=`, use positional). Add `vitest.config.mts` with jsdom env.

Sources:
- https://nextjs.org/docs/app/guides/testing/vitest (official, updated 2026-04-15)
- https://vitest.dev/guide/cli
- https://vitest.dev/guide/filtering
- https://vitest.dev/guide/mocking/timers
- https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464
- https://medium.com/balaena-quant-insights/the-probability-of-backtest-overfitting-pbo-9ba0ac7fb456
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- https://github.com/esvhd/pypbo
- https://cran.r-project.org/web/packages/pbo/vignettes/pbo.html
