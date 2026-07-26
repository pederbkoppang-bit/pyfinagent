---
name: project-session-stampede-donut-80-11
description: phase-80.11 measured facts — getAuthToken stampede needs epoch-not-just-null on 401; LivePortfolioProvider has NO failure counter; donut/NAV mismatch has THREE axes not one
metadata:
  type: project
---

Measured 2026-07-25 while researching phase-80.11 (`/paper-trading/positions`
fires 11x `/api/auth/session` per 20s view).

1. **`getAuthToken` never rejects.** `api.ts:64-87` try/catches to `null`, so the
   dominant literature hazard (Mellman: "important to evict rejected promises from
   the memoization cache") does NOT apply — a plain `.finally()` clear suffices.
   Don't over-engineer the single-flight.
2. **Clearing the in-flight ref on 401 is NOT sufficient.** A probe that started
   before the 401 still writes `sessionTokenCache` at `:82` after it resolves,
   re-caching the just-invalidated token for 60s. Needs an epoch/generation counter
   bumped FIRST. Unbounded on `/login` because `:172` deliberately skips the
   redirect there (phase-75.12 frontend-02 guard), so nothing tears down the JS
   context.
3. **`LivePortfolioProvider` (`live-portfolio-context.tsx:122-166`) has NO
   consecutive-failure counter** and `Promise.allSettled` guarantees it never
   throws — it already violates the frontend.md 5-failure rule. Any step that
   moves the paper-trading layout's ONE-SHOT fetches into it promotes 5 endpoints
   into an unbounded 60s loop. Always pair consolidation with a failure budget.
4. **Donut centre != sum of slices has THREE independent axes**, only one of which
   is provider duplication: (a) root vs layout `/portfolio` responses; (b) cash from
   `/status` `portfolio.cash` (`useLiveNav.ts:32`) vs `/portfolio` `current_cash`
   (`positions/page.tsx:131`) — DIFFERENT ENDPOINTS; (c) `positionMarketValueUsd`
   vs the page-local `mvUsd` (`page.tsx:67-77`). Latent while the book is US-only.
5. **`paper-trading/layout.tsx:137-142` comment is half-true** — phase-72
   consolidated `useLivePrices`/`useLiveNav` only; the layout still runs its own
   `/status` + `/portfolio` + `/snapshots`.

**Why:** these cost a full research session to derive and are invisible from a
casual read; #3 and #4 are the ones that turn a "small fix" into a regression.

**How to apply:** cite these when scoping any paper-trading fetch consolidation.
Related: [[project_frontend_test_env_and_ui_specs_80_5]],
[[feedback_second_next_dev_breaks_operator_3000]].
