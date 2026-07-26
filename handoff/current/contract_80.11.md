# Contract — phase-80.11

**Step id:** `80.11` (phase-80, **P1**, `harness_required: true`)
**Title:** *Self-inflicted load: session-fetch stampede* — 11 `/api/auth/session` requests per
20s page view.

## TIER (assigned before GENERATE)

| field | value |
|---|---|
| **Tier** | **T2** — Opus 5, effort `high` |
| Rationale | P1, not a money surface (the goal assigns T3 to `80.7`/`80.8`/`80.9` only). Not T1: multi-file, research-gated, criteria not mechanically checkable. |

## Research gate

`handoff/current/research_brief_80.11.md` — **`gate_passed: true`**, 10 sources read in full,
25 URLs, recency scan performed, 14 internal files inspected.

## SCOPE DECISION — ship the stampede fix, DEFER the consolidation

The gate surfaced three flags that make the full step materially riskier than its title:

1. **[P0 regression risk] Consolidation would create an unbounded poller.**
   `LivePortfolioProvider` has **no consecutive-failure counter**
   (`live-portfolio-context.tsx:122-166`; `Promise.allSettled` means it never throws), and the
   layout's five fetches are currently **one-shot** (`:186-213`, no interval). Folding them into
   the provider promotes them into an unbounded 60s loop — a **net-new violation of
   `.claude/rules/frontend.md`'s stop-after-5-failures rule, across five endpoints, caused by
   the step meant to reduce traffic.** It also lacks the `document.hidden` guard every sibling
   poller has.
2. **[donut] The centre-vs-slices disagreement has THREE axes, not one.** (a) root-vs-layout
   `/portfolio`; (b) cash read from **two different endpoints** — `status.portfolio.cash`
   (`useLiveNav.ts:32`) vs `portfolio.current_cash` (`positions/page.tsx:131`); (c) two
   market-value formulas — `positionMarketValueUsd` vs page-local `mvUsd` (`page.tsx:67-77`).
   **Shipping axis (a) alone leaves the operator-visible symptom in place.** (b) and (c) are
   latent only while the book is US-only, and `ticker-meta` already shows KR/EU tickers.
3. **[hazard the fix itself creates] The session probe has no timeout.** `api.ts:65` has none;
   the 30s `AbortController` at `:130` guards only the backend fetch. **Under single-flight a
   stalled probe would block *every* `apiFetch` instead of one.** Must be fixed in the same
   change.

**Therefore this cycle ships:** single-flight + the 401 epoch guard + the probe timeout — which
is what actually causes the measured 11 probes. **It does NOT ship the provider consolidation or
the donut axes**, which are queued as their own steps. Criterion 4 is satisfied *because* no
polling loop is touched; consolidating without a failure budget would have broken it.

## Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. `A single 20s page view of /paper-trading/positions issues AT MOST 2 requests to /api/auth/session (down from the measured 11) -- proven with a fresh browser_network_requests capture, not by reading code`
2. `Concurrent apiFetch calls share ONE in-flight session probe (single-flight), and the 401 cache-invalidation path at api.ts:138-139 still works -- test that a 401 still forces a fresh probe`
3. `The duplicate-poller census is MEASURED and listed endpoint by endpoint, and any consolidation preserves each consumer's refresh needs`
4. `Every polling loop touched still honours the .claude/rules/frontend.md rule: stop after 5 consecutive failures, never poll a dead backend forever`

**Verification command (immutable):**
```
cd frontend && grep -n 'inFlight\|sessionTokenCache' src/lib/api.ts
```

**live_check (immutable):** *handoff/current/live_check_80.11.md: before/after browser_network_requests captures for the same 20s page view with per-endpoint counts, plus the 401-revalidation test.*

## Plan

1. **Single-flight** in `getAuthToken` (`api.ts:58-86`): store the **promise** synchronously,
   clear in `.finally()`. Canonical pattern (Go `x/sync/singleflight`: *"only one execution is
   in-flight for a given key at a time … the duplicate caller waits for the original and
   receives the same results"*). All waiters share one settlement.
   **The rejected-promise-cache hazard does not apply here** — `getAuthToken` try/catches to
   `null` and never rejects. Keep the TTL write **inside the async body** (before the `finally`
   microtask); do not invert that ordering.
2. **401 epoch guard.** Clearing the in-flight ref is *necessary but not sufficient*. Failure
   sequence: TTL expires → probe `P` in flight → an earlier request's 401 nulls the cache
   (`:166`) → `P` resolves and writes the just-invalidated token back at `:82` with a fresh 60s
   ts → invalidation undone. Normally cut short by the `/login` redirect, but `:172`
   deliberately skips that when already on `/login`, so **there it is unbounded**.
   Fix: `sessionEpoch += 1` **first**, then null the cache, then null the in-flight ref, with an
   `if (sessionEpoch === myEpoch)` guard on the writes at `:67` and `:82`. Mirrors the repo's
   existing `cancelled` idiom (`useLivePrices.ts:53`).
3. **Probe timeout** — `AbortSignal.timeout` on the `:65` fetch.
4. **Census (criterion 3)** — delivered by §E of the brief: 9 fetch/poll sites with
   owner/endpoint/trigger/interval/duplicate-status, cross-checked line-by-line against the
   archived 20s log (11 session probes + 21 backend requests, all attributed).

## Corrections to this step's own text, from the gate — do not "fix" non-defects

- `positions/page.tsx` issues **zero** fetches of its own.
- The `gate`/`kill-switch`/`freshness`/`cycles-history` ×2 is **one `OpsStatusBar` instance
  running twice** (visibilitychange listener / dev double-mount), **not** a duplicate component.
  `CycleHealthStrip`/`KillSwitchPanel` are not mounted on this route.
- **StrictMode is ON** (measured from Next 15.5.12 source — `define-env.js:111-112`,
  `app-index.js:152`; `reactStrictMode` unset → `null` → `__NEXT_STRICT_MODE_APP = true`). It
  double-invokes effects **in dev only**, which is the leading explanation for the uniform 2×.
  Those are likely **not** real duplication.
- The layout comment at `:137-142` is **half-true**: accurate about `useLivePrices`/`useLiveNav`
  (genuinely single-instance now), but "Now ONE poll instance feeds both" reads as
  "duplication eliminated" while the layout still runs its own `/status` + `/portfolio` +
  `/snapshots` at `:188-194`. Scope any correction to the false half.

## Tests

Unit (vitest/jsdom): single-flight call count via the existing `vi.resetModules()` +
dynamic-import idiom (`api.test.ts:45-47`) — **mutation-resistant: revert the fix and the count
goes 1 → N**; the 401 poison sequence; the probe timeout.
Playwright live_check: the real 11 → ≤2 count, and non-regression of the `OpsStatusBar` ×2.

## Do-no-harm

Frontend-only. No `.env`, no flag flips. UI evidence from the isolated skip-auth `:3100` rig
with its own `distDir`; restore `tsconfig.json` + `next-env.d.ts` after.
**HARD STOP:** any change that leaves a polling loop without a failure budget.
