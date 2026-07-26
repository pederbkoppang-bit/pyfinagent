# Research Brief — phase-80.11 (session-probe stampede + provider consolidation)

**Tier:** T2 (Opus 5, effort high). NOT audit-class.
**Started:** 2026-07-25 · **Status:** IN PROGRESS (write-first; appended incrementally)

## Questions
- A. Single-flight / promise-dedup in TypeScript
- B. Cache-invalidation interaction with the 401 path
- C. React 19 / Next 15 duplicate-provider consolidation
- D. Polling-loop discipline (5-consecutive-failure budget) under consolidation
- E. INTERNAL census of every fetch/poll site on `/paper-trading/positions`

---

## E. INTERNAL CODE CENSUS (criterion 3) — WRITTEN FIRST

### E.0 Component tree actually mounted on `/paper-trading/positions`

```
app/layout.tsx (root)
  └ AuthProvider           -> SessionProvider refetchInterval={15*60}   (components/AuthProvider.tsx:8)
     └ LivePortfolioProvider (lib/live-portfolio-context.tsx:101)       <-- ROOT poller #1
        ├ CommandPalette
        └ app/paper-trading/layout.tsx (PaperTradingLayout)             <-- LAYOUT poller #2
           ├ Sidebar                       (components/Sidebar.tsx:292) <-- poller #3
           ├ OpsStatusBar                  (components/OpsStatusBar.tsx:112) <-- poller #4
           ├ SummaryHero                   (no fetch)
           └ PaperTradingDataContext.Provider
              └ app/paper-trading/positions/page.tsx                    <-- NO fetch of its own
```

`CycleHealthStrip` and `KillSwitchPanel` are **NOT** mounted on this route (grep: no
import in `paper-trading/layout.tsx` or `positions/page.tsx`). `OpsStatusBar` has a
second mount site at `app/page.tsx:360` — that is the **Home** route, not this one.

### E.1 Per-site census

| # | Owner (file:line) | Endpoint(s) | Trigger | Interval | Duplicates? |
|---|---|---|---|---|---|
| 1 | `lib/live-portfolio-context.tsx:164` (`LivePortfolioProvider`) | `/paper-trading/status`, `/paper-trading/portfolio`, `/paper-trading/snapshots` | mount + interval | **60 s** | **YES — all three collide with #2** |
| 2 | `app/paper-trading/layout.tsx:186-213` (`refresh`) | `/paper-trading/status`, `/portfolio`, `/trades`, `/snapshots`, `/performance` | mount only (`useEffect [refresh]`, `refresh` is `useCallback([],)`) + manual Retry / Start / Stop / Run-Now | **none** (one-shot) | **YES — status/portfolio/snapshots duplicate #1** |
| 3 | `components/Sidebar.tsx:292-303` | `/api/health` | mount + interval | 30 s | no |
| 4 | `components/OpsStatusBar.tsx:112-128` | `/paper-trading/gate`, `/kill-switch`, `/freshness`, `/cycles/history?limit=1` | mount + interval + `visibilitychange` | 60 s | no (single owner on this route) |
| 5 | `lib/useLivePrices.ts:73` — instantiated ONLY by `LivePortfolioProvider:178` | `/paper-trading/live-prices` | mount + interval + `visibilitychange` | 60 s | no |
| 6 | `lib/useTickerMeta.ts:22` via `LivePortfolioProvider:196` | `/paper-trading/ticker-meta?tickers=<positions>` | on ticker-set change | n/a | **YES (partial)** — subset of #7 |
| 7 | `lib/useTickerMeta.ts:22` via `paper-trading/layout.tsx:154` | `/paper-trading/ticker-meta?tickers=<positions ∪ trades>` | on ticker-set change | n/a | **YES (partial)** — superset of #6 |
| 8 | `app/paper-trading/layout.tsx:263` (`handleRunNow`) | `/paper-trading/status` | only after "Run Now" click | 10 s, 300 s ceiling | n/a (event-scoped) |
| 9 | `lib/api.ts:58-88` `getAuthToken()` — fires inside **every** `apiFetch` | `/api/auth/session` (Next server, port 3100/3000) | every `apiFetch` on a cold/expired 60 s cache | n/a | **YES — the stampede** |

`positions/page.tsx` itself issues **zero** fetches. It is a pure consumer: it reads
`positions/trades/perf/portfolio/tickerMeta/livePrices/openRationale/activeMarket` from
`usePaperTradingData()` (`:33-42`) and `lp` from `useLivePortfolio()` (`:45`).

### E.2 Cross-check against the measured 20 s log

Raw: `handoff/current/captures_ui_audit_2026-07-25/audit-net-positions-20s.txt`
(requests #11-41; #1-10 were static assets, suppressed by the tool).

| Endpoint | Count | Attributed to |
|---|---|---|
| `:3100/api/auth/session` | **11** | site #9 (10 in the mount-time burst = reqs 11-20, +1 at req 30) |
| `/api/health` | 1 | #3 |
| `/paper-trading/status` | 2 | #2 (req 22) + #1 (req 27) |
| `/paper-trading/portfolio` | 2 | #2 (req 23) + #1 (req 28) |
| `/paper-trading/snapshots?limit=365` | 2 | #2 (req 25) + #1 (req 29) |
| `/paper-trading/trades?limit=100` | 1 | #2 |
| `/paper-trading/performance` | 1 | #2 |
| `/paper-trading/gate` \| `/kill-switch` \| `/freshness` \| `/cycles/history?limit=1` | 2 each | #4, **one instance, two rounds** |
| `/paper-trading/ticker-meta` | 2 | #7 (21 tickers, req 35) + #6 (`AMD,PANW`, req 41) |
| `/paper-trading/live-prices?tickers=PANW,AMD` | 1 | #5 |

Total backend requests = 21; session probes = 11. Ratio confirms the step's framing:
**every backend endpoint is 1-2, `/api/auth/session` is 11.**

### E.3 Which duplicates are REAL vs apparent

**REAL (structural, fix-worthy):**

1. **`status` + `portfolio` + `snapshots` fetched twice** — `LivePortfolioProvider`
   (`:126-130`) and `PaperTradingLayout.refresh` (`:188-194`) independently call the
   same three fetchers. This is the direct cause of the donut-vs-hero disagreement:
   `positions/page.tsx` derives the donut slices from `usePaperTradingData().positions`
   (layout copy, `:33-42`) but the centre label from `useLivePortfolio()` (root copy,
   `:45`), and `useLiveNav` inside the provider runs on the ROOT `positions`/`status`.
   Two `/portfolio` responses landing milliseconds apart with a mid-flight position
   change ⇒ slices sum ≠ centre.
2. **`ticker-meta` fetched twice** with overlapping ticker sets (#6 ⊂ #7 whenever the
   position set is a subset of positions ∪ trades — true in the capture: `AMD,PANW`
   ⊂ the 21-symbol set). Two requests where one superset request suffices.
3. **11 × `/api/auth/session`** — site #9. `getAuthToken` writes
   `sessionTokenCache` only at `:82`, i.e. AFTER `await fetch("/api/auth/session")`
   resolves (`:65`). Between the first caller's `await` and its resolution the cache is
   still `null`, so every concurrent `apiFetch` re-enters the miss branch. With the
   mount-time burst = #1 (3 calls) + #2 (5 calls) + #3 (1 call) + #4 (4 calls) all
   starting in the same tick, 10 probes is exactly what the TTL-only design produces.

**APPARENT ONLY (do NOT "fix"):**

4. **`gate`/`kill-switch`/`freshness`/`cycles-history` ×2** is *one* `OpsStatusBar`
   instance running its `refresh` twice, not two components. `refresh` is
   `useCallback(..., [])` (stable) and the effect deps are `[refresh]`
   (`OpsStatusBar.tsx:113,128`), so it cannot re-run on re-render. The second round
   comes from the `visibilitychange` listener (`:120-124`) or React's dev-mode
   double-invoke — **not** from provider duplication, and consolidating providers will
   not change it. Reported for completeness; out of scope.
5. **`useLivePrices` / `useLiveNav` are already single-instance** — the phase-72
   consolidation is genuinely done for those two. `useLivePrices` is instantiated at
   exactly one place (`live-portfolio-context.tsx:178`); the layout consumes it via
   `useLivePortfolio()` at `:143`.

### E.4 The comment at `paper-trading/layout.tsx:137-142` — VERDICT: **partially false**

Verbatim (`:137-142`):

> `// phase-72: consume live values from the root LivePortfolioProvider`
> `// instead of mounting another useLivePrices + useLiveNav pair here.`
> ... `// Now ONE poll instance feeds both.`

- **TRUE** for what it literally claims: `useLivePrices` + `useLiveNav` are no longer
  mounted in the layout (verified: `layout.tsx` imports neither; only
  `useLivePortfolio` at `:59`).
- **MISLEADING** in effect: the sentence "Now ONE poll instance feeds both" reads as
  "the duplication is eliminated", but the layout **still keeps its own
  `getPaperPortfolio()` / `getPaperTradingStatus()` / `getPaperSnapshots()` calls**
  (`:188-194`) and its own `positions`/`portfolio`/`status`/`snapshots` state
  (`:124-128`), which it then publishes through `PaperTradingDataContext` (`:310-327`).
  So the *upstream* duplication the comment was written to describe was only half
  removed: the price poll was consolidated, the **portfolio poll was not**.

The step's claim that the duplication was NOT eliminated is **correct**; the comment's
claim is correct only about `useLivePrices`/`useLiveNav`.

### E.5 The consolidation is small — CONFIRMED

`paper-trading/layout.tsx:59` already imports `useLivePortfolio` and `:143-146` already
destructures from it. `LivePortfolioValue` (`live-portfolio-context.tsx:53-81`) already
exposes **`status`, `portfolio`, `positions`, `snapshots`, `livePrices`, `tickerMeta`,
`loading`, `error`, `refresh`** — i.e. every field the layout re-fetches except
`trades` and `perf`. The minimal change is therefore:

- delete `getPaperTradingStatus` / `getPaperPortfolio` / `getPaperSnapshots` from the
  layout's `Promise.all` (`:188-194`) and read `lp.status / lp.portfolio / lp.positions /
  lp.snapshots` instead;
- keep the layout's `refresh` for `trades` + `performance` only, and have it also call
  `lp.refresh()` so Start/Stop/Run-Now still force a full round-trip;
- drop the layout's `useTickerMeta` (`:154`) **or** move the union set into the provider
  (see §C recommendation) — a single superset call.

Two caveats to flag for the executor:
- `LivePortfolioProvider.refresh` calls `getPaperPortfolio()` **without** `.catch(() => null)`
  (`:128`), relying on `Promise.allSettled`; the layout's version at `:190` uses
  `.catch(() => null)` inside `Promise.all`. Error semantics differ — the layout sets a
  single top-level `error` when the WHOLE `Promise.all` rejects (`:204-206`); the provider
  only sets `error` when **all three** settle rejected (`:138-148`). Moving the layout onto
  `lp.error` changes when the rose banner appears; the executor must preserve the
  "surface an error when the primary calls all fail" rule from `.claude/rules/frontend.md`.
- `LivePortfolioProvider` is gated on `pathname !== "/login"` (`:110-111,124,156`). The
  layout has no such gate. Consolidating inherits the gate — which is correct/desirable
  but is a behavioural delta worth naming in the contract.

### E.6 The donut/NAV disagreement has THREE axes, not one

`positions/page.tsx` renders `PortfolioAllocationDonut` with:
- `slices={allocationSlices}` (`:124-134`) = Σ `mvUsd(pos)` by sector over `visiblePositions`
  (LAYOUT context) **+ `portfolio?.current_cash`** (LAYOUT `/portfolio` response, `:131`)
- `totalNav={isAllMarkets ? (lp.liveNav ?? portfolio?.total_nav ?? null) : filteredNavUsd}` (`:170`)

`lp.liveNav` comes from `useLiveNav(status, positions, livePrices)`
(`live-portfolio-context.tsx:184`) which computes
`status?.portfolio.cash + Σ positionMarketValueUsd(pos, livePrices[t]?.price)`
(`useLiveNav.ts:32,38-43`).

So centre vs Σslices differ on **three independent axes**:

| Axis | Centre (`lp.liveNav`) | Slices (`allocationSlices`) |
|---|---|---|
| 1. positions array | ROOT provider's `/portfolio` (`live-portfolio-context.tsx:134`) | LAYOUT's `/portfolio` (`layout.tsx:198`) |
| 2. cash source | `status.portfolio.cash` — the **`/status`** endpoint (`useLiveNav.ts:32`) | `portfolio.current_cash` — the **`/portfolio`** endpoint (`page.tsx:131`) |
| 3. market-value formula | `positionMarketValueUsd(pos, livePrice)` (FX-safe helper, `lib/format.ts`) | `page.tsx:67-77 mvUsd` — US: `livePrice × qty`; non-US: `pos.market_value` |

**Consolidating the providers fixes axis 1 ONLY.** Axes 2 and 3 are independent
defects and will keep the centre ≠ Σ slices even after a perfect consolidation.
The executor must be told this explicitly or the step will ship a "fix" that does
not close the visible symptom. (Axis 3 only bites when a non-US position is held —
the capture's book is `AMD, PANW`, both US, so it is currently latent; the KR/EU
tickers in the `ticker-meta` request show the book does go multi-market.)

Note the single-market branch (`:170`, `activeMarket !== "ALL"`) already sidesteps
this: `totalNav={filteredNavUsd}` is derived from the *same* `mvUsd` as the slices,
and cash is excluded from both (`:132`). Only the `"ALL"` view is broken.

---

## EXTERNAL RESEARCH

### Search-query variants run (3-variant discipline)

| Variant | Query |
|---|---|
| year-less canonical | `single flight promise deduplication JavaScript in-flight promise cache stampede` |
| year-less canonical | `promise memoization request coalescing pattern clearing finally race condition` |
| current-year (2026) | `React 19 Next.js 15 App Router share polled state between layout and page single context provider avoid double fetch 2026` |
| current-year (2026) | `"single flight" OR "request deduplication" frontend fetch client 2026 best practice TypeScript` |
| last-2-year (2025) | `next-auth useSession SessionProvider excessive /api/auth/session requests 2025` |

### Read in full (10; gate floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise/finally | 2026-07-25 | official docs (MDN) | WebFetch, full | "A `finally()` call is usually transparent and reflects the eventual state of the original promise." / "The `onFinally` callback does not receive any argument." / "A `throw` (or returning a rejected promise) in the `finally` callback still rejects the returned promise." |
| 2 | https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise/then | 2026-07-25 | official docs (MDN) | WebFetch, full | "All handlers attached to the same promise object are always called in the order they were added." / "the two promises returned by each call of `then()` start separate chains and do not wait for each other's settlement." |
| 3 | https://pkg.go.dev/golang.org/x/sync/singleflight | 2026-07-25 | official docs (canonical prior art) | WebFetch, full | "Do executes and returns the results of the given function, making sure that only one execution is in-flight for a given key at a time. If a duplicate comes in, the duplicate caller waits for the original to complete and receives the same results." + `Forget(key)` = explicit invalidation primitive |
| 4 | https://www.jonmellman.com/posts/promise-memoization/ | 2026-07-25 | authoritative blog | WebFetch, full | **[ADVERSARIAL to the naive fix]** "If our memoization implementation has cached a rejected promise, then all future calls will reject with this same failed promise!" → "it's very important to evict rejected promises from the memoization cache." |
| 5 | https://www.jonmellman.com/posts/singleton-promises/ | 2026-07-25 | authoritative blog | WebFetch, full | "we assign `this.connectionPromise` **synchronously**, repeated calls ... are guaranteed to always reuse the same promise" — the synchronous assignment IS the pattern. Article silently omits rejection handling (gap covered by #4). |
| 6 | https://swr.vercel.app/docs/advanced/performance | 2026-07-25 | official docs (Vercel) | WebFetch, full | 5 components, same key, same tick → "**only 1 network request will be made**" |
| 7 | https://swr.vercel.app/docs/api | 2026-07-25 | official docs (Vercel) | WebFetch, full | `dedupingInterval` default **2000 ms** ("dedupe requests with the same key in this time span"); `errorRetryInterval` 5000; `focusThrottleInterval` 5000; `revalidateOnFocus` true |
| 8 | https://react.dev/reference/react/cache | 2026-07-25 | official docs (React) | WebFetch, full | **"`cache` is for use in Server Components only."** + "`cachedFn` will also cache errors ... the same error is re-thrown" + "React will invalidate the cache for all memoized functions for each server request." |
| 9 | https://nextjs.org/docs/app/getting-started/fetching-data | 2026-07-25 | official docs (Next.js, lastUpdated 2026-07-22, v16.2.11) | WebFetch, full | Client Components: "React's `use` API" or "A community library like SWR or React Query". The `use()`+context pattern documented is for a **server-created promise** passed down, not a client poll. "`React.cache` is scoped to the current request only." |
| 10 | https://tanstack.com/query/latest/docs/framework/react/guides/important-defaults | 2026-07-25 | official docs (TanStack) | WebFetch, full | "Queries that fail are **silently retried 3 times, with exponential backoff delay** before capturing and displaying an error to the UI." / "`refetchInterval` to trigger refetches periodically, which is independent of the `staleTime` setting." / structural sharing keeps the data reference stable when nothing changed |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://1xapi.com/blog/nodejs-cache-stampede-single-flight-pattern-2026 | blog (2026) | **Attempted WebFetch → HTTP 403 Forbidden** |
| https://github.com/nextauthjs/next-auth/issues/10244 | issue tracker | corroborating only ("one page reload sends around 5 requests to retrieve the session"); our root cause is already pinned in `api.ts`, not in SessionProvider |
| https://github.com/nextauthjs/next-auth/discussions/8902 | forum | community tier |
| https://next-auth.js.org/getting-started/client | official docs (v4) | project is on Auth.js v5; v4 page would mislead |
| https://dev.to/serifcolakel/singleflight-smart-request-deduplication-33og | community | community tier, duplicates #3/#4 |
| https://dev.to/karbashevskyi/efficient-request-deduplication-with-createsharedpromise-in-jsts-fbf | community | community tier |
| https://www.npmjs.com/package/fetch-dedupe | package registry | unmaintained; a dep is not warranted for ~12 LOC |
| https://github.com/MattCCC/fetchff | library | new dependency out of scope |
| https://oneuptime.com/blog/post/2026-01-23-nodejs-request-coalescing/view | blog (2026-01) | server-side framing; superseded by #3/#4 |
| https://blog.gaborkoos.com/posts/2026-03-29-One-Cache-to-Rule-Them-All-... | blog (2026-03) | Cloudflare Durable Objects; distributed, not applicable |
| https://softwarepatternslexicon.com/caching-patterns-and-invalidation/consistency-and-stampede-control/stampede-prevention/ | reference wiki | unattributed |
| https://github.com/drizzle-team/drizzle-orm/pull/5862 | PR (2026) | evidence the pattern is being added to mainstream libs in 2026 |
| https://medium.com/talex-global/avoid-double-fetching-using-context-to-share-backend-data-in-next-js-layouts-17a5091d68fa | blog | community tier; same conclusion as #9 |
| https://github.com/vercel/next.js/discussions/50729 | forum | server-side sharing, not client polling |
| https://developer.mozilla.org/.../Promise/race | official docs | not needed once #1/#2 answered the semantics |

**URLs collected: 25** (10 read in full + 15 snippet-only).

### Recency scan (last 2 years, 2024-2026)

Searched the 2025 and 2026 windows explicitly (queries above). Result: **2 relevant
new findings; none supersede the canonical sources.**

1. The single-flight pattern is being *newly adopted into mainstream libraries* in
   2026 — e.g. drizzle-orm PR #5862 "Add single-flight dedup to `$withCache`"
   (2026), and multiple 2026 write-ups (1xapi Jan-2026, oneuptime 2026-01-23,
   gaborkoos 2026-03-29). This is corroboration that the pattern is current best
   practice, not a change to it. The 2026 material adds one caution absent from the
   older sources: **"if an in-flight call stalls, all waiting callers are affected,
   and a failing call causes all waiting callers to see the same error"** — directly
   relevant to question A and to `apiFetch`'s 30 s AbortController.
2. Next.js docs are current as of **2026-07-22 (v16.2.11)** and still name only
   `use()` + SWR/React Query for Client-Component fetching; `React.cache` is still
   documented Server-Components-only. No React 19/20 client-side dedup primitive has
   landed. The "single provider + context" answer is unchanged.

No 2024-2026 source contradicts the Go-singleflight semantics or the MDN promise
semantics relied on below.

---

## ANSWERS

### A. Single-flight / promise dedup in TypeScript

**Name.** The canonical name is **single-flight** (from Go's
`golang.org/x/sync/singleflight`, whose package doc is literally "duplicate function
call suppression"). In JS/TS it is equally called **request coalescing**,
**request deduplication**, or **promise memoization**. All four describe the same
mechanism; `singleflight` is the term to use in the code comment because it points
at an unambiguous reference implementation.

**The mechanism, per-claim.**

1. **Store the PROMISE, assigned synchronously.** Mellman: "we assign
   `this.connectionPromise` **synchronously**, repeated calls to `.getRecord()` are
   guaranteed to always reuse the same promise"
   (https://www.jonmellman.com/posts/singleton-promises/). The synchronicity is the
   whole trick: the bug in `api.ts:58-88` is precisely that the only thing written
   synchronously is *nothing* — the write happens at `:82`, after `await`.
2. **All waiters observe the same settlement.** MDN: "All handlers attached to the
   same promise object are always called in the order they were added"
   (https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise/then).
   Go states the same contract for the equivalent primitive: "the duplicate caller
   waits for the original to complete and **receives the same results**"
   (https://pkg.go.dev/golang.org/x/sync/singleflight).
3. **On rejection, every waiter rejects with the same reason.** That follows from
   (2), and the 2026 practitioner literature states the consequence bluntly: "a
   failing call causes all waiting callers to see the same error."

**Is shared rejection desired here? — NO, and it does not arise.** `getAuthToken`
(`api.ts:64-87`) wraps the whole probe in `try/catch` and *returns* `null` on any
failure; it never rejects. So the shared promise settles **fulfilled-with-null**, and
every waiter gets `null` — i.e. all N calls proceed without an `Authorization`
header, exactly as they do today. This is the important, non-obvious result: **the
rejected-promise-cache hazard that dominates the literature does not apply to this
function as written**, so a plain `.finally()` clear is sufficient. If a future edit
removes the `try/catch`, the Mellman warning becomes live: "If our memoization
implementation has cached a rejected promise, then all future calls will reject with
this same failed promise! ... it's very important to evict rejected promises from the
memoization cache" (https://www.jonmellman.com/posts/promise-memoization/). Add a
comment saying so; a `.finally()` clear (not `.then()`) already satisfies it because
`finally` runs on both outcomes.

**Is there a race where `.finally()` clearing beats a caller that arrived
microseconds earlier? — NO.** Three facts settle it:

- JS is single-threaded; `inflight = null` inside the `finally` callback is atomic
  with respect to every other JS frame. There is no torn read.
- A caller that read `inflight` *before* the clear holds a reference to the
  already-settled promise. Awaiting an already-settled promise resolves on the next
  microtask with the correct value — it does not "miss" the result. So the
  worst case is *one extra microtask of latency*, never a wrong value.
- A caller that reads *after* the clear starts a fresh probe. Because the TTL cache
  at `api.ts:82` is written **inside the async body**, i.e. strictly *before* the
  returned promise settles and therefore before the `finally` microtask runs, that
  caller hits the freshly-populated 60 s TTL cache and issues **no** network request.

  **Ordering requirement for the executor:** the TTL-cache write must stay inside the
  shared async function body (as at `:82`), and the in-flight clear must be in
  `.finally()` on the promise. Do **not** invert this (e.g. clear the in-flight ref
  inside the body before returning, then write the cache in a `.then()`) — that
  creates a real window where `inflight` is null and the cache is not yet warm.

The pathological version of this race exists only when the in-flight entry is cleared
on a *timer* rather than on settlement. Not applicable here.

**MDN caveats on `.finally()` the executor must respect:**
- "The `onFinally` callback does not receive any argument" — so the clear cannot
  inspect the outcome; it clears unconditionally, which is what we want.
- "A `finally()` call is usually transparent and reflects the eventual state of the
  original promise" — so `p.finally(clear)` is safe to hand to callers *if* you hand
  out the same object consistently. **Simplest correct shape: assign the raw promise
  to the module ref, attach `.finally()` for the side-effect only, and return the raw
  promise.** Do not store `p.finally(...)` in the ref and return `p` (or vice-versa)
  — the two are different objects, which is how "cleared twice / never cleared" bugs
  appear.
- "A `throw` ... in the `finally` callback still rejects the returned promise" — the
  clear body must be a bare assignment that cannot throw.

**Reference shape (mirrors the existing style at `api.ts:51-88`):**

```ts
let sessionTokenCache: SessionTokenCache | null = null;
let sessionTokenInflight: Promise<string | null> | null = null;   // single-flight
const SESSION_TOKEN_TTL_MS = 60_000;

async function getAuthToken(): Promise<string | null> {
  if (typeof window === "undefined") return null;
  const now = Date.now();
  if (sessionTokenCache && now - sessionTokenCache.ts < SESSION_TOKEN_TTL_MS) {
    return sessionTokenCache.value;
  }
  if (sessionTokenInflight) return sessionTokenInflight;      // <-- coalesce
  const p = probeSession();                                    // existing :64-87 body
  sessionTokenInflight = p;                                    // SYNCHRONOUS assign
  p.finally(() => { sessionTokenInflight = null; });            // clear on settle
  return p;
}
```

### B. Cache invalidation × the 401 path

Today `api.ts:161-166` does exactly one thing on 401: `sessionTokenCache = null`.

**Concrete poisoning sequence if only the TTL cache is cleared:**

1. `t=0` — `apiFetch(/portfolio)` runs with cached token `T` (still inside its 60 s
   TTL). Request is in flight to the backend.
2. `t=+150 ms` — the TTL expires. `apiFetch(/status)` misses, sets
   `sessionTokenInflight = P`, and fires `/api/auth/session`. `P` is in flight.
3. `t=+200 ms` — step 1's response returns **401**. The branch at `:163-166` sets
   `sessionTokenCache = null`. `sessionTokenInflight` is untouched and still holds `P`.
4. `t=+250 ms` — `P` resolves. Its body reaches `:82` and writes
   `sessionTokenCache = { value: T, ts: now }` — **re-caching the token that was just
   invalidated**, with a fresh 60 s timestamp. The invalidation from step 3 is undone.
5. Every `apiFetch` for the next 60 s sends the dead Bearer token, gets 401, nulls the
   cache — and the next miss re-probes and re-caches. A 401 livelock.

Step 5 is normally cut short because `:172-174` navigates to `/login`, tearing down
the JS context. **But `:172` deliberately skips the redirect when
`window.location.pathname === "/login"`** (the phase-75.12 frontend-02 guard). On
`/login` there is no teardown, so the loop is unbounded — which is the exact page
that guard was added to protect.

**Correct fix — an epoch/generation counter, invalidation-first ordering.** Clearing
`sessionTokenInflight` alone is *necessary but not sufficient*: it stops *new*
callers joining `P`, but does nothing about `P`'s own pending write at `:82`. This is
what Go's `singleflight.Forget` exists for — "Future calls to Do for this key will
call the function rather than waiting for an earlier call to complete"
(https://pkg.go.dev/golang.org/x/sync/singleflight) — note it also only affects
*future* calls, so the in-flight result still needs to be discarded by the caller.

```ts
let sessionEpoch = 0;

// inside the probe body, replacing api.ts:82:
const myEpoch = sessionEpoch;              // snapshot BEFORE the await
const res = await fetch("/api/auth/session");
// ... derive `token` ...
if (sessionEpoch === myEpoch) {            // only write if not invalidated meanwhile
  sessionTokenCache = { value: token, ts: Date.now() };
}
return token;

// inside the 401 branch, replacing api.ts:166 -- ORDER MATTERS:
sessionEpoch += 1;          // 1. poison any in-flight probe's pending write
sessionTokenCache = null;   // 2. drop the resolved cache
sessionTokenInflight = null;// 3. stop new callers joining the doomed probe
```

Epoch bump **first**: if the probe settles between statements 1 and 3, its write is
already rejected by the epoch check. Reversing the order (clear inflight, then bump)
leaves a one-microtask window in which the stale write lands.

This epoch idiom is already the repo's convention for the same class of bug — the
`cancelled` flag in `useLivePrices.ts:53` and `useTickerMeta.ts:29` is the React-effect
form of it. Naming it that way in the comment will make the change legible.

**Also invalidate on the `!res.ok` path.** `api.ts:66-69` caches `{value: null}` when
the session endpoint itself returns non-OK. That sentinel is intentional, but it must
be subject to the same epoch guard, or a 401 arriving during a failing probe will be
overwritten by the negative cache and mask a subsequent successful re-login.

### C. React 19 / Next 15 duplicate-provider consolidation — RANKED

**Ruled out, with citations:**

- **`React.cache()` — NOT APPLICABLE.** "`cache` is for use in **Server Components**
  only" (https://react.dev/reference/react/cache). Every file in play is
  `"use client"`. It also "invalidate[s] the cache for all memoized functions for each
  server request", so it has no notion of a 60 s client poll. And it "will also cache
  errors ... the same error is re-thrown" — the exact hazard §A warns about.
- **`use()` + promise-in-context — NOT APPLICABLE.** The Next.js pattern
  (https://nextjs.org/docs/app/getting-started/fetching-data, "Sharing data with
  context and `React.cache`") passes a **server-created, single-shot** promise into a
  client context. Our data is client-polled every 60 s and must re-render on each
  tick; `use()` reads a promise once. Wrong tool.
- **Server Components — NOT APPLICABLE.** The whole subtree is client-side
  (`"use client"` at `layout.tsx:1`, `positions/page.tsx:1`,
  `live-portfolio-context.tsx:1`) because it polls and holds interactive state.

**Ranked recommendation:**

1. **[RECOMMENDED] Single provider + `useLivePortfolio()`.** This is both the
   documented App-Router idiom ("wrap your provider inside layout … you ensure your
   state is available throughout your entire application") and — critically — an
   almost pure **deletion** in this repo: `LivePortfolioValue`
   (`live-portfolio-context.tsx:53-81`) already exposes `status`, `portfolio`,
   `positions`, `snapshots`, `livePrices`, `tickerMeta`, `loading`, `error`,
   `refresh`; the layout already imports `useLivePortfolio` (`:59`) and already reads
   three fields from it (`:143-146`). See §E.5 for the exact edit and §E.5's two
   behavioural caveats (error-surfacing semantics; the `/login` gate). Verified — the
   step's "the consolidation may be small" hypothesis is correct.
   - Keep `PaperTradingDataContext` as the *view-model* layer (it also carries
     `trades`, `perf`, `activeMarket`, `openRationale`, `tickerMeta`), but have it
     **re-export the provider's objects by reference** so
     `usePaperTradingData().positions === useLivePortfolio().positions`. Referential
     identity is what makes axis 1 of §E.6 impossible to regress. TanStack's
     structural sharing exists for the same reason: "Query results by default are
     structurally shared to detect if data has actually changed and if not, the data
     reference remains unchanged"
     (https://tanstack.com/query/latest/docs/framework/react/guides/important-defaults).
   - Fold the layout's `useTickerMeta` (`:154`, positions ∪ trades) **into the
     provider** as the single superset call, and delete the provider's own
     positions-only call (`:196`). One request instead of two, and the layout's
     superset is what the tables need.
2. **[FUTURE, NOT THIS STEP] Adopt SWR.** It solves both A and C generically —
   "only 1 network request will be made" for N components on one key
   (https://swr.vercel.app/docs/advanced/performance), `dedupingInterval` 2000 ms
   default (https://swr.vercel.app/docs/api), and it is one of the two options Next.js
   itself names for Client Components. But it is a new dependency plus a rewrite of
   every poller in §E.1 — out of scope for a P1 defect fix, and it would not by
   itself fix axes 2/3 of §E.6.
3. **[REJECT] Leave two providers and "sync" them.** Any scheme that keeps two
   `/portfolio` fetches and reconciles afterwards re-creates the phase-72 bug the
   comment at `layout.tsx:137-142` claims to have fixed.

**Do not stop at axis 1.** Per §E.6, also: (a) make the donut's cash come from the
same object as the centre's cash (pick one of `/status`'s `portfolio.cash` or
`/portfolio`'s `current_cash` — they are different endpoints), and (b) make
`page.tsx:67-77 mvUsd` delegate to `positionMarketValueUsd` from `lib/format.ts` so
the slices and the centre use one formula. Otherwise the operator-visible symptom
survives the fix.

### D. Polling discipline under consolidation — THE BIGGEST RISK IN THIS STEP

`.claude/rules/frontend.md` (Conventions): "Polling loops (setInterval) must count
consecutive failures and stop after 5 with an error message. Never poll forever on a
dead backend."

Current compliance, measured:

| Poller | Failure budget | Compliant? |
|---|---|---|
| `useLivePrices.ts:58-67` | `failRef >= 5` → set error, `circuitOpenRef`, `clearInterval` | YES |
| `OpsStatusBar.tsx:95-101` | `failRef >= 5` on all-four-null → `stoppedRef`, interval self-skips | YES |
| `KillSwitchPanel.tsx:52-54` | `failRef >= 5` → error | YES (does not stop the interval, but does surface) |
| `CycleHealthStrip.tsx` | (not mounted on this route) | n/a |
| `Sidebar.tsx:292-303` | none — 30 s `/api/health` forever | pre-existing violation, out of scope |
| **`LivePortfolioProvider` `:122-166`** | **none** | **NO — 60 s forever; `Promise.allSettled` guarantees it never throws** |
| `paper-trading/layout.tsx:186-213` | n/a — **one-shot, no interval** | vacuously compliant |

**The regression hazard, stated plainly:** the layout's `/status`, `/portfolio`,
`/trades`, `/snapshots`, `/performance` calls are currently **one-shot on mount**.
Moving them into `LivePortfolioProvider` promotes them into an **unbounded 60 s loop
with no failure counter**. That is a net-new violation of the 5-failure rule for five
endpoints, introduced by the very step meant to reduce request volume. **The contract
must require a failure counter in `LivePortfolioProvider` in the same change**, or
Q/A should fail the step.

**Correct shape for shared polling with per-consumer failure budgets:** do **not**
give each consumer its own counter once there is one request — a counter per consumer
over a single shared request counts the same failure N times and trips at 5/N rounds.
The industry model is one owner holding the retry policy and every observer reading
the same derived state: TanStack — "Queries that fail are silently retried 3 times,
with exponential backoff delay **before capturing and displaying an error to the
UI**"; the error is a property of the query, and all observers of the key see it.

Concretely: put a single `failRef` in `LivePortfolioProvider.refresh` (increment when
**all** of `status`/`portfolio`/`snapshots` settle rejected — the `allFailed` branch
at `:138-148` already computes exactly this predicate), stop the interval at 5, and
add `stale: boolean` to `LivePortfolioValue` alongside the existing `error`. Consumers
then render their own message from `lp.stale` / `lp.error` — per-consumer *presentation*,
shared *budget*. Mirror the existing `stoppedRef` idiom from `OpsStatusBar.tsx:84,113`
so the code reads like its neighbours.

Two secondary notes:
- `LivePortfolioProvider` has **no `document.hidden` guard** on its 60 s interval,
  unlike `useLivePrices.ts:50`, `OpsStatusBar.tsx:116`, `KillSwitchPanel.tsx:58`.
  Absorbing five more endpoints into it makes a background tab five times noisier.
  Adding the guard is a one-line consistency fix; SWR's analogue is
  `revalidateOnFocus: true` + `focusThrottleInterval: 5000`
  (https://swr.vercel.app/docs/api).
- The single-flight change interacts with the 30 s `AbortController` at
  `api.ts:130-131`: that timeout guards the **backend** fetch, not the session probe
  at `:65`, which has **no timeout at all**. Under single-flight a stalled probe now
  blocks *every* `apiFetch` instead of one — the 2026 caution "if an in-flight call
  stalls, all waiting callers are affected". Recommend adding an `AbortSignal.timeout`
  to the `/api/auth/session` fetch in the same change (it is a same-origin Next.js
  route, so a short 10 s budget is generous). This is a real, new failure mode created
  by the fix; it should be in the contract, not discovered later.

### Testability split (vitest + @testing-library/react in jsdom; no Playwright runner)

Config: `frontend/vitest.config.ts` — `environment: "jsdom"`, `globals: true`,
`include: ["src/**/*.{test,spec}.{ts,tsx}", ...]`.

**Unit-testable (must be), with the idioms already in the repo:**
- **Single-flight count.** `api.test.ts:45-47` already does `vi.resetModules()` +
  dynamic import specifically because "api.ts's module-level `sessionTokenCache` is a
  singleton". Reuse `mockFetchByUrl` (`api.test.ts:33`), fire N `apiFetch` calls
  **concurrently via `Promise.all` without awaiting in between**, assert the
  `/api/auth/session` handler was hit exactly **once**. Mutation check: reverting the
  fix makes the count N — a guard that can fail.
- **401 poison sequence.** Resolve the probe on a deferred, trigger the 401 branch
  mid-flight, then assert `sessionTokenCache` was NOT repopulated (observable as: the
  next `apiFetch` re-probes rather than reusing the dead token). Mutation check: drop
  the epoch guard and the assertion flips.
- **Provider consolidation.** `live-portfolio-context.test.tsx` already mocks
  `@/lib/api` through `vi.hoisted` + `vi.clearAllMocks()`. Render the paper-trading
  layout with the positions page under one `LivePortfolioProvider` and assert
  `getPaperPortfolio` / `getPaperTradingStatus` / `getPaperSnapshots` were each called
  **once**, and that `getTickerMeta` was called once with the **superset** ticker list.
- **Failure budget.** `vi.useFakeTimers()`, make all three fetchers reject, advance
  `5 × 60_000`, assert the mock call count plateaus (interval stopped) and `stale` is
  true. Mutation check: remove the counter and the count keeps climbing.
- **Referential identity.** Assert `usePaperTradingData().positions ===
  useLivePortfolio().positions` — cheap, and it is the guard that prevents axis 1 of
  §E.6 from regressing.

**NOT unit-testable — belongs in the Playwright live_check:**
- The measured request counts on a real page view (11 → 1 session probes; the
  per-endpoint 2 → 1 collapse). jsdom has no network panel, and the count depends on
  real mount ordering and real timing. This must be a fresh
  `audit-net-positions-20s.txt`-style capture compared against the archived one.
- The donut centre visually equalling Σ slices (needs a real render against real BQ
  data, and axes 2/3 only show up with a non-US position held).
- That the `OpsStatusBar` ×2 rounds are unchanged (i.e. the fix did not perturb them).
- Per `.claude/rules/frontend.md` "Live-UI verification" and the recorded
  `feedback_second_next_dev_breaks_operator_3000` incident: run the capture on
  **:3100 with `LIGHTHOUSE_SKIP_AUTH=1` and an isolated `PLAYWRIGHT_DIST_DIR`**, kill
  it afterwards, and re-verify the operator's :3000 still answers.

---

## Application to pyfinagent — change sites

| Concern | File:line | Change |
|---|---|---|
| A. single-flight | `frontend/src/lib/api.ts:55-88` | add `sessionTokenInflight` + synchronous assign + `.finally()` clear; keep the TTL write inside the body at `:82` |
| A. probe timeout | `frontend/src/lib/api.ts:65` | add an `AbortSignal.timeout` to the `/api/auth/session` fetch (new hazard created by coalescing) |
| B. 401 invalidation | `frontend/src/lib/api.ts:163-166` | `sessionEpoch += 1` → `sessionTokenCache = null` → `sessionTokenInflight = null`, in that order; epoch check guarding the writes at `:67` and `:82` |
| C. provider consolidation | `frontend/src/app/paper-trading/layout.tsx:186-213` | drop `getPaperTradingStatus`/`getPaperPortfolio`/`getPaperSnapshots`; read `lp.*`; keep `trades`+`performance`; call `lp.refresh()` from `refresh` |
| C. ticker-meta dedup | `layout.tsx:148-157` + `live-portfolio-context.tsx:191-196` | one superset call in the provider; delete the other |
| C. stale comment | `layout.tsx:137-142` | correct it — it is true only of `useLivePrices`/`useLiveNav` (§E.4) |
| C. donut axis 2 | `positions/page.tsx:131` vs `useLiveNav.ts:32` | one cash source |
| C. donut axis 3 | `positions/page.tsx:67-77` | delegate `mvUsd` to `positionMarketValueUsd` (`lib/format.ts`) |
| D. failure budget | `live-portfolio-context.tsx:122-166` | `failRef` on the existing `allFailed` predicate (`:138`), stop at 5, expose `stale`; add a `document.hidden` guard |

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch — **10**
- [x] 10+ unique URLs total — **25**
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (api.ts, paper-trading
      layout, positions page, LivePortfolioProvider, useLivePrices, useLiveNav,
      useTickerMeta, OpsStatusBar, Sidebar, root layout, AuthProvider, vitest config)
- [x] Contradictions noted — Mellman's rejected-promise hazard vs this codebase's
      never-rejecting `getAuthToken`; `layout.tsx:137-142` comment vs measured state
- [x] All claims cited per-claim
- Gap: the second `OpsStatusBar` round (§E.3 item 4) is attributed to the
  `visibilitychange` listener / dev double-mount by elimination (one mount site,
  stable `useCallback` deps), not by direct instrumentation. It is out of scope for
  this step; if a future step cares, instrument it rather than trusting this inference.

```json
{
  "tier": "T2",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 15,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "summary": "The 11x /api/auth/session stampede is confirmed as a missing single-flight in getAuthToken (api.ts:58-88): the TTL cache is written only at :82, AFTER the await, so every concurrent apiFetch re-enters the miss branch. Fix = store the promise synchronously, clear in .finally(); no rejection hazard because getAuthToken try/catches to null. The 401 path (:166) must ALSO bump an epoch counter BEFORE nulling the cache and the in-flight ref, else an in-flight probe re-caches the just-invalidated token for 60s (livelock on /login, where the redirect guard at :172 suppresses teardown). React.cache() is Server-Components-only and use() is for server-created promises -- neither applies; single provider + useLivePortfolio() is correct and is nearly a pure deletion (LivePortfolioValue already exposes every field the layout re-fetches). BIGGEST RISK: the layout's five fetches are currently ONE-SHOT; moving them into LivePortfolioProvider promotes them into a 60s loop that has NO failure counter, a net-new violation of the 5-consecutive-failure rule. Also: consolidating fixes only ONE of THREE axes of the donut/NAV disagreement (different /portfolio responses; different cash endpoints; different market-value formulas).",
  "brief_path": "handoff/current/research_brief_80.11.md",
  "gate_passed": true
}
```
