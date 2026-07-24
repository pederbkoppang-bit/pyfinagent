# Research Brief — Step 75.12 (Frontend data-plane hardening)

Tier: **moderate** (caller-specified). Executor: sonnet-4.6/high. Priority P1.
Findings: frontend-01, frontend-02, frontend-03, frontend-05, frontend-06,
fe-ts-01, frontend-09. Not audit-class.

---

## 0. LOUD CAVEAT — criterion-1/2 live evidence is VACUOUS on the operator's box

**Measured live 2026-07-24:**
- `:3000/login` → **200**; `:3000/` → **302** (operator instance healthy, untouched); dev server PID 53909.
- `:8000/api/health` → ok, version **6.68.13**.
- `:8000/api/mas/events/stats`, `/api/mas/dashboard`, `/api/paper-trading/status` → **all 200 with NO credentials**; status payload `status="active"`, `loop` present.
- Running uvicorn PID 98681 has **`DEV_LOCALHOST_BYPASS=1`** in its process env (NOT in `backend/.env` — exported into the process). `NEXT_PUBLIC_API_URL=http://localhost:8000`.

**Mechanism (`auth.py:150-153`):** `get_current_user` returns a bypass user when
`DEV_LOCALHOST_BYPASS=1` AND `request.client.host ∈ {127.0.0.1, ::1, localhost}`.
The operator's browser talks to `localhost:8000` → backend sees `127.0.0.1` →
**auth bypassed on every endpoint regardless of credentials.** Therefore:

1. **frontend-01 does NOT reproduce on localhost.** `/agents` SSE already
   connects today with `withCredentials:false`. A Playwright "connected stream"
   capture PASSES with or without the fix → it proves the bypass, not the fix.
2. **frontend-02 does NOT reproduce on localhost.** `/api/paper-trading/status`
   returns 200 (bypass), never 401 → no redirect → no loop. "No reload loop"
   live evidence is equally non-discriminating.
3. **The fixes are still correct** for the non-bypass path: the bypass is
   "NEVER set in production or on a publicly-reachable host" (`auth.py:149`).
   Reached via the **Tailscale CGNAT IP** (`100.64.0.0/10`, matched by
   `_TAILSCALE_ORIGIN_RE`, `main.py:480-491`) the client host is the tailscale
   interface → bypass off → real cookie auth → every finding reproduces.

**⇒ Discriminating proof = vitest behavioral tests, NOT the localhost Playwright
capture** (which the immutable command doesn't run anyway — see §6). Take the
Playwright capture (proves no-regression / renders), but Q/A must NOT accept it
as proof of frontend-01/02/03.

---

## 1. Internal code inventory (re-anchored 2026-07-24)

| File | Anchor | State / drift |
|------|--------|---------------|
| `hooks/useEventSource.ts` | **:91** | `new EventSource(url, { withCredentials: false })` — ACCURATE. Default `maxFailures:3` (:63); `/agents` overrides to 5. |
| `lib/api.ts` getAuthToken | :46-63 | `await fetch("/api/auth/session")` on EVERY call; **no `sessionTokenCache`**. Cookie is HttpOnly so `document.cookie` (:54) can't read it → returns `"session-active"` sentinel; backend skips it (auth.py:182) and auths via cookie. The probe only picks sentinel-vs-null → the redundant round-trip frontend-09 targets. |
| `lib/api.ts` 401 branch | **:112-116; redirect at :113** | `window.location.href="/login"`, no pathname guard. Finding says ":114" → **off-by-one, redirect is :113**. `credentials:"include"` **ALREADY set at :87**. NOTE: the redirect is a SIDE EFFECT inside apiFetch — it fires even when the caller wraps in `Promise.allSettled` (that's why the provider still loops). |
| `lib/useLivePrices.ts` | :48-53 | On 5th fail sets `error` but **interval keeps firing** — NO `clearInterval`/`circuitOpen`. Doc comment (:22-24) claims "stops polling after 5" — **the doc lies**. |
| `lib/live-portfolio-context.tsx` | :100, :142-146 | `LivePortfolioProvider` polls `getPaperTradingStatus`+`getPaperPortfolio`+`getPaperSnapshots` unconditionally every 60s + `void refresh()` on mount. **No pathname/session gate.** |
| `app/layout.tsx` | **:36** | `<LivePortfolioProvider>` wraps ALL routes incl. `/login`, inside `<AuthProvider>`. Comment :34-35 admits gating is a "future hardening pass". |
| `app/agents/page.tsx` | :186-208 SSE; **:221, :229** raw; :241 poll | `fetch(${API_BASE}/api/mas/events/stats)` + `/api/mas/dashboard`, NO creds, silent `catch{}`, **15s poll, NO failure counter**. SSE `maxFailures:5` + error string at :210-213. |
| `components/OpsStatusBar.tsx` | **:78-91** | 4 calls each `.catch(()=>null)` in `Promise.all` → outer `catch` (:89) UNREACHABLE → `failRef` (:74) incremented only in the dead branch, never read. **`failRef` is dead code; no stale segment ever renders.** |
| `lib/types.ts` PaperTradingStatus | **:702-721** | `status: string` (not a union); **`loop` is REQUIRED** → `tsc` never flags `s.loop.running`; the type lies about the `not_initialized` shape. |
| `app/paper-trading/layout.tsx` | **:217, :263** | :217 `!!status?.loop.running` (crashes if status non-null w/o loop — `status?.loop` short-circuits only on nullish `status`); :263 `!s.loop.running` (NO optional chaining at all). :215 already reads `status?.status !== "not_initialized"`. |
| `app/reports/page.tsx` | **:198-200** | `fetch(${API_BASE}/api/charts/${t}?period=1y)`; `if(res.ok)` (:199); `catch { /* ignore chart failures */ }` (:200) → both non-ok and throw swallowed → silently empty chart. No `getChartData` wrapper exists in api.ts. |
| `app/cron/page.tsx` | :59, :161-218, :425-545 | **CANONICAL template**: `MAX_CONSECUTIVE_FAILURES=5`; `failuresRef`+`stoppedRef`; interval guarded `if(!stoppedRef.current) load()`; on 5th fail sets a "polling stopped after 5" error. |
| finding-06 targets confirmed present | — | `components/AutoresearchLeaderboard.tsx` (+ `.test.tsx`), `components/HarnessDashboard.tsx`, `app/observability/page.tsx` (`getObservabilityDataFreshness`), `api.ts::getSeedStability`. Executor must audit each poll loop vs the cron template (not re-read here). |

### Backend trace (load-bearing for frontend-01)
- **CORS** (`main.py:485-491`): `allow_origin_regex=_TAILSCALE_ORIGIN_RE` (`^http://(localhost|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+):\d+$`) + **`allow_credentials=True`** → the finding's "CORS already allows credentials" is CONFIRMED for `localhost:*` and Tailscale.
- **`get_current_user` accepts the NextAuth session COOKIE cross-origin** (`auth.py:169-189`); docstring: *"cross-origin cookies work because fetch uses credentials:'include' + CORS allow-credentials=true"*; reads `__Secure-authjs.session-token` / `authjs.session-token` from `request.cookies`. **⇒ `withCredentials:true` is the correct AND sufficient SSE fix when the bypass is off.**
- **not_initialized backend branch** (`paper_trading.py:134`): `return {"status":"not_initialized","message":...}` — **NO `loop` key** → `layout.tsx:217/:263` TypeError on a fresh/reset install. fe-ts-01 crash is REAL.
- SSE endpoint `mas_events.py:22-44` — `StreamingResponse text/event-stream`; auth via global middleware, not a route dep.

---

## 2. External — Read in full (≥5)

| URL | Accessed | Kind | Finding |
|-----|----------|------|---------|
| developer.mozilla.org/.../HTTP/Guides/Cookies | 2026-07-24 | official | **Port is NOT part of cookie scope** — a cookie set by `localhost:3000` IS sent to `localhost:8000` (scoped by domain+path only). `SameSite=Lax` "same-site" is the **registrable domain**, not origin/port → localhost:3000 & :8000 are same-site. |
| developer.mozilla.org/.../API/Request/credentials | 2026-07-24 | official | `omit` / `same-origin` (default) / **`include`** = "Always include credentials, even for cross-origin requests"; server must respect `Set-Cookie`; cross-origin needs `Access-Control-Allow-Credentials:true`. |
| typescriptlang.org/docs/handbook/2/narrowing.html | 2026-07-24 | official | Discriminated union: shared **literal** discriminant (`kind:"circle"`) lets TS narrow in `switch`/`if`. Must be a literal, NOT plain `string` — "If `kind` were `string`, TypeScript couldn't correlate…". Directly mandates changing `status: string` → a literal union. |
| martinfowler.com/bliki/CircuitBreaker.html | 2026-07-24 | canonical | "Once failures reach a threshold, the circuit breaker trips, and all further calls return with an error, without the protected call being made at all." Half-open reset after a timeout. |
| hahwul.com/sec/web-security/sse/ | 2026-07-24 | practitioner | "EventSource does not standardly support setting custom HTTP headers like Authorization" → **cookie auth is the SSE path**; browser auto-sends cookies on EventSource; CSRF defense = server validates `Origin` + `SameSite`. |
| developer.mozilla.org/.../Server-sent_events/Using_server-sent_events | 2026-07-24 | official (thin) | Confirms cross-origin form `new EventSource(url,{withCredentials:true})`; SSE is one-way; non-HTTP/2 6-connection-per-browser cap. Thin on the header question specifically. |

### Snippet-only (context; not counted): ~20 URLs
MDN EventSource.withCredentials (read-only bool only), whatwg/html#2177 (SPA
shell), Eclipse Ditto SSE, openillumi SSE-auth, Trendyol SSE (Medium), dev.to
cross-domain SSE, sse.js (custom-header lib), Medium/DCHost/jguillaumesio/runebook
SameSite, socket.io #3784, MS Learn SameSite, dev.to "Safe API Response Handling"
DU, oneuptime DU (2026-01-24), convex DU, oneuptime Polly (2026-01-27),
Medium JS circuit breaker, brooker.co.za retries, keyholesoftware retry-storms,
Medium "Efficient Polling in React".

## 3. Recency scan (last 2 years)
Explicit 2025-2026 passes run for SSE-auth, SameSite, DU, and client polling
circuit-breakers. **No new finding supersedes the canonical guidance.** 2025-2026
practitioner sources (oneuptime Polly 2026-01-27, keyholesoftware, brooker 2022→
still cited) reaffirm: "polling should stop once the configurable failure limit
is reached"; modal trip threshold **3-5 consecutive failures** — consistent with
the repo's existing cron/useLivePrices `MAX=5`. DU recency (oneuptime 2026-01-24)
reaffirms the literal-discriminant rule. SSE recency reaffirms cookie-only auth +
Origin/SameSite CSRF defense. Nothing forces a design change from the plan.

## 4. Key findings (external → mapping)
1. **EventSource cannot send Authorization headers; cookie auth is the only
   in-spec path** (HAHWUL; MDN Using-SSE). → `withCredentials:true` + backend
   cookie-read (auth.py:169-189) is the complete frontend-01 fix.
2. **Cross-port cookies work on same-site localhost** (MDN Cookies: port ∉ scope;
   Lax same-site = registrable domain). → the finding's premise holds; NO
   `SameSite=None`/`Secure` change needed for localhost or Tailscale (same host).
3. **`credentials:"include"` + `Access-Control-Allow-Credentials:true`** is the
   required pair (MDN credentials) — both already present (api.ts:87; main.py:488).
4. **DU discriminant must be a literal** (TS Handbook) → `status: string` must
   become `status: "not_initialized" | "active" | …` for narrowing to gate `.loop`.
5. **Circuit breaker = count failures, trip, stop calling** (Fowler) → useLivePrices
   must set a `circuitOpen` flag and `clearInterval`; the cron template is the local idiom.

## 5. Application to pyfinagent (per finding)
- **frontend-01**: `withCredentials:true` in useEventSource:91; route agents
  `fetchStats`/`fetchOpenClaw` (:221/:229) through `apiFetch` (gains
  `credentials:"include"` + Bearer sentinel + typed errors). Consider making
  `withCredentials` an option so non-auth SSE callers (future) can opt out.
- **frontend-02**: (a) api.ts 401 branch — `if (window.location.pathname !==
  "/login") window.location.href="/login"`. (b) Gate `LivePortfolioProvider`:
  skip the poll trio (initial + interval) when `pathname==="/login"` (and ideally
  when no session) — realize the comment's "future hardening pass". Use
  `usePathname()` from `next/navigation`.
- **frontend-03**: add `getChartData(ticker)` → `apiFetch(/api/charts/${t}?period=1y)`
  in api.ts; in reports, catch per-ticker and render a rose partial-failure notice
  (frontend-layout §8 error banner: `border-rose-500/30 bg-rose-950/30`,
  `IconWarning`) when any price series is missing — never a silently empty chart.
- **frontend-05**: OpsStatusBar — after `Promise.all`, if ALL four are null,
  `failRef.current++`; render a stale/error segment after ≥5; back off / stop the
  60s interval (mirror cron `stoppedRef`). Keep the per-call `.catch(()=>null)`
  (graceful-degrade) BUT detect the all-null outcome.
- **frontend-06**: apply the cron `failuresRef`/`stoppedRef` template to agents
  stats/dashboard, observability freshness, HarnessDashboard, AutoresearchLeaderboard;
  give useLivePrices an explicit `circuitOpen` ref that skips `tick()` and clears
  the interval at 5.
- **fe-ts-01**: make `loop?` optional (or a `not_initialized` union member) in
  types.ts; guard `status?.loop?.running` (:217) and `s.loop?.running` (:263).
- **frontend-09**: module-level `sessionTokenCache={value,ts}` in api.ts, ~60s
  TTL; `getAuthToken` returns cache within TTL; **invalidate on 401** (clear cache
  in the 401 branch so a re-login isn't masked by a stale sentinel).

## 6. Vacuous-guard / mutation matrix + boundary/ordering

**Immutable verification command (quoted): a python3 source-SCAN + tsc** —
asserts `'withCredentials: true' in useEventSource`, `'pathname' in api`,
`'sessionTokenCache' in api`, `'failRef' in ops AND 'stale' in ops.lower()`,
`'not_initialized' in types`, then `cd frontend && npx tsc --noEmit`. It does
**NOT run vitest and does NOT run Playwright.** Every asserted token is trivially
satisfiable by writing the string (e.g. `withCredentials:true` present but reload-
loop unfixed; `'stale'` as a comment). **So the source-scan is a weak guard; the
behavioral success_criteria + live_check carry the weight.** Prescribed
discriminating tests (vitest; precedents: `live-portfolio-context.test.tsx` mocks
`@/lib/api`; `RecentTickerChips.test.tsx` uses `Object.defineProperty(window,…)`):
- **reload-loop**: set `window.location.pathname="/login"`, mock fetch→401, call
  an apiFetch fn, assert `window.location.href` was NOT reassigned. Mutation:
  remove the guard → test fails.
- **provider gate**: render `<LivePortfolioProvider>` with `usePathname` mocked to
  `/login`, assert the api mocks were NOT called; mock to `/` → called.
- **useLivePrices stop-at-5**: fake timers, mock reject ×5, advance 6×60s, assert
  `getPaperLivePrices` called exactly 5 (not 6). Mutation: drop `circuitOpen` → 6th call fires.
- **OpsStatusBar**: mock all four fetchers→reject/null, advance 5 intervals,
  assert a stale/error segment renders + no 6th poll.
- **fe-ts-01**: making `loop?` optional makes `npx tsc --noEmit` FAIL on the
  unguarded accesses until guarded — the type change is itself the mutation test.
  Add a render test: status=`{status:"not_initialized"}` renders without throwing.

**Boundary / hot-reload ordering** (dev server PID 53909 picks up edits
instantly; do NOT start a 2nd server that manages :3000 — standing memory
`feedback_second_next_dev_breaks_operator_3000`):
1. **fe-ts-01** (types union + the two layout guards **in one commit**) — pure
   type-safety, no behavior change, but leaving types.ts optional-`loop` without
   the guards = a red `tsc` overlay on the live app until fixed.
2. **frontend-09** + api.ts pathname guard — low risk.
3. **frontend-01** (withCredentials + agents raw→apiFetch) — inert on localhost bypass, safe.
4. **frontend-02b** (`LivePortfolioProvider` gate) — **highest risk**: a wrong
   gate disables Home/Paper-Trading NAV. Gate to skip ONLY on `/login` (or no
   session); land last, with the provider-gate test.
5. **frontend-05/06** circuit breakers — additive.
6. **frontend-03** reports notice — additive.

**UI conventions for the new elements** (partial-failure notice, stale segment):
no emoji; Phosphor icons via `@/lib/icons` (`IconWarning`); navy/slate palette;
rose error banner tokens (frontend-layout §8); OpsStatusBar stale segment mirrors
the existing `slate-600` dot / amber worst-of-N idiom; `scrollbar-thin` on any
new scroll container.

**Playwright**: pinned `@playwright/mcp@0.0.76` (`.mcp.json:79-86`, explicit
`--executable-path` + `--user-data-dir`). Capture `/agents` read-only GET against
the operator :3000 (already authed via bypass — the 75.6 pattern; do NOT use
`LIGHTHOUSE_SKIP_AUTH` against :3000, do NOT spawn a 2nd :3000-managing server).
Move shots to `handoff/current/captures_75.12/`. Remember the capture is
NON-discriminating for the auth fix (§0).

---

## 7. Research Gate Checklist
- [x] ≥5 authoritative external sources READ IN FULL (6: MDN Cookies, MDN
  Request.credentials, TS Handbook, Fowler CircuitBreaker, HAHWUL SSE, MDN Using-SSE)
- [x] 10+ unique URLs (≈26 incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported (§3)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (§1)

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 20,
  "urls_collected": 26,
  "recency_scan_performed": true,
  "internal_files_inspected": 20,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "All 7 findings re-anchored and CONFIRMED with corrected line numbers (api.ts redirect is :113 not :114; credentials:'include' already at :87). Backend trace proves the finding-01 premise sound: get_current_user reads the NextAuth cookie cross-origin (auth.py:169-189) + CORS allow_credentials=True (main.py:488), so withCredentials:true is the correct+sufficient SSE fix; cross-port cookies work because localhost:3000/:8000 are same-site (MDN). BUT the running backend has DEV_LOCALHOST_BYPASS=1 + NEXT_PUBLIC_API_URL=localhost, so auth is bypassed for the operator's browser -> frontend-01/02/03 do NOT reproduce on localhost and the criterion-1 Playwright capture is VACUOUS evidence; discriminating proof must be vitest behavioral tests. fe-ts-01 crash is real (paper_trading.py:134 omits loop). useLivePrices doc lies (no clearInterval at 5). OpsStatusBar failRef is dead code (per-call catches make the outer catch unreachable). Immutable command is a weak source-scan+tsc; success_criteria+live_check carry the weight. Provider-gate is the highest hot-reload risk; land it last.",
  "brief_path": "handoff/current/research_brief_75.12.md",
  "gate_passed": true
}
```
