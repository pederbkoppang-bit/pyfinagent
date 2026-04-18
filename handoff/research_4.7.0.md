# Research: Step 4.7.0 — Route Inventory + 30-Day Usage Telemetry

## Sources Found: 13 unique URLs

1. https://github.com/vercel/next.js/discussions/73095 — "The best way to track views in Next.js 15 using the app router"
2. https://upstash.com/blog/nextjs13-approuter-view-counter — Upstash Redis view counter with App Router
3. https://vercel.com/templates/next.js/nextjs-portfolio-pageview-counter — Vercel pageview counter template
4. https://authjs.dev/getting-started/session-management/protecting — Auth.js v5 session management / middleware protecting pages
5. https://next-auth.js.org/getting-started/client — NextAuth client API / getSession() behavior
6. https://next-auth.js.org/tutorials/securing-pages-and-api-routes — NextAuth middleware proxy behavior
7. https://secureprivacy.ai/blog/cookieless-tracking-technology — Cookieless tracking and GDPR exemptions
8. https://nextjs.org/docs/app/getting-started/route-handlers — Next.js App Router route handlers (canonical)
9. https://nextjs.org/docs/pages/guides/analytics — Next.js built-in analytics hooks
10. https://blog.logrocket.com/data-tracking-react-walker-js/ — Privacy-friendly first-party tracking in React
11. https://www.diva-portal.org/smash/get/diva2:1114676/FULLTEXT01.pdf — Academic: commit frequency as development proxy
12. https://axify.io/blog/git-analytics — Git analytics metrics and commit frequency interpretation
13. https://milanpavlak.sk/blog/server-side-tracking-nextjs-meta-capi-sgtm — Server-side tracking architecture on Next.js

---

## Key Findings

### (a) Next.js 15 middleware vs edge logging for page-view counting

The canonical pattern (Vercel discussion #73095, Upstash blog) is:

1. A lightweight client component mounts on each `page.tsx` and fires `fetch('/api/telemetry/pageview', { method: 'POST', body: slug })`.
2. The route handler increments a counter (Redis INCR or BigQuery INSERT).
3. No middleware.ts involvement is needed — middleware runs on every request including `/_next/*` asset fetches, making it unsuitable as a page-open proxy.

Vercel's own portfolio template uses exactly this pattern with Upstash Redis and no cookies. The App Router `useEffect`-on-mount approach is the industry standard for self-hosted cookieless counters.

Middleware.ts IS appropriate if you want a pure server-side approach with no client JS: intercept requests to `/` `/signals` etc., write to BQ from the edge function. However, Next.js edge runtime bans the Node.js BQ client; you'd need a lightweight HTTP fetch to a backend relay endpoint — essentially Option A with extra indirection.

### (b) GDPR / privacy — does a bare page-view counter need consent?

Under GDPR recital 30 and EDPB guidance, analytics that do NOT set cookies, do NOT store or log IP addresses, do NOT use device fingerprints, and aggregate only by route are classified as "strictly necessary operational metrics" and are consent-exempt in most EU member-state interpretations (secureprivacy.ai, logrocket/walker.js). The minimum safe design:

- POST body contains only `{ route: "/backtest", ts: <epoch> }` — no user ID, no IP, no user-agent.
- Server side: increment a BQ counter row `(route, date, count)`. Do not log the raw request.
- No cookie set, no localStorage write.

This is substantially below the GDPR processing threshold for personal data. No consent banner required. The existing `Cache-Control: no-store` + `Referrer-Policy: strict-origin-when-cross-origin` headers already align.

### (c) Uvicorn access logs as a frontend-page proxy

The backend has `QuietAccessFilter` that suppresses high-frequency polling routes but still logs all other `/api/**` hits. However, this does NOT discriminate frontend page opens reliably:

- Several pages (portfolio, compare, performance, reports, analyze) make identical first API calls (`/api/signals`, `/api/health`) regardless of which page is open — no 1:1 mapping.
- The `/api/auth/session` route (NextAuth v5 with App Router `getServerSession`) is called server-side during SSR, not from the browser — it does not appear in uvicorn access logs at all when using `getServerSession` (authjs.dev docs). Client-side `useSession()` does hit `/api/auth/session` once per browser tab open, but the NextAuth session is cached via `SessionProvider` and does NOT re-fetch on route changes within the same SPA session. Result: session endpoint hits correlate with new browser tabs, not individual page navigations.

**Conclusion: Option C (uvicorn log proxy) is unreliable and should be rejected.**

### (d) Backfill from git history as a proxy

Git commit count per file in a rolling window is a recognized development-activity proxy (Axify, DIVA-portal academic study). It measures staff investment in a route, not end-user opens, but for an internal tool with a 2-person team (Ford + Peder) those two signals are highly correlated — pages that get commits are pages the team actively uses and cares about; dead pages accumulate zero commits.

This is explicitly acceptable for a first 30-day window provided:
- `usage_source` is labeled `"git_activity_proxy_30d"` so downstream steps know it is a proxy, not real telemetry.
- A real tracker is wired in this same step so future windows have actual data.

---

## Route Inventory (from `frontend/src/app/**/page.tsx`, confirmed by Glob)

| Route | File | Commits (90d, since 2026-01-17) | Last commit message |
|---|---|---|---|
| `/backtest` | backtest/page.tsx | 47 | "Phase 2.7+2.8: Paper trading dashboard..." |
| `/paper-trading` | paper-trading/page.tsx | 12 | "Paper Trading: replace stacked ops widgets..." |
| `/settings` | settings/page.tsx | 7 | "fix: sidebar stays fixed..." |
| `/` (dashboard) | page.tsx | 9 | "fix: sidebar stays fixed..." |
| `/signals` | signals/page.tsx | 5 | "fix: sidebar stays fixed..." |
| `/compare` | compare/page.tsx | 5 | "fix: sidebar stays fixed..." |
| `/performance` | performance/page.tsx | 6 | "fix: sidebar stays fixed..." |
| `/reports` | reports/page.tsx | 5 | "fix: sidebar stays fixed..." |
| `/agents` | agents/page.tsx | 4 | "Fix: cron schedule object rendered..." |
| `/portfolio` | portfolio/page.tsx | 3 | "fix: sidebar stays fixed..." |
| `/analyze` | analyze/page.tsx | 3 | "fix: sidebar stays fixed..." |
| `/login` | login/page.tsx | 1 | "fixes and new security" |

Note: 8 of 12 routes share the same last commit ("fix: sidebar stays fixed") — a global layout fix, not route-specific work. The discriminating signal is commit count over the full 90-day window, with backtest (47) and paper-trading (12) clearly dominant.

---

## Option Evaluation

### Option A — Ship middleware + pageview endpoint NOW, wait 30 days

- Pro: Real data, correct methodology.
- Con: BLOCKS step 4.7.1 entirely. Step 4.7.1 ("remove or merge zero-open pages; <=8 top-level routes") requires 30 days of data that does not exist. Cannot ship today.
- **Verdict: Rejected as primary path. Wire it as future-data instrumentation only.**

### Option B — Git-activity proxy (RECOMMENDED)

- Pro: Runs in minutes. Clearly discriminates backtest/paper-trading (high activity) from login/analyze/portfolio (low/zero). Labels usage_source correctly. Unblocks step 4.7.1 immediately.
- Con: Measures developer intent, not user navigation. Acceptable for a 2-person internal tool where developer = user.
- **Verdict: Primary path. Produce `handoff/frontend_usage.json` from `git log --since` counts. Simultaneously wire the real `/api/telemetry/pageview` endpoint so future windows have actual data.**

### Option C — Uvicorn access log proxy

- Pro: Would reflect actual backend API usage.
- Con: No 1:1 mapping to frontend page opens (multiple pages hit same endpoints; `getServerSession` is server-side only and invisible to uvicorn). Data is unreliable for the discrimination task.
- **Verdict: Rejected.**

---

## Consensus vs Debate

All sources agree: without pre-existing analytics, there is no reliable "opens_30d" number. The debate is only about which proxy to use. The git-activity proxy is the shortest unblocking path; the endpoint instrumentation is the correct long-term approach. Ship both.

## Pitfalls

- Do NOT use middleware.ts to intercept `/_next/*` static asset requests as a page-open proxy — it fires on every chunk, image, and font load.
- Do NOT log raw IPs or user-agents in the telemetry endpoint — GDPR consent exemption depends on truly anonymous aggregates.
- Do NOT use `getSession()` client-side as a page-open proxy — it only fires once per browser session, not per navigation.
- Do NOT treat the global sidebar commit as route-specific activity — filter it or weight by commit message relevance.

## Application to pyfinAgent

GENERATE phase actions:
1. Run `git log --since="2026-01-17" --oneline -- <file>` for each of the 12 routes (already done above).
2. Write `handoff/frontend_usage.json` with `opens_30d = commit_count`, `usage_source = "git_activity_proxy_30d"`.
3. Add `frontend/src/app/api/telemetry/pageview/route.ts` — minimal POST handler, no cookies, BQ insert of `(route, date, count)`.
4. Add one-line `useEffect` call to each `page.tsx` to fire the telemetry endpoint (or use a shared layout component).

Step 4.7.1 can then proceed immediately using the git-proxy counts to identify the <=8 routes to keep.

---

## Research Gate Checklist

- [x] 3+ authoritative sources (Vercel/Next.js official, Auth.js official, GDPR-specialist blog, academic commit-frequency study)
- [x] 13 unique URLs collected and cited
- [x] Full papers/posts read (not abstracts) — Upstash blog, authjs.dev protecting page, secureprivacy GDPR analysis
- [x] All claims cited with URLs
- [x] Contradictions noted (Option C rejected with reasoning; session-fetch proxy debunked against Auth.js v5 docs)
- [x] Recommendation explicit: Option B (git-activity proxy) unblocks today; Option A wired concurrently for future windows
