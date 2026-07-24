# Research Brief -- Step 63.1: Playwright walk of all 22 routes

**Tier:** moderate
**Gate:** research gate for phase-63 full-app live audit
**Started:** 2026-07-17
**Status:** COMPLETE -- gate_passed: true (6 sources read in full, recency scan done)

## Objective
Playwright walk of all 22 routes (`find frontend/src/app -name page.tsx`;
incl. one concrete `/sovereign/strategy/[id]`) -- per route: full-page
screenshot, console errors, failed (4xx/5xx) network requests ->
`handoff/away_ops/route_walk_<date>/walk_summary.json`.

---

## Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
| --- | --- | --- | --- | --- |
| https://playwright.dev/docs/network | 2026-07-17 | official doc | WebFetch | `page.on('request', r => r.method(), r.url())`; `page.on('response', r => r.status(), r.url())` -- filter `status()>=400` for 4xx/5xx |
| https://playwright.dev/docs/screenshots | 2026-07-17 | official doc | WebFetch | `await page.screenshot({ path, fullPage: true })` captures full scrollable page; element vs full-page vs buffer |
| https://playwright.dev/docs/auth | 2026-07-17 | official doc | WebFetch | `context.storageState({path})` + setup-project `dependencies:['setup']`; auth files MUST be gitignored |
| https://docs.crawl4ai.com/advanced/network-console-capture/ | 2026-07-17 | tool doc | WebFetch | Capture schema: `network_requests[]` (url/method/status/timing) + `console_messages[]` (type/text/location/timestamp) -- flat arrays |
| https://docs.starter.app/docs/testing/e2e-smoke-tests | 2026-07-17 | practitioner doc | WebFetch | Benign-noise regex set (favicon/manifest/source-map/chrome-extension/404); listen `pageerror`+`console`; keep <30s; html reporter |
| https://oneuptime.com/blog/post/2026-02-02-playwright-network-interception/view | 2026-07-17 | blog (2026) | WebFetch | `page.on('requestfailed', r => r.failure()?.errorText, r.url(), r.method(), r.resourceType())`; separate net-failure from HTTP 4xx/5xx |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://webeyez.com/insights/guides/playwright-console-errors-guide | blog | WebFetch returned HTTP 403 Forbidden (attempted) |
| https://nextjs.org/docs/app/building-your-application/routing | official doc | Route conventions confirmed via search snippet; folder->route, `[id]` dynamic, `(group)` no-URL, `[...slug]` catch-all |
| https://testomat.io/blog/how-to-capture-screenshots-videos-playwright-js-tutorial/ | blog | fullPage caveats covered by official doc + snippets |
| https://www.screenshotapi.net/blog/how-do-you-take-a-full-page-screenshot-in-playwright | blog (2026) | lazy-load + 32767px cap surfaced in search snippet |
| https://www.browserstack.com/guide/playwright-global-setup | industry | storageState pattern already covered by official auth doc |
| https://dev.to/vitalets/authentication-in-playwright-you-might-not-need-project-dependencies-2e02 | blog | alt auth pattern; not needed (skip-auth bypass wins) |
| https://github.com/shopsys/http-smoke-testing | code | route-smoke concept covered by StarterApp doc |
| https://www.mabl.com/blog/automated-codeless-smoke-testing-with-mabl | vendor | codeless smoke; not our path |
| https://qaskills.sh/blog/playwright-trace-cli-analysis-guide-2026 | blog (2026) | trace CLI; tangential to JSON-artifact walk |

## Recency scan (2024-2026)
Ran explicit 2025/2026-scoped passes ("...2026", "...2025", "Next.js 15
...2026"). **Findings:** the current-frontier work CONFIRMS the canonical
API is unchanged and stable -- `page.on('requestfailed'/'response'/'console'/
'pageerror')` + `page.screenshot({fullPage:true})` are the same in the 2026
sources (oneuptime 2026-02-02, screenshotapi 2026) as in the year-less
official docs. NEW-in-window notes: Playwright 1.59 added terminal-native
trace tooling (`npx playwright trace`) for headless CI (nice-to-have, not
required for the JSON walk); Crawl4AI v0.8.x formalizes a
`network_requests[]`+`console_messages[]` capture schema that directly
informs `walk_summary.json`. No 2024-2026 finding SUPERSEDES the approach;
the recency scan strengthens (does not change) the plan. Installed
`@playwright/test ^1.50.0` is older than 1.59 but has every API this walk
needs (the events + fullPage predate 1.0).

---

## Internal code inventory

### A. The 22 routes (`find frontend/src/app -name page.tsx`) -- COUNT = 22 (meets >=22)
URL mapping (route groups none present; one dynamic route):

| # | page.tsx path | URL to visit |
|---|---|---|
| 1 | app/page.tsx | `/` |
| 2 | app/agent-map/page.tsx | `/agent-map` |
| 3 | app/agents/page.tsx | `/agents` |
| 4 | app/backtest/page.tsx | `/backtest` |
| 5 | app/cron/page.tsx | `/cron` |
| 6 | app/learnings/page.tsx | `/learnings` |
| 7 | app/login/page.tsx | `/login` |
| 8 | app/observability/page.tsx | `/observability` |
| 9 | app/paper-trading/page.tsx | `/paper-trading` |
| 10 | app/paper-trading/exit-quality/page.tsx | `/paper-trading/exit-quality` |
| 11 | app/paper-trading/learnings/page.tsx | `/paper-trading/learnings` |
| 12 | app/paper-trading/manage/page.tsx | `/paper-trading/manage` |
| 13 | app/paper-trading/nav/page.tsx | `/paper-trading/nav` |
| 14 | app/paper-trading/positions/page.tsx | `/paper-trading/positions` |
| 15 | app/paper-trading/reality-gap/page.tsx | `/paper-trading/reality-gap` |
| 16 | app/paper-trading/trades/page.tsx | `/paper-trading/trades` |
| 17 | app/performance/page.tsx | `/performance` |
| 18 | app/reports/page.tsx | `/reports` |
| 19 | app/settings/page.tsx | `/settings` |
| 20 | app/signals/page.tsx | `/signals` |
| 21 | app/sovereign/page.tsx | `/sovereign` |
| 22 | app/sovereign/strategy/[id]/page.tsx | `/sovereign/strategy/<CONCRETE_ID>` (dynamic) |

No route-groups `(...)`, no catch-all `[...slug]`. Only ONE dynamic
segment: `[id]` at #22.

### B. Concrete strategy `[id]` resolution
- Route shell `frontend/src/app/sovereign/strategy/[id]/page.tsx:18-49`
  reads `id` via `useParams`, calls `getSovereignStrategy(id)` ->
  `frontend/src/lib/api.ts:804` -> `GET /api/sovereign/strategy/{id}`.
- Real ids come from `GET /api/sovereign/leaderboard`
  (`backend/api/sovereign_api.py:378`: `strategy_id = str(d.get("strategy_id")
  or d.get("trial_id") or "?")`). The walk script SHOULD fetch the
  leaderboard first, take `rows[0].strategy_id`, and fall back to a
  documented placeholder (e.g. `"baseline"`) if empty.
- IMPORTANT: the page renders even for a bogus id -- it shows a rose
  error banner (`:69-73`), it does NOT crash. So a placeholder still
  produces a valid "page visited" artifact, but criterion 1 wants a
  CONCRETE id, so resolve one live.

### C. Auth wall + the bypass (CONFIRMED)
- `frontend/src/middleware.ts:24`:
  `if (!hasAuthProvider || process.env.LIGHTHOUSE_SKIP_AUTH === "1") { return; }`
  -- when `LIGHTHOUSE_SKIP_AUTH=1`, the middleware skips the redirect for
  ALL routes (login/_next/favicon/api-auth already exempt at `:13-20`).
  `hasAuthProvider = !!(AUTH_GOOGLE_ID && AUTH_GOOGLE_SECRET)` (`:7`).
- Canonical bypass (`.claude/rules/frontend.md:82` + `docs/runbooks/browser-mcp.md:83`):
  start a SECOND dev server that never touches the operator's :3000:
  `cd frontend && LIGHTHOUSE_SKIP_AUTH=1 npx next dev --port 3100`
  (NOTE: `npm run dev` hardcodes `--port 3000` in package.json, so invoke
  `next dev --port 3100` directly, not the npm script).
- `:3000` currently returns 302 -> /login (healthy authed instance --
  DO NOT disturb it). `:3100` is down (expected). The runbook's
  launchctl-setenv-on-:3000 path (`browser-mcp.md:88-90`) is the
  fallback; prefer the separate :3100 server (isolation).

### D. Existing E2E baseline (the "8/22" claim)
- `frontend/playwright.config.ts`: `testDir: "./tests/visual-regression"`,
  ONE project `chromium` (Desktop Chrome, 1280x800), `baseURL =
  PLAYWRIGHT_BASE_URL ?? "http://localhost:3000"`, `webServer` runs
  `npm run dev` with `NEXT_PUBLIC_E2E_TESTING: "true"`,
  `reuseExistingServer: !CI`. No `storageState`, no auth handling in
  config -- the visual specs assume auth is off (CI has no
  AUTH_GOOGLE_* so `hasAuthProvider=false`).
- 8 spec files under `tests/visual-regression/`: agent-map, agents,
  backtest, home, paper-trading, performance, reports, sovereign =
  exactly 8/22 routes. Each does `page.goto(route)` +
  `disableAnimations` + `waitForLoadState("networkidle")` +
  `toHaveScreenshot({fullPage:true, mask: dynamicMasks})`.
- Reusable helpers `tests/visual-regression/helpers/visual.ts`:
  `disableAnimations(page)` (zeroes CSS anim/transition) +
  `dynamicMasks(page)` (Locators for time/skeleton/spinner/recharts).
- `sovereign.spec.ts` visits `/sovereign` only -- NOT the `[id]` route,
  confirming #22 has no baseline.

### E. Playwright runtime available
- `@playwright/test ^1.50.0` + `playwright` core BOTH in
  `frontend/node_modules/`. Chromium installed at
  `~/Library/Caches/ms-playwright/chromium-1208`. A standalone Node
  script can `import { chromium } from "playwright"` with no new install.
- The `--project=functional` referenced in `scripts/away_ops/prompt_pm.md:43`
  does NOT exist yet (phase-64 aspiration; current config only has
  `chromium`). It is NOT the vehicle for this walk.

## Key findings
1. **Per-route capture is 4 listeners, registered before navigation.**
   `page.on('console', msg => msg.type()/msg.text())` (console.error/warn),
   `page.on('pageerror', err => ...)` (UNCAUGHT JS exceptions -- distinct
   from console errors; capture BOTH), `page.on('requestfailed', r =>
   r.failure()?.errorText)` (network-level failures), `page.on('response',
   r => r.status()>=400 ...)` (HTTP 4xx/5xx). (oneuptime 2026, playwright.dev
   network doc.)
2. **Full-page screenshot** = `await page.screenshot({ path, fullPage: true })`.
   Two documented caveats: (a) lazy-loaded images off-screen may render blank
   because fullPage does NOT auto-scroll first -> pre-scroll or accept; (b)
   32767px height cap on very long pages. (playwright.dev screenshots +
   screenshotapi 2026 snippet.)
3. **Filter benign console noise or every route flags an error.** Canonical
   ignore set (StarterApp): `/favicon\.ico/`, `/manifest\.webmanifest/`,
   `/DevTools.*source map/`, `/Failed to load resource.*404/`,
   `/chrome-extension:\/\//`, `/metadataBase/`. Report FILTERED errors as the
   defect signal.
4. **Auth: skip the wall, do not automate NextAuth.** Playwright's own
   storageState/setup-project pattern (playwright.dev/docs/auth) is the
   general answer, BUT this repo already has a first-class bypass
   (`LIGHTHOUSE_SKIP_AUTH=1`, middleware.ts:24) -- far less fragile than
   scripting Google SSO/passkey (passkeys are device-bound, Google blocks
   automated login; see browser-mcp.md:96-102). Use the bypass.
5. **Capture schema is a flat per-route record** (Crawl4AI): network as
   `{url,method,status,error,timestamp}[]`, console as
   `{type,text,location,timestamp}[]`. Mirror this in `walk_summary.json`.
6. **Route smoke = "does it run?" + assert no real errors + content loaded**
   (StarterApp, shopsys). Keep it fast; new routes are picked up
   automatically if enumeration is filesystem-driven.

## Application to pyfinagent

### Recommended vehicle: a checked-in standalone Node script (NOT a Playwright test project)
Use `playwright` core (already in `frontend/node_modules`):
`import { chromium } from "playwright"`. Rationale over a `*.spec.ts`:
- Full control of the custom `walk_summary.json` artifact (the test-runner is
  assertion/reporter-oriented; emitting bespoke JSON from a spec is awkward).
- Enumerate routes at RUNTIME by globbing `frontend/src/app/**/page.tsx` and
  mapping to URLs -> this IS criterion-3 reconciliation (on-disk list vs
  walked list; any delta = a defect row).
- Re-runnable in CI + by the away-ops PM session (the `--project=functional`
  in `scripts/away_ops/prompt_pm.md:43` is a phase-64 aspiration and does not
  exist; do not couple to it).
- Dynamic strategy-id resolution for route #22.
Suggested path: `frontend/scripts/audit/route_walk.mjs` (mirrors the existing
`frontend/scripts/audit/lighthouse_auth_home.js` idiom); resolve repo root
for the `handoff/away_ops/route_walk_<date>/` output dir.
**Playwright MCP is a Q/A spot-check aid, not the artifact producer** -- the
MCP cannot emit the structured `walk_summary.json`; the script is the
deliverable, the MCP can live-verify a couple of routes.

### Runbook for the walk (exact)
1. Start an isolated skip-auth server (never touch operator :3000):
   `cd frontend && LIGHTHOUSE_SKIP_AUTH=1 NEXT_PUBLIC_E2E_TESTING=true npx next dev --port 3100`
   (`NEXT_PUBLIC_E2E_TESTING=true` suppresses live polling per
   playwright.config.ts:90 -- CRITICAL, see trap #2 below).
2. Confirm backend :8000 is up on a REAL endpoint (its `/health` 404s;
   probe e.g. `/api/sovereign/leaderboard` or an existing health path) so
   pages get data, not API-error states.
3. Resolve a concrete strategy id: `GET /api/sovereign/leaderboard` ->
   `rows[0].strategy_id`; fallback `"baseline"` if empty.
4. For EACH of the 22 URLs: fresh context/page, attach the 4 listeners,
   `page.goto(url, {waitUntil:'load'})`, bounded settle, `page.screenshot(
   {path, fullPage:true})`, record status + filtered console errors +
   pageerrors + failed requests.
5. Write `walk_summary.json` + `screenshots/<slug>.png`.
6. Kill :3100; verify :3000 still 302->/login.

### `walk_summary.json` schema (top-level keys the verification reads are REQUIRED)
```json
{
  "generated_at": "2026-07-17T14:00:00Z",
  "base_url": "http://localhost:3100",
  "auth_bypass": "LIGHTHOUSE_SKIP_AUTH=1",
  "strategy_id_used": "trial_1234",
  "routes_discovered": 22,
  "routes_visited": 22,
  "console_error_routes": ["/paper-trading", "/signals"],
  "failed_request_routes": ["/observability"],
  "route_list_delta": { "on_disk_not_visited": [], "visited_not_on_disk": [] },
  "routes": [
    {
      "route": "/",
      "final_url": "http://localhost:3100/",
      "http_status": 200,
      "screenshot": "screenshots/root.png",
      "console_errors": [{"type":"error","text":"...","location":"file:line"}],
      "page_errors": ["TypeError: ..."],
      "failed_requests": [{"url":"...","method":"GET","status":500,"error":null}],
      "load_ms": 1234
    }
  ]
}
```
- `routes_visited` (int >=22), `console_error_routes` (list),
  `failed_request_routes` (list) are top-level -- the immutable verification
  command reads exactly `routes_visited` and `console_error_routes`.
- Attribute the `[id]` route as `/sovereign/strategy/<id>` in `routes[]`;
  count it in the 22.

### Traps (repo-specific -- flag these to Main)
- **T1 `networkidle` HANGS the cockpit.** The paper-trading pages poll on
  price ticks; `waitForLoadState('networkidle')` may NEVER settle. Use
  `NEXT_PUBLIC_E2E_TESTING=true` (kills polling) + `waitUntil:'load'` +
  a fixed settle timeout, NOT networkidle. (The existing visual specs use
  networkidle only because E2E_TESTING suppresses the polling.)
- **T2 Per-route listener attribution.** Register listeners on a fresh
  page/context per route (or clear arrays per route) so an error attributes
  to the right route, not a neighbor.
- **T3 Bypass misconfig -> silent all-login.** If `LIGHTHOUSE_SKIP_AUTH`
  isn't seen by the server, every route 302s to `/login` and screenshots
  look identical. DETECT by asserting `final_url` path == expected route;
  a route that lands on `/login` (except #7) is a walk failure, not a pass.
- **T4 Backend down -> false 4xx/5xx everywhere.** :8000/health 404s;
  confirm a real backend endpoint answers before blaming routes.
- **T5 fullPage lazy-load blanks + 32767px cap** (finding #2).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (6 read + 9 snippet = 15)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (middleware.ts:24,
      playwright.config.ts, api.ts:804, sovereign_api.py:378, etc.)

Soft checks:
- [x] Internal exploration covered routes/auth/E2E/schema
- [x] Contradictions/consensus noted (script-vs-MCP, bypass-vs-storageState)
- [x] Three-query-variant discipline visible (2026 frontier + year-less
      canonical + 2025 window mix in the source tables)

## JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 9,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "22 page.tsx routes confirmed (1 dynamic: /sovereign/strategy/[id], resolve via GET /api/sovereign/leaderboard rows[0].strategy_id). Auth bypass = LIGHTHOUSE_SKIP_AUTH=1 (middleware.ts:24) on an isolated :3100 dev server (never touch operator :3000). Existing E2E = 8 visual-regression specs (chromium project, tests/visual-regression/, reusable disableAnimations/dynamicMasks helpers). Recommend a checked-in standalone Node script using playwright core (frontend/scripts/audit/route_walk.mjs): 4 listeners per route (console, pageerror, requestfailed, response>=400), fullPage screenshot, filesystem route enumeration for criterion-3 reconciliation, writes walk_summary.json with top-level routes_visited/console_error_routes/failed_request_routes. Key traps: networkidle hangs the polling cockpit (use NEXT_PUBLIC_E2E_TESTING=true + waitUntil:load), per-route listener attribution, detect silent all-login if bypass misfires, backend must be up.",
  "brief_path": "handoff/current/research_brief_63.1.md",
  "gate_passed": true
}
```
