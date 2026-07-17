# Research Brief — Step 64.1: Functional-E2E Playwright project

**Tier:** moderate
**Gate role:** Research gate before contract.md for phase-64 step 64.1
**Status:** IN PROGRESS (write-first; appended incrementally)

## Objective (verbatim)
"Functional-E2E Playwright project -- new testDir tests/e2e-functional as a
second project in frontend/playwright.config.ts; assertion style: primary data
region renders, zero console.error, zero 5xx -- NO screenshot comparisons, so
the Linux-baseline caveat does not apply and the suite runs on the Mac."

## Immutable success criteria (verbatim)
1. functional project exists in playwright.config.ts with testDir
   tests/e2e-functional and no screenshot assertions
2. a smoke spec passes on the Mac against the port-3100 auth-bypass server
3. NEXT_PUBLIC_E2E_TESTING polling suppression is honored per the existing config note

## Immutable verification command
`cd frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line --grep smoke`

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/playwright.config.ts` | 1-93 | Playwright config, ONE project (`chromium`, testDir `./tests/visual-regression`) + webServer | ACTIVE; must add 2nd project |
| `frontend/scripts/audit/route_walk.mjs` | 1-250 | phase-63.1 route-walk (DIRECT BLUEPRINT: console+5xx+requestfailed capture, benign() allowlist) | ACTIVE; reuse patterns |
| `frontend/tests/visual-regression/helpers/visual.ts` | 1-42 | `disableAnimations(page)`, `dynamicMasks(page)` | Reusable (animations only; NO auth helper exists) |
| `frontend/tests/visual-regression/*.spec.ts` | ~12 ea | 8 screenshot specs (`toHaveScreenshot`) | Do NOT touch; separate project |
| `frontend/src/middleware.ts` | 9-33 | Auth bypass: `LIGHTHOUSE_SKIP_AUTH==="1"` OR no provider → no redirect | Confirmed bypass path |
| `frontend/src/lib/api.ts` | 43-102 | `API_BASE = NEXT_PUBLIC_API_URL || http://localhost:8000` (client hits backend DIRECTLY) | Backend dep for data |
| `frontend/src/lib/live-portfolio-context.tsx` | 84,144 | `setInterval(refresh, 60_000)` — UNCONDITIONAL, does NOT read E2E flag | See criterion-3 finding |

### Existing config anatomy (playwright.config.ts)
- Single project `chromium`, `testDir: "./tests/visual-regression"`, viewport 1280x800, `contextOptions.reducedMotion: "reduce"` at project level.
- `webServer`: `command: "npm run dev"` (→ `next dev --port 3000`), `url: BASE_URL` (`PLAYWRIGHT_BASE_URL ?? http://localhost:3000`), `reuseExistingServer: !CI`, `timeout: 120_000`, `env: { NEXT_PUBLIC_E2E_TESTING: "true" }`.
- `expect.toHaveScreenshot` + `snapshotDir`/`snapshotPathTemplate` are visual-only; the functional project inherits none of these (no `toHaveScreenshot` call = no baseline needed).
- **macOS-baseline caveat is at lines 4-14** (comment): Linux-only baselines. Because the functional project makes NO screenshot assertions, this caveat does not apply to it — confirmed. It runs on the Mac.
- `@playwright/test` INSTALLED version = **1.60.0** (package.json pins `^1.50.0`). All multi-project / webServer-array / tag APIs below are supported.

### CRITICAL finding — criterion 3 (`NEXT_PUBLIC_E2E_TESTING`)
`NEXT_PUBLIC_E2E_TESTING` is **injected** by the existing webServer env (line 90) and by route_walk.mjs (`LIGHTHOUSE_SKIP_AUTH=1 NEXT_PUBLIC_E2E_TESTING=true next dev --port 3100`), and route_walk.mjs:18 calls it "load-bearing" — BUT a repo-wide grep shows **NO app-side consumer**: `live-portfolio-context.tsx:144` polls `setInterval(60_000)` unconditionally, never reading the flag. So "polling suppression is honored per the existing config note" = **replicate the env injection** (set `NEXT_PUBLIC_E2E_TESTING=true` on the functional webServer, matching the config note + 63.1 precedent). It is a HARNESS-side contract, not an app-code consumer today. Practical robustness: the poll interval is 60s, so a fast `@smoke` test (<60s) never triggers an interval poll — only the on-mount fetch fires. Recommend honoring the note by injection AND (optional, low-risk) noting the app-side gap for a future step.

### Backend dependency + "zero 5xx" interaction
`api.ts` fetches go DIRECTLY to `http://localhost:8000` (not proxied through Next). `page.on('response')` in Playwright DOES observe these cross-origin XHR responses. Implication: "zero 5xx" only trips on an actual 5xx status from :8000; a DOWN backend yields `requestfailed`/connection-refused (net::ERR), not a 5xx — but that WOULD surface as a console.error from api.ts's error path. Design guidance below scopes the smoke assertion to the page shell + a `benign()`/expected-API allowlist so the smoke is robust; deep data-region assertions belong in per-route functional specs that assume :8000 is up.

### Route inventory (src/app) + stable selectors
Routes: `/`, `/paper-trading`(+7 subroutes), `/performance`, `/backtest`, `/agents`, `/agent-map`, `/sovereign`, `/reports`, `/signals`, `/observability`, `/cron`, `/learnings`, `/settings`, `/login`.
Stable text anchors confirmed:
- Home `/`: `<h2>MAS Operator Cockpit</h2>` (page.tsx:339) + `role="group" aria-label="Portfolio key performance indicators"` (page.tsx:383).
- `/paper-trading`: `<h2>Paper Trading</h2>` (layout.tsx:334).
These are static (render even if backend is down / data errors), so they are ideal "primary data region renders" anchors for a smoke test.

### Reusable helper API
- `disableAnimations(page): Promise<void>` — injects zero-duration CSS. Reusable in functional specs (harmless).
- `dynamicMasks(page): Locator[]` — screenshot-mask helper; NOT needed for functional (no screenshots).
- No auth/login helper exists — none needed; the bypass is env-var-driven (`LIGHTHOUSE_SKIP_AUTH=1`), handled at the webServer/env layer, not per-spec.

### route_walk.mjs reusable patterns (the blueprint)
- `benign(url)` allowlist: favicon, `.map`, `manifest`, `chrome-extension://`, `hot-update`, `/__nextjs`/`/_next/webpack-hmr` → filter these OUT of the 5xx/failed check (prevents dev-HMR false positives).
- `page.on("console")` → collect `type==='error'` (walk also collects `warning`; smoke should assert on `error` ONLY to avoid React dev-warning flakes).
- `page.on("response")` → `status() >= 500` for the "zero 5xx" assertion (walk uses >=400; 5xx is the criterion's bar).
- `page.on("requestfailed")` → optional; connection-refused noise if backend down.
- Bypass-misfire guard: if final URL is `/login`, the bypass didn't take → fail loudly.

---

## External research (read-in-full below)

### Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://playwright.dev/docs/test-projects | 2026-07-17 | official doc | WebFetch full | Multiple projects supported; `--project=NAME` runs one; per-project `testDir`/`testMatch`/`use` valid; top-level config shared |
| https://playwright.dev/docs/api/class-testproject | 2026-07-17 | official doc | WebFetch full | CONFIRMS per-project `testDir`: "Each project can use a different directory." testDir/testMatch/testIgnore/name/use/grep/retries/dependencies all valid per-project |
| https://playwright.dev/docs/test-webserver | 2026-07-17 | official doc | WebFetch full | webServer keys command/url/reuseExistingServer/env/timeout/cwd/name; `reuseExistingServer:true` reuses if up else starts; `env` inherits process.env + PLAYWRIGHT_TEST=1; webServer CAN be an ARRAY |
| https://playwright.dev/docs/api/class-page | 2026-07-17 | official doc | WebFetch full | `page.on('console', m=>m.type()==='error')` + `page.on('response', r=>r.status()>=500)` + `page.on('pageerror',...)` — exact console/5xx capture |
| https://playwright.dev/docs/test-annotations | 2026-07-17 | official doc | WebFetch full | `@smoke` via title-embed OR `{ tag:'@smoke' }`; `--grep @smoke` / `--grep smoke` selects; OR=`@a\|@b`, AND=lookaheads |
| https://nextjs.org/docs/app/guides/testing/playwright | 2026-07-17 | official doc (updated 2026-02-11) | WebFetch full | Official Next.js Playwright guide; recommends prod build (`next build`+`start`) but `webServer` dev-start is an explicit supported alt; set `baseURL` so specs use relative `page.goto('/')` |
| https://timdeschryver.dev/blog/create-and-run-playwright-test-sets-using-tags-and-grep | 2026-07-17 | practitioner blog | WebFetch full | `--grep` matches the test DESCRIPTION (incl. embedded tags); `@smoke` = "simple, fast, read-only checks"; `test.describe('@smoke', ...)` group-tag works |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://github.com/microsoft/playwright/issues/23770 | issue | Confirms per-project testDir example in search snippet; API doc already authoritative |
| https://qaskills.sh/blog/playwright-test-config-options-complete-reference | blog (2026) | "Complete 2026 Reference" recency signal; official docs preferred |
| https://testdino.com/blog/grouping-playwright-tests | blog (2025) | describe/tags/grep overview; tim-deschryver covers it |
| https://dev.to/playwright/tagging-your-playwright-tests-3omm | blog | Tag-object syntax; annotations doc authoritative |
| https://github.com/vercel/next.js/tree/canary/examples/with-playwright | example repo | Canonical Next+Playwright scaffold; official doc covers config |
| https://playwright.dev/docs/api/class-testconfig | official doc | webServer/projects reference; covered by test-webserver + test-projects |
| https://dev.to/playwright/setup-a-local-dev-server-for-your-playwright-tests-33m9 | blog | webServer local-dev pattern; official doc authoritative |
| https://stevekinney.com/courses/self-testing-ai-agents/playwright-web-server-without-surprises | course | reuseExistingServer gotchas; official doc authoritative |

### Recency scan (2024-2026)
Searched 2026 frontier ("...2026"), 2025 window ("Next.js 15 App Router Playwright ... 2025"), and year-less canonical (webServer, tags/grep). **Result:** no new finding supersedes the canonical multi-project + array-webServer + `--grep` approach — it is stable across 2024-2026. Two recency notes that COMPLEMENT the plan: (1) the tag-OBJECT API `test('name', { tag:'@smoke' }, ...)` is v1.42+ (2024) and is matched by `--grep` alongside title-embedded tags — installed Playwright is 1.60.0 so both styles are available; (2) the official Next.js guide was refreshed 2026-02-11 and still frames `webServer` dev-start as the supported alternative to prod-build for E2E. No 2024-2026 source contradicts running a functional (non-screenshot) suite on macOS.

### Key findings (per-claim cited)
1. **Per-project `testDir` is first-class** — "Each project can use a different directory" (Source: playwright.dev/docs/api/class-testproject, 2026-07-17). So a second project `functional` with `testDir: "./tests/e2e-functional"` sits beside the existing `chromium` visual project in one config. Satisfies criterion 1.
2. **`--project=functional` runs ONLY that project** — "Use the `--project` command line option to run a single project" (Source: playwright.dev/docs/test-projects, 2026-07-17). The immutable command's `--project=functional` is valid.
3. **webServer is a config-level array; entries auto-start/reuse** — "webServer accepts an array"; `reuseExistingServer:true` "re-use an existing server ... if no server is running ... run the command" (Source: playwright.dev/docs/test-webserver, 2026-07-17). Lets the config boot the :3100 skip-auth server standalone, reusing an already-running one.
4. **webServer `env` inherits process.env** — "Defaults to inheriting `process.env`" (Source: playwright.dev/docs/test-webserver, 2026-07-17). So `LIGHTHOUSE_SKIP_AUTH=1` on the immutable command is inherited by the spawned `next dev`; an explicit `env` block double-covers it and adds `NEXT_PUBLIC_E2E_TESTING`.
5. **Console + 5xx are captured via page events** — `ConsoleMessage.type()==='error'` and `Response.status()>=500` (Source: playwright.dev/docs/api/class-page, 2026-07-17). Exact primitives for the two non-visual assertions; `benign()` allowlist from route_walk.mjs filters HMR/favicon noise.
6. **`--grep smoke` matches the test title** — grep "searches test descriptions (including tags)" (Source: timdeschryver.dev, 2026-07-17; playwright.dev/docs/test-annotations, 2026-07-17). A spec whose title contains "smoke" (and/or `{tag:'@smoke'}`) is selected by the immutable `--grep smoke`.
7. **Next.js supports dev-server E2E via webServer, baseURL enables relative goto** — "use the webServer feature to let Playwright start the development server"; add `baseURL` so `page.goto('/')` works (Source: nextjs.org/docs/app/guides/testing/playwright, updated 2026-02-11). pyfinagent's :3100 skip-auth dev server is the established variant (route_walk.mjs); prod-build is the Next default but the auth-bypass tooling is dev-oriented.

### Consensus vs debate (external)
- **Consensus:** multi-project + per-project testDir, `--project` selection, array webServer with `reuseExistingServer:!CI`, `--grep` tag filtering, `expect(locator).toBeVisible()` for functional assertions, page-event capture for console/network — all stable, documented, uncontested 2024-2026.
- **Debate/tradeoff:** Next.js officially prefers testing a PRODUCTION build (`next build`+`next start`) for fidelity; pyfinagent uses `next dev --port 3100` because the auth bypass (`LIGHTHOUSE_SKIP_AUTH`) + polling flag are dev-oriented and 63.1 already proved the dev path. Dev mode adds HMR console/network noise → mitigated by the `benign()` allowlist. This is a deliberate, reasonable deviation for a smoke suite.

### Pitfalls (from literature + internal)
- **webServer is GLOBAL, not per-project** (test-webserver doc is silent on per-project scope; community confirms global). Running `--project=functional` boots ALL array webServers. Locally harmless: :3000 "always running" → `reuseExistingServer:!CI=true` reuses it instantly; :3100 starts fresh or reuses. On CI (`CI=true`→reuse=false) both would boot — but this project is Mac/local-only, so the CI double-boot is moot.
- **Dev-mode console noise** can flake a strict "zero console.error" — filter with the route_walk `benign()` allowlist and assert on `type()==='error'` ONLY (not 'warning').
- **Backend :8000 dependency**: api.ts fetches go directly to :8000; a down/5xx backend trips zero-5xx or emits console.error. CLAUDE.md mandates :8000 always up, so smoke is green in normal ops; note the dependency in the spec header. (Full backend independence would need `page.route('**/api/**', stub)` — heavier than a smoke needs.)
- **`npm run dev` is hardcoded to :3000** — the :3100 webServer entry MUST call `npx next dev --port 3100` directly.
- **Chromium browser install**: `npx playwright test` needs the chromium binary. Visual-regression already uses chromium so it should be present; if a fresh env, `npx playwright install chromium` is the pre-req.

## Application to pyfinagent

### (a) EXACT playwright.config.ts change shape
Two edits, both in `frontend/playwright.config.ts`:

**Edit 1 — add the `functional` project to the `projects` array** (after the existing `chromium` entry):
```ts
const FUNCTIONAL_BASE = process.env.PLAYWRIGHT_FUNCTIONAL_BASE_URL ?? "http://localhost:3100";
// ... inside projects: [ { name:"chromium", ... },  ADD: ]
{
  name: "functional",
  testDir: "./tests/e2e-functional",
  use: {
    ...devices["Desktop Chrome"],
    baseURL: FUNCTIONAL_BASE,
    viewport: { width: 1440, height: 900 },   // matches route_walk.mjs viewport
    contextOptions: { reducedMotion: "reduce" },
  },
},
```
No `expect.toHaveScreenshot`, no `snapshotDir` for this project → zero screenshot assertions → the Linux-baseline caveat (config lines 4-14) does NOT apply. Satisfies criterion 1.

**Edit 2 — convert the single `webServer` object to an ARRAY of two** (keep existing :3000 entry unchanged for visual-regression; add :3100):
```ts
webServer: [
  {
    command: "npm run dev",                       // existing :3000, UNCHANGED
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: { NEXT_PUBLIC_E2E_TESTING: "true" },
  },
  {
    command: "npx next dev --port 3100",          // NOT `npm run dev` (that's :3000)
    url: FUNCTIONAL_BASE,                          // http://localhost:3100
    reuseExistingServer: !process.env.CI,         // reuse an already-running :3100
    timeout: 120_000,
    env: {
      LIGHTHOUSE_SKIP_AUTH: "1",                  // middleware.ts:24 bypass
      NEXT_PUBLIC_E2E_TESTING: "true",            // criterion 3: honor the config note
    },
  },
],
```
Rationale: array is the documented multi-server pattern; `reuseExistingServer:!CI` makes the local Mac run reuse :3000 (always up) and start/reuse :3100. `env` sets both bypass + polling flags explicitly (criterion 3 satisfied by injection — the app has no consumer today, matching the existing note's contract). Alternative if global double-boot ever matters: guard the :3100 entry behind an env flag — not needed for Mac-local.

### (b) Smoke-spec target + assertions
File: `frontend/tests/e2e-functional/smoke.spec.ts`. Target route `/` (home cockpit) — its shell renders independent of backend data.
- **Title carries the tag** so `--grep smoke` selects it: `test.describe("functional smoke", () => { test("@smoke home cockpit renders", ...) })` (title contains "smoke").
- **Register listeners BEFORE `page.goto('/')`** (baseURL=:3100):
  - Primary data region: `await expect(page.getByRole("heading", { name: "MAS Operator Cockpit" })).toBeVisible();` (home page.tsx:339) and optionally `await expect(page.getByRole("group", { name: "Portfolio key performance indicators" })).toBeVisible();` (page.tsx:383).
  - Zero console.error: `page.on("console", m => { if (m.type()==="error" && !benign(m.text())) consoleErrors.push(m.text()); })` then `expect(consoleErrors).toEqual([]);`.
  - Zero 5xx: `page.on("response", r => { if (r.status()>=500 && !benign(r.url())) serverErrors.push({url:r.url(),status:r.status()}); })` then `expect(serverErrors).toEqual([]);`.
  - Bypass sanity (optional, mirrors route_walk guard): assert final URL is not `/login`.
- Copy the `benign(url)` allowlist and the console/response capture idiom from `frontend/scripts/audit/route_walk.mjs:70-144`. NO `toHaveScreenshot`. Optionally `import { disableAnimations } from "../visual-regression/helpers/visual"` (harmless; not required).

### (c) Immutable command runs green on the Mac — confirmation
`cd frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line --grep smoke`
- `--project=functional` → new project, testDir `tests/e2e-functional` (per-project testDir confirmed). GREEN path.
- webServer array auto-starts :3100 with `LIGHTHOUSE_SKIP_AUTH=1`+`NEXT_PUBLIC_E2E_TESTING=true` (and reuses :3000); `LIGHTHOUSE_SKIP_AUTH=1` also inherited from the command's env → bypass double-covered → cockpit renders (no /login redirect).
- baseURL :3100 → `page.goto('/')` → `<h2>MAS Operator Cockpit</h2>` visible → assertion passes.
- `--grep smoke` matches the "smoke"-titled test; `--reporter=line` overrides config reporters.
- No screenshot assertions → no Linux baseline → passes on macOS.
- Pre-reqs to flag in the contract: (1) backend :8000 up (CLAUDE.md always-on rule) so zero-5xx/zero-console-error hold; (2) chromium installed (visual-regression already uses it); (3) first :3100 compile fits the 120s webServer timeout.

### Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 read: 6 official docs + 1 practitioner blog)
- [x] 10+ unique URLs total (7 read-in-full + 8 snippet-only = 15)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
Soft checks:
- [x] Internal exploration covered config, webServer, helpers, middleware, api base, polling source, route_walk blueprint
- [x] Contradictions/consensus noted (Next prod-build vs pyfinagent dev-server)
- [x] Claims cited per-claim

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 8,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Second Playwright project `functional` (testDir ./tests/e2e-functional, no screenshot assertions) is fully supported: per-project testDir is first-class (class-testproject doc), --project=functional selects it, and a webServer ARRAY entry auto-starts the :3100 skip-auth dev server (command `npx next dev --port 3100`, env LIGHTHOUSE_SKIP_AUTH=1 + NEXT_PUBLIC_E2E_TESTING=true, reuseExistingServer:!CI) with baseURL :3100. Criterion 3 note: NEXT_PUBLIC_E2E_TESTING has NO app consumer today (live-portfolio-context.tsx:144 polls unconditionally) so honoring it = env injection matching the existing config note + 63.1 precedent; 60s poll never fires in a <60s smoke. Smoke target: `/` asserting h2 'MAS Operator Cockpit' visible + zero console.error(type==='error') + zero 5xx(status>=500), reusing route_walk.mjs benign() filter; NO toHaveScreenshot so macOS is fine. Immutable command runs green given backend :8000 up + chromium installed.",
  "brief_path": "handoff/current/research_brief_64.1.md",
  "gate_passed": true
}
```

