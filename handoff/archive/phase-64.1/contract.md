# Contract — step 64.1 (Functional-E2E Playwright project + smoke spec)

**Phase:** phase-64 (test matrix build-out) | **Step:** 64.1 | **Priority:** P0 | harness_required: true | depends_on: none
**Cycle:** 1 | Date: 2026-07-17 | **Type:** test-infra (new Playwright project + one smoke spec). $0; local-only;
NO production/live-loop change; historical_macro FROZEN; live book untouched; operator :3000 NEVER disrupted.

## Research-gate summary (gate PASSED)

Researcher subagent (Agent tool, Opus 4.8 effort:max, $0 Max rail), brief `research_brief_64.1.md`. Envelope:
**gate_passed=true**, tier=moderate, **7 external sources read in full** (6 Playwright/Next.js official docs + 1
practitioner), 8 snippet-only, 15 URLs, recency scan, 8 internal files. KEY:
- The existing `playwright.config.ts` is single-project (visual-regression, testDir `./tests/visual-regression`) with a
  single `webServer` object (`npm run dev` → :3000, `reuseExistingServer:!CI`, env NEXT_PUBLIC_E2E_TESTING). macOS
  screenshot-baseline caveat at lines 4-14 — AVOIDED here (no screenshot assertions).
- Heading selector confirmed: `<h2>MAS Operator Cockpit</h2>` at `src/app/page.tsx:339`.
- Criterion 3: `NEXT_PUBLIC_E2E_TESTING` has NO app consumer today (`live-portfolio-context.tsx:144` polls
  unconditionally at 60s). "Honor per config note" = env INJECTION matching the existing webServer + the 63.1
  precedent (a harness contract, not app code); the 60s poll never fires in a <60s smoke → no flake.
- 63.1's `route_walk.mjs:70-144` is the reusable console/5xx capture + `benign()` blueprint.

## Plan

### A. `frontend/playwright.config.ts` — TWO edits [criteria 1 + 3]
1. Add a `functional` project AFTER `chromium`:
   `{ name:"functional", testDir:"./tests/e2e-functional", use:{ ...devices["Desktop Chrome"],
     baseURL:"http://localhost:3100", viewport:{width:1440,height:900}, contextOptions:{reducedMotion:"reduce"} } }`
   (NO `toHaveScreenshot` → the Linux-baseline caveat does NOT apply → runs on Mac.)
2. Convert `webServer` object → an ARRAY of two: KEEP the existing :3000 entry unchanged; ADD a :3100 entry:
   `{ command:"npx next dev --port 3100", url:"http://localhost:3100", reuseExistingServer:!process.env.CI,
     timeout:120_000, env:{ LIGHTHOUSE_SKIP_AUTH:"1", NEXT_PUBLIC_E2E_TESTING:"true" } }`
   (`npm run dev` hardcodes :3000 → the :3100 entry calls `next dev --port 3100` directly. `reuseExistingServer`
   reuses the operator's :3000 [always up] so it is NEVER restarted/disrupted.)

### B. `frontend/tests/e2e-functional/smoke.spec.ts` (NEW) [criteria 1 + 2]
- Test title contains **"smoke"** (so `--grep smoke` matches). Target `/` (baseURL :3100 from the project).
- Assert the primary data region renders: `expect(page.getByRole("heading",{name:"MAS Operator Cockpit"})).toBeVisible()`.
- Zero console.error: `page.on("console", m => m.type()==="error" && !benign(...))` → assert empty.
- Zero 5xx: `page.on("response", r => r.status()>=500 && !benign(...))` → assert empty.
- Inline `benign()` allowlist (favicon/map/manifest/HMR/ext) mirrored from route_walk.mjs. NO screenshots.
- (`/` was proven console/failed-clean by the 63.1 walk → the smoke will pass.)

### C. Run + verify (green on Mac)
Run the immutable command; Playwright's webServer auto-starts :3100 (reuses :3000). After the run, verify operator
:3000 still 302 (untouched).

## Immutable success criteria (verbatim from masterplan.json 64.1)
1. "the functional project exists in playwright.config.ts with testDir tests/e2e-functional and no screenshot assertions"
2. "a smoke spec passes on the Mac against the port-3100 auth-bypass server"
3. "NEXT_PUBLIC_E2E_TESTING polling suppression is honored per the existing config note"

**Verification command (immutable):**
`cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line --grep smoke`

## Boundaries (binding)
$0; local-only; test-infra only (a new Playwright project + one smoke spec + a new testDir). NO production/live-loop
change; NO trade/risk/money touch; kill-switch/stops/caps/DSR/PBO untouched; historical_macro FROZEN; live book
untouched. The functional project targets an ISOLATED :3100 bypass server (Playwright-managed via webServer, reused if
up); the operator's :3000 is reused (never restarted — verified 302 after). Pre-reqs (flagged): backend :8000 up
(confirmed 200), chromium installed (visual-regression already uses it; note: playwright 1.60 needed headless-shell
1223, installed during 63.1), 120s webServer timeout covers the first :3100 compile.

## References
research_brief_64.1.md; frontend/playwright.config.ts (webServer 84-92, macOS caveat 4-14); frontend/src/app/page.tsx:339
(heading); frontend/scripts/audit/route_walk.mjs:70-144 (benign()/capture blueprint); frontend/src/middleware.ts:24
(bypass). Playwright docs: projects/testDir, webServer array + reuseExistingServer, grep tagging.
