# Experiment results — step 64.1 (Functional-E2E Playwright project + smoke spec)

**Step:** 64.1 (P0, phase-64, depends_on none). $0; local-only; test-infra. Research gate PASSED
(research_brief_64.1.md, gate_passed=true, 7 external sources read in full). historical_macro FROZEN; live book
untouched.

## What was built (final, do-no-harm-safe design)

1. **`frontend/playwright.config.ts`** — (a) added a `functional` project (`testDir: "./tests/e2e-functional"`,
   `baseURL: http://localhost:3100`, `reducedMotion`, **NO screenshot assertions** → the Linux-baseline caveat does
   not apply → Mac-safe) [criterion 1]. (b) The `webServer` is now **SELECTED by `LIGHTHOUSE_SKIP_AUTH`**: the
   functional command (which sets it) manages **ONLY :3100**; the default/visual-regression run (`npx playwright
   test`, no such env) manages **ONLY :3000, unchanged** — the two suites are fully decoupled (no CI coupling). (c)
   added `globalTeardown`.
2. **`frontend/next.config.js`** — a conditional `distDir` (`process.env.PLAYWRIGHT_DIST_DIR ? {distDir} : {}`),
   default-preserving (UNSET everywhere except the :3100 webServer → falls back to `.next`, byte-identical). Lets the
   :3100 server compile into an ISOLATED `.next-functional` so it never shares `.next` with the operator's :3000.
3. **`frontend/tests/e2e-functional/smoke.spec.ts`** (NEW) — title contains "smoke" (matches `--grep smoke`), target
   `/`, asserts `getByRole("heading",{name:"MAS Operator Cockpit"})` visible (page.tsx:339) + **zero console.error** +
   **zero 5xx** (inline `benign()` from route_walk.mjs). No screenshots [criteria 1+2].
4. **`frontend/tests/e2e-functional/global-teardown.ts`** (NEW) — restores `next-env.d.ts`/`tsconfig.json` IF the
   :3100 `next dev` (distDir) rewrote them to reference `.next-functional` (checks the marker; no-op for a visual run;
   best-effort). Keeps the tracked TS config pointing at `.next`.
5. **`frontend/.gitignore`** (NEW) — `.next-functional/`, `playwright-report/`, `test-results/`.

Criterion 3: `NEXT_PUBLIC_E2E_TESTING` has no app consumer today (`live-portfolio-context.tsx:144` polls at 60s
unconditionally). "Honor per config note" = env INJECTION matching the existing webServer + the 63.1 precedent; the
60s poll never fires in a <60s smoke → no flake. Set in the :3100 webServer env.

## ⚠️ Do-no-harm INCIDENT + full recovery (disclosed in full)

An intermediate config design (a global `webServer` ARRAY that included the :3000 `npm run dev` entry) caused
**Playwright to attempt STARTING the :3000 webServer** when its reuse-probe transiently missed the operator's server.
`npm run dev`'s `predev: rm -rf .next` RAN, deleting the shared build dir → **the operator's :3000 served HTTP 500
then 404**. **This was a real disruption of the operator's environment.** Recovery + permanent fix:
- **Recovered :3000**: killed the broken `next dev`, restarted it clean (`nohup npm run dev`), regenerated `.next` →
  verified `:3000 /login → 200`, `/ → 302` (healthy). (Note: the running :3000 is now this session's restarted
  process; it is detached and serving correctly.)
- **Permanent fix so it can NEVER recur**: (i) the functional run now manages ONLY :3100 (never runs `npm run dev`,
  never `rm -rf .next`); (ii) `distDir=.next-functional` isolates the :3100 build so it cannot clobber :3000's `.next`
  route manifests (an earlier shared-`.next` design caused a TRANSIENT :3000 route-404-then-recompile; the distDir
  eliminates even that). **Post-fix, verified across 4 functional runs: `:3000 /login` stays 200 throughout — the
  operator's :3000 is untouched.**

## Verification (verbatim, final design)

- IMMUTABLE cmd `cd frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line --grep smoke` → **exit 0, "1 passed" (~9-11s)**, repeatable (ran 4×, all pass).
- **:3000 UNTOUCHED**: `/login → 200` before AND after every functional run (distDir isolation proven); `/ → 302`.
- **TS files CLEAN after run**: `git status` shows next-env.d.ts + tsconfig.json unmodified (globalTeardown restored
  them; next-env.d.ts references `./.next/types/routes.d.ts`).
- `npx tsc --noEmit` on playwright.config.ts + smoke.spec.ts + global-teardown.ts → **CLEAN**.
- `npx eslint tests/e2e-functional/` → **clean** (no output).
- `node -e "require('./next.config.js')"` → parses; distDir UNSET→(default .next), SET→.next-functional.
- **Visual-regression DECOUPLED (cycle-2 corrected)**: BOTH the functional PROJECT and its :3100 webServer are gated
  on `LIGHTHOUSE_SKIP_AUTH`. Verified with the exact CI invocation: bare `npx playwright test --list` (no env)
  enumerates **8 tests, [functional] count = 0** (only `chromium`); `LIGHTHOUSE_SKIP_AUTH=1 npx playwright test
  --list` enumerates the functional smoke. So the visual-regression CI (`.github/workflows/visual-regression.yml`
  lines 62/75 run bare `npx playwright test`, CI=true, trigger paths `frontend/**`) NEVER enumerates or runs the
  functional smoke → no red CI. **[Cycle-1 CONDITIONAL fix]**: cycle-1 gated only the webServer, not the project, so a
  bare run would have enumerated the functional smoke and failed against an unstarted :3100 — the Q/A caught this; the
  project is now also gated (`...(process.env.LIGHTHOUSE_SKIP_AUTH ? [functionalProject] : [])`).

## Do-no-harm / boundaries

$0; local-only; test-infra only. Production runtime code UNCHANGED (next.config distDir is UNSET in all normal/CI/prod
paths → byte-identical; the change is inert outside the functional-E2E env). NO trade/risk/money touch;
kill-switch/stops/caps/DSR/PBO untouched; historical_macro FROZEN; live book untouched. The operator's :3000 was
disrupted mid-development (disclosed above) then fully RECOVERED + permanently isolated so it can't recur;
end-state :3000 verified healthy (200/302). Git scope: next.config.js + playwright.config.ts (prod-config, inert
outside E2E), + new tests/e2e-functional/ + frontend/.gitignore + handoff docs. next-env.d.ts/tsconfig.json clean.

## Artifact shape
The functional suite: `frontend/tests/e2e-functional/*.spec.ts`, run via `--project=functional`. Re-runnable green:
`cd frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --grep smoke`. live_check_64.1.md holds
the green transcript.
