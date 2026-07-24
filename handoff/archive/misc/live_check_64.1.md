# live_check — step 64.1 (Functional-E2E Playwright project + smoke)

## Green smoke-run transcript (immutable command)

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && \
    LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line --grep smoke

Running 1 test using 1 worker
  1 [functional] › tests/e2e-functional/smoke.spec.ts:28:5 › smoke: home cockpit renders, no console errors, no 5xx
  1 passed (9.2s)
# exit 0
```

Ran 4× (design iterations + final), all "1 passed". The functional project + spec are discovered under `--grep smoke`;
the `functional` project uses `testDir tests/e2e-functional` with NO screenshot assertions (Linux-baseline caveat
does not apply → runs on the Mac).

## Do-no-harm verification (operator :3000 untouched by the final design)

The functional command manages ONLY the isolated :3100 skip-auth server, which compiles into `.next-functional`
(`distDir` via `PLAYWRIGHT_DIST_DIR`) so it never shares `.next` with the operator's :3000. Verified `:3000 /login`
before AND after every functional run:

```
baseline :3000 /login -> 200
[functional smoke: 1 passed]
:3000 /login -> 200         # unchanged -- isolation holds
:3000 / -> 302              # healthy authed signature
git status frontend/next-env.d.ts frontend/tsconfig.json -> (clean; globalTeardown restored)
```

**INCIDENT DISCLOSURE:** an intermediate design (global webServer array including the :3000 `npm run dev` entry)
caused Playwright to try to START :3000, running `predev: rm -rf .next` → the operator's :3000 briefly served 500/404.
It was fully RECOVERED (kill + clean `npm run dev` restart → `.next` regenerated → 200/302) and PERMANENTLY fixed
(functional run never touches :3000; distDir isolates the :3100 build). See experiment_results.md "Do-no-harm
INCIDENT + full recovery". End-state :3000 verified healthy.

## Criterion 3 (NEXT_PUBLIC_E2E_TESTING)

Honored via env injection in the :3100 webServer (`env: { LIGHTHOUSE_SKIP_AUTH:"1", NEXT_PUBLIC_E2E_TESTING:"true",
PLAYWRIGHT_DIST_DIR:".next-functional" }`), matching the existing config note + the 63.1 route_walk precedent. The app
has no consumer of the flag today (`live-portfolio-context.tsx:144` polls at 60s unconditionally); the 60s poll never
fires within the <60s smoke, so no flake.

## Method disclosure

Isolated :3100 skip-auth server (Playwright-managed via the gated webServer; `next dev --port 3100`,
`LIGHTHOUSE_SKIP_AUTH=1`, `distDir=.next-functional`). Operator :3000 never managed by the functional run and verified
200/302 before+after. Playwright 1.60.0, chromium-headless-shell 1223. Files: `frontend/playwright.config.ts`,
`frontend/next.config.js` (conditional distDir), `frontend/tests/e2e-functional/{smoke.spec.ts,global-teardown.ts}`,
`frontend/.gitignore`.
