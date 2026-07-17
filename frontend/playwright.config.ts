import { defineConfig, devices } from "@playwright/test";

/**
 * phase-25.A12: Visual regression baseline configuration.
 *
 * Baselines MUST be generated on ubuntu-latest (Linux). macOS-generated
 * snapshots will fail CI due to font rendering and antialiasing
 * differences -- this is a documented Playwright pitfall.
 *
 * First-run operator flow (preferred):
 *   1. From GitHub UI, trigger the `Visual Regression` workflow with
 *      `update_snapshots=true`. The workflow runs Playwright with
 *      --update-snapshots on Linux and commits the baselines via
 *      git-auto-commit-action.
 *   2. Subsequent PR runs verify visual fidelity vs the committed
 *      baselines and fail if `maxDiffPixelRatio > 0.015`.
 *
 * Local fallback (Linux only):
 *   cd frontend && npm ci && npx playwright install --with-deps chromium
 *   npx playwright test --update-snapshots
 *   git add tests/visual-regression/snapshots && git commit
 *
 * Tuning rationale (per research-gate 25.A12 brief):
 *   - maxDiffPixelRatio: 0.015  -- canonical for dark-themed dashboards
 *     (Bug0 / TestDino thresholds for Tailwind dark dashboards).
 *   - threshold: 0.2            -- per-pixel YIQ tolerance; handles
 *     subpixel font rendering on dark backgrounds.
 *   - animations: 'disabled'    -- explicit even though it's the default
 *     since Playwright 1.32.
 *   - reducedMotion: 'reduce'   -- catches `motion`-package JS
 *     animations that Playwright's CSS-only disabling misses.
 *   - NEXT_PUBLIC_E2E_TESTING   -- suppresses live polling, dev
 *     overlays, and timers inside the Next.js app (Ash Connolly).
 */

const BASE_URL = process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000";

export default defineConfig({
  testDir: "./tests/visual-regression",
  // phase-64.1: restores next-env.d.ts/tsconfig.json if the functional :3100
  // server (distDir=.next-functional) rewrote them. No-op for the visual run.
  globalTeardown: "./tests/e2e-functional/global-teardown.ts",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [
    ["html", { outputFolder: "playwright-report", open: "never" }],
    ["list"],
  ],

  snapshotDir: "tests/visual-regression/snapshots",
  snapshotPathTemplate:
    "{snapshotDir}/{projectName}/{testFileName}/{arg}{ext}",

  expect: {
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.015,
      threshold: 0.2,
      animations: "disabled",
    },
  },

  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1280, height: 800 },
        // phase-44.0 motion-stability: reducedMotion lives at the
        // project-level use (per @playwright/test types post-1.40);
        // moved from top-level config use during TanStack/Tremor dep
        // install (cycle 61) when the TS type tightening surfaced.
        contextOptions: {
          reducedMotion: "reduce",
        },
      },
    },
    // phase-64.1: the functional-E2E project is INCLUDED only when
    // LIGHTHOUSE_SKIP_AUTH is set (mirrors the webServer ternary below). A bare
    // `npx playwright test` (the visual-regression CI path, no such env) then
    // enumerates ONLY `chromium` and never runs the functional smoke against an
    // unstarted :3100 -> no regression to visual-regression.yml. The immutable
    // functional command sets LIGHTHOUSE_SKIP_AUTH=1, so the project IS present
    // there. NO screenshot assertions, so the Linux visual-baseline caveat
    // (lines 4-14) does not apply -- this suite runs on the Mac.
    ...(process.env.LIGHTHOUSE_SKIP_AUTH
      ? [
          {
            name: "functional",
            testDir: "./tests/e2e-functional",
            use: {
              ...devices["Desktop Chrome"],
              baseURL: "http://localhost:3100",
              viewport: { width: 1440, height: 900 },
              // `as const` restores the literal type lost when the project
              // object sits inside a conditional-spread array (vs. the inline
              // chromium project, which is contextually typed as Project).
              contextOptions: {
                reducedMotion: "reduce" as const,
              },
            },
          },
        ]
      : []),
  ],

  // phase-64.1: the webServer set is SELECTED by LIGHTHOUSE_SKIP_AUTH so the two
  // suites are FULLY ISOLATED and never share a managed server:
  //   * functional command (LIGHTHOUSE_SKIP_AUTH=1) -> ONLY the isolated :3100
  //     skip-auth server. It deliberately does NOT include the :3000 entry --
  //     `npm run dev` carries a `predev: rm -rf .next` that must NEVER run
  //     against the operator's live :3000 (doing so deletes the shared build
  //     and 500s the running cockpit). `npm run dev` also hardcodes :3000, so
  //     the :3100 server calls `next dev --port 3100` directly (no predev rm).
  //   * default / visual-regression (`npx playwright test`, no such env) ->
  //     ONLY the :3000 dev server, exactly as before (behavior unchanged).
  // reuseExistingServer reuses an already-running target (operator :3000 or a
  // live :3100) rather than starting a duplicate.
  webServer: process.env.LIGHTHOUSE_SKIP_AUTH
    ? [
        {
          command: "npx next dev --port 3100",
          url: "http://localhost:3100",
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
          env: {
            LIGHTHOUSE_SKIP_AUTH: "1",
            NEXT_PUBLIC_E2E_TESTING: "true",
            // phase-64.1: isolated build dir (next.config.js reads this) so the
            // :3100 server never shares .next with the operator's :3000.
            PLAYWRIGHT_DIST_DIR: ".next-functional",
          },
        },
      ]
    : [
        {
          command: "npm run dev",
          url: BASE_URL,
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
          env: {
            NEXT_PUBLIC_E2E_TESTING: "true",
          },
        },
      ],
});
