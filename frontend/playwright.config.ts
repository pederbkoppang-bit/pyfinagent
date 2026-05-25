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
  ],

  webServer: {
    command: "npm run dev",
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: {
      NEXT_PUBLIC_E2E_TESTING: "true",
    },
  },
});
