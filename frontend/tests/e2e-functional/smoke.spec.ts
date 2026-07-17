/*
 * phase-64.1: functional-E2E smoke spec (NO screenshot assertions -> Mac-safe).
 *
 * Runs via the `functional` project in playwright.config.ts against the
 * :3100 auth-bypass server (LIGHTHOUSE_SKIP_AUTH=1 + NEXT_PUBLIC_E2E_TESTING).
 *   cd frontend && LIGHTHOUSE_SKIP_AUTH=1 \
 *     npx playwright test --project=functional --reporter=line --grep smoke
 *
 * Assertion style (per the step): the primary data region renders, zero
 * console.error, zero 5xx. The `/` home cockpit was proven console/failed
 * clean by the 63.1 route walk. benign() mirrors scripts/audit/route_walk.mjs.
 */
import { test, expect, type ConsoleMessage, type Response } from "@playwright/test";

// benign network/console noise -- dev HMR, source maps, favicon, extensions.
// NOT functional failures; excluded from the zero-error assertions.
function benign(u: string): boolean {
  return (
    /favicon\.ico($|\?)/.test(u) ||
    /\.map($|\?)/.test(u) ||
    /manifest\.(webmanifest|json)/.test(u) ||
    /chrome-extension:\/\//.test(u) ||
    /hot-update\.(json|js)/.test(u) ||
    /\/__nextjs|\/_next\/webpack-hmr/.test(u)
  );
}

test("smoke: home cockpit renders, no console errors, no 5xx", async ({ page }) => {
  const consoleErrors: string[] = [];
  const serverErrors: string[] = [];

  page.on("console", (msg: ConsoleMessage) => {
    if (msg.type() === "error") {
      const loc = msg.location()?.url ?? "";
      if (!benign(loc)) consoleErrors.push(msg.text());
    }
  });
  page.on("response", (res: Response) => {
    if (res.status() >= 500 && !benign(res.url())) {
      serverErrors.push(`${res.status()} ${res.request().method()} ${res.url()}`);
    }
  });

  await page.goto("/", { waitUntil: "load" });

  // Primary data region renders (h2 at src/app/page.tsx:339).
  await expect(
    page.getByRole("heading", { name: "MAS Operator Cockpit" }),
  ).toBeVisible({ timeout: 15_000 });

  // Let any late console/network settle before the zero-error assertions.
  await page.waitForTimeout(1500);

  expect(consoleErrors, `console.error(s): ${consoleErrors.join(" | ")}`).toHaveLength(0);
  expect(serverErrors, `5xx response(s): ${serverErrors.join(" | ")}`).toHaveLength(0);
});
