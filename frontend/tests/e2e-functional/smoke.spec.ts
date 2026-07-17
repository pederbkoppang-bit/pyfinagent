/*
 * phase-64.1/64.2: home-family functional spec (`/` route).
 *
 * Keeps the `--grep smoke` canary (64.1 immutable command) AND adds the
 * home-family interaction (64.2). Runs via the `functional` project against
 * the :3100 auth-bypass server:
 *   cd frontend && LIGHTHOUSE_SKIP_AUTH=1 \
 *     npx playwright test --project=functional --reporter=line
 */
import { test, expect } from "@playwright/test";
import { assertFunctionalRoute } from "./_helpers";

test("smoke: home cockpit renders, no console errors, no 5xx", async ({ page }) => {
  await assertFunctionalRoute(
    page,
    "/",
    (p) => p.getByRole("heading", { name: "MAS Operator Cockpit" }),
  );
});

test("home: sidebar navigation to /signals loads that route", async ({ page }) => {
  await page.goto("/", { waitUntil: "load" });
  await page.locator('a[href="/signals"]').first().click();
  await expect(page).toHaveURL(/\/signals$/);
  await expect(page.locator("#signals-ticker-input")).toBeVisible({ timeout: 20_000 });
});
