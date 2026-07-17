/*
 * phase-64.2: settings-family functional specs (/settings, /login).
 */
import { test, expect } from "@playwright/test";
import { assertFunctionalRoute } from "./_helpers";

test("settings: /settings renders the settings header", async ({ page }) => {
  await assertFunctionalRoute(page, "/settings", (p) => p.getByRole("heading", { name: "Settings" }));
});

test("settings: /login renders the branding", async ({ page }) => {
  await assertFunctionalRoute(
    page,
    "/login",
    (p) => p.getByRole("heading", { name: "PyFinAgent" }),
  );
});

test("settings: sidebar navigation from /settings back to the cockpit", async ({ page }) => {
  await page.goto("/settings", { waitUntil: "load" });
  await page.locator('a[href="/"]').first().click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { name: "MAS Operator Cockpit" })).toBeVisible({
    timeout: 20_000,
  });
});
