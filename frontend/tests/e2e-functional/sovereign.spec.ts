/*
 * phase-64.2: sovereign-family functional specs (/sovereign,
 * /sovereign/strategy/[id]). The dynamic id uses the "baseline" fallback
 * (renders 200; see 63.1 route walk), avoiding an API dependency.
 */
import { test, expect } from "@playwright/test";
import { assertFunctionalRoute } from "./_helpers";

test("sovereign: /sovereign renders the control plane", async ({ page }) => {
  await assertFunctionalRoute(page, "/sovereign", (p) => p.getByRole("heading", { name: "Sovereign" }));
});

test("sovereign: /sovereign/strategy/[id] renders the strategy detail", async ({ page }) => {
  await assertFunctionalRoute(
    page,
    "/sovereign/strategy/baseline",
    (p) => p.getByTestId("strategy-detail"),
  );
});

test("sovereign: sidebar navigation from /sovereign to /reports", async ({ page }) => {
  await page.goto("/sovereign", { waitUntil: "load" });
  await page.locator('a[href="/reports"]').first().click();
  await expect(page).toHaveURL(/\/reports$/);
  await expect(page.getByRole("heading", { name: "Reports" })).toBeVisible({ timeout: 20_000 });
});
