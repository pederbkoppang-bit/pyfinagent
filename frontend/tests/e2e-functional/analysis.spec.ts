/*
 * phase-64.2: analysis-family functional specs (/signals, /backtest,
 * /learnings, /reports, /performance).
 */
import { test, expect } from "@playwright/test";
import { assertFunctionalRoute } from "./_helpers";

test("analysis: /signals renders the ticker input", async ({ page }) => {
  await assertFunctionalRoute(page, "/signals", (p) => p.locator("#signals-ticker-input"));
});

test("analysis: /backtest renders the walk-forward header", async ({ page }) => {
  await assertFunctionalRoute(
    page,
    "/backtest",
    (p) => p.getByRole("heading", { name: "Walk-Forward Backtest" }),
  );
});

test("analysis: /learnings renders the virtual-fund learnings", async ({ page }) => {
  await assertFunctionalRoute(page, "/learnings", (p) => p.getByTestId("virtual-fund-learnings"));
});

test("analysis: /reports renders the reports header", async ({ page }) => {
  await assertFunctionalRoute(page, "/reports", (p) => p.getByRole("heading", { name: "Reports" }));
});

test("analysis: /performance renders the recommendation-performance header", async ({ page }) => {
  await assertFunctionalRoute(
    page,
    "/performance",
    (p) => p.getByRole("heading", { name: "Recommendation Performance" }),
  );
});

test("analysis: typing into the /signals ticker input updates its value", async ({ page }) => {
  await page.goto("/signals", { waitUntil: "load" });
  const input = page.locator("#signals-ticker-input");
  await input.fill("AAPL");
  await expect(input).toHaveValue("AAPL");
});
