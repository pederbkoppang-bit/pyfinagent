/*
 * phase-64.2: paper-trading-family functional specs (8 routes). The subpages
 * share the layout's <h2>Paper Trading</h2>, so each asserts its ROUTE-
 * DISTINCTIVE tabpanel (#panel-<subpage>) for real per-route proof. Two
 * redirects: /paper-trading -> /positions, /paper-trading/learnings -> /learnings.
 */
import { test, expect } from "@playwright/test";
import { assertFunctionalRoute } from "./_helpers";

test("paper-trading: /paper-trading redirects to positions", async ({ page }) => {
  const final = await assertFunctionalRoute(
    page,
    "/paper-trading",
    (p) => p.locator("#panel-positions"),
  );
  expect(final).toBe("/paper-trading/positions");
});

test("paper-trading: /positions renders the positions panel", async ({ page }) => {
  await assertFunctionalRoute(page, "/paper-trading/positions", (p) => p.locator("#panel-positions"));
});

test("paper-trading: /trades renders the trades panel", async ({ page }) => {
  await assertFunctionalRoute(page, "/paper-trading/trades", (p) => p.locator("#panel-trades"));
});

test("paper-trading: /nav renders the nav panel", async ({ page }) => {
  await assertFunctionalRoute(page, "/paper-trading/nav", (p) => p.locator("#panel-nav"));
});

test("paper-trading: /reality-gap renders the reality-gap panel", async ({ page }) => {
  await assertFunctionalRoute(
    page,
    "/paper-trading/reality-gap",
    (p) => p.locator("#panel-reality-gap"),
  );
});

test("paper-trading: /exit-quality renders the exit-quality panel", async ({ page }) => {
  await assertFunctionalRoute(
    page,
    "/paper-trading/exit-quality",
    (p) => p.locator("#panel-exit-quality"),
  );
});

test("paper-trading: /manage renders the manage panel", async ({ page }) => {
  await assertFunctionalRoute(page, "/paper-trading/manage", (p) => p.locator("#panel-manage"));
});

test("paper-trading: /paper-trading/learnings redirects to /learnings", async ({ page }) => {
  const final = await assertFunctionalRoute(
    page,
    "/paper-trading/learnings",
    (p) => p.getByTestId("virtual-fund-learnings"),
  );
  expect(final).toBe("/learnings");
});

test("paper-trading: tab navigation from positions to trades", async ({ page }) => {
  await page.goto("/paper-trading/positions", { waitUntil: "load" });
  await page.getByRole("tab", { name: "Trades" }).click();
  await expect(page).toHaveURL(/\/paper-trading\/trades$/);
  await expect(page.locator("#panel-trades")).toBeVisible({ timeout: 20_000 });
});
