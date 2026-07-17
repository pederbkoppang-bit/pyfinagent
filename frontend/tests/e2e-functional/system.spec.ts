/*
 * phase-64.2: system-family functional specs (/agents, /agent-map, /cron,
 * /observability). Load + zero console.error + zero 5xx + zero pageerror per
 * route, plus one family interaction.
 */
import { test, expect } from "@playwright/test";
import { assertFunctionalRoute } from "./_helpers";

test("system: /agents renders the multi-agent system view", async ({ page }) => {
  // The agent-metrics-table testid sits behind a non-default tab; assert the
  // always-rendered page header instead (primary region proof).
  await assertFunctionalRoute(
    page,
    "/agents",
    (p) => p.getByRole("heading", { name: "Multi-Agent System" }),
  );
});

test("system: /agent-map renders the agent map", async ({ page }) => {
  await assertFunctionalRoute(page, "/agent-map", (p) => p.getByTestId("agent-map"));
});

test("system: /cron renders the logs view", async ({ page }) => {
  await assertFunctionalRoute(page, "/cron", (p) => p.getByRole("heading", { name: "Cron / Logs" }));
});

test("system: /observability renders data freshness", async ({ page }) => {
  await assertFunctionalRoute(
    page,
    "/observability",
    (p) => p.getByRole("heading", { name: "Data Freshness" }),
  );
});

test("system: navigation from /agents to /agent-map", async ({ page }) => {
  await page.goto("/agents", { waitUntil: "load" });
  await page.locator('a[href="/agent-map"]').first().click();
  await expect(page).toHaveURL(/\/agent-map$/);
  await expect(page.getByTestId("agent-map")).toBeVisible({ timeout: 20_000 });
});
