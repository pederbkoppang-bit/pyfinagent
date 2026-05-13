import { test, expect } from "@playwright/test";
import { disableAnimations, dynamicMasks } from "./helpers/visual";

test("agent-map page visual baseline", async ({ page }) => {
  await page.goto("/agent-map");
  await disableAnimations(page);
  await page.waitForLoadState("networkidle");
  await expect(page).toHaveScreenshot({
    fullPage: true,
    mask: dynamicMasks(page),
  });
});
