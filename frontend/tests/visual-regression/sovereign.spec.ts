import { test, expect } from "@playwright/test";
import { disableAnimations, dynamicMasks } from "./helpers/visual";

test("sovereign page visual baseline", async ({ page }) => {
  await page.goto("/sovereign");
  await disableAnimations(page);
  await page.waitForLoadState("networkidle");
  await expect(page).toHaveScreenshot({
    fullPage: true,
    mask: dynamicMasks(page),
  });
});
