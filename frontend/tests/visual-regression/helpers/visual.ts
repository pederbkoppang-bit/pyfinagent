import type { Page, Locator } from "@playwright/test";

/**
 * phase-25.A12: shared helpers for visual-regression specs.
 *
 * `disableAnimations` injects a stylesheet that zeros out animation +
 * transition durations and disables smooth scroll -- belt-and-suspenders
 * over Playwright's built-in `animations: 'disabled'` for libraries
 * (e.g. the `motion` package) that use JS animations not caught by
 * the CSS-only path.
 *
 * `dynamicMasks(page)` returns Locator selectors for content that
 * changes per-run (timestamps, live prices, loading spinners, recharts
 * tick labels). `toHaveScreenshot` masks these out with a solid magenta
 * box so the diff doesn't trigger on cosmetic time movement.
 */
export async function disableAnimations(page: Page): Promise<void> {
  await page.addStyleTag({
    content: `
      *, *::before, *::after {
        animation-duration: 0s !important;
        animation-delay: 0s !important;
        transition-duration: 0s !important;
        transition-delay: 0s !important;
        scroll-behavior: auto !important;
      }
    `,
  });
}

export function dynamicMasks(page: Page): Locator[] {
  return [
    page.locator("time"),
    page.locator("[data-dynamic]"),
    page.locator("[class*='skeleton']"),
    page.locator("[class*='spinner']"),
    page.locator("[class*='pulse']"),
    page.locator("[class*='animate']"),
    page.locator(".recharts-text"),
    page.locator(".recharts-cartesian-axis-tick"),
  ];
}
