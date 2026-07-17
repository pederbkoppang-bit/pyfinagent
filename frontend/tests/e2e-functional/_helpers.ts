import { expect, type Locator, type Page } from "@playwright/test";

/**
 * phase-64.2: shared functional-E2E helpers (extends the 64.1 smoke idiom).
 *
 * benign(): network/console noise that is NOT a functional failure (dev HMR,
 * source maps, favicon, extensions).
 */
export function benign(u: string): boolean {
  return (
    /favicon\.ico($|\?)/.test(u) ||
    /\.map($|\?)/.test(u) ||
    /manifest\.(webmanifest|json)/.test(u) ||
    /chrome-extension:\/\//.test(u) ||
    /hot-update\.(json|js)/.test(u) ||
    /\/__nextjs|\/_next\/webpack-hmr/.test(u)
  );
}

/**
 * Load `path`, assert the primary data region renders, and assert ZERO
 * console.error, ZERO uncaught page errors, and ZERO 5xx responses. Mirrors
 * the accepted 64.1 smoke, hardened with page.on("pageerror").
 *
 * @param region either a Locator or a (page)=>Locator that resolves AFTER nav.
 * @returns the final URL path (for redirect assertions).
 */
export async function assertFunctionalRoute(
  page: Page,
  path: string,
  region: Locator | ((p: Page) => Locator),
): Promise<string> {
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const serverErrors: string[] = [];

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      const loc = msg.location()?.url ?? "";
      if (!benign(loc)) consoleErrors.push(`${path}: ${msg.text().slice(0, 300)}`);
    }
  });
  page.on("pageerror", (err) => pageErrors.push(`${path}: ${String(err).slice(0, 300)}`));
  page.on("response", (res) => {
    if (res.status() >= 500 && !benign(res.url())) {
      serverErrors.push(`${path}: ${res.status()} ${res.request().method()} ${res.url()}`);
    }
  });

  await page.goto(path, { waitUntil: "load" });

  const locator = typeof region === "function" ? region(page) : region;
  await expect(locator, `${path}: primary data region did not render`).toBeVisible({
    timeout: 20_000,
  });

  // Let late console/network settle before the zero-error assertions.
  await page.waitForTimeout(1200);

  expect(consoleErrors, `console.error(s) on ${path}: ${consoleErrors.join(" | ")}`).toHaveLength(0);
  expect(pageErrors, `pageerror(s) on ${path}: ${pageErrors.join(" | ")}`).toHaveLength(0);
  expect(serverErrors, `5xx response(s) on ${path}: ${serverErrors.join(" | ")}`).toHaveLength(0);

  return new URL(page.url()).pathname;
}

/**
 * Resolve a concrete strategy id for the dynamic /sovereign/strategy/[id]
 * route via the leaderboard API; fall back to "baseline" (renders 200).
 */
export async function resolveStrategyId(page: Page, baseURL: string): Promise<string> {
  try {
    const r = await page.request.get(`${baseURL}/api/sovereign/leaderboard`, {
      timeout: 8000,
    });
    if (r.ok()) {
      const j = await r.json();
      const rows = j.rows || j.leaderboard || j.data || [];
      if (Array.isArray(rows) && rows.length && rows[0].strategy_id) {
        return String(rows[0].strategy_id);
      }
    }
  } catch {
    /* fall through */
  }
  return "baseline";
}
