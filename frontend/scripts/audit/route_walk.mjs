#!/usr/bin/env node
/*
 * phase-63.1: full-app Playwright route walk (READ-ONLY audit).
 *
 * Visits every src/app/ page.tsx route against an ISOLATED skip-auth dev
 * server (never the operator's :3000), capturing per route: a full-page
 * screenshot, console errors/warnings, page errors, and failed (>=400 /
 * requestfailed) network requests. Emits
 * handoff/away_ops/route_walk_<date>/walk_summary.json + screenshots/.
 *
 * SETUP (operator/Main runs the isolated server first -- middleware.ts:24):
 *   cd frontend && LIGHTHOUSE_SKIP_AUTH=1 NEXT_PUBLIC_E2E_TESTING=true \
 *     npx next dev --port 3100
 *   node scripts/audit/route_walk.mjs            # base http://localhost:3100
 *   node scripts/audit/route_walk.mjs --base http://localhost:3100
 * Then kill :3100 and confirm :3000 still 302s.
 *
 * NEXT_PUBLIC_E2E_TESTING=true is load-bearing: it kills the cockpit's live
 * polling so waitUntil:'load' doesn't hang (research_brief_63.1 T1).
 *
 * Exit 0 = routes_visited>=22 AND bypass active; non-zero otherwise.
 * playwright core only (already installed). No new deps.
 */
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = path.resolve(__dirname, "..", "..");
const REPO_ROOT = path.resolve(FRONTEND_ROOT, "..");
const APP_DIR = path.join(FRONTEND_ROOT, "src", "app");

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}
const BASE = arg("--base", "http://localhost:3100").replace(/\/$/, "");
const NAV_TIMEOUT = 30000;
const SETTLE_MS = 1500;

// date stamp (this is a plain Node script run via Bash -- Date is available)
const DATE = new Date().toISOString().slice(0, 10);
const OUT_DIR = path.join(REPO_ROOT, "handoff", "away_ops", `route_walk_${DATE}`);
const SHOT_DIR = path.join(OUT_DIR, "screenshots");

// ---- route enumeration (glob src/app/**/page.tsx) ------------------------
function findPageFiles(dir) {
  const out = [];
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) out.push(...findPageFiles(full));
    else if (ent.name === "page.tsx") out.push(full);
  }
  return out;
}
function fileToRoute(file) {
  let rel = path.relative(APP_DIR, path.dirname(file));
  if (rel === "") return "/";
  const segs = rel
    .split(path.sep)
    .filter((s) => !/^\(.*\)$/.test(s)); // drop route groups (…)
  return "/" + segs.join("/");
}
function slug(route) {
  return route === "/" ? "root" : route.replace(/^\//, "").replace(/[\/\[\]]/g, "_");
}

// benign network noise -- NOT defects (favicon, source maps, dev HMR, ext)
function benign(u) {
  return (
    /favicon\.ico($|\?)/.test(u) ||
    /\.map($|\?)/.test(u) ||
    /manifest\.(webmanifest|json)/.test(u) ||
    /chrome-extension:\/\//.test(u) ||
    /hot-update\.(json|js)/.test(u) ||
    /\/__nextjs|\/_next\/webpack-hmr/.test(u)
  );
}

async function resolveStrategyId() {
  try {
    const r = await fetch(`${BASE}/api/sovereign/leaderboard`, {
      signal: AbortSignal.timeout(8000),
    });
    if (r.ok) {
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

async function main() {
  fs.mkdirSync(SHOT_DIR, { recursive: true });

  const pageFiles = findPageFiles(APP_DIR);
  const discovered = [...new Set(pageFiles.map(fileToRoute))].sort();
  const strategyId = await resolveStrategyId();

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });

  const results = [];
  let loginRedirects = 0;

  for (const routeTemplate of discovered) {
    const route = routeTemplate.replace(/\[[^\]]+\]/g, strategyId);
    const url = `${BASE}${route}`;
    const consoleErrors = [];
    const pageErrors = [];
    const failedRequests = [];

    const page = await context.newPage();
    page.on("console", (msg) => {
      const t = msg.type();
      if (t === "error" || t === "warning") {
        consoleErrors.push({ type: t, text: msg.text().slice(0, 500) });
      }
    });
    page.on("pageerror", (err) => pageErrors.push(String(err).slice(0, 500)));
    page.on("requestfailed", (req) => {
      const u = req.url();
      if (!benign(u)) {
        failedRequests.push({
          url: u.slice(0, 300),
          method: req.method(),
          status: null,
          error: req.failure()?.errorText || "requestfailed",
        });
      }
    });
    page.on("response", (res) => {
      const st = res.status();
      const u = res.url();
      if (st >= 400 && !benign(u)) {
        failedRequests.push({ url: u.slice(0, 300), method: res.request().method(), status: st, error: null });
      }
    });

    const t0 = Date.now();
    let httpStatus = null;
    try {
      const resp = await page.goto(url, { waitUntil: "load", timeout: NAV_TIMEOUT });
      httpStatus = resp ? resp.status() : null;
      await page.waitForTimeout(SETTLE_MS); // let late console/network settle
    } catch (e) {
      pageErrors.push(`navigation: ${String(e).slice(0, 300)}`);
    }
    const loadMs = Date.now() - t0;
    const finalUrl = page.url();
    const finalPath = (() => {
      try {
        return new URL(finalUrl).pathname;
      } catch {
        return finalUrl;
      }
    })();
    const redirectedToLogin = finalPath.startsWith("/login") && route !== "/login";
    if (redirectedToLogin) loginRedirects++;

    const shotName = `${slug(routeTemplate)}.png`;
    try {
      await page.screenshot({ path: path.join(SHOT_DIR, shotName), fullPage: true });
    } catch {
      // T5: fullPage can exceed the 32767px cap -> fall back to viewport
      try {
        await page.screenshot({ path: path.join(SHOT_DIR, shotName), fullPage: false });
      } catch (e2) {
        pageErrors.push(`screenshot: ${String(e2).slice(0, 200)}`);
      }
    }

    results.push({
      route,
      route_template: routeTemplate,
      final_url: finalUrl,
      http_status: httpStatus,
      redirected_to_login: redirectedToLogin,
      screenshot: `screenshots/${shotName}`,
      console_errors: consoleErrors,
      page_errors: pageErrors,
      failed_requests: failedRequests,
      load_ms: loadMs,
    });
    await page.close();
    process.stdout.write(
      `  visited ${route} (${httpStatus}) console=${consoleErrors.length} failed=${failedRequests.length}${redirectedToLogin ? " [LOGIN-REDIRECT]" : ""}\n`,
    );
  }

  await context.close();
  await browser.close();

  const visitedTemplates = results.map((r) => r.route_template);
  const summary = {
    generated_at: new Date().toISOString(),
    base_url: BASE,
    auth_bypass: "LIGHTHOUSE_SKIP_AUTH=1 + NEXT_PUBLIC_E2E_TESTING=true on :3100 (middleware.ts:24)",
    strategy_id_used: strategyId,
    routes_discovered: discovered.length,
    routes_visited: results.length,
    login_redirect_count: loginRedirects,
    console_error_routes: results.filter((r) => r.console_errors.length).map((r) => r.route),
    failed_request_routes: results.filter((r) => r.failed_requests.length).map((r) => r.route),
    page_error_routes: results.filter((r) => r.page_errors.length).map((r) => r.route),
    route_list_delta: {
      on_disk_not_visited: discovered.filter((r) => !visitedTemplates.includes(r)),
      visited_not_on_disk: visitedTemplates.filter((r) => !discovered.includes(r)),
    },
    routes: results,
  };

  fs.writeFileSync(path.join(OUT_DIR, "walk_summary.json"), JSON.stringify(summary, null, 2));

  // Bypass-misfire guard (T3): if (almost) every non-login route bounced to
  // /login, the skip-auth bypass did not take -> the walk is invalid.
  const nonLogin = discovered.filter((r) => r !== "/login").length;
  const bypassMisfired = loginRedirects >= nonLogin && nonLogin > 0;

  console.log("\n" + JSON.stringify({
    routes_discovered: summary.routes_discovered,
    routes_visited: summary.routes_visited,
    console_error_routes: summary.console_error_routes,
    failed_request_routes: summary.failed_request_routes,
    login_redirect_count: loginRedirects,
    out: path.join(OUT_DIR, "walk_summary.json"),
  }, null, 2));

  if (bypassMisfired) {
    console.error(`\nFAIL: ${loginRedirects}/${nonLogin} non-login routes redirected to /login -- the skip-auth bypass did NOT take. Start :3100 with LIGHTHOUSE_SKIP_AUTH=1.`);
    process.exit(3);
  }
  if (summary.routes_visited < 22) {
    console.error(`\nFAIL: routes_visited=${summary.routes_visited} < 22`);
    process.exit(1);
  }
  process.exit(0);
}

main().catch((e) => {
  console.error("route_walk crashed:", e);
  process.exit(2);
});
