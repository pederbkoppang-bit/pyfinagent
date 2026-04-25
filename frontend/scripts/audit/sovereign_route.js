#!/usr/bin/env node
/*
 * phase-16.33 audit: sovereign route reachability + sidebar entry + page-shell shape.
 *
 * Three checks, JSON output, exit 0 on PASS / 1 on any FAIL:
 *
 *   1. route_reachable -- GET http://localhost:3000/sovereign returns
 *      200 (signed in) or 302 (NextAuth redirect to /login).
 *   2. sidebar_entry_added -- frontend/src/components/Sidebar.tsx contains
 *      a NavSection item with `href: "/sovereign"`.
 *   3. page_shell_conforms_to_frontend_layout -- sovereign/page.tsx
 *      uses the canonical shell (`flex h-screen overflow-hidden` +
 *      `<Sidebar` + `<main`), per .claude/rules/frontend-layout.md.
 *
 * Stdlib only (`http`, `fs`, `path`). No npm deps.
 */
"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");

const FRONTEND_ROOT = path.resolve(__dirname, "..", "..");

const SIDEBAR_FILE = path.join(FRONTEND_ROOT, "src/components/Sidebar.tsx");
const PAGE_FILE = path.join(FRONTEND_ROOT, "src/app/sovereign/page.tsx");

const HOST = "localhost";
const PORT = 3000;
const ROUTE = "/sovereign";
const TIMEOUT_MS = 5000;

function pass(check, detail) {
  return { check, status: "PASS", detail };
}
function fail(check, detail) {
  return { check, status: "FAIL", detail };
}

function readFileSafe(p) {
  try {
    return fs.readFileSync(p, "utf8");
  } catch (e) {
    return null;
  }
}

function probeRoute() {
  return new Promise((resolve) => {
    const req = http.request(
      {
        host: HOST,
        port: PORT,
        path: ROUTE,
        method: "GET",
        timeout: TIMEOUT_MS,
      },
      (res) => {
        const code = res.statusCode || 0;
        if (code === 200 || code === 302 || code === 307) {
          resolve(pass("route_reachable", `HTTP ${code} for http://${HOST}:${PORT}${ROUTE}`));
        } else {
          resolve(fail("route_reachable", `HTTP ${code} for http://${HOST}:${PORT}${ROUTE} (expected 200/302/307)`));
        }
        res.resume();
      },
    );
    req.on("error", (err) => {
      resolve(fail("route_reachable", `request failed: ${err.message} -- is "npm run dev" running on port ${PORT}?`));
    });
    req.on("timeout", () => {
      req.destroy();
      resolve(fail("route_reachable", `request timed out after ${TIMEOUT_MS}ms`));
    });
    req.end();
  });
}

function checkSidebarEntry() {
  const content = readFileSafe(SIDEBAR_FILE);
  if (content === null) {
    return fail("sidebar_entry_added", `cannot read ${SIDEBAR_FILE}`);
  }
  // Looking for an entry like: { href: "/sovereign", label: "Sovereign", icon: NavSovereign }
  // Be lenient on quote style + spacing.
  const re = /href:\s*["'`]\/sovereign["'`]/;
  if (re.test(content)) {
    return pass("sidebar_entry_added", `Sidebar.tsx has href: "/sovereign" entry`);
  }
  return fail(
    "sidebar_entry_added",
    `Sidebar.tsx is missing an entry with href: "/sovereign" -- add one to NAV_SECTIONS`,
  );
}

function checkPageShell() {
  const content = readFileSafe(PAGE_FILE);
  if (content === null) {
    return fail("page_shell_conforms_to_frontend_layout", `cannot read ${PAGE_FILE}`);
  }
  const checks = [
    { token: "flex h-screen overflow-hidden", label: "outer flex+h-screen+overflow-hidden" },
    { token: "<Sidebar", label: "<Sidebar component" },
    { token: "<main", label: "<main wrapper" },
  ];
  const missing = checks.filter((c) => !content.includes(c.token));
  if (missing.length === 0) {
    return pass(
      "page_shell_conforms_to_frontend_layout",
      `sovereign/page.tsx has all 3 canonical shell tokens`,
    );
  }
  return fail(
    "page_shell_conforms_to_frontend_layout",
    `missing: ${missing.map((m) => m.label).join(", ")}`,
  );
}

async function main() {
  const results = [
    await probeRoute(),
    checkSidebarEntry(),
    checkPageShell(),
  ];
  const allPass = results.every((r) => r.status === "PASS");
  console.log(JSON.stringify({
    audit: "sovereign_route",
    timestamp: new Date().toISOString(),
    overall: allPass ? "PASS" : "FAIL",
    checks: results,
  }, null, 2));
  process.exit(allPass ? 0 : 1);
}

main();
