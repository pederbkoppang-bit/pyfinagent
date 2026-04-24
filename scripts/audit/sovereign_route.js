#!/usr/bin/env node
/*
 * phase-10.5.2 audit: verify the /sovereign route shell satisfies the
 * three immutable success criteria via deterministic static analysis:
 *
 *   1. route_reachable                     -- frontend/src/app/sovereign/page.tsx exists
 *   2. sidebar_entry_added                 -- Sidebar.tsx links to /sovereign
 *   3. page_shell_conforms_to_frontend_layout -- page.tsx contains the four
 *      mandatory layout tokens from .claude/rules/frontend-layout.md §1
 *
 * JSON stdout. Exit 0 on PASS, 1 on FAIL.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const REPO_ROOT = path.resolve(__dirname, "..", "..");
const PAGE_PATH = path.join(REPO_ROOT, "frontend", "src", "app", "sovereign", "page.tsx");
const SIDEBAR_PATH = path.join(REPO_ROOT, "frontend", "src", "components", "Sidebar.tsx");

// Mandatory shell tokens from .claude/rules/frontend-layout.md §1.
const SHELL_TOKENS = [
  "flex h-screen overflow-hidden",
  "Sidebar",
  "flex-shrink-0",
  "overflow-y-auto scrollbar-thin",
];

function check(label, condition, detail) {
  return { check: label, status: condition ? "PASS" : "FAIL", detail };
}

function main() {
  const results = [];

  // 1. route_reachable
  const pageExists = fs.existsSync(PAGE_PATH);
  results.push(
    check("route_reachable", pageExists, `${pageExists ? "found" : "missing"}: ${PAGE_PATH}`),
  );

  // 2. sidebar_entry_added
  let sidebarHasLink = false;
  let sidebarDetail = `missing: ${SIDEBAR_PATH}`;
  if (fs.existsSync(SIDEBAR_PATH)) {
    const sidebar = fs.readFileSync(SIDEBAR_PATH, "utf-8");
    sidebarHasLink = sidebar.includes('href: "/sovereign"') || sidebar.includes('href:"/sovereign"');
    sidebarDetail = sidebarHasLink
      ? "Sidebar.tsx contains href: \"/sovereign\""
      : "Sidebar.tsx does NOT contain href: \"/sovereign\"";
  }
  results.push(check("sidebar_entry_added", sidebarHasLink, sidebarDetail));

  // 3. page_shell_conforms_to_frontend_layout
  let shellOk = false;
  let shellDetail = "page missing";
  if (pageExists) {
    const page = fs.readFileSync(PAGE_PATH, "utf-8");
    const missing = SHELL_TOKENS.filter((t) => !page.includes(t));
    shellOk = missing.length === 0;
    shellDetail = shellOk
      ? `all ${SHELL_TOKENS.length} shell tokens present`
      : `missing tokens: ${JSON.stringify(missing)}`;
  }
  results.push(
    check("page_shell_conforms_to_frontend_layout", shellOk, shellDetail),
  );

  const overall = results.every((r) => r.status === "PASS");
  const out = {
    audit: "phase-10.5.2 sovereign_route",
    overall: overall ? "PASS" : "FAIL",
    checks: results,
  };
  console.log(JSON.stringify(out, null, 2));
  process.exit(overall ? 0 : 1);
}

main();
