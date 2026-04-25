#!/usr/bin/env node
/*
 * phase-16.41 (#8): authenticated-home lighthouse harness.
 *
 * Audits the authenticated home view (the Red Line hero embed added
 * in phase-10.5.7) using the existing LIGHTHOUSE_SKIP_AUTH=1
 * middleware bypass at frontend/src/middleware.ts:24.
 *
 * The bypass is server-side, so the Next.js dev server MUST be
 * started with the env var set:
 *
 *   LIGHTHOUSE_SKIP_AUTH=1 npm run dev
 *
 * Then run this audit from the frontend dir:
 *
 *   npm run lighthouse:auth-home
 *
 * Modes:
 *   default       -- probe + lighthouse run + finalUrl check
 *   --probe-only  -- only probe (fast; for masterplan immutable verification)
 *
 * Why bypass instead of JWE token mint:
 *   - Auth.js v5 encode() coupling is fragile (per embracethered.com 2026)
 *   - AUTH_SECRET lives in .env.local (not version-controlled)
 *   - The env-var is the in-app legitimate hatch (server file middleware.ts:24
 *     already exists with comment "perf measurement on cockpit")
 *
 * Output:
 *   stdout: JSON result envelope (matches sovereign_route.js shape)
 *   handoff/lighthouse_authenticated_home.json: full lighthouse report
 *   exit 0 = PASS, 1 = FAIL, 2 = bypass not active (operator setup needed)
 *
 * Stdlib + npm lighthouse only. No new deps.
 */
"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");
const { spawnSync } = require("child_process");

const FRONTEND_ROOT = path.resolve(__dirname, "..", "..");
const REPO_ROOT = path.resolve(FRONTEND_ROOT, "..");

const HOST = "localhost";
const PORT = 3000;
const ROUTE = "/";
const TIMEOUT_MS = 5000;

const OUTPUT_PATH = path.join(REPO_ROOT, "handoff", "lighthouse_authenticated_home.json");

const PROBE_ONLY = process.argv.includes("--probe-only");

function pass(check, detail) {
  return { check, status: "PASS", detail };
}
function fail(check, detail) {
  return { check, status: "FAIL", detail };
}

function probeBypass() {
  // If LIGHTHOUSE_SKIP_AUTH=1 is active in the dev server's env,
  // GET / returns 200. Without it, middleware redirects to /login (302).
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
        if (code === 200) {
          resolve(pass(
            "lighthouse_skip_auth_bypass_active",
            `HTTP 200 for http://${HOST}:${PORT}${ROUTE} (bypass active)`,
          ));
        } else if (code === 302 || code === 307) {
          const location = res.headers.location || "<unknown>";
          resolve(fail(
            "lighthouse_skip_auth_bypass_active",
            `HTTP ${code} -> ${location}. The dev server is NOT running with ` +
            `LIGHTHOUSE_SKIP_AUTH=1. Restart it with: ` +
            `LIGHTHOUSE_SKIP_AUTH=1 npm run dev`,
          ));
        } else {
          resolve(fail(
            "lighthouse_skip_auth_bypass_active",
            `HTTP ${code} for http://${HOST}:${PORT}${ROUTE} (expected 200; bypass inactive)`,
          ));
        }
        res.resume();
      },
    );
    req.on("error", (err) => {
      resolve(fail(
        "lighthouse_skip_auth_bypass_active",
        `request failed: ${err.message} -- is the dev server running on port ${PORT}? ` +
        `Start with: LIGHTHOUSE_SKIP_AUTH=1 npm run dev`,
      ));
    });
    req.on("timeout", () => {
      req.destroy();
      resolve(fail(
        "lighthouse_skip_auth_bypass_active",
        `request timed out after ${TIMEOUT_MS}ms -- dev server not responding`,
      ));
    });
    req.end();
  });
}

function runLighthouse() {
  // Ensure the output dir exists
  const outDir = path.dirname(OUTPUT_PATH);
  try {
    fs.mkdirSync(outDir, { recursive: true });
  } catch (e) {
    return fail("lighthouse_output_dir_writable", `mkdir failed: ${e.message}`);
  }

  // Reuse the existing wrapper for chrome-path discovery + --url translation.
  const wrapper = path.join(FRONTEND_ROOT, "scripts", "audit", "lighthouse-wrapper.js");
  const args = [
    wrapper,
    `--url`, `http://${HOST}:${PORT}${ROUTE}`,
    `--output`, "json",
    `--output-path`, OUTPUT_PATH,
    `--quiet`,
    `--chrome-flags=--headless`,
  ];

  const result = spawnSync("node", args, {
    stdio: "inherit",
    env: { ...process.env },
  });

  if (result.status !== 0) {
    return fail(
      "lighthouse_run_completed",
      `lighthouse exited with status ${result.status}; check stderr above`,
    );
  }
  if (!fs.existsSync(OUTPUT_PATH)) {
    return fail(
      "lighthouse_run_completed",
      `lighthouse exited 0 but ${OUTPUT_PATH} was not produced`,
    );
  }
  return pass(
    "lighthouse_run_completed",
    `report saved to ${OUTPUT_PATH}`,
  );
}

function checkFinalUrl() {
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(OUTPUT_PATH, "utf8"));
  } catch (e) {
    return fail("audit_landed_on_authenticated_view", `cannot parse report: ${e.message}`);
  }
  // Lighthouse v13 puts finalUrl at the top of the report; older versions
  // nest under .lhr.finalUrl. Try both.
  const finalUrl =
    parsed.finalUrl ||
    parsed.finalDisplayedUrl ||
    (parsed.lhr && parsed.lhr.finalUrl) ||
    "";
  if (!finalUrl) {
    return fail(
      "audit_landed_on_authenticated_view",
      `no finalUrl found in report (keys: ${Object.keys(parsed).slice(0, 8).join(", ")})`,
    );
  }
  if (finalUrl.includes("/login")) {
    return fail(
      "audit_landed_on_authenticated_view",
      `lighthouse landed on ${finalUrl} -- bypass did NOT work; check middleware`,
    );
  }
  return pass(
    "audit_landed_on_authenticated_view",
    `finalUrl=${finalUrl} (no /login redirect)`,
  );
}

async function main() {
  const probe = await probeBypass();
  const checks = [probe];

  if (PROBE_ONLY) {
    const overall = probe.status === "PASS" ? "PASS" : "FAIL";
    console.log(JSON.stringify({
      audit: "lighthouse_auth_home",
      mode: "probe-only",
      timestamp: new Date().toISOString(),
      overall,
      checks,
    }, null, 2));
    process.exit(probe.status === "PASS" ? 0 : 2);
  }

  if (probe.status !== "PASS") {
    console.log(JSON.stringify({
      audit: "lighthouse_auth_home",
      mode: "full",
      timestamp: new Date().toISOString(),
      overall: "FAIL",
      checks,
    }, null, 2));
    process.exit(2);
  }

  const lighthouseResult = runLighthouse();
  checks.push(lighthouseResult);
  if (lighthouseResult.status === "PASS") {
    checks.push(checkFinalUrl());
  }

  const allPass = checks.every((r) => r.status === "PASS");
  console.log(JSON.stringify({
    audit: "lighthouse_auth_home",
    mode: "full",
    timestamp: new Date().toISOString(),
    overall: allPass ? "PASS" : "FAIL",
    checks,
    output_report: OUTPUT_PATH,
  }, null, 2));
  process.exit(allPass ? 0 : 1);
}

main();
