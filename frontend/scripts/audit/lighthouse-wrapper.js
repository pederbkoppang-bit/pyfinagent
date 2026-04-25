#!/usr/bin/env node
/*
 * phase-16.33 lighthouse wrapper -- translates `--url X` to positional `X`.
 *
 * Background: the lighthouse v13.x CLI takes the URL POSITIONALLY:
 *   lighthouse <url> [flags]
 *
 * The masterplan-immutable verification command for phase-10.5.7 invokes:
 *   npm run lighthouse -- --url http://localhost:3000 --output json --output-path X
 *
 * `--url` is unrecognized by lighthouse and the run fails. Per CLAUDE.md
 * verification commands are immutable; this wrapper closes the gap by
 * translating `--url X` (or `--url=X`) into the positional argument
 * before invoking lighthouse, while passing every other flag through
 * untouched. Stdlib only (no new npm deps).
 */
"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

function extractUrl(argv) {
  const out = [];
  let url = null;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--url" && i + 1 < argv.length) {
      url = argv[i + 1];
      i++;
      continue;
    }
    if (a.startsWith("--url=")) {
      url = a.slice("--url=".length);
      continue;
    }
    out.push(a);
  }
  return { url, rest: out };
}

function bundledChromePath() {
  // Project ships its own Chrome for Testing under frontend/chrome/...
  // Used by the `axe` script (see package.json:axe). Mirror its discovery
  // so the masterplan-immutable lighthouse command Just Works without
  // requiring CHROME_PATH to be set externally.
  const chromeRoot = path.resolve(__dirname, "..", "..", "chrome");
  if (!fs.existsSync(chromeRoot)) return null;
  try {
    const versions = fs.readdirSync(chromeRoot).filter((d) => d.startsWith("mac_") || d.startsWith("linux") || d.startsWith("win"));
    if (versions.length === 0) return null;
    versions.sort();
    const v = versions[versions.length - 1];
    const candidate = path.join(
      chromeRoot, v,
      "chrome-mac-arm64", "Google Chrome for Testing.app", "Contents", "MacOS",
      "Google Chrome for Testing",
    );
    if (fs.existsSync(candidate)) return candidate;
  } catch (e) {
    // fall through
  }
  return null;
}

const { url, rest } = extractUrl(process.argv.slice(2));
const args = url ? [url, ...rest] : rest;

// Auto-set CHROME_PATH env var if not already set and a bundled Chrome exists.
// chrome-launcher (lighthouse's transitive dep) discovers the binary via the
// CHROME_PATH env var, NOT a CLI flag. The `axe` script in package.json uses
// the same path -- mirror it here.
const env = { ...process.env };
if (!env.CHROME_PATH) {
  const cp = bundledChromePath();
  if (cp) {
    env.CHROME_PATH = cp;
  }
}

// phase-16.37 (#51): expose extractUrl for vitest unit testing without
// launching lighthouse. require.main !== module guards the spawn path so
// `require('./lighthouse-wrapper')` from a test file does NOT side-effect.
if (require.main !== module) {
  module.exports = { extractUrl };
} else {
  // Resolve lighthouse from node_modules/.bin so we don't depend on PATH.
  const lighthouseBin = path.resolve(
    __dirname, "..", "..", "node_modules", ".bin", "lighthouse",
  );
  const result = spawnSync(lighthouseBin, args, { stdio: "inherit", env });
  process.exit(result.status ?? 1);
}
