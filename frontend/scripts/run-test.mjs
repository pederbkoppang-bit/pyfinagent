#!/usr/bin/env node
// Thin wrapper: translate masterplan `--filter=X` to vitest's positional
// file-pattern arg. vitest's native `--filter` does not exist; its CLI
// accepts positional filename patterns. This script bridges the two so
// the masterplan immutable verification command works as-written.
//
//   npm run test -- --filter=AutoresearchLeaderboard
//     ->  vitest run AutoresearchLeaderboard

import { spawn } from "node:child_process";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = resolve(__dirname, "..");
const VITEST_BIN = resolve(FRONTEND_ROOT, "node_modules", ".bin", "vitest");

const raw = process.argv.slice(2);
const rewritten = [];
for (const arg of raw) {
  if (arg.startsWith("--filter=")) {
    rewritten.push(arg.slice("--filter=".length));
  } else if (arg === "--filter") {
    // Next arg is the filter value; consume it in the next iteration.
    // We emulate by pushing a sentinel and peeking; simpler: just skip
    // without value coupling because masterplan uses the `=` form.
    continue;
  } else {
    rewritten.push(arg);
  }
}

const args = ["run", ...rewritten];
const child = spawn(VITEST_BIN, args, {
  stdio: "inherit",
  cwd: FRONTEND_ROOT,
});
child.on("exit", (code) => process.exit(code ?? 1));
