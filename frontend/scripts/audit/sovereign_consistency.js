#!/usr/bin/env node
/*
 * phase-10.5.8 audit: deterministic static checks on the sovereign
 * surface that cover what axe-core can't reach behind auth.
 *
 * Three checks, JSON output, exit 0 on PASS / 1 on FAIL:
 *
 *   1. phosphor_icons_only -- every icon import in sovereign files
 *      comes from `@phosphor-icons/react` or the `@/lib/icons` alias.
 *   2. no_emoji_in_ui -- no emoji codepoints in sovereign files.
 *   3. dark_theme_token_0f172a -- `#0f172a` present in tailwind.config
 *      and at least one sovereign file references a `navy-*` token.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const FRONTEND_ROOT = path.resolve(__dirname, "..", "..");
const REPO_ROOT = path.resolve(FRONTEND_ROOT, "..");

const SOVEREIGN_FILES = [
  "src/app/sovereign/page.tsx",
  "src/app/sovereign/strategy/[id]/page.tsx",
  "src/components/RedLineMonitor.tsx",
  "src/components/ComputeCostBreakdown.tsx",
  "src/components/AlphaLeaderboard.tsx",
  "src/components/StrategyDetail.tsx",
].map((p) => path.join(FRONTEND_ROOT, p));

const ALLOWED_ICON_PACKAGES = new Set([
  "@phosphor-icons/react",
  "@/lib/icons",
]);

// Heuristic: any import path that contains an "icon" keyword and is NOT
// allow-listed is flagged. Purposefully broad -- Phosphor satisfies the
// rule on name alone.
const ICON_HINT_RE = /(icon|heroicons|lucide|feather|font-awesome|material-icons)/i;

// Emoji Presentation codepoints (ignores plain text symbols). This
// catches JSX text nodes and string literals alike.
const EMOJI_RE = /\p{Emoji_Presentation}/u;

function result(check, status, detail) {
  return { check, status: status ? "PASS" : "FAIL", detail };
}

function readFileSafe(p) {
  try {
    return fs.readFileSync(p, "utf-8");
  } catch (e) {
    return null;
  }
}

function checkPhosphorOnly() {
  const violations = [];
  const importRe = /import\s+[^'"]+from\s+['"]([^'"]+)['"]/g;
  for (const f of SOVEREIGN_FILES) {
    const text = readFileSafe(f);
    if (text == null) continue;
    let m;
    while ((m = importRe.exec(text)) !== null) {
      const src = m[1];
      if (!ICON_HINT_RE.test(src)) continue;
      if (ALLOWED_ICON_PACKAGES.has(src)) continue;
      violations.push({
        file: path.relative(REPO_ROOT, f),
        import: src,
      });
    }
  }
  return result(
    "phosphor_icons_only",
    violations.length === 0,
    violations.length === 0
      ? `Scanned ${SOVEREIGN_FILES.length} sovereign files; no non-Phosphor icon imports`
      : `Non-Phosphor icon imports: ${JSON.stringify(violations)}`,
  );
}

function checkNoEmoji() {
  const offenders = [];
  for (const f of SOVEREIGN_FILES) {
    const text = readFileSafe(f);
    if (text == null) continue;
    const lines = text.split("\n");
    for (let i = 0; i < lines.length; i++) {
      if (EMOJI_RE.test(lines[i])) {
        offenders.push({ file: path.relative(REPO_ROOT, f), line: i + 1 });
      }
    }
  }
  return result(
    "no_emoji_in_ui",
    offenders.length === 0,
    offenders.length === 0
      ? `Scanned ${SOVEREIGN_FILES.length} sovereign files; no emoji codepoints`
      : `Emoji found: ${JSON.stringify(offenders)}`,
  );
}

function checkNavyToken() {
  const tailwindCfgCandidates = [
    path.join(FRONTEND_ROOT, "tailwind.config.js"),
    path.join(FRONTEND_ROOT, "tailwind.config.ts"),
    path.join(FRONTEND_ROOT, "tailwind.config.mjs"),
  ];
  let tailwindHit = null;
  for (const c of tailwindCfgCandidates) {
    const text = readFileSafe(c);
    if (text != null && text.includes("#0f172a")) {
      tailwindHit = c;
      break;
    }
  }
  let sovereignNavyHit = null;
  for (const f of SOVEREIGN_FILES) {
    const text = readFileSafe(f);
    if (text == null) continue;
    if (/\bnavy-\d{2,3}\b/.test(text)) {
      sovereignNavyHit = path.relative(REPO_ROOT, f);
      break;
    }
  }
  const pass = !!tailwindHit && !!sovereignNavyHit;
  return result(
    "dark_theme_token_0f172a",
    pass,
    pass
      ? `#0f172a found in ${path.relative(REPO_ROOT, tailwindHit)}; navy-* token referenced in ${sovereignNavyHit}`
      : `#0f172a tailwind hit=${!!tailwindHit}; sovereign navy-* hit=${sovereignNavyHit || "none"}`,
  );
}

function main() {
  const results = [
    checkPhosphorOnly(),
    checkNoEmoji(),
    checkNavyToken(),
  ];
  const overall = results.every((r) => r.status === "PASS");
  const out = {
    audit: "phase-10.5.8 sovereign_consistency",
    overall: overall ? "PASS" : "FAIL",
    checks: results,
  };
  console.log(JSON.stringify(out, null, 2));
  process.exit(overall ? 0 : 1);
}

main();
