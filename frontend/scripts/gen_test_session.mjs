/**
 * Generate a signed Auth.js v5 JWE session cookie for the phase-4.6.6
 * smoketest. Prints JSON to stdout; Playwright injects via addCookies.
 *
 * Run from `frontend/` so Node resolves next-auth/jwt correctly.
 *
 * Usage:
 *   AUTH_SECRET=... node scripts/gen_test_session.mjs
 *
 * Defaults: cookie name authjs.session-token (HTTP). For HTTPS, set
 * AUTH_COOKIE_NAME=__Secure-authjs.session-token.
 *
 * See handoff/current/contract.md for the architectural rationale.
 */
import { encode } from "next-auth/jwt";
import { readFileSync, existsSync } from "node:fs";

// Auto-load .env.local so the AUTH_SECRET the dev server uses is visible
// to this script without shell plumbing. Silently no-ops if the file is
// missing or the env var is already set on the caller side.
if (!process.env.AUTH_SECRET) {
  for (const p of [".env.local", ".env"]) {
    if (!existsSync(p)) continue;
    for (const line of readFileSync(p, "utf8").split("\n")) {
      const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
      if (!m) continue;
      const [, k, vRaw] = m;
      if (!process.env[k]) {
        process.env[k] = vRaw.replace(/^['"]|['"]$/g, "");
      }
    }
  }
}

const secret = process.env.AUTH_SECRET;
if (!secret) {
  console.error("AUTH_SECRET env-var is required");
  process.exit(2);
}

const cookieName = process.env.AUTH_COOKIE_NAME || "authjs.session-token";
const email = process.env.SMOKETEST_EMAIL || "smoketest@pyfinagent.local";

const token = await encode({
  secret,
  salt: cookieName,
  token: {
    sub: "smoketest-user",
    email,
    name: "Smoketest",
    iat: Math.floor(Date.now() / 1000),
  },
  maxAge: 3600,
});

const out = {
  cookie_name: cookieName,
  cookie_value: token,
  domain: "localhost",
  path: "/",
  httpOnly: true,
  secure: cookieName.startsWith("__Secure-"),
  sameSite: "Lax",
  expires: Math.floor(Date.now() / 1000) + 3600,
};
console.log(JSON.stringify(out));
