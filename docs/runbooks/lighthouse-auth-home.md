# Lighthouse Audit: Authenticated Home View

phase-16.41 (closes follow-up #8 from phase-10.5.7).

## Purpose

Run Lighthouse against the **authenticated** home view at
`http://localhost:3000/` -- where the Red Line hero embed
(phase-10.5.7) actually lives. The unauthenticated audit hits
`/login`, so it cannot measure the real cockpit shell.

## Mechanism

`frontend/src/middleware.ts:24` skips the NextAuth redirect when
`process.env.LIGHTHOUSE_SKIP_AUTH === "1"` is set on the **dev
server's** environment. The audit script uses this hatch instead of
minting a JWE session token (which would require AUTH_SECRET access
and is fragile against Auth.js v5 changes).

## How to run

In **terminal 1** (start the dev server with the bypass active):

```bash
cd frontend
LIGHTHOUSE_SKIP_AUTH=1 npm run dev
```

Wait for `Ready in ...` then in **terminal 2**:

```bash
cd frontend
npm run lighthouse:auth-home
```

Output:

- Stdout: JSON envelope with `audit`, `overall`, `checks`,
  `output_report`.
- File: `handoff/lighthouse_authenticated_home.json` (full Lighthouse
  report; ~1-2 MB JSON).
- Exit code: 0 = PASS, 1 = audit ran but a check failed (e.g. landed
  on `/login`), 2 = bypass is not active (operator setup needed).

## Probe-only mode

For CI / masterplan verification, the script supports a fast static
probe:

```bash
node frontend/scripts/audit/lighthouse_auth_home.js --probe-only
```

This only checks that the bypass is active (HTTP 200 for `/`); it
does NOT run Lighthouse. Use this in pre-flight scripts where a
full Lighthouse run is too expensive.

## Reading the report

The Lighthouse JSON has the standard schema:

- `categories.performance.score` — 0-1 (0.9+ is "Good")
- `categories.accessibility.score`
- `categories.best-practices.score`
- `categories.seo.score`
- `audits.<name>.score` — per-audit scores (e.g. `unused-javascript`,
  `largest-contentful-paint`, `cumulative-layout-shift`)
- `finalUrl` — should NOT contain `/login` (the audit script asserts
  this; if it does, the bypass failed).

To extract just the four category scores:

```bash
jq '.categories | to_entries | map({(.key): .value.score}) | add' \
  handoff/lighthouse_authenticated_home.json
```

## Troubleshooting

- **Exit 2 with "bypass not active":** dev server is running without
  `LIGHTHOUSE_SKIP_AUTH=1`. Restart it with the env var.
- **Exit 2 with "request failed: ECONNREFUSED":** dev server isn't
  running on port 3000. Start it.
- **`finalUrl` contains `/login`:** somehow the audit was redirected
  despite the probe passing. Check that `middleware.ts:24` still has
  the `LIGHTHOUSE_SKIP_AUTH` clause; it shouldn't have been removed.
- **Chrome not found:** the wrapper falls back to bundled Chrome at
  `frontend/chrome/`. If missing, install via
  `npx puppeteer browsers install chrome` or set `CHROME_PATH` env.

## Related infrastructure

- `frontend/scripts/audit/lighthouse-wrapper.js` (phase-16.33) -- the
  shared `--url` argv translator + bundled Chrome path. This script
  invokes it via `spawnSync`.
- `frontend/scripts/audit/sovereign_route.js` (phase-16.33) -- the
  unauthenticated sovereign-route audit. Pattern reference for this
  script's shape.
- `frontend/src/middleware.ts:24` -- the bypass clause. Do not remove
  without first updating this script.

## Security note

`LIGHTHOUSE_SKIP_AUTH=1` MUST never be set in production
environments. The middleware bypass is a dev/test-only hatch. The env
var is checked at request time on the server, so any process running
with it set effectively serves the entire app unauthenticated. Audit
your `.env.production` and CI/CD pipelines accordingly.
