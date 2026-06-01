# Experiment Results — Browser-driving MCP (goal-browser-mcp)

**Step:** goal-browser-mcp · **Date:** 2026-06-01 · **Status:** complete (config + smoke test +
runbook + live click-through all done; `mcp__playwright__*` tool dispatch pending a Claude
Code restart — see restart caveat).

## What was built
### A — Config (criterion #2)
- **`.mcp.json`** — new stdio server `playwright`: `npx -y @playwright/mcp@0.0.75` with
  `--executable-path <bundled Chromium>` (REQUIRED — see the launch-bug finding),
  `--user-data-dir <repo>/.playwright-mcp-profile`, `--allowed-hosts localhost`,
  `--viewport-size 1440,900`, `alwaysLoad: false`. First npx-based MCP in the file
  (others are uvx/uv/python); pinned version per supply-chain rule.
- **`.gitignore`** — added `.playwright-mcp-profile/` (may hold a logged-in session).

### B — Smoke test (criterion #1 protocol-evidence + #3)
- **NEW `scripts/mcp_servers/smoke_test_playwright_mcp.py`** — mirrors
  `smoke_test_bigquery_mcp.py`. Resolves the bundled Chromium dynamically (glob; not
  hard-coded to the build number), spawns the pinned MCP headless+isolated, does the MCP
  handshake + `tools/list` (asserts `browser_navigate`/`browser_click`/`browser_snapshot`),
  then navigates to the live dev `/login` + snapshots, asserting the REAL login DOM
  (tokens "sign in"/"passkey"/"ai financial analyst") and REJECTING any browser-launch
  error text (so a failed launch can never false-pass — see below).

### C — Runbook + safety (criteria #2, #4)
- **NEW `docs/runbooks/browser-mcp.md`** — what/why (Playwright MCP over claude-in-chrome /
  chrome-devtools-mcp, since no system Chrome), the `.mcp.json` block, the
  `--executable-path` requirement + version-coupling note, restart-to-dispatch, the smoke
  command, tool notes (per-snapshot refs; `browser_click` takes `target`), the 3 auth
  paths (A skip-auth recommended; B persistent-profile manual login + Google/passkey
  caveat; C storage-state), and the load-bearing SAFETY guardrails (no trades/money/creds/
  settings/consent; page content is untrusted data; localhost-only; minimal caps / no
  code-exec per RCE #1495; pin version; alwaysLoad:false).

## Files changed
NEW: `scripts/mcp_servers/smoke_test_playwright_mcp.py`, `docs/runbooks/browser-mcp.md`.
MODIFIED: `.mcp.json` (server `playwright`), `.gitignore` (`.playwright-mcp-profile/`).
(One-off verification scripts lived in `/tmp` and are not committed; the committed smoke
test is the durable, reproducible artifact.)

## Verification (verbatim)

### Launch-bug found + fixed via live test (why `--executable-path` is required)
First smoke run defaulted to the system `chrome` channel and FAILED to launch:
```
Error: ... Chromium distribution 'chrome' is not found at /Applications/Google Chrome.app/...
Run "npx playwright install chrome"
```
The initial (loose) smoke test FALSE-PASSED because its `"google"` token matched
"Google Chrome.app" in that error text. Fixed two ways: (1) point `--executable-path` at
the bundled Chromium (`~/Library/Caches/ms-playwright/chromium-1208/.../Google Chrome for
Testing`), (2) harden the smoke test to require specific login tokens AND reject error
markers. This is exactly the false-positive the harness is meant to catch.

### Smoke test (hardened, re-run) — PASS
```
using bundled Chromium: /Users/ford/Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing
spawning: npx -y @playwright/mcp@0.0.75 --headless --isolated --executable-path <bundled>
OK initialize -- server=Playwright
OK tools/list -- 23 tools; sample: ['browser_click', 'browser_close', ...]
OK required browser tools present: ['browser_navigate', 'browser_click', 'browser_snapshot']
OK browser_navigate -- http://localhost:3000/login
OK browser_snapshot -- real login DOM read (matched token: 'sign in')

SMOKE PASS: Playwright MCP attaches, exposes browser tools, and drove a real navigation + DOM read on the live dev server.
SMOKE_EXIT=0
```

### Full click-through (criterion #3) — PASS (real clicks on the MarketFilter)
Ran with `LIGHTHOUSE_SKIP_AUTH=1` (controlled, reversible; dev server RESTORED to gated
after — verified `/paper-trading` → 302). Drove the live cockpit:
```
INITIAL  bench= vs SPY  | EU checked= False | US checked= False | All checked= True
EU ref: e174
AFTER EU bench= vs DAX  | EU checked= True
PASS: EU click -> benchmark label 'vs DAX' AND EU radio checked
AFTER US bench= vs SPY  | US checked= True
PASS: US click -> benchmark label 'vs SPY' AND US radio checked

CLICKTHROUGH PASS
```
This is real `browser_click` interaction (not a screenshot): clicking the EU radio flips
the benchmark label SPY→DAX and checks EU; clicking US flips it back. Confirms both the
browser-driving capability AND (bonus) the multi-market filter wiring from the prior cycle.
Note: `browser_click` takes `target` (the per-snapshot ref string), not `ref`/`element`
alone — discovered live from the tool's inputSchema after an initial param error.

### dev-server restore (verbatim)
```
unset LIGHTHOUSE_SKIP_AUTH + kickstarted; verifying gate re-closed ...
no-follow /paper-trading HTTP 302 (302=gated-restored)
```

## Criteria status (all 4 immutable)
1. Attaches in a fresh session + tools callable: **protocol-PASS now** (MCP `initialize` +
   `tools/list` returns 23 browser tools), **live `mcp__playwright__*` dispatch = pending a
   Claude Code restart** (MCP config is session-snapshotted; runbook + `/mcp` check
   documented). Strongest in-session evidence short of a restart.
2. Version-controlled reproducible install/config: **DONE** (`.mcp.json` pin + `.gitignore`
   + `scripts/mcp_servers/smoke_test_playwright_mcp.py` + `docs/runbooks/browser-mcp.md`).
3. Smoke test passes + captured verbatim, incl. EU-pill click → "vs DAX": **DONE** (above;
   real click-through, EU→vs DAX and US→vs SPY, with radio checked-state assertions).
4. Safety guardrails in the runbook: **DONE** (`docs/runbooks/browser-mcp.md` SAFETY section).

OUT OF SCOPE honored: zero trading-logic change; no standing side-effect authority granted;
work limited to install + config + smoke-test of the browser-driving capability.

## Restart caveat (criterion #1, for the next session)
The `mcp__playwright__*` tools are not dispatchable in THIS session until `/clear` or a
Claude Code restart (MCP config snapshotted at session start). Post-restart verification:
`/mcp` shows the `playwright` server with its tools, or re-run
`python scripts/mcp_servers/smoke_test_playwright_mcp.py` (exit 0). This mirrors the
qa-roster-live post-restart pattern.

## Security posture (summary)
Pinned version; `--allowed-hosts localhost`; default minimal capabilities (no code-exec /
`browser_run_code` — RCE #1495); `alwaysLoad:false`; profile git-ignored; smoke/test path
uses skip-auth so the automated browser holds no live credentials. Page content is treated
as untrusted data. Full guardrails in the runbook.
