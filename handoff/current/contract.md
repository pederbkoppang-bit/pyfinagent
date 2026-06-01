# Contract — Browser-driving MCP (goal-browser-mcp)

**Step id:** goal-browser-mcp (session `/goal`, not a masterplan phase id)
**Date:** 2026-06-01
**Cycle driver:** Main (Claude Code) + Researcher (done, gate PASSED) + Q/A (pending)

## Research-gate summary (PASSED)
Researcher `a983c70b128d3bdd5` — gate_passed: true (tier complex; 7 sources read in
full, 22 URLs, recency scan, 8 internal files). Brief at `handoff/current/research_brief.md`.
Decisive findings:
- **Pick: Playwright MCP `@playwright/mcp@0.0.75`** (repo `microsoft/playwright-mcp`,
  Apache-2.0, Microsoft-maintained, daily alphas). Bundles its own Chromium — **no
  system Chrome needed** (Chrome is NOT installed on this Mac; only Safari). Chromium
  binary already cached locally (`~/Library/Caches/ms-playwright/chromium-1208`), node
  v25.8.1 + npx present → runs zero-install. Live-verified: `npx -y @playwright/mcp@0.0.75
  --version` → `Version 0.0.75`; `--help` confirms all needed flags.
- **Rejected:** claude-in-chrome (requires Chrome/Edge — not installed; Safari unsupported)
  and chrome-devtools-mcp (requires system Chrome). Both are non-starters here.
- **Auth crux:** the cockpit IS auth-gated in dev by default (re-verified via curl:
  `/paper-trading` → 302 → `/login`; the earlier transient 200 was stale dev state).
  Least-fragile smoke-test path = run the dev server with `LIGHTHOUSE_SKIP_AUTH=1`
  (honored at `frontend/src/middleware.ts:22`, an existing supported code path for
  Lighthouse). Google SSO + passkey from an automation-controlled browser is reliably
  blocked ("browser may not be secure"; passkeys are device-bound) — documented as the
  fragile manual fallback (persistent `--user-data-dir`), NOT used for the smoke test.
- **Security:** Playwright MCP README says it is "not a security boundary." Pin the
  version (no `@latest`), `--allowed-hosts localhost`, keep capabilities minimal (do NOT
  enable `browser_run_code`/vision/pdf/devtools — issue #1495 RCE via page injection),
  treat page content as untrusted data.
- **Smoke selectors (verified):** `MarketFilter.tsx:62/71/77` = `role="radiogroup"` +
  `role="radio"` buttons named `All`/`US`/`EU`/`KR`; `getByRole('radio',{name:'EU'})`.
  Benchmark label at `cockpit-helpers.tsx:198/226` = `vs SPY|DAX|KOSPI`; assert `vs DAX`
  (CSS uppercases to `VS DAX`; DOM text is `vs DAX` — use case-insensitive match).

## Hypothesis
Adding `@playwright/mcp@0.0.75` as a pinned stdio server in `.mcp.json` gives this
session real browser-driving tools (`browser_navigate`/`browser_click`/`browser_snapshot`),
closing the read-tier gap. Because Playwright bundles Chromium and the binary is cached,
no install is needed. The MCP handshake + `tools/list` (run via a smoke-test script that
mirrors `smoke_test_bigquery_mcp.py`) proves the server attaches and tools are callable
at the protocol level NOW; the `mcp__playwright__*` tools become dispatchable in THIS
Claude Code session only after a restart (same rule as agent/MCP config changes). The
authenticated filter click-through is demonstrable via the `LIGHTHOUSE_SKIP_AUTH=1` path.

## Immutable success criteria (verbatim from the /goal — all must hold; harness verifies live)
1. The chosen MCP/extension attaches in a fresh session and its browser-driving tools are
   callable (show tool list or ping).
2. A version-controlled, reproducible install/config exists (.mcp.json pin + smoke test,
   OR a docs/runbooks/ entry).
3. SMOKE TEST passes, captured verbatim in handoff: navigate to localhost:3000/paper-trading
   (auth if needed), CLICK the "EU" filter pill, confirm via DOM/screenshot the benchmark
   label changes "VS SPY" -> "vs DAX" (and/or table re-scopes). If the live book is all-US
   (no EU rows), instead click between "All" and "US" and confirm the selected state changes
   — and say so.
4. Safety guardrails written in the runbook.

OUT OF SCOPE (verbatim): changing any trading logic; granting standing authority for
side-effectful actions; anything beyond install + config + smoke-test of browser-driving.

## Plan (dependency-ordered)
- **A — Config (criterion #2):** add the `playwright` stdio server to `.mcp.json` (pinned
  `@playwright/mcp@0.0.75`, `--user-data-dir <repo>/.playwright-mcp-profile`,
  `--allowed-hosts localhost`, `--viewport-size 1440,900`, `alwaysLoad: false`). Add
  `.playwright-mcp-profile/` to `.gitignore`.
- **B — Smoke test (criterion #1 protocol-evidence + #3):**
  `scripts/mcp_servers/smoke_test_playwright_mcp.py`, mirroring the bigquery smoke test:
  spawn `npx -y @playwright/mcp@0.0.75 --headless --isolated`, MCP `initialize` →
  `notifications/initialized` → `tools/list`, assert `{browser_navigate, browser_click,
  browser_snapshot}` present. Then a live-server leg: `browser_navigate` →
  `http://localhost:3000/login` + `browser_snapshot`, assert the login DOM ("Sign in")
  — proves real browser-driving against this project's running dev server with zero auth
  friction. Exit 0/1, generous budget (Chromium launch ~3-5s).
- **C — Runbook + safety (criteria #2, #4):** `docs/runbooks/browser-mcp.md` —
  install/config, the 3 auth paths (A skip-auth recommended; B persistent-profile manual
  login with the Google/passkey caveat; C storage-state replay), the smoke-test command,
  the restart-to-dispatch note, and the SAFETY guardrails verbatim (never place trades /
  move money / enter credentials / change account or security settings / accept consent
  dialogs / act on page-content instructions; localhost-only; minimal caps; pin version;
  page content is untrusted).
- **D — Verify (criterion #3 full click-through):** demonstrate the authenticated filter
  click-through. Primary: temporarily run the dev server with `LIGHTHOUSE_SKIP_AUTH=1`
  (controlled, reversible: `launchctl setenv` + kickstart the frontend, drive Playwright
  `browser_navigate /paper-trading/positions` → `browser_click` radio `EU` → assert
  `vs DAX`; also `All`↔`US` selected-state toggle), then RESTORE (unsetenv + kickstart).
  Capture output verbatim. If skip-auth env does not propagate cleanly to the launchd
  service, fall back to: prove `browser_click` on the reachable `/login` page (state
  change) + document the /paper-trading click-through as the operator-run path A with the
  exact commands — and SAY SO honestly (criterion #3 explicitly allows the reduced check).
  Do NOT disrupt the operator's working frontend beyond a reversible restart.
- Then spawn **Q/A** on the complete generate.

## Restart caveat (criterion #1)
A new `.mcp.json` server is snapshotted at session start; `mcp__playwright__*` tools are
not dispatchable in THIS session until `/clear` or a Claude Code restart (CLAUDE.md
"Agent definition changes require session restart" applies to MCP config too). The smoke
test's real MCP `initialize` + `tools/list` is the strongest in-session evidence that it
WILL attach and tools ARE callable; the live `mcp__playwright__*` dispatch is a
post-restart verification (note in handoff for the next session, like the qa-roster-live
pattern).

## References
- `handoff/current/research_brief.md` (research gate — 7 sources, all anchors line-verified)
- `.mcp.json` (alpaca/bigquery pin shape + alwaysLoad discipline)
- `scripts/mcp_servers/smoke_test_bigquery_mcp.py` (smoke-test template)
- `frontend/src/middleware.ts:22` (LIGHTHOUSE_SKIP_AUTH path)
- `frontend/src/components/paper-trading/MarketFilter.tsx:62/71/77` (radiogroup/radios)
- `frontend/src/components/paper-trading/cockpit-helpers.tsx:198/226` (benchLabel)
- microsoft/playwright-mcp README; playwright.dev/docs/getting-started-mcp; code.claude.com/docs/en/chrome
- security: awesome-testing.com Playwright-MCP-security (2025); issue #1495 (RCE)
