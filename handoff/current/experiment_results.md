# Experiment results — step 64.2 (Functional specs for all 22 routes)

**Step:** 64.2 (P0, phase-64, depends_on 64.1). $0; local-only; test-infra. Research gate PASSED
(research_brief_64.2.md, gate_passed=true, 6 external sources read in full). historical_macro FROZEN; live book
untouched; operator :3000 NEVER touched (functional suite runs ONLY on the isolated :3100 per 64.1).

## What was built (pure test-infra — NO frontend changes; all 22 targets are EXISTING selectors)

- **`frontend/tests/e2e-functional/_helpers.ts`** (NEW): `assertFunctionalRoute(page, path, region)` — loads the
  route, asserts the primary data region visible, and asserts **zero console.error + zero pageerror + zero 5xx**
  (extends the 64.1 smoke idiom, HARDENED with `page.on("pageerror")` per the research note). Returns the final URL
  (for redirect assertions). Exports `benign()` (shared) + `resolveStrategyId()`.
- **6 family spec files** (one per route family; ≥22 routes):
  - `smoke.spec.ts` (home family, `/`) — refactored to use the helper; keeps the `--grep smoke` canary (64.1) + a
    home interaction (sidebar nav → /signals).
  - `system.spec.ts` — /agents (heading "Multi-Agent System"), /agent-map (testid `agent-map`), /cron (heading), 
    /observability (heading) + interaction (nav /agents→/agent-map).
  - `analysis.spec.ts` — /signals (`#signals-ticker-input`), /backtest, /learnings (testid), /reports, /performance
    (headings) + interaction (fill the ticker input).
  - `settings.spec.ts` — /settings (heading), /login (heading "PyFinAgent") + interaction (sidebar nav → cockpit).
  - `sovereign.spec.ts` — /sovereign (heading), /sovereign/strategy/baseline (testid `strategy-detail`) +
    interaction (sidebar nav → /reports).
  - `paper-trading.spec.ts` — all 8 routes, each asserting its ROUTE-DISTINCTIVE `#panel-<subpage>` tabpanel (the
    subpages share the layout `<h2>Paper Trading</h2>`, so the panel id is the real per-route proof); the two
    redirects (`/paper-trading`→positions, `/paper-trading/learnings`→/learnings) are asserted via the returned
    final URL + interaction (positions→Trades tab).
- **No new `data-testid`s / no frontend edits** — the 22 targets are all existing selectors (4 component testids,
  distinctive headings, `#signals-ticker-input`, `#panel-<subpage>`). Cleaner scope than the contract's fallback.

## Verification (verbatim)

- IMMUTABLE cmd `cd frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line` →
  **28 passed, exit 0, "28 passed (1.2m)"** (wall 73s). Covers **22 routes** across **6 spec files**, one per family.
- **Timing 73s << 15-min ceiling** (criterion 3).
- Each spec asserts primary-region-renders + **zero console.error + zero 5xx** (+ zero pageerror) — criterion 2. (The
  /agent-map 120 React Flow entries are `type:"warning"`, excluded by the `type()==="error"` filter — confirmed
  green.)
- One fix during GENERATE: /agents' `agent-metrics-table` testid sits behind a non-default tab → switched to the
  always-rendered `<h2>Multi-Agent System</h2>` (a stable primary-region proof). Re-ran → 28/28 green.
- `npx tsc --noEmit tests/e2e-functional/*.ts` → **exit 0**. `npx eslint tests/e2e-functional/` → **exit 0**.
- **:3000 UNTOUCHED**: `/login → 200` before AND after (distDir isolation from 64.1 holds); TS files
  (next-env.d.ts/tsconfig.json) git-clean post-run (globalTeardown).

## Scope honesty — incidental runtime artifacts (NOT part of 64.2)

`git status` also shows backend runtime-loop state files the LIVE autonomous loop touched during this session (NOT my
changes): `handoff/.autonomous_loop.lock` (D — mutex released at cycle end), `handoff/.cycle_heartbeat.json`,
`handoff/away_ops/auth_probe_last.json`, `handoff/cycle_history.jsonl` (M — cycle appended). These are runtime state
(the operator's :8000 backend cycled), not test-infra. The auto-commit hook `git add -A` will include them
incidentally (as with the audit JSONL in prior steps). My 64.2 CODE deliverable is exactly: `_helpers.ts` + the 6
spec files (smoke.spec.ts modified; 5 new). No production/runtime CODE changed.

## Do-no-harm / boundaries

$0; local-only; test-infra ONLY (6 spec files + a shared helper; NO frontend/backend code change). NO trade/risk/money
touch; kill-switch/stops/caps/DSR/PBO untouched; historical_macro FROZEN; live book untouched. The suite runs ONLY on
the isolated :3100 (distDir=.next-functional; operator :3000 verified 200 before+after). Dev-vs-prod-build trade-off
(disclosed, not a blocker): the suite runs against `next dev` (reuses the 64.1 bypass), acceptable given the 73s
runtime + type-filtered warnings; remedy if flake appears = `next build && next start --port 3100`.

## Artifact shape
Re-runnable green: `cd frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional`. 28 tests / 22
routes / 6 family specs. live_check_64.2.md holds the timed transcript.
