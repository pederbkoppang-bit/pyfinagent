# Contract — step 64.2 (Functional specs for all 22 routes)

**Phase:** phase-64 | **Step:** 64.2 | **Priority:** P0 | harness_required: true | depends_on: 64.1 (done)
**Cycle:** 1 | Date: 2026-07-17 | **Type:** test-infra (6 functional spec files + shared helper). $0; local-only; NO
production/live-loop change; historical_macro FROZEN; live book untouched; operator :3000 NEVER touched (functional
suite manages ONLY the isolated :3100, per 64.1).

## Research-gate summary (gate PASSED)

Researcher subagent (Agent tool, Opus 4.8 effort:max, $0), brief `research_brief_64.2.md`. Envelope:
**gate_passed=true**, tier=moderate, **6 external sources read in full** (5 Playwright/Next.js official + 1
practitioner), 14 URLs, recency scan, 12 internal files (incl. all 22 page.tsx + walk_summary.json). KEY:
- **BLOCKER CHECK GREEN**: walk_summary.json (all 22 route objects) → `failed_request_routes: []`, `page_error_routes:
  []`; only `/agent-map` has console entries (120) and EVERY one is `type: "warning"` (React Flow #008) → the
  `type()==="error"` filter excludes them. So all 22 routes pass "zero console.error + zero 5xx" as written. No defect
  blocks "all green".
- **Every route has a stable existing assertion target** (no new testid strictly required): 4 existing testids
  (`agent-metrics-table`, `agent-map`, `virtual-fund-learnings`, `strategy-detail`) + distinctive headings for the
  standalone routes + the shared `<h2>Paper Trading</h2>` (layout:334) for the 8 paper-trading routes.
- **Redirects**: `/paper-trading → /positions`, `/paper-trading/learnings → /learnings`.
- **Timing NON-risk**: ~3-5 min sequential vs the 15-min ceiling. Use **`workers: 1`** on the functional project
  (per-project override; leaves the visual project untouched) — the 6 specs share ONE :3100 dev server compiling
  on-demand; more workers → CPU-thrash flake, not speedup.

## Plan

### A. Shared helper `frontend/tests/e2e-functional/_helpers.ts` (NEW) [criterion 2]
`assertFunctionalRoute(page, path, { primaryRegion })` — mirrors smoke.spec's capture idiom + HARDENS it:
- register `console` (error only, benign()-filtered), `response` (≥500, benign()-filtered), AND **`pageerror`**
  (uncaught React exceptions — the research's hardening note) BEFORE nav.
- `page.goto(path, {waitUntil:"load"})`; assert `primaryRegion` (a Locator) `toBeVisible({timeout:15_000})`;
  settle 1500ms; assert consoleErrors, serverErrors, pageErrors all empty (with descriptive messages).
- export `benign()` (moved from smoke's inline copy → shared).

### B. Six spec files (route-family split; ≥22 routes) [criterion 1]
- `home.spec.ts` — `/` (heading "MAS Operator Cockpit"). **Fold in the 64.1 smoke** OR keep smoke.spec as the `/`
  canary — decide in GENERATE to avoid double-counting `/`. Interaction: click a Sidebar link → assert target heading.
- `system.spec.ts` — `/agents` (testid `agent-metrics-table`), `/agent-map` (testid `agent-map`), `/cron` (heading
  "Cron / Logs"), `/observability` (heading "Data Freshness"). Interaction: nav /agents↔/agent-map → heading changes.
- `analysis.spec.ts` — `/signals`, `/backtest`, `/learnings` (testid `virtual-fund-learnings`), `/reports`,
  `/performance` (distinctive headings). Interaction: on /backtest click a tab → content swaps.
- `settings.spec.ts` — `/settings` (heading "Settings"), `/login` (h1 "PyFinAgent"). Interaction: click a read-only
  settings section (NO POST).
- `sovereign.spec.ts` — `/sovereign` (heading "Sovereign"), `/sovereign/strategy/<id>` (testid `strategy-detail`; id
  via `/api/sovereign/leaderboard` fallback "baseline"). Interaction: click a RedLineMonitor window (getByTestId
  `window-selector`) → chart still visible.
- `paper-trading.spec.ts` — the 8 routes (`/paper-trading`→positions, positions, trades, nav, reality-gap,
  exit-quality, manage, learnings→/learnings). Assert the shared `<h2>Paper Trading</h2>` PLUS a **route-distinctive**
  target per subpage (a distinctive heading/text/element found in GENERATE; add a minimal `data-testid` ONLY if a
  subpage has no distinctive selector — testids are visually inert, won't affect visual-regression). Interaction: from
  /positions click the "Trades" tab (role=tablist, layout:407) → assert URL `/paper-trading/trades`.

### C. Config [criterion 3]
`playwright.config.ts` functional project: add `workers: 1`... NOTE workers is config-level in Playwright, not
per-project. Use the functional project's `fullyParallel: false` + rely on the config `workers: 1` (already set) — the
functional run is single-worker. Confirm the visual project is unaffected (it already runs workers:1).

### D. Criterion-2 "(testid)" interpretation (documented up front)
Criterion 2 says "primary data region renders (testid)". Playwright ranks `getByRole` ABOVE testid, and the ACCEPTED
64.1 smoke asserted a heading (no testid). We read "(testid)" as an EXAMPLE of a stable target, not a literal mandate
— satisfied by a stable testid OR a distinctive role/heading that proves the primary region rendered. Where a route
lacks a distinctive target (paper-trading subpages), we add route-distinctive proof (existing selector or a minimal
inert testid). This matches the 64.1 precedent + Playwright's own locator-priority guidance.

## Immutable success criteria (verbatim from masterplan.json 64.2)
1. "one spec file per route family, >=22 routes covered, all green on the Mac"
2. "each spec asserts: primary data region renders (testid), zero console.error, zero 5xx network responses"
3. "full run completes in under 15 minutes (timed transcript)"

**Verification command (immutable):**
`cd /Users/ford/.openclaw/workspace/pyfinagent/frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line`
(the WHOLE functional project — smoke + all new route specs — must be green.)

## Boundaries (binding)
$0; local-only; test-infra (6 spec files + a shared helper + at most a few inert `data-testid` attributes if a
paper-trading subpage lacks a distinctive selector). NO production/live-loop change; NO trade/risk/money touch;
kill-switch/stops/caps/DSR/PBO untouched; historical_macro FROZEN; live book untouched. The functional suite runs ONLY
against the isolated :3100 (distDir=.next-functional; operator :3000 untouched — verify 200/302 before+after). Any
data-testid added is visually inert (does not change rendering → visual-regression snapshots unaffected). Trade-off
(disclosed, not a blocker): the suite runs against `next dev` (reuses the 64.1 bypass setup), not a prod build —
acceptable given the huge timing margin + the benign()/type-filtered warnings; remedy if dev flake ever appears is
`next build && next start --port 3100`.

## References
research_brief_64.2.md; frontend/tests/e2e-functional/smoke.spec.ts (64.1 template); frontend/playwright.config.ts;
handoff/away_ops/route_walk_2026-07-17/walk_summary.json (per-route console/5xx evidence); the per-route page.tsx
targets (agent-metrics-table page:629, agent-map, virtual-fund-learnings :45, strategy-detail :57, Paper Trading
layout:334, tablist layout:407). Playwright docs (locators, parallel, network, best-practices); Next.js Playwright guide.
