# Contract — step 63.1 (Playwright walk of all 22 routes → walk_summary.json)

**Phase:** phase-63 (full-app live audit) | **Step:** 63.1 | **Priority:** P0 | harness_required: true | depends_on: none
**Cycle:** 1 | Date: 2026-07-17 | **Type:** live-app AUDIT (read-only walk; produces evidence artifacts, no production code change).
$0; local-only; historical_macro FROZEN; live book untouched; **NEVER touch the operator's :3000 server**.

## Research-gate summary (gate PASSED)

Researcher subagent (Agent tool, Opus 4.8 effort:max, $0 Max rail), brief `research_brief_63.1.md`. Envelope:
**gate_passed=true**, tier=moderate, **6 external sources read in full** (3 playwright.dev official docs [network,
screenshots, auth] + Crawl4AI + StarterApp smoke + oneuptime-2026), 9 snippet-only, 15 URLs, recency scan, 9 internal
files inspected (with file:line). KEY findings:
- **22 routes** confirmed (21 static + 1 dynamic `/sovereign/strategy/[id]`). Full list below.
- **Auth bypass** = `LIGHTHOUSE_SKIP_AUTH=1` (frontend/src/middleware.ts:24) on an ISOLATED `:3100` dev server.
  `NEXT_PUBLIC_E2E_TESTING=true` is LOAD-BEARING (kills cockpit live polling so `networkidle`/load doesn't hang).
- **Dynamic id** resolved at walk-time via `GET /api/sovereign/leaderboard` → `rows[0].strategy_id` (fallback
  "baseline"; the page renders a rose error banner for a bogus id, never crashes).
- **Walk vehicle** = a checked-in standalone Node script using `playwright` core (already installed; chromium-1208
  present) — NOT a `*.spec.ts`, NOT the MCP (MCP can't emit the JSON artifact). Glob `frontend/src/app/**/page.tsx`
  at runtime = criterion-3 reconciliation.

## Hypothesis / objective

A checked-in, re-runnable Playwright route-walk script visits every `page.tsx` route against an isolated bypass dev
server, captures per-route screenshot + console errors + failed (≥400) requests, and emits
`handoff/away_ops/route_walk_<date>/walk_summary.json` with `routes_visited >= 22`. This replaces screenshot
interpretation of the operator's reported bugs with live evidence (the phase-63 defect-register input).

## Plan

### A. `frontend/scripts/audit/route_walk.mjs` (NEW — checked-in, re-runnable)
- Enumerate routes by globbing `frontend/src/app/**/page.tsx` at runtime → map file paths → URL paths (strip route
  groups `(...)`, map `[id]` dynamic). Report `routes_discovered` + reconcile vs the walk (`route_list_delta`).
- Resolve the strategy id: `GET http://localhost:3100/api/sovereign/leaderboard` → `rows[0].strategy_id` (fallback
  "baseline").
- Per route: a FRESH context/page; 4 listeners registered BEFORE navigation (`console` [error/warning], `pageerror`,
  `requestfailed`, `response` status≥400); `page.goto(url, {waitUntil:'load'})` (NOT networkidle — T1);
  `page.screenshot({fullPage:true})` → `screenshots/<slug>.png`; record `final_url`, `http_status`, `load_ms`.
- **Benign-noise filter**: drop favicon/manifest/source-map/chrome-extension/analytics 404s from `failed_requests`.
- **Bypass-misfire guard (T3)**: assert `final_url` path == route (except `/login`); if every route 302s to /login,
  FAIL loudly (the bypass didn't take).
- Emit `walk_summary.json` (schema below). Exit non-zero if `routes_visited < 22`.

### B. Run the walk (live, isolated)
1. Preflight: confirm operator `:3000` is up (302) and will be left UNTOUCHED; confirm `:3100` is free.
2. Spin up the isolated bypass server (background):
   `cd frontend && LIGHTHOUSE_SKIP_AUTH=1 NEXT_PUBLIC_E2E_TESTING=true npx next dev --port 3100`
3. Wait for `:3100` ready; run `node scripts/audit/route_walk.mjs` (base_url http://localhost:3100).
4. **Kill `:3100`** after; verify `:3000` still 302s (untouched).

### C. Evidence
- `handoff/away_ops/route_walk_<date>/walk_summary.json` + `screenshots/*.png` (the deliverable).
- `handoff/current/live_check_63.1.md` = walk_summary.json (verbatim, minus the big per-route array if large) +
  the artifact directory listing + the routes_visited count + console_error_routes/failed_request_routes.

### walk_summary.json schema (top-level keys the immutable verification reads are REQUIRED)
```
{ generated_at, base_url, auth_bypass, strategy_id_used, routes_discovered,
  routes_visited (int>=22), console_error_routes:[...], failed_request_routes:[...],
  route_list_delta:{on_disk_not_visited:[], visited_not_on_disk:[]},
  routes:[{route, final_url, http_status, screenshot, console_errors:[{type,text,location}],
           page_errors:[...], failed_requests:[{url,method,status,error}], load_ms}] }
```

### Route list (22)
`/`, `/agent-map`, `/agents`, `/backtest`, `/cron`, `/learnings`, `/login`, `/observability`, `/performance`,
`/reports`, `/settings`, `/signals`, `/sovereign`, `/sovereign/strategy/<ID>`, `/paper-trading`,
`/paper-trading/exit-quality`, `/paper-trading/learnings`, `/paper-trading/manage`, `/paper-trading/nav`,
`/paper-trading/positions`, `/paper-trading/reality-gap`, `/paper-trading/trades`.

### Traps to build in (from the research brief)
- **T1** networkidle hangs the cockpit → `waitUntil:'load'` + `NEXT_PUBLIC_E2E_TESTING=true`.
- **T2** register listeners per FRESH page/context (correct attribution).
- **T3** bypass-misfire → assert `final_url` path==route (except /login); all-login = FAIL.
- **T4** backend `:8000/health` 404s → probe a REAL endpoint, don't treat 404-on-health as a defect.
- **T5** fullPage lazy-load blanks + 32767px cap → scroll/settle before screenshot; cap height.

## Immutable success criteria (verbatim from masterplan.json 63.1)
1. "every page.tsx route visited via Playwright against the running app (NextAuth wall or the documented
   LIGHTHOUSE_SKIP_AUTH=1 port-3100 bypass), including one concrete strategy [id]"
2. "per-route artifacts: screenshot, console messages, failed-request list; walk_summary.json enumerates
   routes_visited, console_error_routes, failed_request_routes"
3. "the on-disk route list is reconciled against the walk (any delta is itself a defect row)"

**Verification command (immutable):**
`python3 -c "import json,glob; d=json.load(open(sorted(glob.glob('handoff/away_ops/route_walk_*/walk_summary.json'))[-1])); assert d['routes_visited']>=22, d; print(d['routes_visited'], d.get('console_error_routes'))"`

## Boundaries (binding)
$0; local-only. READ-ONLY audit — the walk only NAVIGATES the app; the only new files are the checked-in script +
the evidence artifacts. NO production code change; NO trade/risk/money touch; kill-switch/stops/caps/DSR/PBO
untouched; historical_macro FROZEN; live book untouched. **The operator's :3000 server is NEVER touched** — the walk
runs against an isolated :3100 bypass server that is spun up and torn down. Console errors / failed requests found are
RECORDED as evidence (the defect-register input for later phase-63 steps), not fixed here (63.1 is the audit; fixes
are 63.4, post-66.2).

## References
research_brief_63.1.md; frontend/src/middleware.ts:24 (bypass); frontend/playwright.config.ts +
frontend/tests/visual-regression/helpers/visual.ts (E2E baseline); backend/api/sovereign_api.py:378,698 (strategy id);
Playwright docs (network events, screenshots, auth). Phase-63 audit_basis: operator screenshot report; 8/22 routes had
prior E2E baseline.
