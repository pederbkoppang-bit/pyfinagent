# Experiment results — step 63.1 (Playwright walk of all 22 routes)

**Step:** 63.1 (P0, phase-63 full-app live audit, depends_on none). $0; local-only; READ-ONLY audit; live book
untouched; historical_macro FROZEN; **operator :3000 NEVER touched**. Research gate PASSED (research_brief_63.1.md,
gate_passed=true, 6 external sources read in full + recency scan).

## What was built + run

1. **`frontend/scripts/audit/route_walk.mjs`** (NEW, checked-in, re-runnable) — playwright-core standalone script.
   Globs `frontend/src/app/**/page.tsx` at runtime (criterion-3 reconciliation), resolves the dynamic
   `/sovereign/strategy/[id]` id via `GET /api/sovereign/leaderboard` (fallback "baseline"), and per route: a fresh
   page with 4 listeners (console error/warning, pageerror, requestfailed, response≥400) registered BEFORE nav +
   `page.goto(waitUntil:'load')` + `page.screenshot({fullPage:true})`. Benign-noise filter (favicon/map/manifest/HMR/
   ext). Emits `walk_summary.json` + `screenshots/`. Exit non-zero if <22 routes or bypass-misfire (all-login).
2. **Live run (isolated, teardown-clean):**
   - Preflight: :3000 = 302 (healthy, left UNTOUCHED); :3100 free.
   - Spun up isolated bypass server: `LIGHTHOUSE_SKIP_AUTH=1 NEXT_PUBLIC_E2E_TESTING=true npx next dev --port 3100`
     (Ready in 1156ms; bypass probe `GET :3100/` → **HTTP 200**, not 302).
   - One-time dev-browser install: `npx playwright install chromium-headless-shell` (playwright 1.60 wanted build
     1223; cache had 1208 — free dev tooling, like npm install; user-level ms-playwright cache, repo untouched).
   - Ran `node scripts/audit/route_walk.mjs --base http://localhost:3100`.
   - **Killed :3100; verified :3000 still 302** (operator instance untouched).

## Results (walk_summary.json)

- `routes_discovered`: **22**, `routes_visited`: **22** (criterion 1 — every page.tsx route visited).
- `login_redirect_count`: **0** (bypass active on every route — no silent all-login misfire).
- `route_list_delta`: `{on_disk_not_visited: [], visited_not_on_disk: []}` — **fully reconciled** (criterion 3; no
  delta defect rows).
- `failed_request_routes`: **[]** (no 4xx/5xx after benign filter).
- `page_error_routes`: **[]** (no uncaught page exceptions).
- `console_error_routes`: **["/agent-map"]** — **DEFECT SURFACED**: /agent-map emits **120 React Flow warnings**
  `"[React Flow]: Couldn't create edge for source handle id: 'null'"` (edges main→researcher, main→qa,
  multi_agent_orchestrator→planner_agent, etc.). Root cause shape: the agent-graph edges reference source handles that
  render as `null`, so React Flow rejects them. This is a concrete defect-register row for a later phase-63 fix step
  (63.4, post-66.2). **63.1 is the AUDIT — the defect is RECORDED, not fixed here** (per the contract boundary).
- `strategy_id_used`: "baseline" (the leaderboard fallback; the concrete-`[id]` route `/sovereign/strategy/baseline`
  loaded HTTP 200 with no console/page errors — criterion 1 "including one concrete strategy [id]" satisfied).
- Per-route artifacts: 22 full-page screenshots + per-route console/failed-request arrays (criterion 2).

## Verification (verbatim)

- `node --check scripts/audit/route_walk.mjs` → syntax OK.
- IMMUTABLE cmd `python3 -c "import json,glob; d=json.load(open(sorted(glob.glob('handoff/away_ops/route_walk_*/walk_summary.json'))[-1])); assert d['routes_visited']>=22, d; print(...)"` → **routes_visited: 22 | console_error_routes: ['/agent-map']**, exit 0 (PASS).
- Artifacts: `handoff/away_ops/route_walk_2026-07-17/walk_summary.json` (34KB) + `screenshots/` (22 PNGs, 1.6M).
- Operator :3000 verified 302 AFTER teardown (untouched).

## Do-no-harm / boundaries

$0 metered (Playwright browser download is free dev tooling, not an API cost). READ-ONLY audit — the only new files
are the checked-in script + the evidence artifacts (walk_summary.json + 22 screenshots). NO production code change; NO
trade/risk/money touch; kill-switch/stops/caps/DSR/PBO untouched; historical_macro FROZEN; live book untouched. The
walk ran against an ISOLATED :3100 bypass server (spun up + torn down); the operator's :3000 was never touched
(verified 302 before and after). The /agent-map console-error defect is RECORDED for the phase-63 defect register, not
fixed here.

## Artifact shape
`handoff/away_ops/route_walk_2026-07-17/walk_summary.json` (top-level: routes_discovered, routes_visited,
login_redirect_count, console_error_routes, failed_request_routes, page_error_routes, route_list_delta, routes[] with
per-route final_url/http_status/screenshot/console_errors/page_errors/failed_requests/load_ms) + `screenshots/<slug>.png`.
Re-runnable: `node frontend/scripts/audit/route_walk.mjs --base http://localhost:3100` against the isolated bypass server.
