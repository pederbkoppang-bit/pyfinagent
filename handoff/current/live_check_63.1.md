# live_check — step 63.1 (Playwright walk of all 22 routes)

## Method disclosure (canonical 55.1 §A template)

- **Operator :3000 instance UNTOUCHED.** Verified `curl :3000 → HTTP 302` (healthy authed signature) BEFORE and
  AFTER the walk. The walk ran against an ISOLATED second dev server on **:3100** spun up with the documented
  skip-auth bypass (`frontend/src/middleware.ts:24`):
  `cd frontend && LIGHTHOUSE_SKIP_AUTH=1 NEXT_PUBLIC_E2E_TESTING=true npx next dev --port 3100`
  (`NEXT_PUBLIC_E2E_TESTING=true` kills the cockpit's live polling so `waitUntil:'load'` doesn't hang). Bypass probe
  `GET :3100/ → HTTP 200` (not 302→/login). :3100 killed after capture.
- **Capture stack:** a checked-in standalone script `frontend/scripts/audit/route_walk.mjs` using **playwright
  1.60.0** core (chromium-headless-shell build **1223**, installed this run via `npx playwright install
  chromium-headless-shell` — free dev tooling). NOT the Playwright MCP (the MCP cannot emit the JSON artifact) and
  NOT a `*.spec.ts`. Re-runnable: `node frontend/scripts/audit/route_walk.mjs --base http://localhost:3100`.

## Immutable verification command output

```
$ python3 -c "import json,glob; d=json.load(open(sorted(glob.glob('handoff/away_ops/route_walk_*/walk_summary.json'))[-1])); assert d['routes_visited']>=22, d; print('routes_visited:', d['routes_visited'], '| console_error_routes:', d.get('console_error_routes'))"
routes_visited: 22 | console_error_routes: ['/agent-map']
# exit 0 (PASS)
```

## walk_summary.json — top-level (verbatim)

```json
{
  "generated_at": "2026-07-17T18:49:49.037Z",
  "base_url": "http://localhost:3100",
  "auth_bypass": "LIGHTHOUSE_SKIP_AUTH=1 + NEXT_PUBLIC_E2E_TESTING=true on :3100 (middleware.ts:24)",
  "strategy_id_used": "baseline",
  "routes_discovered": 22,
  "routes_visited": 22,
  "login_redirect_count": 0,
  "console_error_routes": ["/agent-map"],
  "failed_request_routes": [],
  "page_error_routes": [],
  "route_list_delta": { "on_disk_not_visited": [], "visited_not_on_disk": [] }
}
```

All 22 routes returned HTTP 200 (bypass active; `login_redirect_count: 0`). `route_list_delta` empty = the on-disk
`page.tsx` route list is fully reconciled against the walk (criterion 3; no delta defect rows).

## Artifact directory listing

```
handoff/away_ops/route_walk_2026-07-17/
├── walk_summary.json            (34 KB — 22 routes with per-route console/failed-request/screenshot/load_ms)
└── screenshots/                 (22 PNGs, 1.6 MB total — one full-page shot per route)
    ├── root.png  agent-map.png  agents.png  backtest.png  cron.png  learnings.png  login.png
    ├── observability.png  performance.png  reports.png  settings.png  signals.png  sovereign.png
    ├── sovereign_strategy__id_.png
    └── paper-trading{,_exit-quality,_learnings,_manage,_nav,_positions,_reality-gap,_trades}.png
```

## Defect surfaced (the audit's purpose — recorded, NOT fixed in 63.1)

**/agent-map — 120 React Flow console warnings**, all of the form:
`[React Flow]: Couldn't create edge for source handle id: "null", edge id: <edge>` (e.g. `main-researcher`,
`main-qa`, `multi_agent_orchestrator-planner_agent`, `autonomous_loop-multi_agent_orchestrator`,
`planner_agent-evaluator_agent`, `communication_agent-slack_bot`). The agent-graph edges reference source handles that
render as `null`, so React Flow drops them — the graph likely renders with missing/broken edges. This is a concrete
defect-register row for phase-63 (63.3 register / 63.4 fix queue, post-66.2). All other 21 routes: clean (0 console
errors, 0 page errors, 0 failed requests).
