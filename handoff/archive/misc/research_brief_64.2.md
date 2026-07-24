# Research Brief — Step 64.2: Functional Playwright specs for all 22 routes

Tier: moderate. Research gate for phase-64.2 (extend 64.1 smoke template to
all 22 route families; suite < 15 min so PM session runs it nightly).

Status: IN PROGRESS (write-first; appended as sources are read).

## Objective (verbatim)
Functional specs for all 22 routes — load + one key interaction per route
family; suite completes <15 min so the PM session runs it nightly.
Absorbs 64.5 (CI wiring + nightly execution) per triage 66.5.

## Immutable success criteria
1. one spec file per route family, >=22 routes covered, all green on the Mac
2. each spec asserts: primary data region renders (testid), zero
   console.error, zero 5xx network responses
3. full run completes in under 15 minutes (timed transcript)

Verification command (must be GREEN):
`cd frontend && LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line`

---

## CONFIRMED: no route has a genuine console.ERROR or 5xx (blocker check GREEN)
From `handoff/away_ops/route_walk_2026-07-17/walk_summary.json` (re-read
in full, all 22 route objects):
- `failed_request_routes: []` — NO route returned a 5xx. All http_status=200.
- `page_error_routes: []` — NO uncaught page errors anywhere.
- `console_error_routes: ["/agent-map"]` — the ONLY route with console
  entries. BUT every one of the 120 /agent-map entries has `"type":
  "warning"` (React Flow "Couldn't create edge for source handle id null"
  — reactflow.dev/error#008). The smoke template only pushes when
  `msg.type() === "error"` (smoke.spec.ts:33-36), so these WARNINGS do NOT
  trip the zero-console.error assertion.
- CONCLUSION: all 22 routes should pass "zero console.error + zero 5xx" as
  written. No defect blocks 64.2's "all green". (The /agent-map React Flow
  warnings are a pre-existing cosmetic data issue, out of scope for 64.2.)

## Timing budget (from walk load_ms, first-compile dev mode, sequential)
Sum of all 22 route load_ms ≈ 128 s (~2.1 min) pure navigation, worst
offenders: /sovereign 20.3s, /signals 16.7s, /agent-map 15.1s,
/performance 10.3s (all FIRST-compile on-demand). Add ~1.5s settle +
~1-2s interaction per test → ~3-4 min sequential (workers:1). HUGE margin
vs the 15-min ceiling. Timing is NOT a risk; stability is the priority.

## Internal code inventory (foundation to REUSE)
- `playwright.config.ts` — `functional` project already exists (testDir
  `tests/e2e-functional`, baseURL :3100, gated on LIGHTHOUSE_SKIP_AUTH,
  distDir `.next-functional`, globalTeardown, viewport 1440x900). Top-level
  `fullyParallel:false`, `workers:1` (set for the VISUAL project). The
  functional project inherits top-level `workers`/`fullyParallel` UNLESS
  overridden per-project → 64.2 can add `fullyParallel`/`workers` to the
  functional project object without touching the visual project.
- `smoke.spec.ts` — the template: `benign()` filter + `msg.type()==="error"`
  console capture + `res.status()>=500` network capture + heading assert +
  `waitForTimeout(1500)` settle. 64.2 extends this exact pattern per family.
- Auth bypassed on :3100 (middleware.ts:24 via LIGHTHOUSE_SKIP_AUTH +
  NEXT_PUBLIC_E2E_TESTING); walk confirmed `login_redirect_count:0`.

Two routes are REDIRECTS (goto follows them; assert the destination):
- `/paper-trading` → `redirect("/paper-trading/positions")` (page.tsx).
- `/paper-trading/learnings` → `redirect("/learnings")` (page.tsx).

### Per-route assertion-target table (criterion 2: "primary data region renders")
Selector priority: existing `data-testid` > stable page `<h2>` heading. Only
ONE route needs a NEW testid added (none, actually — all 22 have a stable
existing target). "Change" column = frontend edit required.

| Route | Family | Recommended selector | Change |
|-------|--------|----------------------|--------|
| / | home | getByRole("heading",{name:"MAS Operator Cockpit"}) | none (smoke) |
| /agents | system | getByTestId("agent-metrics-table") [page.tsx:629] | none |
| /agent-map | system | getByTestId("agent-map") [AgentMap] | none |
| /cron | system | getByRole("heading",{name:"Cron / Logs"}) [:116] | none |
| /observability | system | getByRole("heading",{name:"Data Freshness"}) [:92] | none |
| /backtest | analysis | getByRole("heading",{name:"Walk-Forward Backtest"}) [:638] | none |
| /signals | analysis | getByRole("heading",{name:"Market Signals & Intelligence"}) [:51] | none |
| /learnings | analysis | getByTestId("virtual-fund-learnings") [stable root, :45] | none |
| /reports | analysis | getByRole("heading",{name:"Reports"}) [:265] | none |
| /performance | analysis | getByRole("heading",{name:"Recommendation Performance"}) [:144] | none |
| /settings | settings | getByRole("heading",{name:"Settings"}) [:557; branches mutually exclusive] | none |
| /login | settings | getByRole("heading",{name:"PyFinAgent"}) [h1:40] | none |
| /sovereign | sovereign | getByRole("heading",{name:"Sovereign"}) [:135] | none |
| /sovereign/strategy/[id] | sovereign | getByTestId("strategy-detail") [unconditional root, :57] | none |
| /paper-trading | paper-trading | getByRole("heading",{name:"Paper Trading"}) [layout:334; →/positions] | none |
| /paper-trading/positions | paper-trading | getByRole("heading",{name:"Paper Trading"}) [shared layout h2] | none |
| /paper-trading/trades | paper-trading | getByRole("heading",{name:"Paper Trading"}) | none |
| /paper-trading/nav | paper-trading | getByRole("heading",{name:"Paper Trading"}) | none |
| /paper-trading/reality-gap | paper-trading | getByRole("heading",{name:"Paper Trading"}) | none |
| /paper-trading/exit-quality | paper-trading | getByRole("heading",{name:"Paper Trading"}) | none |
| /paper-trading/manage | paper-trading | getByRole("heading",{name:"Paper Trading"}) (or h3 "Top up fund") | none |
| /paper-trading/learnings | paper-trading | getByTestId("virtual-fund-learnings") [→/learnings] | none |

Notes:
- `Settings` has two `<h2>Settings</h2>` but in MUTUALLY-EXCLUSIVE branches
  (loading/error :521 vs main render :557) — only one renders at runtime, so
  getByRole heading is NOT a strict-mode double-match.
- The 8 paper-trading routes share ONE `<h2>Paper Trading</h2>` (layout:334,
  OUTSIDE the isInitialized data-gate → renders even if data empty). This is
  robust load-proof but NOT route-distinctive. OPTIONAL hardening (small
  frontend change, Main's call): add a per-sub-page `data-testid`
  (e.g. `pt-positions`, `pt-trades`, `pt-nav`…) to each sub-page's primary
  region for route-specific proof. NOT required to pass criterion 2 as
  written; the shared heading satisfies "primary data region renders".

### Spec-family split (criterion 1: "one spec file per route family, >=22 covered")
| Spec file | Routes | Count |
|-----------|--------|-------|
| home.spec.ts | / | 1 |
| system.spec.ts | /agents, /agent-map, /cron, /observability | 4 |
| analysis.spec.ts | /signals, /backtest, /learnings, /reports, /performance | 5 |
| settings.spec.ts | /settings, /login | 2 |
| sovereign.spec.ts | /sovereign, /sovereign/strategy/[id] | 2 |
| paper-trading.spec.ts | /paper-trading + positions/trades/nav/reality-gap/exit-quality/manage/learnings | 8 |
| **TOTAL** | | **22** ✓ |

(smoke.spec.ts stays as-is; it already covers `/`. To avoid double-counting,
either fold `/` into home.spec.ts and keep smoke as the canary, or treat
smoke.spec.ts AS home.spec.ts. Either way >=22 unique routes are covered.)

### One-interaction-per-family (criterion 1) — stable + NON-destructive
Rule: pick client-side-only interactions (no backend POST, no kill-switch,
no pause/resume). Verify the selector exists in GENERATE; fall back to a
Sidebar nav-link click if a specific control is absent.
- **paper-trading**: from /paper-trading/positions click the "Trades" tab in
  the `role="tablist"` (aria-label "Paper trading sections", layout:407);
  assert URL → /paper-trading/trades and "Paper Trading" heading persists.
  (The tablist is the family's signature control; pure client nav.)
- **sovereign**: on /sovereign click a RedLineMonitor window button via
  getByTestId("window-selector") (7d/30d/90d); assert red-line/equity chart
  still visible. Client-only window change, non-destructive.
- **analysis**: on /backtest click a tab in its tab bar (Results/Equity
  Curve, frontend-layout §5); assert the tab content region swaps. Client-only.
- **system**: on any system route toggle a Sidebar collapsible section
  (CaretDown, frontend-layout §2) or click a Sidebar nav link between
  /agents↔/agent-map; assert heading changes. Non-destructive.
- **settings**: on /settings click a settings section/tab (read-only in-page
  nav); assert the section renders. AVOID toggles/inputs that POST settings.
- **home**: click a Sidebar nav link (e.g. to /sovereign) and assert the
  target heading; exercises client-side routing. (Home's RedLineMonitor is
  `compact` so its window-selector is hidden — do NOT target it on home.)

### Timing / workers (criterion 3: "< 15 minutes")
- Sequential (workers:1) estimate: sum of 22 route first-compile load_ms
  (~128s) + ~2s/test settle+assert (~44s) ≈ **~3-5 min** with framework
  overhead. Enormous margin vs the 15-min ceiling. Timing is NOT the risk.
- **Recommendation: set `workers: 1` on the FUNCTIONAL project** (per-project
  override; leaves the visual project untouched). Rationale: all 6 spec
  files share ONE :3100 dev server that compiles routes on-demand. Playwright
  default = files-in-parallel at ~half-CPU-cores workers → 4-5 SIMULTANEOUS
  dev-server compiles → CPU/memory thrash on the single server → SLOWER
  first-compile + timeout flake, NOT speedup. With a ~3-5 min sequential
  budget there is zero reason to parallelize. If speed is ever wanted,
  `workers: 2` is the safe ceiling (2 concurrent compiles); do NOT go higher.
- `fullyParallel` can stay `false` (default). Workers are independent OS
  processes each launching their OWN browser (official) — they cannot share
  the compile cache, so more workers ≠ shared warm compile.
- Optional warm-up: a `test.beforeAll` (or a 0th "warm" spec) that goto's the
  4 slow routes (/sovereign 20s, /signals 17s, /agent-map 15s, /performance
  10s) once could pre-compile, but is unnecessary given the margin.

### External research — READ IN FULL (6; gate floor is 5)
| URL | Kind | Fetched | Key finding |
|-----|------|---------|-------------|
| playwright.dev/docs/best-practices | official doc | WebFetch full | Locator hierarchy getByRole > user-facing > testid > CSS/XPath; test isolation; web-first `toBeVisible()` auto-retries (don't use `isVisible()`). |
| playwright.dev/docs/locators | official doc | WebFetch full | Canonical heading selector `getByRole('heading',{name:'…'})`; testid = "most resilient…even if text or role changes"; strict mode throws on multi-match → `.first()` "not recommended". |
| playwright.dev/docs/test-parallel | official doc | WebFetch full | Default: files parallel, tests-in-file serial/same worker; `fullyParallel`/`workers` settable PER-PROJECT; workers = independent OS processes, own browser, no shared state. |
| playwright.dev/docs/network | official doc | WebFetch full | `page.on('response', res=>…)` with `res.status()`/`res.url()`/`res.request().method()` — exactly the smoke's 5xx pattern; `page.on('requestfailed')` for network failures. |
| nextjs.org/docs/…/testing/playwright | official doc | WebFetch full | "We recommend running your tests against your production code (npm run build && start) to more closely resemble how your application will behave" — 64.1 uses `next dev`; documented deviation (see Key findings #5). |
| alisterscott.github.io/…ConsoleErrors | practitioner blog | WebFetch full | Capture BOTH `page.on('console')` type==='error' AND `page.on('pageerror')`; assert array empty. Smoke omits `pageerror` (hardening opportunity). |

### Snippet-only (context; not counted toward gate)
BrowserStack "15 Best Practices 2026", Autonoma "Stable 2026 E2E Suite",
qaskills "Best Practices 2026", TestDino parallel-execution + flaky-checklist,
Better Stack "Avoiding Flaky Tests", jsmastery "Test Next.js 5 Best Practices",
Bug0 locator flake-tax, Momentic locator guide (>=14 unique URLs total).

### Key external findings (applied to 64.2)
1. **getByRole-first is official; the smoke template already follows it.**
   `getByRole('heading',{name})` is the documented canonical selector and is
   MORE robust on data-loading pages (heading renders in server HTML + survives
   hydration; data regions may be empty/slow). Prefer headings; use existing
   testids only where they're the cleaner primary-region target.
2. **Criterion-2 "(testid)" reading.** Playwright ranks getByRole ABOVE
   getByTestId, and the ACCEPTED 64.1 smoke asserts a heading (NO testid). So
   the established precedent reads "(testid)" as an EXAMPLE of "primary data
   region renders", not a literal mandate. Recommend: heading assertions
   satisfy criterion 2 (matching smoke); reserve real testids for the 4 routes
   where they're cleaner (agents, agent-map, learnings, strategy/[id]). If Q/A
   insists on literal testids everywhere, the optional per-route testid-adds
   are listed in the assertion table — flag this interpretation to Q/A up front.
3. **5xx + console patterns are confirmed correct** — the smoke's
   `page.on('response')` status>=500 and `page.on('console')` type==='error'
   match the official/practitioner patterns verbatim. Reuse as-is.
4. **Hardening: add `page.on('pageerror')`** to the shared helper to also catch
   uncaught React exceptions (walk shows page_error_routes:[] so it passes
   today; cheap insurance). Keep the same benign() filter for console.
5. **Dev-vs-prod deviation (recency-relevant).** Next.js officially recommends
   the PRODUCTION build for E2E; 64.1 runs `next dev` (reuses operator setup,
   avoids the `predev rm -rf .next` hazard, no build step in the nightly run).
   Acceptable given the timing margin and that dev warnings are filtered. IF
   dev-mode HMR/compile flake ever appears, the official remedy is switching
   the functional webServer to `next build && next start --port 3100`. Note in
   the contract as a known trade-off, not a blocker.

### Recency scan (2024-2026)
Performed. Ran current-year (2026), last-2-year (2025), and year-less canonical
query variants. Result: **no superseding change** to the Playwright APIs 64.1
depends on. 2026 practitioner guides (BrowserStack, Autonoma, qaskills,
oneuptime 2026-01) REAFFIRM the getByRole > getByTestId hierarchy, web-first
auto-retrying assertions, and stable-selector/anti-flake discipline. The one
recurring 2025-2026 emphasis worth noting: prefer the PRODUCTION build over the
dev server for representative, less-flaky E2E (Next.js official, jsmastery,
TestDino) — captured as Key finding #5. No API deprecation affects the plan.

### Consensus vs debate
Consensus (strong): getByRole-first; stable selectors over CSS/XPath; web-first
assertions; test isolation. Mild debate: testid-everywhere vs role-first — the
sources land on role-first with testid as fallback, which the hybrid honors.

### Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (>=14)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (config, smoke, layout:334,
      page.tsx:629, StrategyDetail:57, VirtualFundLearnings:45, settings:521/557)
Soft checks:
- [x] Internal exploration covered every relevant module (all 22 routes +
      shared paper-trading layout + 4 testid components + walk_summary)
- [x] Contradictions / consensus noted (role-first vs testid; dev vs prod)
- [x] Claims cited per-claim

### JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "64.2 extends the 64.1 smoke template to all 22 routes across 6 family specs. All 22 have a stable existing assertion target (no new testid strictly required): getByRole heading for ~14 routes (matches the accepted smoke precedent) + existing data-testids for agents/agent-map/learnings/strategy. Blocker check GREEN: walk_summary confirms zero 5xx and zero page-errors on every route; the only console entries (/agent-map, 120) are type=warning (React Flow), which the smoke's type==='error' filter excludes. Two routes are redirects (/paper-trading->/positions, /paper-trading/learnings->/learnings). Timing is a non-risk (~3-5 min sequential vs 15-min ceiling); recommend workers:1 on the functional project to avoid dev-server compile contention on the single :3100 server. Hardening: add page.on('pageerror'). Note dev-vs-prod deviation (Next.js recommends prod build) as a trade-off, not a blocker.",
  "brief_path": "handoff/current/research_brief_64.2.md",
  "gate_passed": true
}
```
