# Phase C - phase-10.5: Sovereign Dashboard (Net System Alpha cockpit)

Status: proposal (pending integration into `.claude/masterplan.json`)
Owner: Peder / harness / principal sovereign systems architect
Depends on: phase-4.7 (UI baseline), phase-4.8.2 (CVaR feed), phase-8.5 (autoresearch leaderboard data)
Gate: none (read-only dashboard; does not gate any cron)

## Red Line contract

- Objective monitored: `Net System Alpha = Profit - (Risk Exposure + Compute Burn)`.
- `Profit`: realized NAV P&L over the window (7/30/90 day).
- `Risk Exposure`: daily 97.5% portfolio CVaR (from phase-4.8.2) converted to
  dollars at current NAV.
- `Compute Burn`: rolling window LLM + BigQuery slot + alt-data vendor spend
  in USD (from billing export + provider invoices).
- `cron_slots: 0` - browser-driven reads only. This phase consumes ZERO of
  the 15-per-day Claude scheduled routine slots in the live environment.

## Relationship to prior phases

- `phase-4.7` (UI/UX Optimization & Frontend Audit) defines the HOMEPAGE,
  OpsStatusBar, and the MAS operator cockpit baseline. See
  `handoff/phase-proposals/phase-3.7-4.7-8.5-mas-ux-autoresearch.md`
  (Phase Y, lines 137-219) for the established baseline.
- `phase-4.5` (Paper Trading Dashboard v2) already ships realized-metrics
  tear sheets, kill-switch audit, reconciliation, and regime buckets. Phase
  C does NOT rebuild those views.
- `phase-4.8.2` adds portfolio CVaR and factor-exposure gates. Phase C
  surfaces the already-computed CVaR as the "Risk Exposure" axis on the
  Red Line; it does NOT recompute CVaR.
- `phase-8.5` ships the Autoresearch Run view (candidate leaderboard with
  DSR/PBO columns). Phase C extends that concept into a LIVE champion
  vs challenger leaderboard scoped to deployed strategies, not backtest
  candidates.

What is NEW and unique to Phase C:

1. A top-level `/sovereign` route with a two-hero layout.
2. **Alpha Leaderboard** - live champion (Master Baseline) + N challengers
   ranked by the same Net System Alpha metric, with live P&L and compute
   cost columns, and champion-flip history.
3. **Red Line Monitor** - the single most prominent widget on `/sovereign`
   AND on the post-4.7 homepage: a time-series chart of Net System Alpha
   at 7/30/90 day windows, annotated with kill-switch triggers and
   champion/challenger flips, with a compute-cost stacked-bar breakdown
   by provider directly below.

## Goal

Give the operator a single screen that answers "are we winning, net of all
costs, after risk?" without requiring cross-reference between the backtest
page, the paper-trading page, the costs page, and the harness log. Elevate
the `Net System Alpha` formula to the most prominent visual artefact in
the entire product, so that every strategy change, every vendor invoice,
and every kill-switch trip is visible in the same frame.

## Success criteria

1. `/sovereign` route landed and reachable from sidebar under a new "Sovereign"
   group; uses the standard page shell in `frontend-layout.md` Section 1 with
   `h-screen overflow-hidden` and `scrollbar-thin`.
2. **Alpha Leaderboard** renders at least 1 champion row + up to N challenger
   rows with the exact columns
   `strategy_id, deployed_since, live_pnl_30d, realized_sortino_30d, dsr,
   pbo, max_dd_30d, compute_cost_30d, status`
   where `status in {champion, challenger_live, challenger_shadow, retired}`;
   columns are sortable client-side and filterable by status.
3. Clicking a row opens a strategy-detail drawer (or navigates to
   `/sovereign/strategy/{id}`) showing the per-strategy equity curve,
   the last 10 parameter overrides, and the kill-switch history for that
   strategy only.
4. **Red Line Monitor** renders a Recharts line of `Net System Alpha` across
   a user-selectable 7/30/90 day window, with dollar units on the Y-axis,
   annotation markers for (a) kill-switch trips and (b) champion/challenger
   flips, and a zero reference line.
5. Directly below the Red Line chart, a **compute-cost stacked bar**
   segmented by provider: Anthropic, Vertex AI / Gemini, OpenAI, BigQuery
   slot-hours ($), alt-data vendor $. Daily buckets over the same selected
   window. Bar colors are deterministic per provider and locked in
   `frontend/src/lib/costColors.ts`.
6. The Red Line Monitor is ALSO embedded as the hero widget on the
   post-phase-4.7 homepage, above the fold, taking at least 55% of the
   home-page vertical real estate. Homepage layout is coordinated with
   phase-4.7.2; Phase C adds the widget, phase-4.7 reserves the slot.
7. All data is fetched browser-side via three endpoints:
   `GET /api/sovereign/leaderboard`,
   `GET /api/sovereign/red-line?window=30d`,
   `GET /api/sovereign/compute-cost?window=30d`.
   Each endpoint has a 30s timeout, returns cached BQ results with
   max-age 5 min, and declares `cron_slots: 0` (read-on-demand).
8. Page passes Lighthouse performance >= 0.9 and WCAG 2.1 AA, consistent
   with phase-4.7.2 and 4.7.6 success criteria. No emoji; Phosphor icons
   from `@/lib/icons.ts` only; dark theme `#0f172a` background.
9. Loading uses `PageSkeleton`; every chart has an empty-state placeholder
   (`frontend-layout.md` Section 8); every fetch group surfaces a rose
   error banner with retry if ALL primary calls fail.
10. Added to cross-page consistency pass (phase-4.7.5) so that a subsequent
    lint/audit enforces Sovereign page conformance.

## Step-by-step plan

### 10.5.0 - Backend read endpoints (stateless, no cron)

Add three FastAPI routes under `backend/api/sovereign.py`:
- `GET /api/sovereign/leaderboard` - reads deployed strategies from
  `pyfinagent_pms.strategy_deployments` (new view) joined with live P&L
  from the paper-trading tables and 30-day CVaR/compute cost aggregates.
- `GET /api/sovereign/red-line?window={7d|30d|90d}` - returns a daily
  series of `{date, profit_usd, risk_usd, compute_usd, net_alpha_usd}`.
- `GET /api/sovereign/compute-cost?window=...` - returns per-day per-provider
  spend from `all_billing_data` (GCP), Anthropic usage API, OpenAI usage
  API, Vertex AI invoice, and alt-data vendor CSV drops.

Verification: pytest hitting each endpoint with a BQ emulator fixture;
p95 latency < 800ms for 30-day window; `cron_slots: 0` annotation in the
router module docstring.

### 10.5.1 - BQ view `strategy_deployments`

Create `pyfinagent_pms.strategy_deployments` with columns:
`strategy_id STRING, deployed_since TIMESTAMP, retired_at TIMESTAMP,
status STRING, allocation_pct FLOAT64, source_config STRING`.
Back-fill from `optimizer_best.json` history + champion-challenger
roll-out state (phase-4.8.6). Read-only from Phase C.

Verification: `bq show` returns the view; seed script populates >= 1
champion row before the page ships.

### 10.5.2 - Frontend route `/sovereign` (shell)

Create `frontend/src/app/sovereign/page.tsx` using the "New Page Template"
section of `frontend-layout.md`. Sidebar entry added under a new
"Sovereign" group above "Reports". Tier 1 header: "Sovereign Dashboard".
No tabs; the page is a two-hero stack (Red Line on top, Leaderboard
below) because both heroes demand full width (Few 2006, Tufte 1983).

### 10.5.3 - `RedLineMonitor` component

`frontend/src/components/sovereign/RedLineMonitor.tsx`:
- Recharts `ComposedChart` with `<Area>` for Net System Alpha,
  `<ReferenceLine y=0>`, and `<ReferenceDot>` markers for kill-switch
  trips and champion-flip events.
- Window selector pill group `7d | 30d | 90d` (default 30d) above the
  chart, right-aligned, styled per `frontend-layout.md` Section 5.
- Legend fixed below the chart: "Profit - (Risk + Compute)" formula
  shown as literal text.
- Tooltip shows all four series values in USD with 2-decimal precision.

### 10.5.4 - `ComputeCostBreakdown` component

`frontend/src/components/sovereign/ComputeCostBreakdown.tsx`:
- Recharts `BarChart` stacked, one bar per day, segments per provider.
- Color map from `@/lib/costColors.ts` (deterministic; never
  auto-assigned).
- Hovering a segment shows provider, USD, and percent of that day's
  total. Bar sits in a `BentoCard` directly below the Red Line chart.

### 10.5.5 - `AlphaLeaderboard` component

`frontend/src/components/sovereign/AlphaLeaderboard.tsx`:
- Table component following `frontend-layout.md` Section 7.
- Columns exactly as Success criterion 2. Numeric columns right-aligned,
  `status` rendered as a Phosphor-icon pill (Crown for champion,
  GitBranch for challenger_live, Eye for challenger_shadow, Archive for
  retired). No emoji.
- Sort: click column header; persists via `useState`; default sort by
  `live_pnl_30d` desc.
- Filter: pill row above the table filters by `status`.
- Row click routes to `/sovereign/strategy/{id}` or opens a `Drawer`.

### 10.5.6 - Strategy detail view `/sovereign/strategy/[id]`

Minimal detail page reusing existing paper-trading equity-curve component
with a `strategyId` filter, plus a parameter-override timeline (reads
`quant_results.tsv` + `optimizer_best.json` history), plus the kill-switch
events from phase-4.8 filtered to this strategy.

### 10.5.7 - Homepage Red Line embed

Add a compact variant of `RedLineMonitor` (same component, prop
`variant="compact"`) as the hero widget on the post-phase-4.7 homepage.
Coordinates with phase-4.7.2 redesign: homepage reserves the slot; Phase
C fills it. OpsStatusBar stays above; Red Line takes >= 55% of remaining
vertical space; secondary MAS stats go below.

### 10.5.8 - Accessibility + cross-page consistency

Run `npm run axe`; keyboard navigation across window selector, row sort,
and row drawer. Add Sovereign page to the phase-4.7.5 consistency lint
allowlist. Add e2e Playwright spec asserting Red Line renders within
5s on a seeded BQ emulator.

### 10.5.9 - Docs + handoff

Update `frontend-layout.md` with a new subsection "Sovereign two-hero
layout" (the only sanctioned page that intentionally skips a tab bar on
a primary route). Append `handoff/harness_log.md` cycle entry.

## Research findings

Deep-read (full article / paper) marked [R]. Survey-level marked [S].

### Sovereign wealth fund dashboards + institutional public reporting

1. [R] Norges Bank Investment Management, Government Pension Fund Global -
   Annual Report 2024. https://www.nbim.no/en/publications/reports/
   - Public dashboard architecture: single "Return on the fund" hero,
     benchmark-relative return below, cost-per-AUM as third tier.
2. [R] NBIM - Responsible investment and risk section.
   https://www.nbim.no/en/responsible-investment/
   - Risk-adjusted return is top-of-page; costs broken out by asset class.
3. [S] GIC (Singapore) - Report on the Management of the Government's
   Portfolio 2023/24. https://www.gic.com.sg/our-performance/
4. [S] CalPERS - Trust Level Review Q4 2024.
   https://www.calpers.ca.gov/page/investments/asset-classes/performance
5. [R] CPP Investments Annual Report 2024 (net-of-cost return framing).
   https://www.cppinvestments.com/public-media/annual-report/
6. [S] Yale Endowment Report 2023.
   https://investments.yale.edu/reports

### Institutional risk dashboards

7. [R] Bloomberg Professional - RISK<GO> function overview.
   https://www.bloomberg.com/professional/solution/risk-and-valuations/
   - Canonical "one big risk number at top, component breakdown below"
     layout; VaR / CVaR at 97.5% and 99% are standard.
8. [R] MSCI Barra Portfolio Manager - factor exposure dashboards.
   https://www.msci.com/our-solutions/analytics/barra
9. [S] MSCI RiskMetrics - daily VaR and ES reporting patterns.
   https://www.msci.com/riskmanager
10. [S] Axioma Portfolio Analytics (Qontigo) - risk-and-return cockpit.
    https://qontigo.com/product/axioma-portfolio-analytics/
11. [R] J.P. Morgan Markets - Morgan Markets risk dashboards (public
    marketing doc). https://www.jpmorgan.com/markets

### Champion / challenger ML ops dashboards

12. [R] DataRobot - Champion / Challenger model monitoring.
    https://docs.datarobot.com/en/docs/mlops/monitor/challenger.html
    - Columns ship with leaderboard exactly: strategy_id analogue,
      deployed_since, live KPI delta vs champion, promotion button.
13. [R] H2O MLOps - Model Lifecycle and A/B.
    https://docs.h2o.ai/mlops/userguide/model-lifecycle/
14. [R] AWS SageMaker Model Dashboard.
    https://docs.aws.amazon.com/sagemaker/latest/dg/model-dashboard.html
15. [S] Vertex AI Model Registry + continuous evaluation.
    https://cloud.google.com/vertex-ai/docs/model-registry/introduction
16. [S] Weights & Biases - Launchpad and production comparison.
    https://docs.wandb.ai/guides/launch/
17. [S] Fiddler AI - production model monitoring cards.
    https://docs.fiddler.ai/product-guide/monitoring-platform

### Operator cockpit UI patterns

18. [R] Stripe Dashboard - Sigma and the status strip pattern.
    https://stripe.com/docs/sigma
19. [R] Linear Insights.
    https://linear.app/docs/insights
20. [R] Grafana 12 dynamic dashboards.
    https://grafana.com/blog/2025/05/07/dynamic-dashboards-grafana-12/
21. [S] Vercel Analytics dashboard.
    https://vercel.com/docs/analytics
22. [S] QuantConnect Cloud live trading results.
    https://www.quantconnect.com/docs/v2/cloud-platform/live-trading/results

### Risk-adjusted return visualization

23. [R] Edward Tufte - The Visual Display of Quantitative Information
    (2nd ed.) - data-ink ratio, sparkline argument, small multiples.
    https://www.edwardtufte.com/tufte/books_vdqi
24. [R] Stephen Few - Information Dashboard Design (2006).
    https://www.stephen-few.com/idd.php
25. [R] Colin Ware - Information Visualization: Perception for Design
    (4th ed., 2020). https://www.sciencedirect.com/book/9780128128756
26. [R] WSJ Graphics - 2024 portfolio drawdown interactives (data-driven
    narrative framing). https://www.wsj.com/news/types/graphics
27. [S] Bloomberg Graphics best-of 2024 on cost-of-trade visualization.
    https://www.bloomberg.com/graphics/

### Net-of-cost alpha framing (academic)

28. [R] Bailey, D. and Lopez de Prado, M. - "The Deflated Sharpe Ratio"
    (2014). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
29. [R] Bailey, D. and Lopez de Prado, M. - "The Probability of
    Backtest Overfitting" (2015).
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
30. [S] Harvey, C. R. and Liu, Y. - "Backtesting" (2015, JPM).
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2345489

### Compute-cost and FinOps references

31. [R] FinOps Foundation - Framework 2024.
    https://www.finops.org/framework/
    - Daily stacked-bar by provider is the canonical presentation
      for multi-cloud + SaaS AI spend (Inform phase).
32. [S] Anthropic Usage API.
    https://docs.anthropic.com/en/api/admin-api/usage-cost/get-cost-report
33. [S] OpenAI Usage API.
    https://platform.openai.com/docs/api-reference/usage
34. [S] GCP Billing Export to BigQuery.
    https://cloud.google.com/billing/docs/how-to/export-data-bigquery

### Accessibility + frontend conventions

35. [S] WCAG 2.1 AA success criteria.
    https://www.w3.org/TR/WCAG21/
36. [S] Phosphor Icons.
    https://phosphoricons.com/
37. [S] Recharts docs (ComposedChart + ReferenceDot).
    https://recharts.org/en-US/api/ComposedChart

## Proposed masterplan.json snippet

```json
{
  "id": "phase-10.5",
  "name": "Sovereign Dashboard (Net System Alpha cockpit)",
  "status": "pending",
  "depends_on": ["phase-4.7", "phase-4.8", "phase-8.5"],
  "gate": null,
  "cron_slots": 0,
  "steps": [
    {
      "id": "10.5.0",
      "name": "Backend read endpoints: leaderboard, red-line, compute-cost",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "cd backend && pytest tests/api/test_sovereign.py -q && python -c \"import json,urllib.request as u; r=json.load(u.urlopen('http://localhost:8000/api/sovereign/red-line?window=30d',timeout=10)); assert len(r['series'])>=25\"",
        "success_criteria": [
          "three_endpoints_landed",
          "p95_latency_under_800ms",
          "cron_slots_zero_declared"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.5.1",
      "name": "BQ view pyfinagent_pms.strategy_deployments",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "python scripts/migrations/create_strategy_deployments_view.py --verify",
        "success_criteria": [
          "view_exists",
          "at_least_one_champion_row"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.5.2",
      "name": "Route /sovereign shell (two-hero layout)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "cd frontend && npm run build && node scripts/audit/sovereign_route.js",
        "success_criteria": [
          "route_reachable",
          "sidebar_entry_added",
          "page_shell_conforms_to_frontend_layout"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.5.3",
      "name": "RedLineMonitor component (7/30/90 windows + event annotations)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "cd frontend && npm run test -- --filter=RedLineMonitor",
        "success_criteria": [
          "window_selector_7_30_90",
          "reference_line_zero",
          "kill_switch_and_flip_markers_rendered",
          "recharts_composed_chart"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.5.4",
      "name": "ComputeCostBreakdown stacked-bar by provider",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "cd frontend && npm run test -- --filter=ComputeCostBreakdown",
        "success_criteria": [
          "deterministic_color_map_present",
          "providers_cover_anthropic_vertex_openai_bq_altdata",
          "tooltip_shows_usd_and_percent"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.5.5",
      "name": "AlphaLeaderboard table (sortable, filterable)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "cd frontend && npm run test -- --filter=AlphaLeaderboard",
        "success_criteria": [
          "columns_match_spec",
          "status_pill_phosphor_only",
          "sort_persists_client_side",
          "filter_by_status_pill_row"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.5.6",
      "name": "Strategy detail route /sovereign/strategy/[id]",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "cd frontend && npm run test -- --filter=StrategyDetail",
        "success_criteria": [
          "equity_curve_scoped_by_strategy",
          "param_override_timeline_rendered",
          "kill_switch_events_scoped"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.5.7",
      "name": "Homepage Red Line hero embed (compact variant)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "cd frontend && npm run lighthouse -- --url http://localhost:3000 --output json --output-path handoff/lighthouse_home_sovereign.json && python -c \"import json; d=json.load(open('frontend/handoff/lighthouse_home_sovereign.json')); assert d['categories']['performance']['score'] >= 0.9\"",
        "success_criteria": [
          "red_line_hero_present_on_home",
          "takes_at_least_55pct_vertical",
          "lighthouse_perf_ge_90"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.5.8",
      "name": "Accessibility + consistency pass (WCAG 2.1 AA + frontend.md lint)",
      "status": "pending",
      "harness_required": true,
      "verification": {
        "command": "cd frontend && npm run axe && npm run lint && node scripts/audit/sovereign_consistency.js",
        "success_criteria": [
          "wcag_2_1_aa_pass",
          "phosphor_icons_only",
          "no_emoji_in_ui",
          "dark_theme_token_0f172a"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "10.5.9",
      "name": "Docs + handoff log cycle entry",
      "status": "pending",
      "harness_required": false,
      "verification": {
        "command": "grep -q 'Sovereign two-hero layout' .claude/rules/frontend-layout.md && grep -q 'phase-10.5' handoff/harness_log.md",
        "success_criteria": [
          "layout_docs_updated",
          "harness_log_cycle_entry_appended"
        ]
      },
      "contract": null,
      "retry_count": 0,
      "max_retries": 3
    }
  ]
}
```

## Implementation notes

- **No new cron.** Phase C consumes 0 of the 15 daily Claude scheduled
  routine slots. All reads are browser-driven with 5-minute cache on the
  BQ views. If a compute-cost fetch requires hitting Anthropic/OpenAI
  usage APIs, that call is server-side inside the FastAPI route, cached
  for 5 min, and STILL not a scheduled routine (event-driven on user
  load).
- **Shell conformance.** The page MUST use
  `<div className="flex h-screen overflow-hidden">` + Sidebar +
  `<main className="flex flex-1 flex-col overflow-hidden">` per
  `frontend-layout.md` Section 1. No `min-h-screen` anywhere. Fixed
  header zone holds Tier 1 title + optional window selector; scrollable
  zone holds the two heroes stacked.
- **No tabs.** The Sovereign page deliberately has no tab bar because
  both heroes answer the same question at different time resolutions,
  and splitting them across tabs would defeat the Red Line's purpose
  (Few 2006: one screen, one story).
- **Icons.** Use `Crown`, `GitBranch`, `Eye`, `Archive`, `ArrowUp`,
  `ArrowDown`, `Warning`, `Lightning` from `@/lib/icons.ts`. Add any
  missing aliases there, not at the callsite. No raw
  `@phosphor-icons/react` imports. No emoji.
- **Charts.** Recharts only. Dark theme `#0f172a` page background; card
  background `bg-navy-800/60`; grid color `#1f2937`. Tooltip background
  `#0f172a` with `border-navy-700`.
- **CVaR conversion.** `Risk Exposure = cvar_97_5_pct * NAV` where both
  inputs come from existing phase-4.8.2 outputs. If CVaR is missing for
  a day (pre-4.8.2 backfill), render that point as null and the chart
  shows a dashed gap, not zero.
- **Compute cost sources.**
  - Anthropic: Admin Usage / Cost API.
  - OpenAI: Usage API.
  - Vertex AI / Gemini: GCP billing export dataset
    `sunny-might-477607-p8.all_billing_data.gcp_billing_export_v1_*` with
    service filter on Vertex AI and BigQuery.
  - BigQuery slot-hours: same billing export, service = BigQuery.
  - Alt-data vendor: CSV drops in `gs://pyfinagent-ops/altdata_invoices/`
    with monthly normalized to daily pro rata.
- **Mutation rules (BQ).** The phase is read-only; use
  `execute_sql_readonly` for ad-hoc inspection during development. The
  one DDL statement (creating `strategy_deployments`) lives in
  `scripts/migrations/create_strategy_deployments_view.py`, not in an
  ad-hoc MCP call (per `CLAUDE.md` rule 5).
- **State.** Window selection persisted in `localStorage` under
  `sovereign.window` so the user's 30d vs 90d preference survives
  reloads.
- **Testing.** E2E Playwright spec seeds a BQ emulator with a champion
  row + 2 challenger rows + 30 days of P&L / CVaR / cost, loads
  `/sovereign`, asserts Red Line Monitor and Alpha Leaderboard render
  within 5s, and exercises window-switch + sort + row-click.
- **Future extension (out of scope).** A "promote challenger" action
  button is deliberately deferred to phase-4.8.6 (champion/challenger
  rollout). Phase C is observation-only.

## References

- `handoff/phase-proposals/phase-3.7-4.7-8.5-mas-ux-autoresearch.md` -
  phase-4.7 (UI baseline) and phase-8.5 (autoresearch leaderboard).
- `handoff/phase-proposals/phase-4.8-risk-compliance-hardening.md` -
  phase-4.8.2 supplies the CVaR feed consumed here; phase-4.8.6
  supplies the champion/challenger rollout state.
- `.claude/rules/frontend.md` - Next.js 15 + React 19 + Tailwind +
  Phosphor + Recharts conventions; scrollbar-thin; no emoji.
- `.claude/rules/frontend-layout.md` - page shell (Section 1), sidebar
  (Section 2), 6-tier anatomy (Section 3), metric grids (Section 4),
  OpsStatusBar pattern (Section 4.5), tab bar (Section 5), collapsibles
  (Section 6), content blocks (Section 7), empty states (Section 8),
  hierarchy principles (Section 9), "New Page Template" section.
- `CLAUDE.md` - live cap of 15 Claude scheduled routine runs per day;
  BigQuery MCP rules; 30s timeout.
- Public reports, risk platforms, and academic sources enumerated in
  the Research findings section above.
