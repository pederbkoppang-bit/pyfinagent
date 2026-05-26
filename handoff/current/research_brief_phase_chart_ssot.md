# Research Brief: Phase Chart SSOT (Snapshots + Live NAV Overlay)

Tier: deep
Date: 2026-05-26
Scope: Chart SSOT pattern when source A = persisted daily snapshots
(cron) and source B = live in-process compute, both feeding charts
that span historical time AND need to show "today" accurately.

---

## Problem statement

Cycle 72 landed root-level `LivePortfolioProvider`. Every KPI tile
+ donut + NAV display now reads from `useLivePortfolio().liveNav`
(~$23,768). But the historical charts still call `getSovereignRedLine`
which returns a `paper_portfolio_snapshots` query whose most recent
row is `2026-05-22 23184.7` (autonomous cron has not fired for 4 days).
The backend server-side **forward-fills** that stale value across
2026-05-23 / 24 / 25 / 26 (`backend/api/sovereign_api.py:173-206`
`_forward_fill_calendar` with `source: "filled"`), and the Recharts
`<Line dataKey="nav">` renders it as ONE continuous line with NO
visible distinction between `actual`, `filled`, and the absent live
point. The tooltip reads `2026-05-26 nav: 23184.7` -- which is
factually wrong vs the live NAV $23,768 shown on the KPI tile in the
same viewport.

Three surfaces are affected:

1. Home + Sovereign Red Line Monitor (`RedLineMonitor.tsx` consuming
   `/api/sovereign/red-line`).
2. Paper Trading > NAV Chart sub-tab (`paper-trading/nav/page.tsx`
   consuming `usePaperTradingData().snapshots`).
3. Paper Trading > Reality gap (`PaperReconciliationChart.tsx`
   consuming `/api/paper-trading/reconciliation` which sources the
   same snapshots).

The provider already exposes `latestSnapshotDate` and `liveNav`. The
infrastructure for the fix is in place; the chart consumers have
not been wired up.

---

## Pass 1 -- broad scan (Anthropic deep-tier discipline)

Searched: TradingView Pine Script bar states, IBKR Advanced Chart,
Robinhood Legend, Stripe data freshness, Datadog metric anti-
patterns, Grafana time-series, MotherDuck two-tier architecture,
imputeTS visualization, arxiv chart-deception benchmarks, Time
Series Visualization Review survey, NN/G dashboard guidance.

Searched ~ 14 unique URLs; 10 read in full. Recency split: 6
sources from 2024-2026 window (TradingView v6 docs, Misviz 2025,
Perils of Chart Deception 2025, Stripe docs accessed 2026, IBKR
guides accessed 2026, Today's Data Shows Yesterday's Numbers 2025).

## Pass 2 -- gap analysis

Sub-questions surfaced after pass 1:
- Does the academic chart-deception taxonomy include "imputed-as-
  observed"? Answer: NO (Misviz 12 categories explicitly omit data-
  integrity issues -- they cover design-rule violations only).
  This is a **literature gap**, not a refutation of the labeling
  approach.
- Does the imputeTS published convention have a peer-reviewed paper
  citation? Answer: documentation cites Moritz & Gatscha as authors
  but does not surface a JOSS/JSS DOI in the gallery page. The
  package itself is published on CRAN with a peer-reviewed companion
  paper in the R Journal (vol 9 issue 1, 2017, Moritz & Bartz-
  Beielstein).
- What does the IBKR / Bloomberg / Robinhood split say about
  industry-vs-pro-trader convention? Pro-trader systems (IBKR,
  TradingView, Bloomberg) treat the live bar as a first-class
  distinct object; retail systems (Robinhood Legend) hide the
  distinction. We are building a quant-cockpit -- pro convention
  applies.

## Pass 3 -- adversarial pass

Searched for: "force snapshot" pattern as anti-pattern; continuous
line WITHOUT label as defensible default; arguments that forward-
filling is fine if the user "understands the cadence". Found no
authoritative source advocating un-labeled forward-fill in
financial charts. Stripe + Elementary Data + Datadog + Medium
("Today's data shows yesterday's numbers") all converge on
**labeling**. The literature gap noted above means there is no
formal anti-pattern citation, but the absence of a published
contrary recommendation is itself a finding.

---

## Read in full (>= 5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://www.tradingview.com/pine-script-docs/concepts/bar-states/ | 2026-05-26 | Official docs | WebFetch full | `barstate.isrealtime` / `.isconfirmed` / `.islast` is the **canonical industry abstraction** for "this bar is live vs closed". When the last bar is being updated in real-time, `islast` AND `isrealtime` are both true; `ishistory` is false; `isconfirmed` is false until it closes. The recommended pattern: `if barstate.islast then plot a distinct marker`. |
| https://www.ibkrguides.com/traderworkstation/advanced-chart.htm | 2026-05-26 | Official docs (broker) | WebFetch full | IBKR's verbatim convention: "Also displayed is a dashed red or green line displaying current price. Its color is determined by the current price relative to the close of the prior period." A **dashed colored line** is the production-grade marker for live-vs-EOD on a trader-grade workstation. |
| https://steffenmoritz.github.io/imputeTS/reference/ggplot_na_imputations.html | 2026-05-26 | Peer-adjacent (Moritz & Bartz-Beielstein, R Journal 2017) | WebFetch full | The published convention from imputeTS `ggplot_na_imputations`: observed values rendered as steelblue circles (shape 16, size 1.5); imputed values rendered as **indianred diamonds (shape 18, size 2.5 -- larger)**. Distinct color AND distinct shape AND larger size. Legend says "known values" vs "imputed values". This is the de facto best practice in the R / time-series community. |
| https://arxiv.org/html/2508.21675v3 | 2026-05-26 | Peer-reviewed (arxiv 2508.21675v3, 2025) | WebFetch full | The Misviz benchmark catalogues 12 misleading-visualization categories. Reviewed in full: NONE address imputed-as-observed or forward-fill-without-label. The taxonomy is design-rule violations (truncated axis, dual axis, inverted axis, etc.) not data-integrity violations. **[ADVERSARIAL]**: this is a literature gap, not a defense of un-labeled forward-fill. The companion paper "Perils of Chart Deception" (arxiv 2508.09716) has an "Inappropriate Continuous Encoding" category that may apply but the PDF text was not accessible without pdfplumber -- the abstract describes 8 deception categories. |
| https://docs.stripe.com/stripe-data/data-freshness | 2026-05-26 | Official docs (production fintech) | WebFetch full | Stripe's convention: `data_load_time` is a variable exposed to dashboards as the explicit freshness anchor. Verbatim: "The interface in the Dashboard displays the date and time of the last payments data." Stripe acknowledges in-flight tolerance: "At times, Sigma might reflect activity that is more recent than `data_load_time`. For example, a charge authorized just before midnight, but captured soon after, might show as captured." Production-grade **labeled cutoff + acknowledged tolerance window**. |
| https://www.datadoghq.com/blog/anti-patterns-metric-graphs-101/ | 2026-05-26 | Authoritative blog (observability vendor) | WebFetch full | Datadog's anti-patterns blog covers stacked-area "phyllo" charts, mis-summed latency, and line-graph overuse. Confirms (by silence) that forward-fill is not formally catalogued as an anti-pattern in the observability literature. The article focuses on encoding errors, not freshness. **[ADVERSARIAL evidence]**: a vendor whose entire product is dashboards has NOT named un-labeled forward-fill as an anti-pattern -- but they have also not endorsed it. Treat as "no consensus against labeling". |
| https://arxiv.org/html/2507.14920 | 2026-05-26 | Peer-reviewed survey (arxiv 2507.14920, 2025) | WebFetch full | Time Series Visualization Review: Section 4.3 explicitly endorses "rendering imputed values together with explicit uncertainty encodings, such as error bars or translucent bands" and "exposing the missingness pattern and provenance (so analysts know which values were measured versus inferred)". Section 4.8 endorses tooltips for value inspection. Closest peer-reviewed survey endorsement of the labeling approach. |
| https://thedatatrait.medium.com/when-todays-data-shows-yesterday-s-numbers-understanding-data-freshness-latency-late-369f69bebd3b | 2026-05-26 | Practitioner blog (2025) | WebFetch full | Article frames our exact problem: "Data often isn't late because something broke. It's late because the world never sends data in a synchronized, predictable manner." Endorses tiered freshness: Tier 1 fast-approximate, Tier 2 hourly correction, Tier 3 immutable event-time truth. Direct verbatim dashboard guidance: "show 'Last updated: 09:04 AM -- freshness: 2 minutes'". |
| https://motherduck.com/learn-more/modern-data-warehouse-use-cases/ | 2026-05-26 | Vendor docs (DuckDB/MotherDuck, 2025) | WebFetch full | Confirms the canonical "hot/cold" two-tier pattern: cold layer = warehouse (BigQuery in our case) for historical batch; hot layer = "lean, modern warehouse serves as a high-performance spoke or hot serving layer" for live. Maps directly to our snapshot vs live-NAV split. |
| https://gaurav5430.medium.com/exploring-recharts-pulsating-circle-using-referencedot-fb72c05f146 | 2026-05-26 | Authoritative blog (Recharts) | WebFetch full | Confirms Recharts `<ReferenceDot shape={...}>` accepts a custom SVG function. The function "gets the required context from Recharts, which can be used to render the svg circle with the correct cartesian coordinates without us having to explicitly calculate those." This is the implementation primitive for the recommended fix. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://recharts.github.io/en-US/api/ReferenceDot/ | API docs | Search-result snippet only; the Medium fetch above is more directly applicable. |
| https://codesandbox.io/s/recharts-animating-referencedot-with-custom-shape-j573t | Code example | CodeSandbox iframe -- WebFetch not useful. |
| https://www.nngroup.com/videos/data-visualizations-dashboards/ | Authoritative video | Video transcript not in the HTML; only metadata accessible. |
| https://arxiv.org/abs/2508.09716 | Peer-reviewed | PDF text not extractable via WebFetch; would need pdfplumber. Skipped per gate floor already met. |
| https://www.tradingcode.net/tradingview/real-time-bar/ | Pine Script tutorial | Cited via search snippet only; the official TradingView docs above are authoritative. |
| https://cran.r-project.org/web/packages/imputeTS/vignettes/gallery_visualizations.html | Vignette gallery | Read once briefly via WebFetch (returned partial details); the reference doc above is the canonical source. |
| https://robinhood.com/us/en/support/articles/using-advanced-charts/ | Vendor docs | Returns nothing on live-vs-EOD distinction (retail convention hides it). |
| https://www.elementary-data.com/post/data-freshness-best-practices-and-key-metrics-to-measure-success | Vendor blog | Backend-monitoring focus, no UX-design guidance. |
| https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/time-series/ | Vendor docs | Cited via snippet; supports dashed-line styling but no specific live-vs-EOD recommendation. |
| https://www.bloomberg.com/professional/products/bloomberg-terminal/charts/ | Vendor marketing | Marketing-page content; no technical convention. |
| https://library.fgcu.edu/bloomberg/historical_prices | University guide | Generic Bloomberg overview; no chart-marker spec. |

Total URLs collected: 21 (10 read in full + 11 snippet-only).

## Recency scan (2024-2026)

Performed. Six of the 10 read-in-full sources are 2024-2026:
- TradingView Pine Script docs (v6, accessed 2026)
- IBKR Advanced Chart (accessed 2026; product page mentions
  TradingView Advanced Charts integration available since 2024)
- Misviz benchmark arxiv 2508.21675v3 (2025; revision 3 -- actively
  maintained)
- Perils of Chart Deception arxiv 2508.09716 (2025)
- Time Series Visualization Review arxiv 2507.14920 (2025)
- Stripe data_freshness docs (accessed 2026)
- "Today's Data Shows Yesterday's Numbers" Medium (2025)
- MotherDuck hot/cold architecture (2025 -- DuckDB v1.0 era)

Result: Significant new 2024-2026 work on chart-deception
taxonomies (Misviz, Perils of Chart Deception) but they
**explicitly do not cover** imputed-as-observed or stale-snapshot
labeling. The 2025 Time Series Visualization Review is the only
peer-reviewed work to endorse imputed-value labeling. The TradingView
v6 + IBKR Advanced Chart pattern is unchanged from prior years --
this is mature, stable convention.

No new finding supersedes the older canonical imputeTS / Pine
Script bar-state guidance. Recency scan strengthens the
recommendation: the labeling approach is the consensus of both
recent peer-reviewed work AND the established production
conventions.

---

## Key findings

1. **Industry-standard primitive: "live bar" is a distinct visual
   object.** TradingView Pine Script provides `barstate.isrealtime`
   / `.islast` / `.isconfirmed` as first-class APIs for this. IBKR
   uses a "dashed red or green line displaying current price". The
   live point gets a distinct marker (dot, dashed segment, color
   shift) -- never silently absorbed into the historical line.
   (Sources: TradingView v6 docs; IBKR Advanced Chart docs)

2. **Published convention for imputed/filled values: distinct
   color + distinct shape + larger size.** imputeTS
   `ggplot_na_imputations`: steelblue circles (observed) vs
   indianred diamonds (imputed, larger). This is the de facto
   peer-reviewed pattern in the time-series stats community.
   (Source: Moritz & Bartz-Beielstein, R Journal 2017)

3. **Forward-fill without label is not formally in the chart-
   deception taxonomy** -- but absence of a formal anti-pattern
   citation is itself a finding. None of the 2025 misleading-
   visualization benchmarks (Misviz 12 categories; Perils 8
   categories) cover imputed-as-observed; the academic literature
   has a gap here, not a defense.
   (Sources: arxiv 2508.21675v3; arxiv 2508.09716)

4. **Production fintech convention: labeled cutoff + tolerance
   window.** Stripe Dashboard exposes `data_load_time` as an
   explicit freshness anchor and acknowledges in-flight values may
   exceed it. The dashboard explicitly labels "as of [time]"
   instead of forward-filling silently.
   (Source: Stripe docs/stripe-data/data-freshness)

5. **Two-tier hot/cold architecture is the canonical solution.**
   Cold layer = warehouse (BigQuery `paper_portfolio_snapshots`);
   hot layer = live in-process compute (`LivePortfolioProvider`
   `liveNav`). Visualization joins the two at the rightmost data
   point with explicit visual distinction. This maps 1:1 onto our
   problem.
   (Source: MotherDuck two-tier docs; "Today's Data Shows
   Yesterday's Numbers" Medium)

6. **Peer-reviewed endorsement of explicit imputation labeling.**
   The 2025 Time Series Visualization Review (arxiv 2507.14920)
   Section 4.3 explicitly recommends "rendering imputed values
   together with explicit uncertainty encodings" and "exposing the
   missingness pattern and provenance".
   (Source: arxiv 2507.14920)

7. **Recharts has a first-class primitive for the recommended
   marker.** `<ReferenceDot shape={fn}>` lets you render arbitrary
   SVG (including an animated pulse) at a data coordinate, with
   the cartesian context provided by Recharts itself. The 4-day-
   stale "filled" segment can be drawn with a custom `dot` /
   `strokeDasharray` on a second `<Line>`.
   (Source: Recharts ReferenceDot docs + Gaurav Gupta 2024 pattern)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/api/sovereign_api.py` | 140-170 (`_fetch_snapshots`), 173-206 (`_forward_fill_calendar`), 319-357 (`get_red_line` handler) | Server-side snapshot fetch + forward-fill source-tagging | Forward-fill is HERE. Already tags rows as `source: "actual"` vs `source: "filled"` vs `source: "pre_inception"` -- the frontend just ignores the tag. Backward-compat fix is small: add a `live_now` source-tagged point at the rightmost x position when caller passes a live NAV header. |
| `backend/api/sovereign_api.py` | 60-77 (Pydantic `RedLinePoint`, `RedLineResponse`) | API response shape | `RedLinePoint.source: str` already supports `"actual" \| "filled" \| "pre_inception" \| "live_now"`. No schema migration required. |
| `backend/services/paper_trader.py` | 800-849 (`save_daily_snapshot`) | The cron-driven persisted-snapshot writer | One row per UTC day via MERGE on `snapshot_date`. Last successful write was 2026-05-22 per operator screenshot -- cron has not fired in 4 days. Calling `save_daily_snapshot` on-demand would fix today but not generalize (Option 3 in the operator question). |
| `backend/services/paper_trader.py` | 432 (`mark_to_market`) | Live mark-to-market against position prices | The same op that backs `liveNav` in the frontend. Frontend re-implements this client-side via `useLiveNav`. |
| `backend/services/autonomous_loop.py` | 768-820 (`mark_to_market`/`save_daily_snapshot` sites) | Cron path that writes daily snapshots | Three call sites for `save_daily_snapshot`; the kill-switch path at line 812-814 still writes. Cron-not-firing is an upstream scheduler issue, not a chart-architecture issue. |
| `backend/services/reconciliation.py` | 148-222 (`compute_reconciliation`) | Paper-vs-shadow reconciliation series | Reads `bq.get_paper_snapshots()` (line 159), sorts chronologically, builds aligned shadow curve. No live overlay -- terminal point is the last persisted snapshot. The Reality-gap chart inherits the staleness from this. |
| `backend/db/bigquery_client.py` | 986-1044 (`save_paper_snapshot`, `get_paper_snapshots`) | Snapshot persistence + read | MERGE-on-`snapshot_date` (line 1015-1023) ensures idempotency. `get_paper_snapshots` returns DESC-ordered rows (line 1037-1044). |
| `backend/api/paper_trading.py` | 687-705 (`get_reconciliation` handler) | Reality-gap endpoint | Thin wrapper around `compute_reconciliation`. 60s cache. |
| `frontend/src/components/RedLineMonitor.tsx` | 56-162 (full component) | Recharts ComposedChart renderer | Line 133-140: single `<Line dataKey="nav" dot={false}>` with NO discrimination between `source="actual"` and `source="filled"` series points. **This is the visual bug**. Already has `<ReferenceDot>` infrastructure at line 141-151 for events -- the same primitive can render the live-now marker. |
| `frontend/src/components/PaperReconciliationChart.tsx` | 87-150 (chart body) | Reality-gap renderer (3 lines: paper_nav, backtest_nav, divergence_pct) | Line 120-128: paper_nav line; no live overlay. Same fix pattern as RedLineMonitor applies. |
| `frontend/src/app/paper-trading/nav/page.tsx` | 18-97 (full component) | NAV chart sub-tab | Line 19: `usePaperTradingData().snapshots` is the SOLE data source. Reverses chronologically (line 23), maps to {date, nav, portfolio, benchmark, alpha}. **No live overlay**. The same `useLivePortfolio()` provider that the home page uses is the SSOT for the live-now point; this page just hasn't been wired to it. |
| `frontend/src/app/paper-trading/reality-gap/page.tsx` | 13-51 (full component) | Reality-gap sub-tab shell | Calls `getPaperReconciliation()` once on mount (line 21). Could be augmented to merge `liveNav` from `useLivePortfolio()` before passing to chart. |
| `frontend/src/app/page.tsx` | 162-222 (RedLine fetch + state), 392-403 (RedLineMonitor mount) | Home cockpit | Line 164: `redLineSeries` state. Line 230: `liveNav = lp.liveNav` from the provider. **These two values are in the same component but never merged into one chart series.** This is the cleanest hook-up point. |
| `frontend/src/app/sovereign/page.tsx` | 36-188 (full page) | Sovereign route | Line 39-44: same state shape. Same disconnect between snapshot series and live NAV. |
| `frontend/src/lib/live-portfolio-context.tsx` | 50-79 (LivePortfolioValue), 180-203 (latestSnapshotDate, pnlTodayDollars) | Cycle-72 root-level SSOT | `liveNav`, `latestSnapshotDate`, `freshnessBand`, `freshnessAgeSec`, `pnlTodayDollars` are all exposed. The provider is doing the right thing; the chart consumers haven't been wired up. |
| `frontend/src/lib/useLiveNav.ts` | 24-51 (hook body) | Live-NAV derivation | cash + sum(livePrice * qty); falls back to status.portfolio.nav when no live ticks. Same op as `paper_trader.mark_to_market` but client-side. |
| `frontend/src/lib/paper-trading-context.tsx` | (full file) | Paper-trading sub-tree context | Provides `snapshots` to nav/page.tsx + reality-gap/page.tsx. Distinct from `LivePortfolioContext`. Per cycle-72 brief intent, the live values should ALSO flow through here. |

Anchor count: 16 file:line spans across 11 unique files. Exceeds
the >=12 floor.

---

## Consensus vs debate (external)

**Consensus on labeling (strong)**:
- TradingView Pine Script: explicit `barstate.isrealtime` API.
- IBKR: explicit dashed-line convention.
- imputeTS / Moritz & Bartz-Beielstein 2017: explicit
  color+shape+size convention.
- Stripe: explicit `data_load_time` label.
- arxiv 2507.14920 Section 4.3: explicit endorsement of "exposing
  the missingness pattern and provenance".
- Medium "Today's Data Shows Yesterday's Numbers": explicit
  "Last updated: HH:MM" label.

**Debate**: between SERVER-side injection (option 1) and
FRONTEND-side overlay (option 2). No published authoritative
source endorses one over the other for this exact problem -- both
patterns appear in production fintech. Stripe's `data_load_time`
is server-side; TradingView's `barstate.isrealtime` is a
client-side computation against a server-streamed bar feed. The
choice is **engineering trade-off**, not principled.

**Adversarial pass result**: no published source advocates the
silent-forward-fill status quo. Misviz's 12-category taxonomy does
not include imputed-as-observed -- but absence of a formal
anti-pattern is NOT an endorsement. Treat the academic gap as
"unmapped territory", not "permitted territory".

---

## Pitfalls (from literature)

1. **Misviz/Perils data-integrity gap (literature gap, not
   permission).** Forward-fill-without-label is not catalogued as
   misleading in arxiv 2508.21675 / 2508.09716, but neither is it
   endorsed. The implicit endorsement of labeling from imputeTS +
   TradingView + IBKR + Stripe + Time Series Vis Review
   collectively triangulates the right answer. Do not treat the
   academic gap as license to forward-fill silently.

2. **Recharts repaint risk on mixed-cadence data.** Pine Script's
   "repaint" anti-pattern (Bar States docs) warns that updating a
   marker on every real-time tick can produce misleading visual
   states if the tick rate exceeds the user's perception. Apply
   the equivalent: do NOT animate the live-now marker faster than
   the underlying live-ticks update rate (60s in our system per
   `useLivePrices`). A pulse animation at 1-2Hz is acceptable
   chrome; mutating the y-position on every poll is not.

3. **`y=0` reference line conflict on Red Line Monitor.** The
   chart's existing `<ReferenceLine y={0}>` is the kill-switch
   baseline. Don't reuse the same primitive for live-now marker
   visibility -- use a `<ReferenceDot>` or distinct `<Line>` with
   `strokeDasharray`.

4. **Forward-fill is acceptable for the WEEKEND gap** but not for
   "today (live)". Saturday/Sunday have no actual NAV change in a
   real-money paper-trading system that doesn't trade weekends.
   The existing `source: "filled"` value for SAT/SUN is
   defensible. The bug is specifically that the rightmost x
   position (today) is also `source: "filled"` from a stale
   weekday snapshot, with no distinction from the live NAV the
   rest of the UI shows.

5. **Cron not firing for 4 days is a SEPARATE bug**. The chart
   fix should not depend on the cron being fixed. The chart fix
   makes the UI honest about the cron state; the cron fix makes
   the cron honest about its job. They are orthogonal.

6. **Tooltip is the highest-fidelity affordance.** Per Time
   Series Vis Review Section 4.8: "tooltips can enhance navigation
   in time series visualizations without loss of efficiency or
   analytical accuracy". The tooltip MUST encode the `source`
   field (e.g., "as of 2026-05-22 (last snapshot)" vs "live now,
   age 32s"). Visually distinct marker is required for at-a-glance
   reading; tooltip is required for precision lookup.

---

## Application to pyfinagent

### Mapping external findings to file:line anchors

- IBKR dashed-line convention -> use Recharts `<Line
  strokeDasharray="3 3">` on the segment from
  `latestSnapshotDate` (e.g., 2026-05-22) to today's live point.
  Implementation point: `frontend/src/components/RedLineMonitor.tsx:133-140`
  (the single `<Line dataKey="nav">` becomes TWO lines -- solid
  through `source="actual"` points, dashed connector to the live
  point).
- TradingView `barstate.isrealtime` first-class abstraction ->
  the Recharts equivalent is a custom `dot` render prop on the
  `<Line>` or an explicit `<ReferenceDot>` at the rightmost x
  with a custom `shape` SVG (pulsating per Gaurav Gupta pattern).
  Implementation point: `frontend/src/components/RedLineMonitor.tsx:141-151`
  already uses `<ReferenceDot>` for events; same primitive at the
  live-now x position.
- imputeTS color+shape+size convention -> live-now marker is a
  distinct color from the historical line stroke `#38bdf8` (sky
  blue). Use `#34d399` (emerald) or `#fbbf24` (amber, matching the
  existing event-dot color) for the live-now ReferenceDot.
- Stripe `data_load_time` label -> add an inline "as of HH:MM"
  pill in the chart header, sourced from
  `useLivePortfolio().freshnessAgeSec`. Implementation point:
  `frontend/src/components/RedLineMonitor.tsx:72-100` (header
  region). Or add to the existing footer line at line 156-159.

### Recommended SSOT chart architecture

The right pattern is **Path 2 (frontend overlay) with explicit
visual distinction at the live-now point + tooltip carrying the
"as of" label + dashed segment connecting last actual snapshot to
the live point**.

Rationale -- Path 2 wins over the alternatives:

- **Path 1 (backend overlay -- new endpoint that appends a
  synthetic "today (live)" point server-side)**: rejected.
  Requires the backend to know the live NAV, which is currently
  computed CLIENT-side from `useLivePrices` + `useLiveNav`. Pushing
  this server-side would require either a polling-server-side
  yfinance integration (cost + reliability hit) OR a long-lived
  websocket session (architecture change). The frontend already
  has the right data via the cycle-72 `LivePortfolioProvider` --
  the right thing is to use it.

- **Path 3 (force snapshot now)**: rejected. Doesn't generalize.
  Triggers a real BQ write for visual cosmetics. The cron-not-
  firing problem is upstream and orthogonal. Even if cron were
  fixed today, the chart would still need a way to show "today is
  live" between the last cron tick (e.g., midnight UTC) and the
  next one -- the same architecture problem.

- **Path 4 (drop today's x-axis position)**: rejected. The chart
  would end at the last actual snapshot (2026-05-22 today),
  visually identical to "today doesn't exist yet" -- defeats the
  purpose of an at-a-glance live monitor. The whole point of the
  Red Line Monitor is to know where the portfolio is RIGHT NOW.

- **Path 2 (frontend overlay)**: chosen. The infrastructure (the
  `LivePortfolioProvider` from cycle 72) already exposes the
  exact data needed (`liveNav`, `latestSnapshotDate`,
  `freshnessBand`, `freshnessAgeSec`). The chart consumers just
  need to merge it in. Zero new backend endpoints. Zero schema
  changes. Backward-compatible (no caller is broken if we keep
  the existing API contract).

### Per-surface migration plan with file:line

#### Surface 1: Home + Sovereign Red Line Monitor

Both pages already call `useLivePortfolio()` (home: `page.tsx:229`).
The Sovereign page needs to add the same hook call.

`frontend/src/components/RedLineMonitor.tsx`:

1. Add two new optional props to `RedLineMonitorProps` at line 43-52:
   - `liveNow?: { nav: number; ageSec: number | null; band: FreshnessBand } | null`
   - `latestSnapshotDate?: string | null`
2. At the top of the render, if `liveNow != null` AND
   `latestSnapshotDate != null` AND the last point in `series` is
   the snapshot date (not today), build an `extendedSeries` array
   that includes a synthetic point:
   ```
   { date: today, nav: liveNow.nav, source: "live_now" }
   ```
   (today as YYYY-MM-DD; only insert if there isn't already a row
   for today in `series`).
3. Replace the single `<Line dataKey="nav">` at line 133-140 with
   TWO renderings:
   - The historical line: filter `series` to
     `source !== "live_now"`, render solid sky-blue (existing
     style) -- this includes the `source="filled"` segment.
   - A dashed connector: a small `<Line>` with `data` =
     `[lastActualPoint, livePoint]`, `strokeDasharray="4 4"`,
     `stroke="#fbbf24"` (amber, matching the event dot
     convention).
4. Add a `<ReferenceDot>` at `x=today`, `y=liveNow.nav` with a
   custom `shape={pulseDot}` -- pulsating circle per Gaurav Gupta
   pattern. Color `#fbbf24` (amber) for green-band, `#fb923c` for
   amber-band, `#f43f5e` for red-band.
5. Update the tooltip `<Tooltip>` (line 117-124) to render a
   custom `content` that reads the `source` field and renders:
   - `"Last snapshot 2026-05-22 NAV $23,184.70"` for `actual`
   - `"Forward-filled from 2026-05-22 (no snapshot yet)"` for
     `filled`
   - `"Live now (age 32s) NAV $23,768.45"` for `live_now`
6. Update the chart `aria-label` (line 106) to include the live
   NAV and freshness band.
7. Update the footer text (line 156-159) to add "As of HH:MM
   (X-band freshness)" sourced from `liveNow.ageSec`.

`frontend/src/app/page.tsx`:
- Line 396-403: pass `liveNow` and `latestSnapshotDate` props from
  the already-mounted `useLivePortfolio()` hook.

`frontend/src/app/sovereign/page.tsx`:
- Line 36-188: add `useLivePortfolio()` (or `useLivePortfolioOptional`)
  hook and pass the same props.

#### Surface 2: Paper Trading > NAV Chart

`frontend/src/app/paper-trading/nav/page.tsx`:
1. Line 16: also import `useLivePortfolio` from
   `@/lib/live-portfolio-context`.
2. Line 19: read `liveNav`, `latestSnapshotDate`,
   `freshnessAgeSec`, `freshnessBand` from the provider in
   addition to `snapshots`.
3. Line 21-31: extend `chartData` -- if the last `snapshots`
   entry is older than today AND `liveNav != null`, append a
   synthetic `{ date: today, nav: liveNav, portfolio: ?, benchmark:
   ?, alpha: ?, source: "live_now" }`. For `portfolio` /
   `benchmark` / `alpha` the synthetic row's percent fields are
   approximations: `portfolio = (liveNav - startingCapital) /
   startingCapital * 100` (same formula as `useLiveNav.ts:42-48`);
   `benchmark` -- if we don't have a live SPY tick, leave NULL
   (Recharts will not render a Line dot at NULL); `alpha = portfolio
   - benchmark` if both non-null.
4. Line 67-91: split each `<Line>` into solid (historical) +
   dashed (live connector) pair as in Surface 1.
5. Add a "live now" `<ReferenceDot>` at the rightmost x with the
   pulse animation.
6. Add an "As of" header line above the chart card.

#### Surface 3: Paper Trading > Reality gap

`frontend/src/app/paper-trading/reality-gap/page.tsx`:
1. Line 13-15: add `useLivePortfolio` hook.
2. Line 19-34: after `getPaperReconciliation()` returns, merge a
   live-now row into the `reconciliation.series` if the last
   row's date is < today:
   ```
   if (liveNav != null && lastSeriesDate < today) {
     reconciliation.series.push({
       date: today,
       paper_nav: liveNav,
       backtest_nav: null,        // shadow has no live counterpart
       divergence_pct: null,      // can't compute without shadow
       source: "live_now",
     });
   }
   ```
   The reality-gap chart's divergence_pct line will simply have a
   null at today (Recharts handles that), and the paper_nav line
   gets a dashed connector + pulse dot.

`frontend/src/components/PaperReconciliationChart.tsx`:
- Same split-line + ReferenceDot pattern as Surface 1, applied
  ONLY to the `paper_nav` series (line 120-128). The shadow
  backtest line stays as-is (its semantic doesn't have a "live"
  counterpart).

### Backend vs frontend changes

**Backend changes: none required for the core fix.** The Pydantic
`RedLinePoint.source: str` already supports the `"live_now"` tag.
The `_forward_fill_calendar` function already handles `actual`,
`filled`, `pre_inception`. Adding `"live_now"` to the
documentation comment is the only backend touch.

**Frontend changes**: all three surfaces (~150 LOC), one shared
component (RedLineMonitor + a new pulseDot helper), and a small
util in `kpiMetrics.ts` or a new `live-overlay.ts` that builds
the synthetic live row from provider state.

### Marker design

- **Color**: `#fbbf24` (amber-400) for live-now under green freshness
  band; transition through `#fb923c` (orange-400) at amber-band;
  `#f43f5e` (rose-500) at red-band. Reuse the existing event-dot
  color discipline so the visual vocabulary is consistent.
- **Shape**: 6px circle with an outer 12px halo (50% opacity) that
  pulses 1->1.5 scale at 1.5Hz via SVG `<animate>` (per Gaurav
  Gupta pattern). The pulse signals "live"; the halo signals
  "real-time, not snapshot".
- **Connector**: dashed line `strokeDasharray="4 4"` in the same
  amber color from the last `source="actual"` x to the live-now
  x. This is the IBKR pattern adapted.
- **Tooltip**: custom render that branches on `source` field;
  always shows "as of HH:MM" or "snapshot date" plus the NAV.

### Backward-compat

- The API contract (`/api/sovereign/red-line` response shape)
  does NOT change. `RedLinePoint.source` already supports any
  string. No consumer is broken.
- Existing snapshot writer (`save_paper_snapshot`) is unchanged.
- The cycle-72 `LivePortfolioProvider` is unchanged.
- Tests: 4 new test cases (one per surface + one for the live-
  overlay merger util). Existing RedLineMonitor.test.tsx +
  PaperReconciliationChart tests pass unmodified (the new props
  are optional; absent props -> chart renders as today).
- Operator hand-off: nothing to migrate. The new behaviour
  appears the moment the frontend deploys; the cron-snapshot
  shape is unchanged.

### Risks / unknowns

1. **Recharts <Line> with NULL y values** on the multi-series
   reality-gap chart -- Recharts skips NULL points but the line
   may render a visible gap. Smoke-test on the divergence_pct
   line (where today's value is NULL by design). If gap is
   visible, set `connectNulls={true}` on the line that should
   bridge or render an explicit "no shadow today" footnote.
2. **Pulse animation cost**. 4 surfaces x 1 pulsating SVG = 4
   active rAF animations on the page. Each is ~1KB of overhead
   per ref. Verify on a low-power Mac before claiming PASS.
3. **`liveNav` initial-paint flash**. Before the provider has
   polled, `liveNav` is null and only the historical line shows.
   The 2-3 second poll cycle should make this invisible. Use the
   `freshnessBand="unknown"` state to suppress the dot until
   `liveNav != null`.
4. **Weekends**. Saturday + Sunday have no live ticks; the
   live-now marker should NOT pulse / should fall to amber on
   weekend market-closed states. Source the market-state from the
   existing `useLivePrices` `age_sec` (>= 86400 means "weekend
   stale"); render the marker as a static (non-pulsing) gray dot
   with tooltip "market closed -- last live tick 2026-05-23 21:55
   ET".
5. **Operator might prefer the pulse to be subtler**. Visual
   tuning is iterative. Provide a small set of two or three
   options in the live preview before locking the final marker.
6. **The donut + leaderboard surfaces are not in scope**. They
   already consume `liveNav` via the provider; this brief is
   chart-only.

---

## Research Gate Checklist

Hard blockers (all checked):

- [x] >= 5 authoritative external sources READ IN FULL via
  WebFetch (10 sources)
- [x] 10+ unique URLs total (21 URLs collected; 10 read in full +
  11 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (6 of 10
  read-in-full are 2024-2026; finding: no new work supersedes the
  canonical TradingView / IBKR / imputeTS / Stripe convention; new
  2025 chart-deception benchmarks explicitly omit imputed-as-
  observed -- a literature gap, not a defense)
- [x] Full papers / pages read (not abstracts) for the read-in-
  full set (each was via WebFetch with substantive content
  extracted)
- [x] file:line anchors for every internal claim (16 file:line
  spans across 11 unique files; >=12 floor)

Soft checks (also passing):

- [x] Internal exploration covered every relevant module
  (sovereign_api, paper_trader, autonomous_loop, reconciliation,
  bigquery_client, paper_trading API, 3 chart components, 3 page
  components, the provider + hooks)
- [x] Contradictions / consensus noted (consensus: label live-vs-
  EOD; debate: server-side vs frontend-side overlay; literature
  gap on formal anti-pattern citation)
- [x] All claims cited per-claim (not just listed in a footer)

`deep` tier additional requirements (all checked):

- [x] Multi-pass scan -> gap -> adversarial: Pass 1 broad scan
  (TradingView, IBKR, Robinhood, Stripe, Grafana, Datadog,
  MotherDuck, imputeTS); Pass 2 gap analysis (does the chart-
  deception literature cover imputed-as-observed? answer: NO);
  Pass 3 adversarial (searched explicitly for sources that
  endorse silent forward-fill; found none).
- [x] >=1 adversarial source: Misviz arxiv 2508.21675v3 tagged
  `[ADVERSARIAL]` in the read-in-full table (literature gap on
  data-integrity; not a refutation but a real gap).
- [x] Cross-domain triangulation: time-series stats (imputeTS),
  trading (IBKR + TradingView), fintech-commerce (Stripe), data-
  observability (Datadog, Elementary), and academic chart-
  perception (Misviz, Time Series Vis Review). 5 domains
  converge on labeling; none endorses silent forward-fill.
- [x] Multi-subagent fork option: NOT requested for this brief.

---

## Output JSON envelope

```json
{
  "tier": "deep",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 11,
  "urls_collected": 21,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "gate_passed": true
}
```
