# Research Brief — `goal-market-filter-in-gate-bar`

**Tier:** simple. **Date:** 2026-06-01.
**Objective:** Fold the paper-trading market filter (`All·US·EU·KR`
radiogroup, `MarketFilter.tsx`) INTO the operator status bar
(`OpsStatusBar.tsx`) as a conditional segment; retire the standalone filter
row at `layout.tsx:483-490`; fold the `MarketSessionStrip` open/closed
signal into the pills. `OpsStatusBar` is SHARED with the homepage
(`page.tsx:360`) — the new segment MUST be conditional or the homepage
breaks (auto-FAIL per acceptance criterion 4).

**Bottom line:** the change is low-risk and well-supported. Three findings
decide the design: (1) the repo's own §4.5 doctrine endorses *folding a
signal into the bar* ("fold its signal into the bar itself"), so a
*controls-in-status-bar* tension is resolvable, not blocking; (2) because
`OpsStatusBar` is a `<section>` and NOT `role="toolbar"`, the moved
radiogroup keeps its native four-direction arrow-key model with zero
conflict (promoting to `role="toolbar"` would BREAK it — do not); (3) the
existing mount-guarded `useState<Date|null>` is exactly React's documented
two-pass pattern and stays correct in React 19, so the session dot folds in
safely. `gate_passed: true`.

---

## Source table

### Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://www.w3.org/WAI/ARIA/apg/patterns/radio/examples/radio/ | 2026-06-01 | standard (W3C APG) | WebFetch (full) | Standalone radiogroup: container `role="radiogroup"` (not focusable), options `role="radio"`+`aria-checked`, roving tabindex (one `0`, rest `-1`), Arrow keys (all 4 dirs) move+check with **selection-follows-focus**, Home/End, Space. Exactly what `MarketFilter.tsx` implements. |
| 2 | https://www.w3.org/WAI/ARIA/apg/patterns/toolbar/ | 2026-06-01 | standard (W3C APG) | WebFetch (full) | `role="toolbar"` is OPTIONAL (use only for grouping 3+ controls to reduce tab stops). **Explicit warning:** "Avoid including controls whose operation requires the pair of arrow keys used for toolbar navigation." A radiogroup needs arrows -> nesting in a `toolbar` conflicts. Generic container w/o `toolbar` role -> each widget keeps native keyboard model. |
| 3 | https://react.dev/reference/react-dom/client/hydrateRoot | 2026-06-01 | official doc (React) | WebFetch (full) | For inherently client-only values (`new Date()`), two valid patterns: (a) `suppressHydrationWarning` (one level deep, escape hatch), (b) **two-pass render via `useState`+`useEffect`** (initial render matches server, updates post-hydration). Holds for React 19. |
| 4 | https://github.blog/changelog/2026-04-16-rule-insights-dashboard-and-unified-filter-bar/ | 2026-06-01 | vendor changelog (GitHub) | WebFetch (full) | Apr-2026 peer precedent: GitHub replaced per-page custom dropdowns with ONE "unified filter bar component" across alert pages. Stated rationale = **consistency** ("consistent filtering experience across all of these pages"). Consolidating filter controls into a shared bar is shipping practice. |
| 5 | https://grafana.com/blog/2025/05/07/dynamic-dashboards-grafana-12/ | 2026-06-01 | vendor blog (Grafana) | WebFetch (full) | Grafana 12 reduces clutter via **conditional rendering** ("panels or entire rows shown/hidden based on variable selections... reduces clutter") + context-aware side pane. Directly supports the conditional-segment design + the row-removal density win. |
| 6 | https://tailkits.com/blog/tailwind-dynamic-classes/ | 2026-06-01 | practitioner blog | WebFetch (full) | Tailwind JIT static-scans for literal class strings; `bg-${color}-500` is NOT generated. Fix = **static literal lookup `Record`** (preferred) or safelist. Current for v3/v4 (updated Feb-2025). Confirms `MARKET_DOT_CLASS` (`format.ts:100`) is the correct, JIT-safe pattern to reuse. |

(6 read in full; floor is 5.)

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|
| https://www.w3.org/WAI/ARIA/apg/practices/keyboard-interface/ | standard | Roving-tabindex practice; covered by sources 1+2 |
| https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/Reference/Roles/toolbar_role | doc (MDN) | Toolbar role corroboration; source 2 authoritative |
| https://www.w3.org/WAI/ARIA/apg/patterns/toolbar/examples/toolbar/ | standard example | Concrete toolbar w/ nested alignment radiogroup using Up/Down (proves the arrow conflict); snippet sufficient |
| https://www.pencilandpaper.io/articles/ux-pattern-analysis-data-dashboards | practitioner | "Top-rail consolidates nav+filters+KPIs into a horizontal header"; snippet |
| https://www.aufaitux.com/blog/dashboard-filter-design-guide/ | practitioner | "Consolidate filters into one single, consistent panel"; snippet |
| https://blog.logrocket.com/ux-design/dashboard-ui-best-practices-examples/ | practitioner | Linear/Stripe minimal-chrome density study; snippet |
| https://www.gitnexa.com/blogs/saas-dashboard-ux-patterns | practitioner | 2026 SaaS dashboard patterns; snippet |
| https://grafana.com/blog/2025/05/07/dynamic-dashboards-grafana-12/ (already full) | — | (listed in read-in-full) |
| https://github.com/tailwindlabs/tailwindcss/discussions/14050 | issue | Tailwind safelist for dynamic classes; corroborates source 6 |
| https://blogs.perficient.com/2025/08/19/understanding-tailwind-css-safelist-keep-your-dynamic-classes-safe/ | blog | Safelist alternative (Aug-2025); snippet |
| https://opensource.adobe.com/spectrum-web-components/tools/roving-tab-index/ | doc | Roving-tabindex impl reference; snippet |
| https://www.uxpin.com/studio/blog/keyboard-navigation-patterns-complex-widgets/ | blog | Keyboard-nav patterns for composite widgets; snippet |
| https://www.datacamp.com/tutorial/dashboard-design-tutorial | tutorial | Operational dashboards = big status indicators + low latency; snippet |
| https://medium.com/@achronus/solving-a-niche-frontend-problem-dynamic-tailwind-css-classes-in-react-da5f513ecf6a | blog | Lookup-map pattern in React; corroborates source 6 |

**Unique URLs collected: 20** (6 read-in-full + 14 snippet-only). Floor is 10.

---

## Recency scan (last 2 years, 2024-2026)
**Performed.** Findings in the 2024-2026 window:
1. **GitHub "unified filter bar" (Apr 16 2026)** — a fresh, dated peer
   precedent for consolidating view-filter controls into ONE shared bar
   component; rationale = consistency. COMPLEMENTS the older Few/Stripe
   density doctrine the repo already cites. (source 4)
2. **Grafana 12 dynamic dashboards (May 7 2025)** — conditional
   rendering of rows/panels to "reduce clutter"; already cited in
   `frontend-layout.md` §4.5 and re-verified current. COMPLEMENTS the
   conditional-segment + row-removal design. (source 5)
3. **Tailwind dynamic-class guidance (updated Feb 2 2025)** — static
   literal lookup map remains the recommended fix for v3/v4; no
   deprecation. CONFIRMS the existing `MARKET_DOT_CLASS` approach. (source 6)
4. **W3C APG radio + toolbar patterns** — current living standard; the
   arrow-key-conflict warning for controls nested in a `toolbar` is
   unchanged. SUPERSEDES nothing; it is the canonical a11y authority and
   directly shapes the "keep `<section>`, don't promote to `toolbar`"
   decision. (sources 1, 2)
5. **React hydration guidance** — `react.dev` hydrateRoot doc is
   version-current; the two-pass `useState/useEffect` pattern is still the
   recommended way to render client-only time values. No React-19 change
   that affects the `MarketSessionStrip` mount-guard. (source 3)

No source CONTRADICTS the planned approach. The recency hits all reinforce
it (consolidation is accepted; conditional rendering reduces clutter; the
JIT + hydration patterns already in the repo are still correct).

## 3-query-variant evidence
- **Current-year (2026):** "operator status bar control consolidation 2026
  dashboard segment filter best practice" -> surfaced the GitHub Apr-2026
  unified-filter-bar changelog (source 4) + 2026 SaaS dashboard guides.
- **Last-2-year (2024-2025):** "Next.js 15 React 19 hydration mismatch
  new Date() useEffect mount guard" + Grafana-12 (May-2025) + Tailwind
  (Feb-2025 update). Covers the hydration + JIT + conditional-render leg.
- **Year-less canonical:** "WAI-ARIA APG radiogroup roving tabindex inside
  toolbar" + "dense status bar toolbar consolidating view controls
  Linear Stripe" -> surfaced the W3C APG radio/toolbar living standards
  (sources 1, 2) and the canonical Few/Stripe/Linear density literature
  (Pencil&Paper, LogRocket, AufaitUX snippets).

---

## Key findings (external)

1. **Folding a signal into the operator bar is explicitly endorsed by the
   project's own §4.5 doctrine** — and consolidation is mainstream 2026
   practice. The goal flags a "tension" with §4.5 ("the dense bar is for
   status, not controls"). The tension is OVER-STATED: §4.5's *forbidden*
   list targets rendering status as separate *cards/bento*, and its
   prescriptions literally say "fold its signal into the bar itself" and
   put a `Next run` segment with `ml-auto`. The market filter is a global
   view-scope control; the open/closed dot is a status signal. Both fit
   the dense-bar mandate; GitHub's unified filter bar (source 4) and
   Grafana 12 (source 5) show consolidation is the shipping norm. The
   contract should cite §4.5's "fold its signal into the bar" sentence to
   justify, and explicitly note this is a *global* control (not a
   per-panel one) so it stays consistent with "globally relevant content
   above the tab bar" (§3).

2. **a11y is SAFE because `OpsStatusBar` is `<section>`, not
   `role="toolbar"` — and it must STAY that way.** (sources 1, 2)
   - APG Toolbar pattern: "Avoid including controls whose operation
     requires the pair of arrow keys used for toolbar navigation." A
     radiogroup needs Left/Right (and Up/Down) — nesting it inside a
     `role="toolbar"` would force the toolbar to steal Left/Right and the
     radiogroup to fall back to Up/Down only, changing its keyboard model.
   - Because the bar is a generic `<section aria-label="...">`, NOT a
     toolbar, the nested `radiogroup` keeps its full native model
     (all four arrows + Home/End + selection-follows-focus, per source 1).
   - **Design rule for the contract:** keep `OpsStatusBar` as `<section>`.
     Do NOT add `role="toolbar"`. The radiogroup's existing keyboard code
     in `MarketFilter.tsx:44-58` moves in verbatim and remains spec-correct.
   - Focus order is DOM order: placing the Market segment first means
     Tab order = Market radios -> Gate info -> Kill buttons -> ... which is
     a sane "scope before status" reading order. Acceptable per APG (no
     toolbar-level focus management needed since it's not a toolbar).

3. **The open/closed dot folds in with zero hydration risk if the existing
   two-pass pattern is preserved.** (source 3) React's documented fix for
   `new Date()`-dependent UI is exactly `MarketSessionStrip.tsx:24-29`'s
   `useState<Date|null>(null)` + `useEffect(setNow(new Date()))`. Whether
   the open/closed computation lives in a retired strip or inside
   `MarketFilter`/a new `MarketSegment`, the SAME mount guard must wrap it:
   render the "unknown" dot color on the server/first paint, then color
   emerald/slate after mount. Folding into a child component does NOT
   change this — it just moves where the `useState<Date|null>` lives.
   - Recommended: compute the session map inside the segment and pass it
     down, OR add an optional `sessionOpen?: Record<string,boolean>` prop
     to `MarketFilter`. EITHER way the time read must be mount-guarded.
   - `LastSegment` already uses `suppressHydrationWarning`
     (`OpsStatusBar.tsx:341`) for `formatRelativeTime` — that escape hatch
     is the alternative, but for a boolean open/closed the two-pass guard
     is cleaner and matches the existing strip.

4. **Dot colors must stay in a static literal map (JIT).** (source 6)
   `MARKET_DOT_CLASS` (`format.ts:100`) is the correct, JIT-safe pattern
   for the per-market pill dot. For the open/closed state, the existing
   strip uses LITERAL ternary classes (`bg-emerald-400` / `bg-slate-600`,
   `MarketSessionStrip.tsx:46`) — also JIT-safe. Reuse literals; never
   build `bg-${...}` strings. (No new safelist entry needed.)

---

## Internal code inventory
| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/src/components/OpsStatusBar.tsx` | 1-374 | Dense status `<section>`; segments Gate/Kill/Cycle/Last/Next; `Divider`+`SegmentLabel` helpers | In scope — add conditional Market segment |
| `frontend/src/components/paper-trading/MarketFilter.tsx` | 1-99 | WAI-ARIA radiogroup of pills; roving tabindex + arrows | In scope — moves into bar; optional session-dot prop |
| `frontend/src/components/paper-trading/MarketSessionStrip.tsx` | 1-56 | Open/closed indicator; mount-guarded `useState<Date|null>` | In scope — retire (fold into pills) or re-home inside bar |
| `frontend/src/app/paper-trading/layout.tsx` | 137,164-177,323-324,478-505 | Owns `activeMarket`/`availableMarkets`; renders bar + filter row + hero | In scope — delete row 483-490; pass market props into bar at 478 |
| `frontend/src/app/page.tsx` | 8,360 | Homepage; renders `OpsStatusBar nextRunAt=...` ONLY | MUST stay byte-identical (no market props) |
| `frontend/src/lib/format.ts` | 100,77,51,128,212 | `MARKET_DOT_CLASS`,`MARKET_ORDER`,`MARKET_EXCHANGE`,`resolveMarket`,`isMarketOpen` | Read-only — reuse exports |

### Verification of goal-prompt claims (file:line) — all CONFIRMED
- **`OpsStatusBar` is `<section aria-label="Paper-trading operator
  status">`** — confirmed `OpsStatusBar.tsx:116-119`. Container classes:
  `mb-6 flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border
  border-navy-700 bg-navy-800/60 px-4 py-3` (so the wrap behaviour the
  goal relies on already exists). **NOT `role="toolbar"`** (grep: zero
  `role="toolbar"` in the codebase) — this is the load-bearing a11y fact.
- **Segments Gate | Kill | Cycle | Last | Next** — confirmed
  `OpsStatusBar.tsx:120-132` (`GateSegment`,`KillSegment`,`CycleSegment`,
  `LastSegment`,`NextSegment`, each separated by `<Divider/>`).
- **`SegmentLabel` / `Divider` helpers** — confirmed
  `OpsStatusBar.tsx:139-141` (`Divider`, `hidden ... sm:block`),
  `:143-149` (`SegmentLabel`, `text-[10px] uppercase tracking-wider
  text-slate-500`). Reuse both for the Market segment.
- **`LastSegment`'s `ml-auto`** — confirmed `OpsStatusBar.tsx:339`
  (`<div className="ml-auto flex items-center gap-2">`). NOTE: this is on
  *Last*, not *Next* as a casual read of §4.5 might suggest. Implication:
  inserting a Market segment as the LEFT-MOST child does not disturb the
  `ml-auto` right-push (Last+Next still right-align). Safe.
- **`layout.tsx:478`** `<OpsStatusBar nextRunAt={status?.next_run} />` —
  confirmed (no market props today).
- **`layout.tsx:483-490`** the standalone filter row
  `<div className="mb-4 flex flex-wrap items-center justify-between gap-3">`
  wrapping `<MarketFilter .../>` (`:484-488`) + `<MarketSessionStrip
  markets={availableMarkets} />` (`:489`) — confirmed verbatim. This is the
  row to delete.
- **`layout.tsx:499-504`** the `activeMarket !== "ALL"` filtered note —
  confirmed; it explains hero/table scope, stays put.
- **Market state** — `activeMarket`/`setActiveMarket`
  (`layout.tsx:137`), `availableMarkets` (`layout.tsx:164-169`, built from
  `["US","EU","KR"]` + held markets, filtered through `MARKET_ORDER`),
  auto-fallback-to-ALL effect (`layout.tsx:173-177`). Shared via
  `PaperTradingDataContext` (used at `:477` `ctxValue`). Confirmed.
- **`page.tsx:360`** homepage `<OpsStatusBar nextRunAt={ptStatus?.next_run
  ?? null} />` — confirmed it passes ONLY `nextRunAt`. So a Market segment
  gated on `markets && activeMarket && onMarketChange` ALL being present
  renders NOTHING extra on the homepage. The conditional-prop design
  satisfies acceptance criterion 4 cleanly.

### Consumers of OpsStatusBar / MarketFilter / MarketSessionStrip
- **`OpsStatusBar`** — TWO render sites: `page.tsx:360` (homepage, only
  `nextRunAt`) and `layout.tsx:478` (cockpit). Confirms the shared-component
  constraint. (Also referenced in non-render comments: `Button.tsx:3`,
  `paper-trading-utils.ts:30`, `design-tokens.ts:38` — comments only, no
  behavioural coupling.)
- **`MarketFilter`** — ONE render site: `layout.tsx:484`. Safe to change
  its props (add optional `sessionOpen`/dot map) — no other caller.
- **`MarketSessionStrip`** — ONE render site: `layout.tsx:489`. Safe to
  retire entirely; nothing else imports it.
- **`isMarketOpen`** — ONE consumer: `MarketSessionStrip.tsx:37`. If the
  strip is retired, the new consumer (MarketFilter or MarketSegment) keeps
  `isMarketOpen` live; if no consumer remains it becomes dead code (flag,
  but it's a tiny pure export and likely still used by the folded logic).

### format.ts exports (signatures)
- `isMarketOpen(market: string, now: Date = new Date()): boolean`
  (`format.ts:212`) — weekday + local-tz cash-session window; holiday-blind
  (UI hint only; backend gate is authoritative). Unknown market -> false.
- `MARKET_DOT_CLASS: Record<string,string>` (`format.ts:100`) — per-market
  Tailwind dot bg literal (US `bg-sky-400`, EU `bg-amber-400`, KR
  `bg-violet-400`, ...). STATIC literal map (JIT-safe).
- `MARKET_EXCHANGE: Record<string,string>` (`format.ts:51`) — friendly
  exchange name (US "NYSE/Nasdaq", EU "XETRA", KR "KRX", ...); used as the
  pill `title` tooltip.
- `MARKET_ORDER: string[]` (`format.ts:77`) — canonical display order
  `["US","EU","NO","SE","DK","FI","IS","CA","KR"]`.
- `resolveMarket(opts:{market?,ticker?}): string` (`format.ts:128`) —
  explicit market wins else derive from ticker suffix.
- (bonus) `MARKET_BENCHMARK_LABEL` (`format.ts:38`) — the `vs SPY/DAX/KOSPI`
  label the Playwright click-through asserts (acceptance criterion 3).

### Test coverage
- **No test references `OpsStatusBar`, `MarketFilter`, `MarketSessionStrip`,
  "Filter by market", or "operator status"** (grep over all `*.test.tsx`).
- **`layout-tablist.test.tsx` is misnamed** — it is a `DataTable`
  meta-support smoke test (align/className/onRowClick); it does NOT assert
  anything about the market filter row or the status bar. My change cannot
  break it. The file's own comment (line 7-8) says full layout/tablist a11y
  is exercised via Playwright in a separate cycle — i.e. the visual
  click-through IS the coverage (matches acceptance criterion 3 + frontend.md
  rule 5).
- 22 test files exist total; none touch the in-scope components. `npm run
  build` + the existing suite passing (criterion 6) is therefore a
  regression guard, not a direct assertion of the move.

---

## Recommended approach + risks

### Recommended approach (aligns with the goal's recommended design)
1. **Add an optional `MarketSegment` to `OpsStatusBar`**, gated on
   `markets && activeMarket && onMarketChange` all being present. Render it
   as the **left-most** child of the `<section>`, followed by a `<Divider/>`
   before `GateSegment`. Reuse `SegmentLabel` ("Market") + the existing
   `MarketFilter` radiogroup. Props:
   `markets?: string[]; activeMarket?: string; onMarketChange?: (m)=>void;`
   (all optional, additive — `nextRunAt` unchanged).
2. **Keep `OpsStatusBar` as `<section>` — do NOT promote to
   `role="toolbar"`.** This is the single most important a11y decision
   (source 2): a toolbar would hijack the radiogroup's arrow keys. As a
   plain section, `MarketFilter`'s native roving-tabindex + arrow model
   (`MarketFilter.tsx:44-58`) survives verbatim.
3. **Fold the session signal into the pills.** Color each non-`All` pill's
   dot emerald when `isMarketOpen(market, now)` is true, slate when closed,
   using a mount-guarded `useState<Date|null>` (lift the
   `MarketSessionStrip.tsx:24-29` guard into `MarketFilter` or the
   `MarketSegment`). Render the neutral `MARKET_DOT_CLASS` color until
   mount to avoid the hydration mismatch (source 3). Keep the exchange name
   in each pill's `title` (`MarketFilter.tsx:80`). This retires
   `MarketSessionStrip` and removes BOTH the filter and the strip from
   `layout.tsx:483-490` (net −1 full row = the density win, criterion 2).
   - **Fallback** (if pills get cramped): keep a compact `MarketSessionStrip`
     INSIDE the same bar segment. Still −1 row; criterion 1 + 5 satisfied.
4. **In `layout.tsx`:** delete the `<div className="mb-4 ...">` row
   (`483-490`) and pass `markets={availableMarkets}
   activeMarket={activeMarket} onMarketChange={setActiveMarket}` into the
   cockpit `<OpsStatusBar>` at `:478`. Leave `page.tsx:360` untouched.
5. **Keep the filtered note** (`layout.tsx:499-504`) where it is.
6. **Palette + JIT:** navy/slate only (`bg-navy-800/60`, `text-slate-*`),
   never zinc (frontend.md rule 1); dot classes from the static
   `MARKET_DOT_CLASS` / literal emerald-slate ternary, never `bg-${...}`
   (frontend.md rule 3, source 6). No emoji anywhere (colored dots +
   `SegmentLabel` text only) — confirmed zero emoji in the 3 files today.

### Risks / watch-items
- **R1 (homepage regression — highest):** the Market segment MUST be
  conditional. If it renders on `page.tsx` the step FAILs (criterion 4).
  Mitigation: gate on all three market props; homepage passes none. Verify
  by reading the rendered homepage bar (5 segments, no Market) in Q/A.
- **R2 (hydration warning):** if the open/closed color is computed from
  `new Date()` WITHOUT the mount guard, React 19 will warn and criterion 5
  fails ("no hydration warning in the console"). Mitigation: reuse the
  `useState<Date|null>(null)` two-pass guard verbatim (source 3).
- **R3 (a11y keyboard regression):** if someone "tidies" the bar by adding
  `role="toolbar"`, the radiogroup arrows break (source 2). Mitigation:
  leave the `<section>` role as-is; note this explicitly in the contract.
- **R4 (wrap / density):** the Market segment adds width; on ≥1280px it
  must not force a *permanent* 2nd line beyond today's existing `Next`
  wrap (goal guardrail). The existing `flex flex-wrap gap-x-6 gap-y-3`
  handles graceful wrap; visual click-through at 1440px is the check
  (criterion 3 + frontend.md rule 5).
- **R5 (dead export):** if pills fully absorb sessions and
  `MarketSessionStrip` is deleted, double-check `isMarketOpen` still has a
  consumer (it will, inside the folded logic). Don't leave an unused import.
- **R6 (visual-only correctness):** unit tests + grep cannot see the moved
  control or the dot colors (frontend.md rule 5; no unit test covers these
  components). The Playwright click-through (EU -> `vs DAX`, POSITIONS
  change, filtered note, reset to All, restore auth gate) is MANDATORY and
  is the real acceptance evidence (criterion 3).

### Mapping external findings -> internal anchors
| External finding | pyfinagent anchor / action |
|---|---|
| §4.5 "fold its signal into the bar"; GitHub unified filter bar; Grafana conditional render | Justify folding filter+session into `OpsStatusBar.tsx:116`; delete `layout.tsx:483-490` |
| APG: radiogroup keeps native arrows only outside a `toolbar` | Keep `OpsStatusBar.tsx:116` as `<section>`; `MarketFilter.tsx:44-58` arrow code unchanged |
| React two-pass guard for `new Date()` | Lift `MarketSessionStrip.tsx:24-29` `useState<Date\|null>` into `MarketFilter`/`MarketSegment` |
| Tailwind JIT static literal map | Reuse `MARKET_DOT_CLASS` (`format.ts:100`) + literal emerald/slate ternary |
| Conditional segment on shared component | Gate Market segment on 3 optional props; `page.tsx:360` passes none -> unchanged |
| Visual verification mandatory | Playwright click-through per `docs/runbooks/browser-mcp.md` (criterion 3) |

---

## Research Gate Checklist

Hard blockers — `gate_passed` false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 read in full)
- [x] 10+ unique URLs total incl. snippet-only (20 collected)
- [x] Recency scan (last 2 years) performed + reported (section above)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (OpsStatusBar.tsx:116/120-132/139-149/339/341,
      MarketFilter.tsx:44-58/62/80, MarketSessionStrip.tsx:24-29/37/46, layout.tsx:137/164-177/478/483-490/499-504,
      page.tsx:360, format.ts:38/51/77/100/128/212)

Soft checks:
- [x] Internal exploration covered every relevant module (3 components + layout + page + format + tests)
- [x] Contradictions/consensus noted (no source contradicts; §4.5 tension resolved)
- [x] Claims cited per-claim with URL + file:line

---

```json
{"tier":"simple","external_sources_read_in_full":6,"snippet_only_sources":14,"urls_collected":20,"recency_scan_performed":true,"internal_files_inspected":7,"report_md":"handoff/current/research_brief.md","gate_passed":true}
```
