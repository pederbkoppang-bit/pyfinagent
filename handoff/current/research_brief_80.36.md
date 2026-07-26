# Research Brief — phase-80.36: Risk Monitor fabricates SAFE/OK with zero data

Tier: T2 (Opus 5, effort high). 2026-07-26. Status: COMPLETE — `gate_passed: true`.

Caller question: with the backend unreachable, several widgets on
`/paper-trading/positions` assert facts they cannot know (`SAFE`, `OK`, `0% / -15%`,
`+0,00 %` in positive-green, `Positions 0` when 2 are held). Need (A) UX/HCI guidance
on unknown-vs-zero-vs-nominal, (B) prior art on stale/unknown rendering in dashboards,
(C) a React/TS tri-state that makes the bad state unrepresentable, (D) internal
inventory + per-surface minimal fix, (E) the highest-risk way the fix changes the
healthy path.

---

## Search queries run (3-variant discipline)

| # | Query | Variant |
|---|-------|---------|
| 1 | `dashboard UI distinguishing "no data" from zero null vs zero rendering 2026` | current-year (2026) |
| 2 | `Nielsen Norman Group dashboard status indicator unknown state fail loud safety` | YEAR-LESS canonical |
| 3 | `Grafana alert rule "No Data and Error handling" configure no data state default` | YEAR-LESS canonical |
| 4 | `FAA AC 25-11B electronic flight displays "misleading information" hazardous failure condition vs loss of function` | YEAR-LESS canonical |
| 5 | `"no data" state design system 2025 status badge unknown vs healthy observability dashboard` | last-2-year (2025) |
| 6 | `React TypeScript discriminated union "make illegal states unrepresentable" component props 2026` | current-year (2026) |
| 7 | `2025 dashboard "unknown state" safety indicator must not default to healthy null coalescing bug` | last-2-year (2025) |

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://www.faa.gov/documentlibrary/media/advisory_circular/ac_25-11b.pdf | 2026-07-26 | regulatory standard (tier 2) | `curl` + `pdfplumber`, 135pp / 279,831 chars | Tables 4-1/4-2: "Display of misleading attitude information on one primary display" = **Hazardous**; "Loss of all primary attitude displays" = "Major – Hazardous". Misleading ≥ loss, always. |
| 2 | https://grafana.com/docs/grafana/latest/alerting/fundamentals/alert-rule-evaluation/nodata-and-error-states/ | 2026-07-26 | official doc | WebFetch | No Data and Error are FIRST-CLASS states, each its own DEFAULT; mapping to Normal is an explicit opt-out that "Ignores missing data". |
| 3 | https://grafana.com/docs/grafana/latest/alerting/guides/missing-data/ | 2026-07-26 | official doc | WebFetch | "No Data: the query runs successfully but returns no data at all" vs "Missing Series"; turning nothing into 0 requires an explicit `OR on() vector(0)` rewrite. |
| 4 | https://carbondesignsystem.com/patterns/status-indicator-pattern/ | 2026-07-26 | official design system | `curl` + tag-strip (WebFetch truncated) | `Unknown` is a NAMED status — "Gray 60, Gray 50 — Indicates that the status of an object is unknown" — distinct from `Normal`, which "implies no issues are present". "for WCAG compliance, at least three of these elements must be present". |
| 5 | https://prometheus.io/docs/prometheus/latest/querying/basics/#staleness | 2026-07-26 | official doc | WebFetch | Stale markers; "If a query is evaluated at a sampling timestamp after a time series is marked as stale, then no value is returned"; carry-forward bounded at 5 min. |
| 6 | https://www.typescriptlang.org/docs/handbook/2/narrowing.html | 2026-07-26 | official doc | WebFetch | Optional-property modelling fails even under `strictNullChecks`; discriminated union + `const _exhaustiveCheck: never` is the prescribed fix. |
| 7 | https://www.nngroup.com/articles/visibility-system-status/ | 2026-07-26 | authoritative (NN/g) | WebFetch | "A lack of information often equates to a lack of control"; predictability creates trust. |
| 8 | https://oneuptime.com/blog/post/2026-01-15-typescript-discriminated-unions-react-props/view | 2026-07-26 | industry blog (2026-01-15) | WebFetch | RECENCY ANCHOR — current-year restatement of the same union + `never` switch pattern for React props. |
| 9 | https://sre.google/workbook/alerting-on-slos/ | 2026-07-26 | authoritative (Google SRE) | WebFetch | Precision/recall definitions obtained verbatim. **Honest note: this chapter does NOT address missing telemetry or monitoring-pipeline failure — it did not support the claim it was fetched for.** |
| 10 | https://www.dundas.com/support/learning/documentation/data-visualizations/how-to/handling-null-value-in-a-state-indicator-column | 2026-07-26 | vendor doc | WebFetch | **Fetched, zero relevant content** — the page title promises null-in-state-indicator guidance but the body covers only state styles. Recorded for auditability; contributes no claim. |
| — | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-07-26 | authoritative | WebFetch | Also fetched; likewise silent on monitoring-pipeline self-failure. No claim rests on it. |

## Identified but snippet-only

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://community.dynatrace.com/t5/Dashboarding/Replace-quot-no-data-quot-with-quot-0-quot-in-a-dashboard/m-p/181861 | community (tier 5) | Corroborates the semantic distinction only; lowest weight |
| https://community.esri.com/t5/arcgis-dashboards-questions/indicator-show-0-and-icon-instead-of-quot-no-data/td-p/1310195 | community (tier 5) | Same |
| https://gitlab.com/gitlab-org/gitlab-design/-/issues/462 | industry issue | "pattern to denote null versus zero in line graphs" — corroborating |
| https://github.com/google/site-kit-wp/issues/4226 | industry issue | Corroborating |
| https://grafana.com/docs/grafana/latest/alerting/fundamentals/alert-rule-evaluation/state-and-health/ | official doc | Superseded by #2/#3 for this question |
| https://github.com/grafana/grafana/issues/75594 | industry issue | Bug report, not guidance |
| https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_23.1311-1B.pdf | regulatory | Part 23 analogue; AC 25-11B (#1) is the governing one |
| https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_25-11A.pdf | regulatory | **Cancelled** — superseded by AC 25-11B |
| https://en.wikipedia.org/wiki/AC_25.1309-1 | community | Tertiary summary of the primary read at #1 |
| https://v10.carbondesignsystem.com/patterns/status-indicator-pattern/ | official design system | v10 mirror of #4 (this URL is what `curl` actually resolved; same content) |
| https://designsystem.ny.gov/components/badge/ , https://design-system.agriculture.gov.au/components/status-badge , https://nordhealth.design/components/badge/ , https://horizon.servicenow.com/workspace/components/now-badge | design systems | Badge mechanics; none add an unknown-state rule beyond Carbon |
| https://www.developerway.com/posts/advanced-typescript-for-react-developers-discriminated-unions , https://www.totaltypescript.com/workshops/advanced-react-with-typescript/advanced-props/type-checking-react-props-with-discriminated-unions/solution , https://stevekinney.com/courses/react-typescript/typescript-discriminated-unions , https://dev.to/acsreedharreddy/why-make-illegal-states-impossible-to-represent-1dkn | blog / course | All restate the handbook pattern read at #6 |
| https://www.nngroup.com/articles/dashboards/ (via X/Twitter pointer), https://uxdesign.cc/loading-progress-indicators-ui-components-series-f4b1fc35339a | authoritative / blog | Loading-indicator scope, not unknown-status scope |
| https://dart.dev/null-safety , https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/null-safety/null-operators , https://github.com/php/php-src/issues/8661 | official docs | Query-7 drift into language-level null semantics; off-topic |

~45 unique URLs surfaced across the 7 queries; the table above lists the ones that were
actually evaluated rather than every hit.

## Recency scan (2024-2026)

**Performed.** Two explicitly year-scoped passes (query 5 → 2025, query 7 → 2025) and two
current-year passes (queries 1, 6 → 2026), run alongside three deliberately year-less
canonical queries (2, 3, 4).

**Result: no 2024-2026 finding supersedes the canonical sources; one confirms them.**

1. **Confirming (read in full).** oneuptime, **2026-01-15**, "How to Use TypeScript
   Discriminated Unions for React Component Props" — restates the exact pattern the TS
   handbook prescribes ("If you add a new type and forget to handle it, TypeScript will
   error here"), including the loading/error/success `switch`. The §C recommendation is
   therefore current practice as of 2026, not a stale idiom.
2. **No newer standard displaces the safety argument.** The 2025-scoped pass for
   "unknown state must not default to healthy" surfaced nothing on topic — results
   drifted to language-level null-coalescing semantics (Dart, C#, PHP). Absence of
   newer guidance here is itself the finding: **AC 25-11B (dated 10/07/14) remains the
   FAA's ACTIVE advisory circular** — its predecessor AC 25-11A is listed as *Cancelled*
   in the FAA library, and no post-2024 replacement appeared.
3. **No design-system change.** The 2025-scoped design-system pass returned current
   badge/status documentation (NY State, DAFF, Nord, ServiceNow, Carbon). None
   introduces a rule beyond Carbon's `Unknown`-is-its-own-status; several omit an
   unknown state entirely, which strengthens rather than weakens the recommendation to
   copy Carbon.
4. **Grafana docs read are the `latest` channel** (fetched 2026-07-26), so the No Data /
   Error defaults cited are current, not historical.

## Internal code inventory

### The fabricating component: `RiskMonitorCard`

`frontend/src/components/paper-trading/cockpit-helpers.tsx:298-409`.
Single consumer: `frontend/src/app/paper-trading/positions/page.tsx:151`.

ROOT CAUSE, line 309:
```ts
const maxDd = perf?.max_drawdown_pct ?? 0;
```
`perf: PaperPerformance | null`. When the backend is unreachable `perf` is `null`,
`maxDd` becomes the literal `0`, and `0 > -10` is TRUE — so every downstream
threshold reads "best case". The `?? 0` converts *absence* into the *most
reassuring possible observation*.

(full per-row table below, pending)

### The correct siblings (in-repo convention)

| Component | file:line | Guard condition | Exact string |
|---|---|---|---|
| `SectorBarList` | `frontend/src/components/SectorBarList.tsx:82-87` | `if (items.length === 0)` | `"No positions yet."` (default via `emptyState ?? `; positions page passes the same literal at `positions/page.tsx:161`) |
| `PortfolioAllocationDonut` | `frontend/src/components/PortfolioAllocationDonut.tsx:218-222` | `if (data.length === 0 \|\| totalValue <= 0)` | `"No allocation data yet."` |
| `MultiCurrencyNavBreakdown` | `frontend/src/components/MultiCurrencyNavBreakdown.tsx:70-74` | `if (rows.length === 0)` | `"No holdings yet."` |

Also already-correct INSIDE the offending component:
- `Max position` row, `cockpit-helpers.tsx:316` + `:362` —
  `const maxPos = concentrations.length > 0 ? Math.max(...concentrations) : null;`
  then `{maxPos != null ? \`${maxPos.toFixed(1)}%\` : "—"}`. This is the honest
  pattern, sitting three lines above the dishonest one.
- `PnlBadge` (`:45-47`) and `Dollar` (`:73-81`) and `SharpeValue` (`:165-168`) all
  do `if (value == null) return <span className="text-slate-500">—</span>;`.
  **The em-dash convention already exists and is already correct** — the defect is
  that the risk rows never route through it.

### Test coverage

`grep -rn "SAFE\|WARNING\|DANGER" --include='*.test.tsx' --include='*.test.ts' frontend/src`
returns exactly ONE hit and it is unrelated
(`components/cron/density-helpers.test.ts:37-38`, a cron log-level parser).
**There is NO test anywhere asserting the Risk Monitor's `SAFE`/`OK` strings, and no
test file for `cockpit-helpers.tsx` at all.** Consequence for the step: a fix cannot
"silently weaken coverage" by editing an existing assertion (there is none) — but it
ALSO means the healthy path is currently unguarded, so criterion 5 (healthy path
byte-for-byte unchanged) has no existing regression net. New tests must pin BOTH the
unknown render and the healthy render.


---

## A. The "unknown vs zero vs nominal" display problem — ranked findings

**A1 (strongest, safety-critical). Misleading information is classified AS SEVERE AS
OR MORE SEVERE THAN loss of the display.** FAA AC 25-11B *Electronic Flight Displays*,
Table 4-1 (Attitude), read in full via pdfplumber (135pp):

| Failure Condition | Hazard Classification |
|---|---|
| "Loss of all primary attitude displays" | "Major – Hazardous¹" |
| "Display of misleading attitude information on one primary display" | **"Hazardous"** |
| "Loss of all attitude displays, including standby display" | "Catastrophic" |
| "Display of misleading attitude information on both primary displays" | **"Catastrophic"** |

Table 4-2 (Airspeed) repeats the identical structure. The regulator's judgment is
explicit: *showing a wrong reading is never a lesser failure than showing nothing* —
at the both-displays level the two are equally Catastrophic, and at the single-display
level "misleading" (Hazardous) sits at the TOP of the range assigned to "loss"
(Major–Hazardous). AC 25-11B §4.7.1 further requires the safety assessment to consider
"failure modes, **failure detection and annunciation**, redundancy management".
Source: https://www.faa.gov/documentlibrary/media/advisory_circular/ac_25-11b.pdf
(accessed 2026-07-26).

**Application:** a Risk Monitor that prints `SAFE` in green while it can observe
nothing is the software equivalent of a primary attitude display showing wings-level
after its sensor died. Blanking it is strictly the safer failure mode.

**A2. "No Data" is a first-class state in the reference implementation, and it is the
DEFAULT.** Grafana Alerting defines two distinct non-normal states:
- No Data — "occurs when the alert rule query runs successfully but returns no data points at all."
- Error — "triggered when the alert rule fails to evaluate its query or queries successfully."

Both default to their own dedicated state (`DatasourceNoData` / `DatasourceError`),
NOT to Normal. Mapping no-data to Normal is an explicit opt-out the operator must
choose: "**Set Normal state** — Transitions immediately to Normal", described as
"Ignores missing data". Grafana warns that "in situations where strict monitoring is
critical, relying solely on the 'Keep Last State' option may not be appropriate."
Sources: https://grafana.com/docs/grafana/latest/alerting/fundamentals/alert-rule-evaluation/nodata-and-error-states/
and https://grafana.com/docs/grafana/latest/alerting/guides/missing-data/ (accessed 2026-07-26).

**Application:** pyfinagent's Risk Monitor currently hard-codes the opt-out
(`?? 0` == "Set Normal state") for a *kill switch*, i.e. exactly the "strict monitoring
is critical" case the vendor warns about — and does it silently, with no operator choice.

**A3. Absence must propagate as absence, not as a value.** Prometheus: "If a target
scrape or rule evaluation no longer returns a sample for a time series that was
previously present, this time series will be marked as stale," and "If a query is
evaluated at a sampling timestamp after a time series is marked as stale, then no
value is returned for that time series." The engine emits a *stale marker* rather
than carrying the last value forward indefinitely; the carry-forward window is bounded
(5 min default `--query.lookback-delta`). Note the deliberate contrast Grafana draws:
turning nothing into a number requires an EXPLICIT rewrite —
"`your_metric_query OR on() vector(0)`" — you have to ask for it.
Source: https://prometheus.io/docs/prometheus/latest/querying/basics/#staleness (accessed 2026-07-26).

**Application:** `perf?.max_drawdown_pct ?? 0` is an implicit, unbounded, undisclosed
`OR on() vector(0)` applied to a safety threshold.

**A4. HCI framing (why this is a usability defect and not only a safety one).**
NN/g heuristic #1: systems "should always keep users informed about what is going on,
through appropriate feedback within reasonable time"; "**A lack of information often
equates to a lack of control**"; and "When we understand the system's state, we feel
in control — we can rely on the system to act as expected in all circumstances. The
predictability of the interaction creates trust." A widget that reports `SAFE` when
its input is absent does not merely withhold status — it actively *misinforms*, which
is the trust-destroying case, not the trust-neutral one.
Source: https://www.nngroup.com/articles/visibility-system-status/ (accessed 2026-07-26).

## B. Prior art — how mature tools render an unreachable source

| Strategy | Who does it | Right for | Wrong for |
|---|---|---|---|
| **Dedicated no-data state** (own label + own colour, never the OK colour) | Grafana (`No Data`, default); Prometheus (stale marker → series simply absent) | **Safety rows** (kill switch). The state is nameable, alertable, and cannot be confused with Normal. | — |
| **Error state distinct from no-data** | Grafana (`DatasourceError`, default) | Distinguishing "backend answered, empty" from "backend unreachable" | Over-engineering for an informational row |
| **Keep last value + explicit age** | Prometheus 5-min bounded lookback; Grafana `Keep Last State` | Informational rows where a slightly stale number still informs | Safety rows — Grafana explicitly warns it "may not be appropriate" for strict monitoring |
| **Coerce to zero** (`OR on() vector(0)`, `ZN()` in Tableau) | Opt-in only, everywhere | Counters where zero is genuinely the measured value | **Anything thresholded.** This is precisely the current bug. |
| **Em-dash / blank** | pyfinagent's own `Dollar`/`PnlBadge`/`SharpeValue`, and its `Max position` row | Numeric readouts | Verdict badges — a blank badge is ambiguous; needs a word |

**Recommendation split (per the caller's ask):**
- **Safety row (Kill switch, Position size, Sector concentration, Drawdown bar):** dedicated
  UNKNOWN state — neutral slate treatment, explicit word (`UNKNOWN` / `NO DATA`), never the
  emerald token. Follows A1 + A2. Do **not** keep-last-state here.
- **Informational row (vs SPY, Positions):** em-dash via the component's existing null
  convention (`PnlBadge` already renders `—` for null — the bug is that the call site
  computes a non-null `0` before it ever reaches `PnlBadge`).

## C. React/TypeScript — making the bad state unrepresentable

The TS handbook's own framing of this exact anti-pattern (optional fields on one
interface) is that **`strictNullChecks` does not save you**: with
`interface Shape { kind: "circle" | "square"; radius?: number }`, "The type-checker
doesn't have any way to know whether or not `radius` or `sideLength` are present based
on the `kind` property." The prescribed fix is a discriminated union plus `never`-based
exhaustiveness:

```ts
default:
  const _exhaustiveCheck: never = shape;
  return _exhaustiveCheck;
```
Source: https://www.typescriptlang.org/docs/handbook/2/narrowing.html (accessed 2026-07-26).

**Ranked recommendation for 80.36:**

1. **(Recommended) Discriminated union at the verdict boundary.** Introduce a local
   type in `cockpit-helpers.tsx`:
   ```ts
   type RiskVerdict =
     | { state: "unknown" }
     | { state: "ok"; label: string }
     | { state: "warn"; label: string }
     | { state: "breach"; label: string };
   ```
   A single `RiskPill({ verdict }: { verdict: RiskVerdict })` renderer switches on
   `verdict.state` with a `never` default. Why this is structurally safe: the
   *derivation* functions must return `{state:"unknown"}` — they cannot return a
   label without also picking a non-unknown state, so "unknown silently rendered as
   nominal" stops being expressible. The `never` arm makes a future 5th state a
   compile error rather than a silent fallthrough.
2. **(Necessary, insufficient alone) Delete the `?? 0` defaults** and thread
   `number | null` to the threshold sites. This fixes today's bug but leaves the same
   trap for the next `??` author — it is a value fix, not a shape fix. Do this AS PART
   of (1), not instead of it.
3. **(Reject) `?? "—"` sprinkled at the JSX leaves.** The caller explicitly rejected
   this; it is also wrong on the merits — the coercion happens upstream at line 309,
   so a leaf-level `??` never sees a null to catch.

Note there is a stronger option — `NonNullable`-typed props with the null handled by
an early `return <UnknownCard/>` at the top of `RiskMonitorCard` — but it is WORSE
here: it would blank the honest `Max position` row too, and it cannot express
"drawdown unknown but positions known", which is a real partial-failure state given
`layout.tsx:196-202` (see D).

---

## D. INTERNAL — per-surface inventory and minimal fix

### D0. The ENABLING condition (not in the caller's table — this is why the card renders at all)

`frontend/src/app/paper-trading/layout.tsx:215`
```ts
const isInitialized = status?.status !== "not_initialized";
```
When the backend is unreachable, `getPaperTradingStatus()` (`layout.tsx:189`, the ONLY
member of the `Promise.all` without a `.catch`) rejects, the `catch` at `:204` sets
`error`, and `status` stays `null`. `null?.status` is `undefined`, and
`undefined !== "not_initialized"` is **`true`** — so `isInitialized` is TRUE with zero
knowledge, and the full cockpit (`layout.tsx:474-508`) renders instead of the
"No paper portfolio initialized" placeholder. This is the same absence-becomes-
affirmative shape as the `?? 0` bugs, one level up. It is the reason the error banner
and the fabricated widgets appear on screen simultaneously.

### D1. Per-surface table

Legend: **src** = what the value reads from; **cold** = backend never answered
(all state at initial values); **warm** = a previous fetch succeeded, then the
backend died (`Promise.all` rejects at `:188`, so `setStatus`/`setTrades`/`setSnapshots`
never run and **every previously-fetched value is silently retained with no age
disclosure** — Grafana "Keep Last State" without the label).

| # | Surface | file:line | Precise fabrication condition | Renders | Minimal fix |
|---|---|---|---|---|---|
| 1 | **Kill switch (-15%)** | `cockpit-helpers.tsx:309` → `:344-357` | `perf == null` ⇒ `maxDd = 0` ⇒ `0 > -10` true | `SAFE` on `bg-emerald-500/10 text-emerald-400` | Derive `maxDd: number \| null` (drop `?? 0`); when `null` emit `{state:"unknown"}` → neutral slate pill reading `UNKNOWN` |
| 2 | **Position size** | `:316-317` → `:365-377` | `maxPos == null` (no positions) ⇒ `concentrationHigh = false` ⇒ falls to the emerald `else` | `OK` in emerald | `concentrationHigh` must become `"unknown" \| boolean`: unknown when `positions.length === 0` **or** `portfolio == null` (see #6) |
| 3 | **Sector concentration** | `:333-337` → `:378-390` | `positions.length >= 3` is false ⇒ `sectorConcentrationHigh = false` ⇒ emerald `"OK"` | `OK` in emerald | unknown when `positions.length === 0`. **Caveat below (D2).** |
| 4 | **Drawdown label + bar** | `:391-405` | `perf?.max_drawdown_pct?.toFixed(1) ?? "0"` (note: `?? "0"`, not `?? "—"`); bar colour from `maxDd`; width `Math.abs(maxDd)` | `0% / -15%` + zero-width **emerald** bar | label → `—`; bar → neutral slate track with no fill, or hide the fill entirely |
| 5 | **KPI "vs SPY"** | `layout.tsx:495-501` → `cockpit-helpers.tsx:189` + `:217` | `bench = status?.portfolio.benchmark_return_pct ?? 0` **and** `vsValue = (pnlDisplay ?? 0) - bench` — two `?? 0` producing a non-null `0` | `+0,00 %` in emerald (`PnlBadge`: `isPositive = value >= 0`) | Compute `vsValue = null` when `pnlDisplay == null \|\| bench == null`. `PnlBadge` **already** renders `—` for null (`:47`) — no component change needed |
| 6 | **KPI "Positions"** | `cockpit-helpers.tsx:197` → `:242-244` | `status?.position_count ?? 0` | `0` (2 actually held) | `positionCount: number \| null`; render `—` when `status == null` |

### D2. Findings the caller's table does NOT list (all in the same component)

- **`navDenom` fabricates a $10,000 fund.** `cockpit-helpers.tsx:310`:
  `const navDenom = portfolio?.total_nav ?? 10000;`. In the *warm* scenario
  (`positions` retained, `portfolio` null) every concentration percentage is computed
  against a fictional denominator, so rows #2 AND the "honest" **Max position** row
  (`:362`) both emit confident numbers derived from a made-up NAV. **Max position is
  only honest against empty `positions`, not against a null `portfolio`.** Any fix that
  claims "Max position was already correct" is overstating it.
- **Sector concentration under-reports even when HEALTHY.** `:334` requires
  `positions.length >= 3`, so a genuinely 100%-single-sector 2-position book renders
  emerald `OK` with a live backend. That is a separate (real) defect; the 80.36 fix must
  not be described as "sector concentration is now honest".
- **`PaperVsBacktestCard` has the identical bug on a different route.**
  `cockpit-helpers.tsx:256-257`: `const sharpe = perf?.sharpe_ratio ?? 0; const maxDd =
  perf?.max_drawdown_pct ?? 0;` — at `:282` the Max-DD number correctly renders `—` but
  is coloured `text-emerald-400` because `0 > -15`. The *number* is honest and the
  *colour* lies. Rendered on `/paper-trading/reality-gap` (`reality-gap/page.tsx:48`).
  **Out of 80.36's scope** (different route, not in the criterion) → per the standing
  rule, queue as its own masterplan step rather than silently widening this one.

### D3. In-repo conventions the fix must reuse (do not invent new ones)

1. **`—` for absent numerics** — already the rule in this very file:
   `PnlBadge` `:45-47`, `Dollar` `:73-81`, `SharpeValue` `:165-168`, and the
   `Max position` row `:362`. All use `value == null → <span className="text-slate-500">—</span>`.
2. **`"unknown"` as an explicit union member** — `FreshnessBand = "green" | "amber" |
   "red" | "unknown"` is declared THREE times
   (`lib/paper-trading-utils.ts:33`, `lib/types.ts:1243`, `lib/live-portfolio-context.tsx:51`)
   and `deriveFreshness` returns `{ band: "unknown", ageSec: null }` when there is
   nothing to measure (`live-portfolio-context.tsx:94`). The codebase already has the
   idiom this fix needs.
3. **A rendered unknown badge already exists** — `components/states/StaleDataState.tsx`
   has an `isUnknown = ageSeconds === null` branch emitting the label `"no data"`, a
   NEUTRAL palette, `role="status"`, and `aria-label="No data freshness available"`.
   (Note: it uses `zinc` tokens, which violates `.claude/rules/frontend.md` §1
   "Use the project's navy + slate palette, NOT Tailwind's default zinc palette" —
   copy the *shape*, not the tokens.)
4. **Sibling empty states on the same page**: `"No positions yet."`
   (`SectorBarList.tsx:82-87`), `"No allocation data yet."`
   (`PortfolioAllocationDonut.tsx:218-222`), `"No holdings yet."`
   (`MultiCurrencyNavBreakdown.tsx:70-74`).

### D4. The binding in-repo rule

`.claude/rules/frontend.md`, "Conventions" — verbatim:
> **Error states**: Never use `.catch(() => null)` on ALL calls in a group. If all
> calls in a `Promise.all` fail, surface an error banner (rose-900 border,
> rose-950/50 bg) with retry button.

and

> **Color coding**: green=bullish, red=bearish, amber=neutral, **gray=error/unavailable**

The second line is decisive and already binding: **gray is the project's designated
token for "unavailable"**, and the Risk Monitor is using green for it. `frontend-layout.md`
§8 adds: "Never show blank space. Every conditional render (`{data && ...}`) must have a
corresponding empty state for when data is null/empty." Carbon corroborates the colour
choice — its `Unknown` status is "Gray 60, Gray 50 — Indicates that the status of an
object is unknown", filed under *Low attention*, and distinct from `Normal`, which
"Indicates stability ... **implies no issues are present**".

### D5. Test coverage — what exists and what a fix could quietly break

- **No test asserts `SAFE` / `OK` / `WARNING` / `DANGER` anywhere.** The only
  `SAFE|WARNING|DANGER` hit under `src/**/*.test.*` is
  `components/cron/density-helpers.test.ts:37-38` (a cron log-level parser, unrelated).
- **No test file exists for `cockpit-helpers.tsx`.**
- The only paper-trading layout test is
  `src/app/paper-trading/layout.test.tsx:52-53` —
  `describe("PaperTradingLayout not_initialized payload (phase-75.12 fe-ts-01)")` /
  `it("renders the placeholder without throwing")`. It feeds a REAL `not_initialized`
  payload, so it does not pin the `status == null` branch at `:215` and would not catch
  a regression there.
- **Implication:** there is no existing assertion for a fix to weaken — but equally
  there is no net under the healthy path. New tests must pin BOTH directions, and per
  the standing mutation-test rule the unknown-state guard must be shown to FAIL when
  the `?? 0` is reintroduced.

---

## E. Regression risk — how this fix could break the healthy path

Ranked, most dangerous first. Items 1 and 2 are **hard stops**.

**E1 (HARD STOP — turns a genuine SAFE into UNKNOWN). Discriminating on the VALUE
instead of on PRESENCE.** `max_drawdown_pct === 0` is a legitimate, common healthy
reading (a fund that has not yet drawn down; also true immediately after a peak reset).
Any implementation shaped as `if (!maxDd)`, `if (maxDd === 0)`, `if (!perf?.max_drawdown_pct)`,
or `maxDd || null` treats real zero as absent and flips a healthy `SAFE` to `UNKNOWN`.
The discriminator MUST be `perf == null` / `perf.max_drawdown_pct == null` — a
presence test, never a truthiness or equality test. Same trap on
`benchmark_return_pct` (`:189`) and `position_count` (`:197`), where `0` is likewise a
real value. Note `Math.abs(null) === 0` in JS, so a half-applied fix leaves the
drawdown bar's width computation silently working on a null.

**E2 (HARD STOP — makes a genuine breach LESS visible). Hoisting the unknown check to
the CARD level.** An early `if (!perf) return <UnknownCard/>` at the top of
`RiskMonitorCard` is the tempting one-line fix and it is wrong: **`Position size` and
`Sector concentration` do not read `perf` at all** — they read `positions` + `portfolio`
+ `tickerMeta`. In the real partial-failure state (`getPaperPerformance()` has its own
`.catch(() => null)` at `layout.tsx:193`, so `perf` alone can be null while portfolio
data is fine), a card-level bail would **suppress a live `HIGH (>20%)` concentration
warning**. Unknown must be derived per row, from that row's own inputs.

**E3. Changing `isInitialized` (`layout.tsx:215`) without scoping it.** Fixing D0 to
`status != null && status.status !== "not_initialized"` is correct, but it changes what
renders for EVERY paper-trading sub-route (positions, trades, nav, manage,
reality-gap, exit-quality, learnings), not just the Risk Monitor — the whole cockpit
would be replaced by the "No paper portfolio initialized" placeholder, which is itself
a *wrong* message (the fund exists; we just can't see it). If D0 is touched at all in
80.36, the placeholder copy must change too, and the blast radius is 7 routes.
**Recommendation: leave D0 alone in 80.36** (fix the widgets, which is what criterion
5 scopes) and queue D0 as its own step with its own copy decision.

**E4. `clsx` ternary chains silently reordering.** Rows 1-3 currently build their class
string with nested ternaries inside `clsx(...)`. Inserting an `unknown` arm at the wrong
position (e.g. `unknown ? slate : high ? amber : emerald` vs
`high ? amber : unknown ? slate : emerald`) changes which class wins for a **healthy
breaching** book. The unknown arm must be FIRST and must be mutually exclusive with the
others by construction — which is precisely what the discriminated union in §C buys.

**E5. Byte-identity of the healthy render.** Criterion 5 is satisfiable only if the
healthy branch emits the *same* strings and the *same* class list. Concrete hazards:
switching `SAFE` to a `<RiskPill>` component that adds a wrapper element, an icon, an
`aria-label`, or reorders `clsx` arguments will change the DOM even though the visible
text matches. If a shared pill component is introduced, its healthy output must be
asserted against the current literal markup, not merely against the visible text.

**E6. Carbon/WCAG addendum (do not regress accessibility while fixing).** Carbon:
of its four ingredients (Symbols, Shapes, Colors, Type) "for WCAG compliance, at least
three of these elements must be present." Today the pills carry text + colour (2). If
the fix distinguishes unknown by colour alone (grey pill still reading `SAFE`), it
fails both WCAG and the point of the step. The unknown state must change the **word**,
not just the token.

---

## Ranked recommendation (A-C, condensed)

1. **Per-row tri-state derived from PRESENCE, rendered via a discriminated union with a
   `never` exhaustiveness arm** (§C option 1). Unknown → neutral slate + an explicit
   word (`UNKNOWN`), never emerald. Safety rows get the dedicated unknown state
   (Grafana default, FAA A1); informational rows (vs SPY, Positions) get the existing
   `—` convention (`PnlBadge:47` already does it).
2. **Do NOT keep-last-state on the safety rows.** Grafana explicitly: "in situations
   where strict monitoring is critical, relying solely on the 'Keep Last State' option
   may not be appropriate." The warm-failure path already keeps last state invisibly
   (`layout.tsx:188-208`); that is a second, separate defect worth queueing, not
   something to lean on.
3. **Reuse `FreshnessBand`'s `"unknown"` idiom and `StaleDataState`'s shape**, but with
   navy/slate tokens per `.claude/rules/frontend.md` §1 — do not copy its `zinc`.

## Hard stops flagged (per the caller's instruction)

- **E1** — value-based discrimination would turn a genuine `SAFE` (`max_drawdown_pct === 0`)
  into `UNKNOWN`: changes the healthy path.
- **E2** — a card-level `if (!perf)` bail would hide a LIVE `HIGH (>20%)` position-size
  breach, because that row does not depend on `perf`: makes a genuine breach less visible.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (9 with usable content; 2 further
      fetched-but-empty are disclosed above rather than padded into the count)
- [x] 10+ unique URLs total (~45 surfaced; evaluated set tabled above)
- [x] Recency scan (2024-2026) performed + reported, including a negative finding
- [x] Full pages/documents read (AC 25-11B read as 135pp of extracted text, not the abstract)
- [x] file:line anchors on every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the request, plus 3 the request
      did not name (`layout.tsx` `isInitialized`, `navDenom`, `PaperVsBacktestCard`)
- [x] Contradictions / consensus noted (Google SRE chapters did NOT support the claim
      they were fetched for — reported, not hidden)
- [x] Per-claim citation with URL + access date

## Files inspected (internal)

| File | Lines read | Role |
|---|---|---|
| `frontend/src/components/paper-trading/cockpit-helpers.tsx` | 1-621 (full) | Hosts `RiskMonitorCard`, `SummaryHero`, `PaperVsBacktestCard`, `PnlBadge`, `Dollar`, `SharpeValue` |
| `frontend/src/app/paper-trading/positions/page.tsx` | 1-201 (full) | Sole consumer of `RiskMonitorCard`; wires the three correct siblings |
| `frontend/src/app/paper-trading/layout.tsx` | 180-219, 460-520 | `Promise.all` fetch, `isInitialized`, `SummaryHero` call site, error banner |
| `frontend/src/components/SectorBarList.tsx` | 30, 61, 82-93 | Empty-state convention |
| `frontend/src/components/PortfolioAllocationDonut.tsx` | 218-222 | Empty-state convention |
| `frontend/src/components/MultiCurrencyNavBreakdown.tsx` | 35-74 | Empty-state convention |
| `frontend/src/components/states/StaleDataState.tsx` | 1-66 (full) | Existing `isUnknown` / "no data" badge |
| `frontend/src/components/states/index.ts` | 1-10 (full) | States barrel |
| `frontend/src/lib/live-portfolio-context.tsx` | 51, 68-98, 165-205 | `FreshnessBand` union incl. `"unknown"`; stale-positions comment |
| `frontend/src/lib/paper-trading-utils.ts` | 33-35 | Second `FreshnessBand` declaration |
| `frontend/src/lib/types.ts` | 1243-1261 | Third `FreshnessBand` declaration |
| `frontend/src/lib/format.ts` | 140-157 | `positionMarketValueUsd` |
| `frontend/src/lib/paper-trading-context.tsx` | 1-47 | Context shape (`perf`/`portfolio` nullable, `positions` non-nullable array) |
| `frontend/src/app/paper-trading/layout.test.tsx` | 52-53 | The only paper-trading layout test |
| `.claude/rules/frontend.md` | Conventions section | Binding error/empty/colour rules |
| `.claude/rules/frontend-layout.md` | §8 | "Never show blank space" empty-state rule |

## JSON envelope

```json
{
  "tier": "T2",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 26,
  "urls_collected": 45,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "The Risk Monitor's SAFE/OK verdicts come from one line: cockpit-helpers.tsx:309 `perf?.max_drawdown_pct ?? 0`, which converts absence into the most reassuring observation; the vs-SPY KPI has the same shape twice (:189 and :217), and Positions once (:197). The card renders at all only because layout.tsx:215 `status?.status !== \"not_initialized\"` is TRUE when status is null. External consensus is one-directional: FAA AC 25-11B classifies misleading display information as Hazardous where loss of the same display is Major-Hazardous; Grafana makes No Data and Error first-class DEFAULT states and calls mapping to Normal 'ignores missing data'; Carbon defines Unknown (gray) as a status distinct from Normal, which 'implies no issues are present'; the project's own frontend.md already assigns gray to 'error/unavailable'. Fix as a per-row discriminated union with a never arm. Two hard stops: discriminating on value rather than presence would flip a genuine max_drawdown_pct===0 SAFE to UNKNOWN, and a card-level `if (!perf)` bail would hide a LIVE position-size breach because that row never reads perf. No test asserts SAFE/OK anywhere, so the healthy path has no regression net today.",
  "brief_path": "handoff/current/research_brief_80.36.md",
  "gate_passed": true
}
```
