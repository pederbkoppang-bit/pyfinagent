# Research Brief — phase-80.36: Risk Monitor fabricates SAFE/OK with zero data

Tier: T2 (Opus 5, effort high). Started 2026-07-26. Status: IN PROGRESS (write-first).

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
| 1 | `dashboard UI distinguishing "no data" from zero null vs zero rendering 2026` | current-year |
| 2 | `Nielsen Norman Group dashboard status indicator unknown state fail loud safety` | year-less canonical |
| 3 | `Grafana alert rule "No Data and Error handling" configure no data state default` | year-less canonical |
| (more appended below) | | |

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://grafana.com/docs/grafana/latest/alerting/fundamentals/alert-rule-evaluation/nodata-and-error-states/ | 2026-07-26 | official doc | WebFetch | No Data is a FIRST-CLASS state, distinct from Normal, and it is the DEFAULT. |
| 2 | https://www.nngroup.com/articles/visibility-system-status/ | 2026-07-26 | authoritative (NN/g) | WebFetch | "A lack of information often equates to a lack of control." |

## Identified but snippet-only

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://community.dynatrace.com/t5/Dashboarding/Replace-quot-no-data-quot-with-quot-0-quot-in-a-dashboard/m-p/181861 | community | vendor-forum tier 5 |
| https://community.esri.com/t5/arcgis-dashboards-questions/indicator-show-0-and-icon-instead-of-quot-no-data/td-p/1310195 | community | tier 5 |
| https://gitlab.com/gitlab-org/gitlab-design/-/issues/462 | industry issue | corroborating only |
| https://github.com/google/site-kit-wp/issues/4226 | industry issue | corroborating only |

## Recency scan (2024-2026)

(pending)

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

