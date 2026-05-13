---
step: 25.C12
slug: cross-tab-sharpe-kpi-reconciliation
tier: moderate
cycle_date: 2026-05-13
---

# Research Brief — phase-25.C12: Cross-tab Sharpe KPI Reconciliation (Backend Authoritative)

**Gate status:** PASS

---

## Three-variant search queries run

| Variant | Query |
|---------|-------|
| Current-year frontier | `single source of truth KPI dashboard backend authoritative client-local divergence pattern 2026` |
| Last-2-year window | `frontend migration client-computed to API-served metric backwards compatibility fallback TypeScript 2025` |
| Year-less canonical | `JSDoc @deprecated TypeScript TSDoc convention ESLint no-deprecated rule` |
| Year-less canonical | `Sharpe ratio annualization daily returns sqrt 252 convention finance` |
| Year-less canonical | `cache invalidation derived KPI multiple API endpoints performance portfolio shared computation pattern` |

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://tsdoc.org/pages/tags/deprecated/ | 2026-05-13 | Official doc (TSDoc) | WebFetch | "@deprecated communicates that an API item is no longer supported and may be removed in a future release; should be followed by a sentence describing the recommended alternative" |
| https://typescript-eslint.io/rules/no-deprecated/ | 2026-05-13 | Official doc (typescript-eslint) | WebFetch | "rule reports on any references to code marked as @deprecated; included in strict-type-checked config; accepts `allow` option using TypeOrValueSpecifier" |
| https://njakob.com/shorts/deprecate-symbols-in-typescript | 2026-05-13 | Authoritative blog | WebFetch | Exact syntax: `/** @deprecated Zod schemas with {@link z} should be used instead. */`; "the function will be struck through" in IDE intellisense |
| https://www.typescriptlang.org/play/4-0/new-js-features/jsdoc-deprecated.ts.html | 2026-05-13 | Official docs (TypeScript) | WebFetch | "@deprecated added to TypeScript 4.0 type system; non-blocking warning; VS Code shows in intellisense, outlines, and inline annotations" |
| https://www.sixfigureinvesting.com/2013/09/daily-scaling-sharpe-sortino-excel/ | 2026-05-13 | Authoritative practitioner | WebFetch | "average of the daily returns is divided by sampled std, multiplied by sqrt(252) — typical trading days per year in USA markets; the key thing is the consistency of approach between the investments you are comparing" |
| https://datahubanalytics.com/metric-chaos-to-metric-clarity-why-enterprises-need-a-single-source-of-truth-for-kpis/ | 2026-05-13 | Industry practitioner blog | WebFetch | "metrics drift... definitions diverge across platforms... Manual reconciliations before leadership meetings... Declining trust in analytics outputs; one authoritative definition of each metric, consistently calculated and reused everywhere" |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://strapi.io/blog/what-is-single-source-of-truth | Blog | SSOT concept covered by datahubanalytics fetch |
| https://www.databricks.com/blog/semantic-layer-architecture-components-design-patterns-and-ai-integration | Industry blog | Semantic layer; SSOT point already covered by deeper fetch |
| https://github.com/gajus/eslint-plugin-jsdoc | Code repo | JSDoc linting plugin; not needed — step uses plain @deprecated tag only |
| https://www.kpifire.com/blog/ssot-single-source-of-truth/ | Blog | SSOT definition; covered by datahubanalytics |
| https://www.redhat.com/en/blog/single-source-truth-architecture | Engineering blog | Enterprise SSOT architecture; general; not dashboard-specific |
| https://oneuptime.com/blog/post/2026-02-02-fastapi-cache-invalidation/view | Blog | FastAPI cache invalidation; existing api_cache.py already implements TTL-based approach |
| https://blog.quantinsti.com/sharpe-ratio-applications-algorithmic-trading/ | Industry | Sharpe annualization; confirmed by sixfigureinvesting fetch |
| https://en.wikipedia.org/wiki/Sharpe_ratio | Reference | Annualization formula confirmed by practitioner fetch |
| https://quantnet.com/threads/sharpe-ratio-question.3217/ | Forum | Community tier; canonical confirmed elsewhere |
| https://medium.com/@drv.muk/understanding-fastapi-caching-and-asynchronous-processing-77608b0dd474 | Blog | FastAPI cache pattern; existing api_cache.py covers this |

---

## Recency scan (2024-2026)

Searched: "single source of truth KPI dashboard backend authoritative 2026", "JSDoc deprecated TypeScript 2025", "FastAPI cache invalidation derived metric 2025", "frontend API migration backwards compatibility TypeScript 2025".

**Result:** No findings that supersede canonical sources. The TSDoc `@deprecated` tag convention and Sharpe sqrt(252) annualization are stable conventions with no material changes in the 2024-2026 window. A 2026 FastAPI article (oneuptime.com, 2026-02-02) confirms the existing TTL-based `api_cache.py` pattern in pyfinagent is current best practice. The 2026 datahubanalytics article on KPI SSOT corroborates the backend-authoritative approach. No new findings supersede the canonical sources.

---

## Key findings

1. **The actual numeric divergence source is the risk-free rate, not the sqrt(252) factor.** Backend `analytics.compute_sharpe` (`backend/backtest/analytics.py:135`) subtracts `risk_free_rate / periods_per_year` from each daily return before computing the ratio: `excess = returns - 0.04/252`. Frontend `kpiMetrics.ts::sharpe` (lines 57-64) computes `annualized = (mu / sd) * sqrt(252)` with no risk-free subtraction. Both use sqrt(252), but the numerator differs by approximately 0.04 effective annual drag. This is the numeric divergence bucket-24.12 F-4 observed. Wiring the home page to the API value eliminates it by construction. (Sources: `backend/backtest/analytics.py:135,144`; `frontend/src/lib/kpiMetrics.ts:63`)

2. **TSDoc `@deprecated` is the canonical marker for TypeScript >= 4.0.** Syntax: `/** @deprecated Use <alternative> instead. */` immediately above the export declaration. TypeScript renders strikethrough in IDE intellisense with no compiler error. The `@typescript-eslint/no-deprecated` rule (strict-type-checked) escalates callers to a lint error. The existing codebase already uses this exact pattern at `frontend/src/lib/types.ts:10` (`/** @deprecated Use monthly_nok */`). No new tooling needed. (Sources: tsdoc.org; typescript-eslint.io; `frontend/src/lib/types.ts:10`)

3. **`compute_sharpe_from_snapshots` is already imported at the top of `paper_trading.py`.** The import at `backend/api/paper_trading.py:30` is: `from backend.services.perf_metrics import compute_sharpe_from_snapshots, compute_alpha`. No new import is required. Only a snapshot fetch and a result-dict field addition are needed inside `get_portfolio`. (Source: `backend/api/paper_trading.py:30`)

4. **`get_portfolio` does not currently fetch snapshots; the insertion pattern is established by `get_performance`.** The `/performance` endpoint at `paper_trading.py:264` calls `snapshots = await asyncio.to_thread(bq.get_paper_snapshots, limit=365)` then passes to `compute_sharpe_from_snapshots(snapshots)` at line 276. The portfolio endpoint should replicate this exact pattern. This adds one BQ round-trip per uncached portfolio request; the enriched result is stored under the existing `paper:portfolio` cache key. (Source: `backend/api/paper_trading.py:264,276`)

5. **Home page currently discards the `portfolio` sub-object; only `positions` is captured from the portfolio response.** At `frontend/src/app/page.tsx:105`: `if (portfolio.status === "fulfilled") setPositions(portfolio.value.positions ?? [])`. The `portfolio.value.portfolio` sub-object (which will carry `sharpe_ratio`) is not stored in any state variable. The fix requires adding a `useState<number | null>` and capturing the field alongside `positions`. (Source: `frontend/src/app/page.tsx:96-105`)

6. **`PaperPortfolio` interface lacks `sharpe_ratio`; `PaperPerformance` already has it.** `frontend/src/lib/types.ts:608-617` defines `PaperPortfolio` with no performance fields. `types.ts:691` shows `PaperPerformance.sharpe_ratio: number`. Adding `sharpe_ratio?: number | null` to `PaperPortfolio` is the minimal type change; optional with null union provides backwards compat during rolling deploy. (Source: `frontend/src/lib/types.ts:608-617,691`)

7. **Backend-authoritative SSOT for KPIs prevents metric drift and reconciliation overhead.** Distributed client-local computation creates "Multiple definitions of the same KPI" and "Conflicting numbers across dashboards" — the exact symptom this step fixes. Moving the home page to read `sharpe_ratio` from the `/portfolio` API response ensures both tabs render the same number from the same BQ snapshot series computed by the same formula. (Source: datahubanalytics.com fetch; `backend/services/perf_metrics.py:87-115`)

---

## Internal code inventory

| File | Lines inspected | Role | Status |
|------|-----------------|------|--------|
| `backend/api/paper_trading.py` | 1-33 (imports), 160-214 (`get_portfolio`), 249-302 (`get_performance`) | Portfolio + performance API endpoints | Active; `get_portfolio` lacks `sharpe_ratio`; `get_performance` has it at line 294 |
| `backend/services/perf_metrics.py` | 1-30 (header), 87-115 (`compute_sharpe_from_snapshots`) | Canonical Sharpe computation for API layer | Active; risk-free adjusted, sqrt(252) annualized |
| `backend/backtest/analytics.py` | 124-144 (`compute_sharpe`) | Base annualized Sharpe with risk-free subtraction | Active; `excess = returns - rf/252`; `* sqrt(periods_per_year)` |
| `frontend/src/lib/kpiMetrics.ts` | 1-107 (full file) | Client-local KPI math; `sharpe()` at lines 57-65 | Active; raw mean only — no risk-free subtraction — diverges from backend |
| `frontend/src/app/page.tsx` | 1-28 (imports), 85-125 (load effect), 143-168 (KPI computation), 217-251 (KPI tile render) | Home page; consumes `kpiSharpe` at line 161 | Active; `sharpe90 = kpiSharpe(navSeries)` at line 161 needs replacement |
| `frontend/src/lib/types.ts` | 606-617 (`PaperPortfolio`), 685-698 (`PaperPerformance`) | TypeScript interfaces | `PaperPortfolio` missing `sharpe_ratio`; `PaperPerformance` has it at line 691 |
| `frontend/src/lib/api.ts` | 276-290 (`getPaperPortfolio`, `getPaperPerformance`) | API client functions | `getPaperPortfolio` return type flows through `PaperPortfolio`; no function body change needed |

---

## Backend response shape change — `/portfolio` endpoint

**File:** `backend/api/paper_trading.py`
**Insertion point:** After the sector_breakdown try/except block (after line 205), before `result = {` at line 208.

Add inside `get_portfolio`:
```python
# phase-25.C12: authoritative Sharpe from NAV snapshots — same formula and
# window as /performance so home page and /paper-trading page always match.
snapshots_for_sharpe = await asyncio.to_thread(bq.get_paper_snapshots, limit=365)
portfolio_sharpe = compute_sharpe_from_snapshots(snapshots_for_sharpe)
```

Add one field to the `result` dict:
```python
result = {
    "portfolio": portfolio,
    "positions": positions,
    "sector_breakdown": sector_breakdown,
    "sharpe_ratio": portfolio_sharpe,          # phase-25.C12: backend-authoritative
}
```

`compute_sharpe_from_snapshots` is already imported at `paper_trading.py:30` — no new import needed.

---

## Frontend wire path — `page.tsx` swap

**File:** `frontend/src/app/page.tsx`

**Step 1 — add state variable** (near line 87, alongside existing `useState` declarations):
```typescript
const [apiSharpe, setApiSharpe] = useState<number | null>(null);
```

**Step 2 — capture `sharpe_ratio` from portfolio response** (replace lines 104-106):
```typescript
// Before:
if (portfolio.status === "fulfilled") setPositions(portfolio.value.positions ?? []);

// After:
if (portfolio.status === "fulfilled") {
  setPositions(portfolio.value.positions ?? []);
  setApiSharpe(portfolio.value.portfolio?.sharpe_ratio ?? null);
}
```

**Step 3 — replace `sharpe90` computation** (line 161):
```typescript
// Before:
const sharpe90 = kpiSharpe(navSeries);

// After (backwards-compat fallback: if API field absent, fall back to local):
const sharpe90 = apiSharpe ?? kpiSharpe(navSeries);
```

The fallback `kpiSharpe(navSeries)` keeps the tile functional if the backend is not yet deployed or the cache still holds the old response shape.

---

## JSDoc `@deprecated` shape for `kpiMetrics.ts::sharpe`

**File:** `frontend/src/lib/kpiMetrics.ts`
**Location:** line 57 — replace the bare export declaration with:

```typescript
/**
 * @deprecated Use the `sharpe_ratio` field from `/api/paper-trading/portfolio`
 * (or `/api/paper-trading/performance`) instead. This client-local computation
 * omits the risk-free rate adjustment and diverges from the backend canonical
 * value (`analytics.compute_sharpe`, risk-free adjusted, sqrt(252) annualized).
 * See phase-25.C12.
 */
export function sharpe(series: NavPoint[], periodsPerYear = 252): number | null {
```

This matches the existing codebase pattern at `types.ts:10`: `/** @deprecated Use monthly_nok */`. TypeScript 4.0+ will render strikethrough on all call sites in VS Code intellisense. The function body is NOT removed — it remains as the fallback path in `page.tsx` and may be needed by any future caller during a transition window.

---

## Consensus vs debate (external)

**Consensus:** Backend-authoritative KPI is the industry standard. Client-local duplication is universally acknowledged as a maintenance liability causing metric drift. The sqrt(252) Sharpe annualization with risk-free subtraction is the Lo (2002) canonical form; the frontend's zero-risk-free shortcut is an approximation that diverges materially at a 4% risk-free rate (current US environment). No debate on the `@deprecated` tag syntax or TypeScript 4.0 support.

---

## Pitfalls

1. **Do not call `bq.get_paper_snapshots` twice in `get_portfolio` if a snapshot list is already being fetched elsewhere in the same request.** The current `get_portfolio` body does not fetch snapshots, so the new call is the first and only one. Pattern: `asyncio.to_thread` consistent with lines 173 and 177.

2. **Cache keys are independent: `paper:portfolio` and `paper:performance` cache separately.** The two endpoints compute Sharpe independently from the same BQ source. This is intentional — do NOT attempt to have `/portfolio` read `/performance`'s cached value; circular dependency risk.

3. **`kpiMetrics.ts::sharpe` deprecation must NOT remove the function.** `sortino`, `maxDrawdownPct`, `dailyDelta`, and `categorizePositions` in the same file are all still active on the home page. Only `sharpe` is deprecated; the function body is kept as a fallback.

4. **The `KpiTile` label "Sharpe (90d)" may not reflect the 365-day window used by the API.** Both `/portfolio` and `/performance` pass `limit=365` to `bq.get_paper_snapshots`. The label cosmetic update is out of scope for the three immutable criteria but is a follow-on note.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total incl. snippet-only (16 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (7 files inspected)
- [x] Critical contradiction noted: frontend omits risk-free rate; backend applies 4% annual drag — this IS the numeric divergence
- [x] All claims cited per-claim with URL or file:line anchor

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
