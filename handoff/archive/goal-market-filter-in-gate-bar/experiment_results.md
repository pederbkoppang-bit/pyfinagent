# Experiment Results — `goal-market-filter-in-gate-bar` (Cycle 34)

**Date:** 2026-06-01. **Status:** complete (implemented + built + tested + live
Playwright click-through verified; auth gate restored).

## What was built

Folded the paper-trading market filter (`All·US·EU·KR` radiogroup) INTO the operator
status bar (`OpsStatusBar`) as its left-most **Market** segment, and retired the
standalone filter+session-strip row. Each pill's dot now doubles as that market's
open/closed indicator (emerald=open, slate=closed), folding the deleted
`MarketSessionStrip` signal into the pills. The segment is conditional on three new
optional props, so the homepage (which shares `OpsStatusBar`) is unchanged.

## Files changed

| File | Change |
|------|--------|
| `frontend/src/components/OpsStatusBar.tsx` | +`useMemo` import; +`MarketFilter`/`isMarketOpen` imports; +optional props `markets`/`activeMarket`/`onMarketChange`; +`MarketSegment` helper (mount-guarded `useState<Date\|null>` clock → `sessionOpen` map); conditional left-most render of `<MarketSegment/> + <Divider/>`. Bar stays `<section>` (NOT `role="toolbar"`). |
| `frontend/src/components/paper-trading/MarketFilter.tsx` | +optional `sessionOpen?: Record<string,boolean>` prop; dot color = emerald(open)/slate(closed) when supplied, else per-market `MARKET_DOT_CLASS` (pre-mount fallback → no hydration mismatch); folded OPEN/CLOSED into the pill `title`. Radiogroup role + roving-tabindex/arrow code unchanged. |
| `frontend/src/app/paper-trading/layout.tsx` | Deleted standalone filter row (old `483-490`); passed `markets`/`activeMarket`/`onMarketChange` into the cockpit `<OpsStatusBar>`; dropped now-unused `MarketFilter`/`MarketSessionStrip` imports. Filtered note kept. |
| `frontend/src/components/paper-trading/MarketSessionStrip.tsx` | **Deleted** (`git rm`) — signal folded into pills; no remaining importer. |
| `frontend/src/app/page.tsx` | **Unchanged** (homepage `OpsStatusBar` keeps only `nextRunAt`). |
| `.gitignore` | +`cockpit-*.png` (ignore loose Playwright verification screenshots at repo root; mirrors the goal-browser-mcp `.playwright-mcp/` block). Housekeeping. |
| `handoff/current/{research_brief,contract,experiment_results,evaluator_critique}.md` | Cycle-34 rolling harness artifacts. |

## Verification command output (verbatim)

### `npx tsc --noEmit`
```
EXIT_TSC=0
```
(no type errors)

### `npx eslint <3 changed files>`
```
✖ 4 problems (0 errors, 4 warnings)
EXIT_ESLINT=0
```
All 4 are `react-hooks/set-state-in-effect` warnings (not errors): 3 pre-existing
(`layout.tsx:173` fallback-to-ALL, `layout.tsx:212` refresh, `OpsStatusBar.tsx:96`
refresh); 1 is the new `OpsStatusBar.tsx:197` `setNow(new Date())` mount-guard — the
documented two-pass hydration pattern, identical to the one the deleted
`MarketSessionStrip` already carried (net-zero new warning class; `next build` treats
as warning, build stays green).

### `npm run build` (next build)
```
 ✓ Generating static pages (24/24)
Route (app)                                 Size  First Load JS
├ ○ /                                    11.2 kB         154 kB
├ ○ /paper-trading/positions               11 kB         137 kB
... (all 24 routes compiled) ...
ƒ Middleware                             85.3 kB
```
Build GREEN (route table prints only on success).

### `npm run test` (vitest)
```
 Test Files  23 passed (23)
      Tests  178 passed (178)
EXIT_TEST=0
```
Includes `layout-tablist.test.tsx`.

### Emoji grep (3 changed files)
```
(no emoji)
```

## Live Playwright click-through (skip-auth Path A; gate restored after)

Driven via the `mcp__playwright__*` MCP against the running dev server. Evidence
screenshots (gitignored, repo root): `cockpit-after-all.png` (filter in the bar).

**DOM containment (`browser_evaluate`) on `/paper-trading/positions`:**
```json
{
  "insideBar": true,
  "oldStandaloneRowStillPresent": false,
  "segmentOrder": ["Market","Gate","Kill","Cycle","Last","Next"],
  "pills": [
    {"label":"All","title":"All markets","dotClass":null},
    {"label":"US","title":"NYSE/Nasdaq — CLOSED","dotClass":"...bg-slate-600"},
    {"label":"EU","title":"XETRA — OPEN","dotClass":"...bg-emerald-400"},
    {"label":"KR","title":"KRX — CLOSED","dotClass":"...bg-slate-600"}
  ]
}
```
→ Criterion 1 (radiogroup INSIDE the `<section aria-label="Paper-trading operator
status">`) ✓; old row gone ✓; Market segment left-most ✓; session folded into pills ✓.

**Click EU (`browser_evaluate`):**
```json
{"checkedMarket":"EU","hasVsDAX":true,"hasVsSPY":false,
 "filteredNote":"Filtered to EU. NAV / Cash / Sharpe are fund-level USD; the table, allocation, and sector breakdown below show EU only."}
```
→ Criterion 3 ✓ (benchmark flips VS SPY→VS DAX; filtered note; scoping intact). Reset
to All confirmed.

**Console (`browser_console_messages`, all levels):** 0 React errors, 0 warnings, **no
hydration mismatch** text. Only a pre-existing `favicon.ico` 404 + Fast-Refresh/DevTools
info logs. → Criterion 5 (no hydration warning) ✓ + Criterion 6 (no new console errors) ✓.

**Homepage `/` (`browser_evaluate`):**
```json
{"barPresent":true,"hasMarketSegment":false,"segmentOrder":["Gate","Kill","Cycle","Last","Next"]}
```
→ Criterion 4 (homepage bar structurally identical, no Market segment) ✓.

**Auth gate restore (mandatory, per `docs/runbooks/browser-mcp.md`):**
```
skip-auth OFF; verifying gate
GATE RESTORED after 2s (302)
LIGHTHOUSE_SKIP_AUTH now: (unset)
```
→ Criterion 3 restore clause ✓.

## Acceptance-criteria summary

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Radiogroup inside `OpsStatusBar` `<section>`; old row gone | PASS (`insideBar:true`, `oldStandaloneRowStillPresent:false`) |
| 2 | One fewer row (gate-bar→NAV distance smaller) | PASS (standalone row removed; screenshot shows NAV tiles directly below bar) |
| 3 | EU→`vs DAX` + scope; All restores; auth gate restored (302) | PASS |
| 4 | Homepage bar structurally identical (no Market segment) | PASS |
| 5 | Session open/closed still visible; no hydration warning | PASS (emerald/slate pill dots + title; console clean) |
| 6 | `npm run build` green; tests pass; zero emoji; no console errors | PASS (24 pages; 178 tests; 0 emoji; 0 errors) |

## Notes / residue

- `isMarketOpen` (`format.ts:212`) retains a live consumer (the new `MarketSegment`); no
  dead export. `MarketFilter` now imported by `OpsStatusBar` only.
- The bar wraps gracefully at 1440px (existing `flex flex-wrap gap-x-6 gap-y-3`); the
  Market segment sits left, `Last`+`Next` keep their `ml-auto` right alignment.
- Dev server `.next` kickstarted after the production build to restore dev state.
