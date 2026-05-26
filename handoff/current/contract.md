# Contract -- phase-44.2.X UX audit fix bundle (Cycle 68)

**Cycle:** 68 (2026-05-26). UX-quality follow-up to phase-44.2 closure; no separate masterplan step (the underlying step flipped done in cycle 67; this cycle polishes operator-flagged issues).

## Research gate

- Researcher `af5fa1f8484539e6d`, tier=moderate.
- 10 external sources read in full (Tailwind dark-mode + TanStack global filtering + Tremor DonutChart + WCAG 2.2 + WebAIM contrast).
- 24 URLs / 14 snippet-only / recency scan / 12 internal files.
- **gate_passed: true.**
- Brief: `handoff/current/research_brief_phase_44_2_uxaudit.md`.

## Hypothesis (single load-bearing finding)

Five UX issues operator reported (hover-row near-white, search doesn't match company name, Sector card unequal height, no portfolio allocation chart, headers still hard to read) are mostly SYMPTOMS of one root cause: `tailwind.config.js` is missing the `darkMode` key, defaulting to `'media'` strategy. The `dark:*` variants in DataTable / SectorBarList / cockpit-helpers only fire when the OS itself is in dark mode -- they don't reliably activate on this Mac. Fixing the dark-mode strategy unlocks ALL the cycle-67 `dark:` color work in one 2-line change.

## N* delta

- **B (Burn) primary:** cockpit readability is operator-blocking. Every dark-mode token landed in cycles 63-67 currently doesn't apply reliably. After this cycle: those tokens activate correctly + 4 additional improvements (filter-by-company, donut, equal-heights via items-start, header bump) compound.
- **R speculative:** correct allocation visibility helps spot concentration breaches faster.
- **P:** marginal.

## Scope (5 fixes)

| # | Fix | Approach |
|---|-----|----------|
| 1 | Hover-row near-white + every dark-mode token unreliable | Add `darkMode: 'selector'` to `tailwind.config.js` + `className+=" dark"` on `<html>` in `app/layout.tsx`. 2-line patch. Unlocks all existing `dark:*` variants. |
| 2 | Filter only matches ticker; should match company name | Custom `globalFilterFn` closes over `tickerMeta` + matches ticker OR company_name OR sector (case-insensitive substring). DataTable foundation gains optional `globalFilterFn` prop (TanStack's default behavior when omitted). |
| 3 | Sector card unequal-height with Risk Monitor | 3-col `items-start` row: Risk Monitor + Sector Concentration + new PortfolioAllocationDonut. Variable heights acceptable with items-start per frontend-layout.md §4.5. |
| 4 | No portfolio allocation chart | New `PortfolioAllocationDonut.tsx`: Tremor DonutChart wrapper; per-sector + Cash slices; normalized to NAV%; center label "NAV $X". |
| 5 | Headers still hard to read | DataTable header `dark:text-slate-300` -> `dark:text-slate-200`. The "still hard to read" perception is largely Fix #1's symptom: when darkMode='media' doesn't fire, `dark:text-slate-300` collapses to `text-zinc-700` (3.7:1 fail AA on navy-800/70). |

## Plan steps

1. `tailwind.config.js` -- add `darkMode: 'selector'` line.
2. `app/layout.tsx` -- add `dark` token to html className.
3. `DataTable.tsx` -- optional `globalFilterFn` prop wired to useReactTable; header text dark:text-slate-300 -> dark:text-slate-200; sanity-check hover class.
4. `positions/page.tsx` -- restructure to 3-col `grid items-start`; pass `globalFilterFn` closing over tickerMeta.
5. New `PortfolioAllocationDonut.tsx` + `.test.tsx` (Tremor DonutChart).
6. Verify all gates.

## Files

NEW:
- `frontend/src/components/PortfolioAllocationDonut.tsx`
- `frontend/src/components/PortfolioAllocationDonut.test.tsx`

MODIFIED:
- `frontend/tailwind.config.js`
- `frontend/src/app/layout.tsx`
- `frontend/src/components/DataTable.tsx`
- `frontend/src/app/paper-trading/positions/page.tsx`

ZERO backend changes.

## /goal integration-gate plan

| # | Gate | Plan |
|---|------|------|
| 1 | pytest >= 614 + vitest >= 158 | Run both; no backend touches. |
| 2 | TS build green | tsc + build. |
| 3 | Flag default OFF | N/A (UX polish). |
| 4-5 | N/A | |
| 6 | N* delta | DONE. |
| 7 | Zero emojis | Grep. |
| 8 | ASCII loggers | N/A. |
| 9 | SSOT | DonutChart wrapper reusable for /performance criterion 7 if needed. |
| 10 | log first / flip last | Yes -- no masterplan flip; just log + commit. |

## Sign-off

Authored AFTER researcher gate_passed=true. Operator explicitly requested "full harness with our mas agents" -- protocol observed.
