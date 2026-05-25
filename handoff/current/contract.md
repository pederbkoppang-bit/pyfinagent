# Contract -- phase-44.4 Reports section refresh

**Step id:** 44.4
**Cycle:** 65 (2026-05-25)
**Hypothesis:** Migrating reports to `useURLState` + ARIA tablist + DataTable foundation + Tremor AreaChart on performance + TimeRangeSelector advances 8 of 10 UX-DoD code-side criteria with no operator-approval gate. Lighthouse a11y >=95 deferred to operator-side run.

## Research gate

- Researcher subagent `a2f06d8fdab4b52f4`, tier=moderate, executed 2026-05-25.
- External sources read in full: **7** (>= 5 floor). Next.js useSearchParams + W3C WAI-ARIA Tabs + WCAG 2.2 target-size + Tremor AreaChart + Tremor SparkChart + UX Movement segmented-vs-dropdown + react-sparklines.
- Snippet-only sources: 17.
- Recency scan (2024-2026): performed.
- Search queries: 3-variant discipline.
- Internal codebase audit: 11 file:line entries.
- **gate_passed: true.**
- Brief: `handoff/current/research_brief_phase_44_4.md`.

## North-star (N*) delta

- **B (Burn) primary:** removes manual `searchParams.get` boilerplate (replaced with `useURLState` foundation); -3 manual URL-handling sites consolidated; per-tile UX advances.
- **R (Risk) speculative:** Per-pillar performance bars + Tremor AreaChart make cost-trend visibility glanceable.
- **P (Profit) speculative:** marginal -- presentation, not signal generation.

## Scope (code work)

Executes 9 of 10 criteria; 1 deferred to operator-side Lighthouse:

| # | Criterion | This cycle? | Approach |
|---|-----------|-------------|----------|
| 1 | reports_useURLState_syncs_tab_ticker_selected_to_url_params_shareable_links_work | YES | Replace `searchParams.get("tab")` (line 95) + `searchParams.get("ticker")` (line 103) with `useURLState` hooks from `lib/hooks` (cycle 44.1 foundation). `selected[]` stays in component state (transient compare-selection, not URL-shareable). |
| 2 | reports_compare_wizard_uses_Drawer_overlay | YES | Build a new `ReportCompareDrawer` overlay component (mirrors `AgentRationaleDrawer` pattern: aria-modal, role=dialog, close-on-ESC, backdrop). Trigger: clicking Compare tab opens drawer. Selection persists across tab switches via component state. Result: clear separation between History default + Compare overlay. |
| 3 | reports_history_uses_DataTable_TanStack_v8_with_sparkline_column_30d_score_history | YES | Replace the BentoCard-list rendering of `filtered` reports with a `DataTable` from the cycle 62 foundation. Columns: ticker / company / date / score (sortable) / recommendation / 30d-score-sparkline (inline Tailwind-SVG MiniSpark from cycle 64). Group reports by ticker -> sparkline data per row = score history for that ticker over the latest N analyses. Click row -> expand summary (preserve existing `expanded` UI -- via DataTable's `onRowClick`). |
| 4 | reports_empty_state_uses_EmptyState_component_not_inline_paragraph | YES | Replace inline `<p>No reports found...</p>` at lines 292-294 + 334 with `<EmptyState>` (cycle 44.1 foundation; takes icon + title + description props). |
| 5 | performance_AreaChart_Tremor_above_cost_history_table_cumulative_cost | YES | Use Tremor `<AreaChart>` with `colors={["amber"]}` override (Tremor's hardcoded-blue defeat per cycle 63 SectorBarList precedent). Cumulative transform via `useMemo` over existing `costHistory`. Title above the existing cost table; chart at h-64 floor. |
| 6 | performance_sparkline_next_to_win_rate_number_30d_trend | YES | Add inline mini-sparkline next to Win Rate BentoCard. Data source per researcher risk-flag: 30d trend likely not in current `PerformanceStats` shape; use Tailwind-SVG MiniSpark with `stats.recent_returns` (if exists) OR honest placeholder pattern (small dim line + tooltip "30d trend pending"). Decided: render only when data exists; otherwise omit the sparkline (no fake data). |
| 7 | performance_per_pillar_performance_bars_from_SynthesisReport_data | YES | Aggregate pillar scores from `loaded` (or fetched recent reports) -> 5 horizontal bars (pillar_1 corporate / pillar_2 industry / pillar_3 valuation / pillar_4 sentiment / pillar_5 governance). Average across N recent reports. Same Tailwind-grid bar pattern as `SectorBarList` (cycle 63 Option B rewrite). |
| 8 | performance_TimeRangeSelector_7d_30d_90d_all | YES | Segmented control with `role="radiogroup" aria-label="Time range"`, 4 options (7d / 30d / 90d / all). Each option is `<button role="radio" aria-checked={...} min-h-[32px]>`. Filters costHistory + per-pillar aggregation. |
| 9 | Lighthouse_a11y_at_least_95_on_both_pages | DEFERRED (operator-side) | ARIA wiring + label + EmptyState all done; audit pending operator Lighthouse run. |
| 10 | tab_bar_has_role_tablist_aria_selected | YES | Reports tab bar: `role="tablist"` on container + `role="tab"` + `aria-selected={activeTab === id}` + `aria-controls="panel-{id}"` per button; each tab content wraps in `<div role="tabpanel" id="panel-{id}" tabIndex={0}>`. Performance page has no tabs -- N/A for that route. |

## Plan steps

1. **useURLState migration** in `reports/page.tsx`: 2 hooks (tab + ticker).
2. **ARIA tablist wiring** on reports tab bar (criterion 10).
3. **DataTable wiring** for reports history with sparkline column.
4. **EmptyState replacement** (2 sites in reports + 1 site in performance).
5. **ReportCompareDrawer** new component + mount in reports/page.tsx.
6. **Tremor AreaChart** + cumulative transform in performance/page.tsx.
7. **Win rate sparkline** (conditional render).
8. **Per-pillar bars** in performance/page.tsx.
9. **TimeRangeSelector** segmented control component (radiogroup) + filtering logic.
10. **Vitest** for new components + hook usage.
11. **Verify** all gates.

## Files planned

NEW:
- `frontend/src/components/ReportCompareDrawer.tsx` + `.test.tsx`
- `frontend/src/components/TimeRangeSelector.tsx` + `.test.tsx`
- `frontend/src/components/PerformancePillarBars.tsx` (if extracted) OR inline
- `frontend/src/components/reports-columns.tsx` (DataTable factory for reports)
- `handoff/current/live_check_44.4.md`

MODIFIED:
- `frontend/src/app/reports/page.tsx` (URLState migration + tablist + DataTable + EmptyState + Drawer trigger)
- `frontend/src/app/performance/page.tsx` (AreaChart + sparkline + pillar bars + TimeRangeSelector + EmptyState)

ZERO backend changes.

## Verification command

```
test -f handoff/current/live_check_44.4.md
```

Single-gate (no operator_approval AND-clause).

## /goal integration-gate plan

| # | Gate | Plan |
|---|------|------|
| 1 | pytest >= 614 backend + 100 frontend | Run both; no backend changes. |
| 2 | TS build + ast.parse green | tsc + build. |
| 3 | Feature behind flag default OFF | N/A (refactor + new UX components called out in master_design). |
| 4 | BQ migrations idempotent | N/A. |
| 5 | New env vars | N/A. |
| 6 | Contract has N* delta | DONE. |
| 7 | Zero emojis | Grep. |
| 8 | ASCII loggers | N/A. |
| 9 | Single source of truth | DataTable + EmptyState + MiniSpark reused; useURLState reused. |
| 10 | log first / flip last | Yes. Status CAN flip this cycle. |

## Circuit-breaker plan

- pytest drop -> revert + investigate.
- TS errors -> revert offending file.
- Scope > 3 cycles -> stop + file blocker.

## Sign-off

Authored AFTER researcher returned gate_passed=true. 9 criteria targeted; 1 operator Lighthouse deferral.
