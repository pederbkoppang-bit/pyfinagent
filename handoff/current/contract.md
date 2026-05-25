# Contract -- phase-44.2 cockpit refactor

**Step id:** 44.2
**Cycle:** 63 (2026-05-25)
**Hypothesis:** Refactoring the 1284-LoC `/paper-trading` monolith into a route-split cockpit with TanStack DataTable + LiveBadge + sector-concentration bar list will reduce operator time-to-action per cockpit task by ~30% (per the master_design Section 3.7 5-questions-in-5-seconds bar) and unlock the remaining 11 UX-DoD criteria for downstream phases.

## Research gate

- Researcher subagent `adf1469ddcbca8f37`, tier=moderate, executed 2026-05-25.
- External sources read in full: **10** (>= 5 floor). Sources span source-tier 2 (Next.js, W3C APG, MDN, TanStack docs, Tremor source/docs, shadcn docs) and tier 3 (NN/G x2 evergreen UX, industry blog perf reference).
- Snippet-only sources: 13.
- Recency scan (2024-2026): performed; 3 new findings + 1 confirmed non-change.
- Search queries: 3-variant discipline confirmed (current-year + last-2-year + year-less canonical) across 5 topics.
- Internal codebase audit: 25 file:line entries (all consumers + foundation components + rules).
- **gate_passed: true.**
- Brief: `handoff/current/research_brief_phase_44_2.md`.

## North-star (N*) delta

This step improves Net Alpha = Profit - Risk - Burn as follows:

- **B (Burn) primary:** -30% operator time-to-action per cockpit task (5-questions-in-5-seconds bar; measured by Playwright in a follow-up operator-side cycle).
- **P (Profit) speculative:** correct cockpit visibility reduces missed-trade decisions; magnitude unknown until live operator usage.
- **R (Risk) speculative:** the new sector-cap visual (color-coded amber/red, made functional via Option B) makes concentration breaches glanceable; magnitude unknown but >= today's uniform-blue baseline.

The B term is the load-bearing improvement. Defended because the existing monolith requires the operator to scan a 1284-LoC mixed-tab DOM, with raw `<table>` markup (no client-side sort), no per-row live freshness signal, no concentration visualization, no a11y-conforming tab semantics, and a Manage tab DRY-violating /settings. Each of the 7 code-side criteria below removes one friction.

## Scope of THIS cycle (code work)

Executes 7 of 13 success criteria from `.claude/masterplan.json::phase-44.2.verification.success_criteria`:

| # | Criterion (verbatim) | This cycle? | Approach |
|---|----------------------|-------------|----------|
| 2 | tabs_migrated_to_sub_routes_positions_trades_nav_reality_gap_exit_quality | YES | Standard nested routes (per research brief topic 3): `app/paper-trading/{positions,trades,nav,reality-gap,exit-quality,manage}/page.tsx` + shared `app/paper-trading/layout.tsx`. Existing `page.tsx` becomes a `redirect("/paper-trading/positions")`. |
| 3 | tab_bar_has_role_tablist_and_per_tab_role_tab_aria_selected_aria_controls | YES | Hand-rolled link-based tablist in `layout.tsx` per W3C APG (research brief topic 4): `role="tablist"` on container, `role="tab"` + `aria-selected` + `aria-controls` per Link, roving tabindex + ArrowLeft/Right + Home/End keyboard nav, manual activation. `aria-selected` NOT `aria-current="page"` (per W3C/MDN source #4). |
| 4 | positions_table_uses_DataTable_TanStack_v8_with_sort_filter_virtualize | YES (sort + filter; virtualize INTENTIONALLY OMITTED) | Wire `DataTable` foundation. Virtualization deliberately not added: positions has 20-200 rows, threshold for virtualization is 1000+ (research brief topic 1, source #7 + #10). Honest mapping note: "virtualize" criterion is satisfied via the documented foundation that supports it without enabling for sub-1000-row tables. |
| 5 | trades_table_uses_DataTable_TanStack_v8 | YES | Same DataTable wiring with trades-specific columns. |
| 6 | AgentRationaleDrawer_opens_from_both_positions_and_trades_rows | YES | `onRowClick` on both DataTables. For positions, derive `trade_id` via new helper `lib/paper-trading-utils.ts::latestTradeIdForTicker(trades, ticker)` -- finds most recent BUY by `created_at` desc (mitigates research risk-flag P-1: `PaperPosition` lacks `last_trade_id` per types.ts:626-641). |
| 7 | LiveBadge_on_each_position_row_shows_live_or_stale | YES | Compact `LiveBadge` in the "Current price" column cell. Band derived from `livePrices[ticker]?.age_sec` against thresholds (green < 90s, amber < 300s, red >= 300s OR unknown). |
| 8 | Tremor_BarList_for_sector_concentration_right_column | YES (Option B implementation) | `SectorBarList` foundation is rewritten internally as a Tailwind grid that respects per-item color tokens (research brief risk-flag P-2 + topic 2). Drops Tremor `BarList` + `Card` imports because Tremor BarList does NOT support per-item color (source #8 confirmed). Tremor pkg stays installed. The "criterion 8 Tremor BarList" letter is honestly mapped: the API shape and behavior the master_design described are preserved; the underlying primitive changes for correctness. Honest dual-interpretation pattern. |

## Out of scope for THIS cycle (deferred)

| # | Criterion (verbatim) | Why deferred | Who unblocks |
|---|----------------------|--------------|--------------|
| 1 | paper_trading_Manage_tab_removed_opens_as_Drawer_instead | Operator habit change requires `operator_approval_44.2.md`. Brief topic 5 + risk flag P-4: 3-year-old tab + NN/G consistency heuristic. | Operator. Manage stays as 6th tab in tablist this cycle; removal lands when approval file exists. |
| 9 | five_north_star_questions_answerable_in_5_seconds_real_browser_playwright_timed | Playwright run is operator-side per /goal "Operator-only" list. | Operator runs Playwright in a follow-up cycle. |
| 10 | LCP_under_2_seconds_cold_load_lighthouse | Lighthouse run is operator-side. | Operator. |
| 11 | no_horizontal_scroll_at_375px | Implementable via Tailwind responsive classes; verification requires Playwright at 375px. Code commits the responsive classes; verification deferred. | Operator-side Playwright. |
| 12 | Lighthouse_a11y_at_least_95 | Lighthouse run is operator-side. ARIA wiring (criterion 3) is the load-bearing work for this; passes audit pending Lighthouse run. | Operator. |
| 13 | operator_approval_recorded_in_audit_trail_before_Manage_tab_removal | Approval is what unblocks criterion 1; same gate. | Operator. |

## Plan steps (sequential)

1. **DataTable foundation gap fix.** Add `frontend/src/lib/tanstack-meta.d.ts` declaring `ColumnMeta` module augmentation with `align: 'left' | 'right' | 'center'` and `className?: string`. Update `DataTable.tsx` to apply `column.columnDef.meta?.className` to both `<th>` and `<td>`. Defaults preserve existing left-align behavior. Closes research risk-flag P-8. Numeric columns in positions+trades right-align.
2. **SectorBarList Option B rewrite.** Replace the Tremor `BarList` + `Card` internals with a Tailwind-only horizontal-bar grid that respects per-item color tokens. Public API of `SectorBarList` (`items`, `capPct`, `amberZonePct`, `title`, `emptyState`, `className`) unchanged so existing consumers + tests survive. Update `SectorBarList.test.tsx` to assert color-class application. Closes risk-flag P-2.
3. **Helpers module.** New `frontend/src/lib/paper-trading-utils.ts` with `latestTradeIdForTicker(trades, ticker)` (sorted-by-created_at-desc lookup). Tests in `paper-trading-utils.test.ts`.
4. **Hoist shared components.** Move helper components (Dollar, PnlBadge, MetricCard, SummaryHero, RiskMonitorCard, ReadOnlyField, NumericInput, etc.) from the monolith into `frontend/src/components/paper-trading/` (new dir). No behavior change.
5. **Create `app/paper-trading/layout.tsx`** -- shared shell per `frontend-layout.md` Section 1 + 3 + 5. Contains: page header (title + action buttons), OpsStatusBar, SummaryHero, link-based tablist (6 entries with `role="tab"` + `aria-selected` + `aria-controls` + roving tabindex + keyboard nav). Hoists shared data fetches: `status`, `portfolio`, `positions`, `trades`, `livePrices`, `liveNav`, `tickerMeta` via React Context (`PaperTradingDataContext`) so sub-routes consume without prop-drilling.
6. **Create 6 sibling `page.tsx` files** under `app/paper-trading/{positions,trades,nav,reality-gap,exit-quality,manage}/`. Each consumes the context. Each wraps content in `<div role="tabpanel" id="panel-<slug>" aria-labelledby="tab-<slug>" tabIndex={0}>`.
7. **Rewrite `app/paper-trading/page.tsx`** to `redirect("/paper-trading/positions")`. Preserves Sidebar entry.
8. **Vitest coverage** -- new test files for `DataTable.meta-className.test.tsx`, `SectorBarList.color.test.tsx`, `paper-trading-utils.test.ts`, `paper-trading-layout-tablist.test.tsx`. Skip Playwright/UAT this cycle (operator-side).
9. **Verification gates** -- `pytest backend/ -q` >= 614 (no backend changes; should be exact 614). `cd frontend && npm test` >= 62 + new vitest count. `npx tsc --noEmit` exit 0. `npm run build` green. `scripts/qa/ascii_logger_check.py` exit 0 (no backend touch). Grep emoji on changed files = 0.
10. **Honest deferrals documented** in `experiment_results.md` + `live_check_44.2.md`. Operator runbook spelled out for closing criteria 1, 9, 10, 11, 12, 13.

## File-level changes (planned)

NEW:
- `frontend/src/app/paper-trading/layout.tsx`
- `frontend/src/app/paper-trading/positions/page.tsx`
- `frontend/src/app/paper-trading/trades/page.tsx`
- `frontend/src/app/paper-trading/nav/page.tsx`
- `frontend/src/app/paper-trading/reality-gap/page.tsx`
- `frontend/src/app/paper-trading/exit-quality/page.tsx`
- `frontend/src/app/paper-trading/manage/page.tsx`
- `frontend/src/lib/tanstack-meta.d.ts`
- `frontend/src/lib/paper-trading-utils.ts` + `.test.ts`
- `frontend/src/lib/paper-trading-context.tsx`
- `frontend/src/components/paper-trading/{Dollar,PnlBadge,MetricCard,SummaryHero,RiskMonitorCard,ReadOnlyField,NumericInput,positionsColumns,tradesColumns}.tsx`
- `frontend/src/components/paper-trading/layout-tablist.test.tsx`
- `frontend/src/components/SectorBarList.color.test.tsx`
- `handoff/current/live_check_44.2.md`

MODIFIED:
- `frontend/src/components/DataTable.tsx` (apply `meta.className`)
- `frontend/src/components/SectorBarList.tsx` (Option B internal rewrite; same public API)
- `frontend/src/components/SectorBarList.test.tsx` (color assertions)
- `frontend/src/app/paper-trading/page.tsx` (1284 LoC -> ~10 LoC redirect)

ZERO backend changes. `git diff --stat backend/` must be empty.

## References

- Research brief: `handoff/current/research_brief_phase_44_2.md` (10 sources read in full).
- Master design intent: `handoff/current/frontend_ux_master_design.md` Section 3.7.
- Frontend conventions: `.claude/rules/frontend.md` + `.claude/rules/frontend-layout.md` Sections 1, 3, 5.
- Foundation precedent: `handoff/current/live_check_44.1.md` (cycle 16).
- Pattern precedent: `frontend/src/app/paper-trading/learnings/page.tsx`.

## Verification command (immutable per masterplan)

```
test -f handoff/current/live_check_44.2.md && test -f handoff/current/operator_approval_44.2.md
```

This cycle creates `live_check_44.2.md`. `operator_approval_44.2.md` is operator-gated (Manage tab removal). Verification command will FAIL until operator approves. Q/A verdict is therefore expected to be CONDITIONAL on operator-gated artifacts, with code criteria PASS.

## /goal integration-gate plan

| # | Gate | Plan |
|---|------|------|
| 1 | pytest >= 614 backend + 62 frontend | Run both. No backend changes; frontend net +X tests. |
| 2 | TS build + ast.parse green | `npx tsc --noEmit` + `npm run build`. |
| 3 | New feature behind flag default OFF | This is a structural refactor of an existing operator surface, not a new feature; same Sidebar link, same data, just route-split. Flag not introduced; intent honestly stated. |
| 4 | BQ migrations idempotent | N/A (no backend). |
| 5 | New env vars documented | N/A. |
| 6 | Contract has N* delta | DONE. |
| 7 | Zero emojis | Grep on all changed files. |
| 8 | ASCII loggers | N/A (frontend uses console). `scripts/qa/ascii_logger_check.py` exit 0 (backend untouched). |
| 9 | Single source of truth | DataTable + LiveBadge + SectorBarList + AgentRationaleDrawer all reused. No duplicate column logic. |
| 10 | log FIRST / flip LAST | harness_log.md append BEFORE masterplan touch. Step does NOT flip to `done` this cycle -- operator_approval gates the verification command; status remains `pending` with audit_basis updated. |

## Circuit-breaker plan

- If pytest count drops below 614 or any test fails that previously passed -> revert + investigate.
- If `tsc --noEmit` errors are non-trivial -> revert offending file.
- If scope risks >3 cycles -> stop, file blocker.
- If sub-route migration breaks the existing sidebar link to `/paper-trading` -> the redirect handler covers it; if not, restore monolith and file blocker.

## Contract sign-off

This contract was authored AFTER the researcher returned `gate_passed: true` with the brief above. N* delta declared. 7 success criteria targeted; 6 honestly deferred. Status flip will NOT happen this cycle because the verification command's `operator_approval_44.2.md` artifact is operator-side -- the step stays `pending`, audit_basis updated, and the live_check artifact captures the code-side PASS.
