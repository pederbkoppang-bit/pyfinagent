# Sprint Contract — phase-25.B12 — Missing states + tab icons sweep

**Cycle:** phase-25 cycle 11
**Date:** 2026-05-12
**Step ID:** 25.B12
**Priority:** P1

## Research-gate
Reuses phase-24.12 cycle 11 researcher gate.

## Hypothesis
Three small frontend fixes close phase-24.12 F-2 (degraded states on perf/sovereign pages) + F-3 (paper-trading tab icons missing).

## Success criteria (verbatim)
1. performance_page_uses_pageskeleton_and_error_banner
2. sovereign_page_surfaces_redline_api_errors_in_ui
3. paper_trading_tabs_array_has_icon_field_for_each_tab

## Plan
1. `performance/page.tsx`: replace `<p>Loading...</p>` with `<PageSkeleton/>`; replace `<p>` error with rose-bordered banner + Retry button
2. `sovereign/page.tsx`: add `redLineError` state; capture in .catch; render rose banner when set
3. `paper-trading/page.tsx`: add `icon` field to TABS (Wallet/Receipt/ChartLineUp/ChartBar/TrendUp/Gear), import via canonical `@/lib/icons` barrel
4. `icons.ts`: add `TabPositions/TabTrades/TabNavChart/TabRealityGap/TabExitQuality/TabManage` aliases
5. Verifier `tests/verify_phase_25_B12.py` (9 claims)

## References
- `docs/audits/phase-24-2026-05-12/24.12-ui-ux-presentation-findings.md` F-2 + F-3
- `.claude/rules/frontend-layout.md` §5 (tab icon mandate)
