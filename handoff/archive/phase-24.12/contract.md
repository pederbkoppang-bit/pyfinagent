# Sprint Contract — phase-24.12 — Frontend UI/UX Presentation Layer

**Cycle:** phase-24 cycle 11
**Date:** 2026-05-12
**Step ID:** 24.12
**Priority:** P2

## Research-gate
`gate_passed: true` (tier=moderate). 6 sources: WCAG 2.2, Playwright screenshots + visual regression guides, ESLint no-restricted-imports, WAI-ARIA authoring practices, design-system enforcement.

```json
{"tier":"moderate","external_sources_read_in_full":6,"snippet_only_sources":7,"urls_collected":13,"recency_scan_performed":true,"internal_files_inspected":16,"gate_passed":true}
```

## Hypothesis
Strict frontend rules but inconsistent enforcement. Phosphor icon bypass. Missing loading/error/empty states. Cross-tab KPI mismatches.

**Researcher verdict: SURPRISINGLY GOOD with specific gaps:**
- **Icon imports: ZERO violations** — `no-restricted-imports: "error"` ESLint rule (phase-16.39) enforced; barrel-only exemption; grep returns empty
- **2 degraded-state pages** — `performance/page.tsx:65-66` uses bare `<p>` instead of `<PageSkeleton/>` + rose-border error banner; `sovereign/page.tsx:63-68` silently swallows RedLine API failures
- **Tab icons missing** — `paper-trading/page.tsx:383-390` TABS array lacks `icon` field (violates `frontend-layout.md §5`)
- **Cross-tab Sharpe mismatch** — home uses local `kpiSharpe()` over NAV; paper-trading uses `perf.sharpe_ratio` from API; sources not reconciled
- **Polling discipline gap** — `paper-trading/page.tsx:534-550` RunNow interval has no fail counter; other pages comply

## Success criteria (verbatim)
1. findings_md_exists
2-10. common pack
11. findings_audits_design_system_conformance_phosphor_dark_scrollbar
12. findings_audits_per_page_error_loading_empty_states
13. findings_audits_a11y_keyboard_aria_contrast
14. findings_audits_responsive_design_breakpoints
15. findings_audits_cross_tab_kpi_reconciliation
16. screenshots_dir_contains_at_least_14_images

**Verifier:** `python3 tests/verify_phase_24_12.py`

## Plan
1. Findings
2. Results
3. Q/A
4. Cycle 52 log
5. live_check_24.12.md
6. Flip
