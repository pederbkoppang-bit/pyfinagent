# Cycle 9 Contract -- Step 38.11 (Recent Reports table display + normalization fixes)

**Generated:** 2026-05-27T19:30+02:00.

**Step id:** `38.11` -- Recent Reports table -- Alpha=0.00 + recommendation casing mix + company column shows ticker.

**Cycle class:** Frontend display + small backend normalization. NOT a trading-policy change. Citation floor (>=2 AI-in-trading + >=2 academic) does NOT apply per goal precedent. Researcher floor (>=5 sources read in full, >=10 URLs, recency scan, write-first) IS satisfied.

## Research gate
- Researcher: `a2fe26dfb8e227db7`, tier=simple, gate_passed=true.
- Output: `handoff/current/research_brief_phase_38_11_recent_reports.md`.
- Sources read in full: 5 (TanStack column-defs, React useMemo, WCAG 2.2 1.1.1, MDN toUpperCase, GitHub Primer DataTable).
- URLs collected: 12.
- Recency scan: performed (three-variant query discipline).
- Internal files inspected: 8 with file:line anchors.

## Findings from researcher
1. **ALPHA=0.00 root cause**: ALREADY fixed by phase-71 at `backend/services/autonomous_loop.py:1309-1311` (reads `final_weighted_score` then `final_score` fallback). Operator's 2026-05-26 23:55 screenshot captures stale pre-fix rows. New rows from cycle 7/8 onwards will have non-zero scores. No code change needed for this symptom.

2. **Recommendation casing mix** (HOLD vs Hold vs Buy): Backend write-path bug at `autonomous_loop.py:1845`. The fallback writes `"HOLD"` (uppercase) but the source-of-truth `Recommendation` enum in `backend/api/models.py:21-26` defines Title-case (`"Hold"`, `"Buy"`, `"Sell"`, ...). The LLM emits in mixed case, so persisted rows are inconsistent.
   - **Backend fix**: change default `or "HOLD"` to `or "Hold"` to align with enum.
   - **Frontend fix (defensive)**: normalize ALL recommendation display to UPPERCASE in the three table consumers (RecentReportsTable, reports-columns, ReportCompareDrawer). This makes display consistent regardless of stored casing.

3. **Company column shows ticker**: Backend write-path bug at `autonomous_loop.py:1843`. The fallback `market_data.get("name") or ticker` lets the ticker leak into the company column when `name` is empty.
   - **Backend fix**: change `or ticker` to `or None`. Let frontend show em-dash placeholder (already implemented at `RecentReportsTable.tsx:125`).

## Hypothesis
Two backend write-path normalization edits + one new frontend formatter helper applied at three call sites fully resolves all three reported display issues for fresh rows. Old rows in BQ (pre-phase-71) keep their stale `final_score=0` but cannot be retroactively fixed without a one-off backfill migration (out of scope per researcher).

## Plan steps
1. `backend/services/autonomous_loop.py:1843` -- `company_name=market_data.get("name") or ticker` -> `company_name=market_data.get("name") or None`.
2. `backend/services/autonomous_loop.py:1845` -- `recommendation=analysis.get("recommendation") or "HOLD"` -> `recommendation=(analysis.get("recommendation") or "Hold")`.
3. NEW file `frontend/src/lib/formatRecommendation.ts` -- helper that normalizes recommendation strings: uppercase + underscores->spaces.
4. `frontend/src/components/RecentReportsTable.tsx:132` -- replace inline `.replace(/_/g, " ")` with helper.
5. `frontend/src/components/reports-columns.tsx:103` -- wrap `row.original.recommendation` in helper.
6. `frontend/src/components/ReportCompareDrawer.tsx:134` -- wrap `r.recommendation` in helper.

## Success criteria (verbatim from masterplan 38.11)
- `alpha_column_displays_nonzero_value_when_analysis_has_nonzero_final_score` -- already satisfied by phase-71; cycle-8 fresh cycle (in flight) will validate.
- `recommendation_column_normalized_to_single_case_across_all_rows` -- frontend uppercase normalization.
- `company_column_displays_company_name_not_ticker_symbol` -- backend ticker-fallback removal + frontend em-dash fallback.
- `live_check_38_11_captures_post_fix_screenshot_path_or_html_snippet` -- live_check_38.11.md will reference dev-server screenshot + verbatim rendered HTML.

## Out-of-scope (flagged by researcher)
- `autonomous_loop.py:1932` `risk_judge_decision or "HOLD"` fallback (writes to paper_trades, not analysis_results). Follow-up normalization.
- BQ backfill for stale `final_score=0` rows: not in scope per researcher (write-path fix stops the bleeding).
- No frontend "—" guard for `final_score === 0` -- would mask legitimate 0.0 alpha rows.

## Memory-rule compliance
- ZERO new npm deps.
- NO `npm install`, NO `npm run build`, NO `rm -rf .next/*`.
- ZERO emojis introduced.
- Frontend dev server stays up; helper file is a pure addition (TS compiler picks up on next HMR).
- Phosphor icons convention respected (no icon changes).

## References
- `handoff/current/research_brief_phase_38_11_recent_reports.md`
- `backend/api/models.py:21-26` (Recommendation enum -- Title case canonical)
- `backend/services/autonomous_loop.py:1843,1845` (write-path bugs)
- `frontend/src/components/RecentReportsTable.tsx:131-134` (defensive em-dash already in place)
