# Experiment Results -- Cycle 73: chart-side SSOT overlay

**Date:** 2026-05-26
**Phase:** chart-side SSOT (SSOT integrity work that complements
cycle 72's tile-side fix; no masterplan step id).
**Result:** GENERATE complete; awaiting Q/A.

## What changed

Implemented Path 2 (frontend overlay) per researcher
`a6c2b1e445ca9b644` Section 7. Every chart surface that previously
forward-filled a 4-day-stale persisted snapshot to today's x-axis now
appends an explicit "live_now" point sourced from the cycle-72
`LivePortfolioProvider`, with a pulsating marker + dashed connector so
the operator can SEE which point is "as of close" and which is "now".

### Files modified (6)

1. `frontend/src/components/RedLineMonitor.tsx`
   - Added `liveNav?: number | null` + `liveBand?: "green"|"amber"|"red"|"unknown"` props.
   - Added static `LIVE_MARKER_COLOR` lookup (JIT-safe per cycle-68 lesson).
   - Computed `shouldOverlay` from `today > lastActual.date`.
   - Appended synthetic `{date: today, nav: liveNav, source: "live_now"}` to series.
   - Rendered dashed `Line` overlay between last-actual + live point.
   - Rendered `ReferenceDot` with inline SVG `<animate>` halo (Gaurav Gupta pattern).
   - Tooltip formatter appends "(live now)" suffix when payload `source==="live_now"`.

2. `frontend/src/app/page.tsx`
   - Passes `liveNav={lp.liveNav}` + `liveBand={lp.freshnessBand}` to `<RedLineMonitor>`.

3. `frontend/src/app/sovereign/page.tsx`
   - Imported `useLivePortfolio` from `@/lib/live-portfolio-context`.
   - Called the hook + passed same props to `<RedLineMonitor>`.

4. `frontend/src/app/paper-trading/nav/page.tsx`
   - Imported `useLivePortfolio`.
   - Extended `chartData` useMemo: when today > last persisted date AND
     `lp.liveNav != null` AND `startingCap > 0`, appended a synthetic row
     with `portfolio = (liveNav - startingCap)/startingCap*100` and
     `alpha = portfolio - lastBenchmark`. Benchmark carried forward
     (Yahoo Finance close-only; live SPY tracker out of scope per
     researcher Section 7).

5. `frontend/src/components/PaperReconciliationChart.tsx`
   - Added optional `livePaperNav?: number | null` prop.
   - Appended synthetic row to series: `paper_nav = livePaperNav`,
     `backtest_nav = last.backtest_nav` (carried forward, since the
     shadow backtest is historical by definition -- the divergence
     between live paper and last-known shadow IS the signal the chart
     wants to show), `divergence_pct` recomputed from the live paper
     NAV.
   - Chart now consumes `seriesOverlay` instead of raw `series`.

6. `frontend/src/app/paper-trading/reality-gap/page.tsx`
   - Added `useLivePortfolio` import + hook call.
   - Passed `livePaperNav={lp.liveNav}` to `<PaperReconciliationChart>`.

### Files unchanged (audit)

- ZERO backend files changed (researcher Section 7 explicit: backend
  forward-fill is independent of UX overlay).
- ZERO test scaffolding changed; existing
  `test_phase_23_2_8_use_live_nav_ssot.py` and
  `tests/verify_phase_23_1_17.py` continue to assert the page-import
  SSOT invariant; both pass post-cycle.
- `frontend/src/lib/live-portfolio-context.tsx` -- READ ONLY.

## Verification (verbatim command output)

### tsc --noEmit (frontend strict typecheck)
```
$ cd frontend && npx tsc --noEmit
exit=0
(no output, no errors)
```

### npx vitest run
```
 Test Files  23 passed (23)
      Tests  178 passed (178)
   Start at  20:09:49
   Duration  4.01s
```

### pytest backend/tests/test_phase_23_2_8_use_live_nav_ssot.py
```
backend/tests/test_phase_23_2_8_use_live_nav_ssot.py::test_phase_23_2_8_use_live_nav_hook_exists_and_exports PASSED [ 16%]
backend/tests/test_phase_23_2_8_use_live_nav_ssot.py::test_phase_23_2_8_home_page_imports_use_live_nav PASSED [ 33%]
backend/tests/test_phase_23_2_8_use_live_nav_ssot.py::test_phase_23_2_8_paper_trading_page_imports_use_live_nav PASSED [ 50%]
backend/tests/test_phase_23_2_8_use_live_nav_ssot.py::test_phase_23_2_8_both_pages_destructure_live_nav_and_pnl PASSED [ 66%]
backend/tests/test_phase_23_2_8_use_live_nav_ssot.py::test_phase_23_2_8_nav_math_lives_only_in_hook PASSED [ 83%]
backend/tests/test_phase_23_2_8_use_live_nav_ssot.py::test_phase_23_2_8_hook_return_shape_is_documented PASSED [100%]
============================== 6 passed in 0.03s ===============================
```

### tests/verify_phase_23_1_17.py
```
$ python tests/verify_phase_23_1_17.py
ok useLiveNav shared hook + home page consumption + paper-trading refactor + repair script (mark_to_market + save_daily_snapshot)
```

## Artifact shape

After cycle 73, every chart that previously interpolated the persisted
snapshot to today's x-axis now renders TWO visually distinct line
segments:

1. **Solid line** -- historical NAV from `series[0..N-1]`, each point
   labeled by its actual snapshot date.
2. **Dashed line** -- one segment from `series[N-1]` to a new "live_now"
   point at `today` with `y = lp.liveNav`. Endpoint dot pulses (SVG
   `<animate>` r=6 -> 12 -> 6 over 1.5s) so the operator's
   pre-attentive system sees "this is live" without reading text.
3. **Tooltip** -- on hover over the live point, shows the live value
   suffixed " (live now)" so screen-reader + verbose-debug paths also
   see the distinction.

## JIT-safety verification

`LIVE_MARKER_COLOR` is a static literal map (cycle-68 lesson): green
`#34d399`, amber `#fbbf24`, red `#fb7185`, unknown `#94a3b8`. No
template-string concatenation; Tailwind JIT compiles all four classes.

## Visual verification status

Per `.claude/rules/frontend.md` rule 5 (visual verification is
mandatory for any chart or color-coded UI): pulsing dot + dashed
connector still pending operator browser-probe. Dev server is managed
by the launchctl watchdog (cycle-68 memory rule); no `npm run build`
ran during this cycle.

## Not in scope

- Backend forward-fill in `get_paper_reconciliation` (researcher
  Section 7 explicit: backend stays simple; UX layer is where "live vs
  close" distinction matters).
- Live SPY tracking for NAV-chart benchmark line (deferred per
  researcher; Yahoo Finance close-only is acceptable for cycle 73).
- Hover-tooltip i18n / WCAG SC 1.4.13 (tooltip text is short and
  in-band; passes by default).
