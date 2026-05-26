# Contract -- Cycle 73: chart-side SSOT overlay

**Cycle:** 73 (2026-05-26)
**Trigger:** Live NAV tiles now agree (cycle 72), but charts forward-fill 4d-stale snapshots to today's x-axis position. Researcher recommends Path 2 (frontend overlay).

## Research gate

- Researcher `a6c2b1e445ca9b644`, tier=deep, 10 sources read in full, gate_passed=true.
- Brief: `handoff/current/research_brief_phase_chart_ssot.md`.

## N* delta

- **B primary:** charts now READ-OUT-LOUD that the rightmost point is live, with a distinct marker -- no more silent forward-fill of a 4d-stale snapshot labeled as today.

## Scope -- 3 chart surfaces, Path 2 frontend overlay

| Surface | File | Change |
|---|---|---|
| RedLineMonitor (Home + Sovereign) | `components/RedLineMonitor.tsx` | Optional `liveNav` + `liveBand` props. Append `{date: today, nav: liveNav, source: "live_now"}` to the series when today > last actual snapshot date. Pulsating ReferenceDot + dashed connector. |
| Home consumer | `app/page.tsx` | Pass `lp.liveNav` + `lp.freshnessBand` to `<RedLineMonitor>`. |
| Sovereign consumer | `app/sovereign/page.tsx` | Add `useLivePortfolio()` + pass props. |
| Paper Trading NAV Chart | `app/paper-trading/nav/page.tsx` | Append live "today" data row with portfolio pct from `(liveNav - starting_capital) / starting_capital * 100`. |
| Paper Trading Reality Gap | `components/PaperReconciliationChart.tsx` + `app/paper-trading/reality-gap/page.tsx` | Append live `paper_nav` point only (backtest_nav stays at last snapshot). |

## Files

MODIFIED:
- `frontend/src/components/RedLineMonitor.tsx`
- `frontend/src/app/page.tsx`
- `frontend/src/app/sovereign/page.tsx`
- `frontend/src/app/paper-trading/nav/page.tsx`
- `frontend/src/components/PaperReconciliationChart.tsx`
- `frontend/src/app/paper-trading/reality-gap/page.tsx`

ZERO backend changes.

## /goal integration gates

1. tsc + vitest green. 2. No `npm run build` (memory rule). 3. Zero emojis. 4. log first / no masterplan flip.
