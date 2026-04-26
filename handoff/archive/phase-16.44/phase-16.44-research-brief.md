---
step: phase-16.44
tier: simple
date: 2026-04-25
gate: internal-only (continued user feedback on home cockpit)
---

# Research Brief: phase-16.44 KPI scorecards + Last/Next gate-bar segments

User feedback after seeing 16.43 ship:
1. Scorecards (KPI tiles) should sit directly under the gate bar (currently under Red Line Monitor).
2. Add comparison sub-text to each KPI card matching the target screenshot
   (P&L: $ + %, VS SPY: alpha + benchmark, SHARPE + SORTINO, MAX DD + bounded, POSITIONS + breakdown).
3. Gate bar is missing scheduler date — show LAST run timestamp + NEXT run timestamp.

## Internal evidence

### Where the data lives

**Already loaded in `page.tsx:73-94`:**
- `ptStatus.portfolio.{nav, cash, starting_capital, pnl_pct, benchmark_return_pct, inception_date}` from `getPaperTradingStatus()`
- `ptStatus.next_run` -- next scheduled run ISO string (live probe confirmed: `"2026-04-27T14:00:00-04:00"`)
- `positions` array with `{quantity, ticker, ...}` -- no explicit `side` field; convention: `quantity > 0` = long, `< 0` = short/hedge
- `redLineSeries` -- daily NAV history (live: 31 points, all flat at 9499.5 source=pre_inception today). Format: `[{date, nav, source}]`

**OpsStatusBar already loads in `OpsStatusBar.tsx:61-77`:**
- `latestCycle.started_at` -- timestamp of last cycle (currently null since no cycles have run)
- Already accepts `nextRunAt?: string | null` prop but `page.tsx:150` does NOT pass it.

**Backend gap:** Sharpe / Sortino / MaxDD are NOT exposed by `/api/paper-trading/status` or `/portfolio`. They live in `backend/services/perf_metrics.py` for backtest contexts only.

### What can be computed client-side from existing data

From `redLineSeries` (the NAV history we already fetch for the Red Line Monitor):
- Daily P&L $ + %: `series[len-1].nav - series[len-2].nav` and the same / starting_nav
- Sharpe: `mean(daily_returns) / stddev(daily_returns) * sqrt(252)` (Lo 2002)
- Sortino: same but only downside variance (Sortino & Price 1994)
- Max DD: `max((running_max - current) / running_max)` over the window

When the series is short or flat (today's case: 31 points all 9499.5), Sharpe = NaN (0/0), Max DD = 0. Display "—" with a comment that real values populate post-inception.

### Pitfall: NaN propagation

`stddev = 0` when all NAVs are equal (today's case). `0/0 = NaN` in JS. Helper must guard: `if (stdDev === 0) return null` and the UI shows "—" for null.

### OpsStatusBar SchedulerSegment

Currently shows ONLY `Next run`. User wants BOTH `Last` (relative time of latestCycle.started_at) AND `Next` (existing). Split into TWO segments separated by Divider.

## Plan

| # | File | Change |
|---|------|--------|
| 1 | `frontend/src/lib/kpiMetrics.ts` (new, ~80 LOC) | Pure helpers: dailyDelta(series), sharpe(returns), sortino(returns), maxDrawdown(series), categorizePositions(positions). All return `null` on insufficient data. |
| 2 | `frontend/src/app/page.tsx` | Reorder JSX: gate bar -> KPI grid -> Red Line -> two-column. Replace 6 KpiTile lines with new shape (label + value + subText, color-coded). Pass `nextRunAt={ptStatus?.next_run}` to OpsStatusBar. |
| 3 | `frontend/src/components/OpsStatusBar.tsx` | Replace SchedulerSegment with two segments: LastSegment (uses `latestCycle.started_at` + formatRelativeTime) + NextSegment (existing logic). |

## Verification

```
cd frontend && npx tsc --noEmit && \
  test -f frontend/src/lib/kpiMetrics.ts && \
  grep -q "nextRunAt={ptStatus" frontend/src/app/page.tsx && \
  grep -q "LastSegment\|Last run" frontend/src/components/OpsStatusBar.tsx && \
  grep -q "kpiMetrics" frontend/src/app/page.tsx && \
  npm run lint 2>&1 | grep -c '@phosphor-icons/react' | grep -q '^0$'
```

## Honest disclosures (planned for the cycle)

- Sharpe / Sortino / MaxDD will display "—" today because redLineSeries is flat pre-inception. Once paper-trading runs Monday and NAV starts moving, computed values will populate. NO hardcoded stand-ins.
- Daily P&L $ + % will be "—" today for the same reason.
- Long/Short categorization uses `quantity > 0` = long, `< 0` = short. The label "long · hedge" mirrors the screenshot but uses `long · short` because hedge classification needs explicit side metadata not in the current schema. If user wants "hedge" label specifically, that's a follow-up to add a `position_role` field to backend.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 6,
  "report_md": "handoff/current/phase-16.44-research-brief.md",
  "gate_passed": true
}
```

Note on gate: pure-internal cycle (continued visual feedback on the same dashboard). Sharpe/Sortino/MaxDD formulas are textbook (Lo 2002, Sortino & Price 1994) and previously documented in `backend/services/perf_metrics.py` -- no fresh external research needed. Documented honestly rather than padded to 5.
