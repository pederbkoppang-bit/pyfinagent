# Pre-checks -- 2026-05-19 morning smoketest

| # | Check | Result | Notes |
|---|-------|--------|-------|
| 1 | pytest 49+ green | **PASS** | `49 passed, 1 warning in 3.58s` -- cycle_heartbeat (7) + step_5_6 (7) + observability (12) + price_tolerance (6) + strategy_decisions (4) + sector_concentration (13) |
| 2 | backend /api/health == "ok" | **PASS** | `health status: ok` |
| 3 | kill_switch.paused == true | **PASS** | `paused: True | pause_reason: manual` |
| 4 | phase-30.4 BQ migration | NOT APPLIED | `external_flow_today` column absent on `financial_reports.paper_portfolio_snapshots`. Per goal directive: Sharpe-pollution anomaly persists; proceed with smoketest. |
| 5 | researcher.md deep tier active | **PASS** | `\| deep \| <=3500 w \| <=200 \| 40+ \| at least 20 (typically 20-50) \|` -- the row is present at the documented line. |

**Overall verdict: PROCEED.** 4 of 5 PASS; pre-check 4 is documented-but-not-blocking per the goal.

Smoketest directory: `handoff/smoketest_20260519/`. NO production BQ writes. Loop STAYS PAUSED.
