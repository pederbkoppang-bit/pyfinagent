---
name: metric-source-paths
description: The three distinct Sharpe/maxDD code paths feeding the cockpit vs the go-live gate, and the get_paper_snapshots DESC-order trap that bit phase-47.4
metadata:
  type: project
---

There are THREE distinct performance-metric code paths in pyfinagent, and they do NOT agree by construction. This is the meta-cause behind phase-47.4's Sharpe/maxDD inconsistencies.

1. **Cockpit Sharpe** (`SHARPE(90D)`): backend `compute_sharpe_from_snapshots` (`perf_metrics.py:87`) via `/api/paper-trading/portfolio` (`paper_trading.py:219`). RFR=4%, 365-day window.
2. **Cockpit Sortino + maxDD**: LOCAL TypeScript (`frontend/src/lib/kpiMetrics.ts` `sortino`/`maxDrawdownPct`) on `redLineSeries` from `/api/sovereign/red-line` (chronological ASC, 30d default). NO risk-free subtraction.
3. **Gate maxDD + PSR/DSR/rolling_sharpe**: backend `paper_go_live_gate.py` (`_snapshot_max_dd_pct`) + `paper_metrics_v2.py` (`compute_metrics_v2` -> `_nav_to_returns`) on `get_paper_snapshots(limit=30)`.

**Why:** features were added incrementally (phase-16.44 local TS KPIs, phase-25.C12 backend-authoritative Sharpe, phase-4.5 gate). They were never unified.

**THE TRAP:** `bq.get_paper_snapshots()` (`bigquery_client.py:1035`) returns rows `ORDER BY snapshot_date DESC` (NEWEST FIRST). Any consumer that does `np.diff(navs)` or a peak-to-trough walk WITHOUT re-sorting gets a reversed series -> Sharpe sign flips, drawdown reads growth-as-crash. The CORRECT reference is `_nav_to_returns` (`paper_metrics_v2.py:65-66`) which sorts by snapshot_date asc AND subtracts `external_flow_today` (GIPS TWR, phase-30.4). By contrast `_fetch_snapshots` in `sovereign_api.py:161` already has `ORDER BY snapshot_date` ASC.

**How to apply:** When auditing or adding ANY metric that consumes `get_paper_snapshots`, FIRST confirm it sorts chronologically. The single-source-of-truth rule (`backend-services.md:22`: never compute Sharpe/drawdown/alpha outside perf_metrics.py) is only half-enforced -- the gate re-implements maxDD in `_snapshot_max_dd_pct` and the frontend re-implements all three in TS. Prefer delegating to `analytics.compute_max_drawdown` / `analytics.compute_sharpe` (note: compute_max_drawdown returns NEGATIVE %, gate wants positive magnitude). See [[project_psr_dsr_formulas]] for the DSR/PSR side.

**n_obs reality:** paper history is tiny (28 snapshots / 27 returns as of 2026-05-28). metrics-v2 returns None below MIN_OBS_FOR_PSR=30 (`paper_metrics_v2.py:33`). Per Lopez de Prado (ADIA Lab / PSR-MinTRL), a point Sharpe at n=27 is "a source of systematic error" -- don't trust any single Sharpe number until the track record is months long.
