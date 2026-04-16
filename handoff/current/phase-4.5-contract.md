# Phase 4.5 Contract — Paper Trading Dashboard v2 (Evaluation-Grade)

**Drafted:** 2026-04-16
**Goal:** Upgrade the paper trading page from "status viewer" to "go/no-go evaluation instrument" for live-capital promotion.
**Why:** Current page has Sharpe + NAV + positions but lacks PSR/DSR, no reality-gap reconciliation, no go-live gate, no progressive-disclosure agent rationale, no production-grade kill-switch. Without these, the decision to promote to live capital is vibes, not evidence.
**Gate:** Blocks phase-4 step 4.4 (Go-Live Checklist).

## Success Criteria (phase-level)
1. All 11 substeps (4.5.0 – 4.5.10) reach status="done" with evaluator_critique_pass.
2. Go-Live Gate widget renders 5 deterministic booleans; "Promote to live" button disabled unless all green.
3. PSR computed per Bailey & Lopez de Prado (2012) formula; DSR corrects for N strategies tested.
4. Paper-live equity curve overlays parallel OOS backtest; divergence >5% alerts.
5. Kill-switch flatten-all closes every open position within one cycle; daily loss + trailing DD limits enforced.
6. Agent rationale drawer surfaces Analyst → Bull/Bear → Trader → Risk hierarchy on click (progressive disclosure, not inline).
7. No regression in existing paper trading endpoints.

## Out of scope (explicit)
- Real-time tick-by-tick ladder (noise for cycle-based strategy).
- Per-agent token-count UI on main page (move to debug sub-route if added later).
- Corporate actions handling (splits/dividends) — separate phase.
- Live brokerage integration — that's phase-4.4 go-live.

## Dependencies
- phase-2 complete (harness infrastructure).
- `backend/services/autonomous_loop.py` stable (phase 2.14 done).

## Risk register
- **yfinance rate limits** on live-price endpoint (4.5.6) → cache + rate-gate.
- **BigQuery write amplification** from snapshot metrics (4.5.2) → batch updates.
- **PSR formula edge case** at N<30 trades → disable gate until threshold met.

## API surface (new or modified endpoints)

| # | Method | Path | Step | Purpose |
|---|---|---|---|---|
| 1 | GET | `/api/paper-trading/metrics-v2` | 4.5.1 | PSR, DSR, Sortino, Calmar, rolling-Sharpe 95% bootstrap CI |
| 2 | GET | `/api/paper-trading/round-trips` | 4.5.2 | win_rate, profit_factor, expectancy, MFE, MAE, median holding_days |
| 3 | GET | `/api/paper-trading/reconciliation` | 4.5.3 | paper-live NAV vs parallel OOS backtest NAV series, divergence_pct, alert flag |
| 4 | GET | `/api/paper-trading/gate` | 4.5.4 | 5 booleans (trades≥100, psr≥0.95 sustained 30d, dsr≥0.95, sr_gap≤30%, max_dd within tolerance) + `promote_eligible` |
| 5 | GET | `/api/paper-trading/trades/{trade_id}/rationale` | 4.5.5 | Agent-attribution tree: Analyst → Bull/Bear → Trader → Risk |
| 6 | GET | `/api/paper-trading/live-prices` | 4.5.6 | Per-ticker intraday price, cache-gated, 60s TTL |
| 7 | POST | `/api/paper-trading/flatten-all` | 4.5.7 | Close every open position at next mark-to-market; requires confirmation token |
| 8 | POST | `/api/paper-trading/pause` | 4.5.7 | Halt new entries; existing positions marked-to-market normally |
| 9 | POST | `/api/paper-trading/resume` | 4.5.7 | Resume entries if daily loss + trailing DD limits are healthy |
| 10 | GET | `/api/paper-trading/cycles/history` | 4.5.8 | Last 10 autonomous-loop runs with timings + status |
| 11 | GET | `/api/paper-trading/freshness` | 4.5.8 | Per-source `last_tick_age_sec` + loop heartbeat + BQ ingest lag |
| 12 | GET | `/api/paper-trading/mfe-mae-scatter` | 4.5.9 | Per closed round-trip: mfe, mae, pnl, capture_ratio (pnl/mfe) |

Existing endpoints (`/status`, `/portfolio`, `/trades`, `/snapshots`, `/performance`, `/start`, `/stop`, `/run-now`) must continue returning 200 with unchanged response shapes (no regressions).

## Schema deltas (BigQuery, dataset `pyfinagent_pms`)

**Modified tables:**
- `paper_trades`
  - ADD `signals` ARRAY<STRUCT<agent STRING, role STRING, rationale STRING, weight FLOAT64>> NULLABLE — per-trade agent attribution (4.5.5)
  - ADD `mfe_pct` FLOAT64 NULLABLE — max favorable excursion % across holding period (4.5.2, 4.5.9)
  - ADD `mae_pct` FLOAT64 NULLABLE — max adverse excursion % across holding period (4.5.2, 4.5.9)
  - ADD `holding_days` INT64 NULLABLE — inclusive day count buy→sell (4.5.2)
  - ADD `round_trip_id` STRING NULLABLE — groups paired buy+sell rows (4.5.2)
  - ADD `capture_ratio` FLOAT64 NULLABLE — realized_pnl_pct / mfe_pct (4.5.9)

- `paper_positions`
  - ADD `mfe_pct` FLOAT64 NULLABLE — tracked live, reset each new position (4.5.2)
  - ADD `mae_pct` FLOAT64 NULLABLE — tracked live (4.5.2)

**New tables:**
- `paper_metrics_v2` (one row per daily snapshot): date DATE, nav FLOAT64, rolling_sharpe FLOAT64, psr FLOAT64, dsr FLOAT64, sortino FLOAT64, calmar FLOAT64, rolling_sharpe_ci_low FLOAT64, rolling_sharpe_ci_high FLOAT64, n_strategies_tested INT64, trades_to_date INT64.
- `paper_reconciliation` (one row per cycle): cycle_ts TIMESTAMP, date DATE, paper_nav FLOAT64, backtest_nav FLOAT64, divergence_pct FLOAT64, alert_fired BOOL.
- `paper_cycle_history` (one row per autonomous-loop run): cycle_id STRING, started_at TIMESTAMP, duration_ms INT64, status STRING, data_source_ages_json STRING, bq_ingest_lag_sec INT64, error_count INT64.
- `paper_gate_snapshots` (one row per daily snapshot): date DATE, trades_ge_100 BOOL, psr_ge_95_sustained_30d BOOL, dsr_ge_95 BOOL, sr_gap_le_30pct BOOL, max_dd_within_tolerance BOOL, promote_eligible BOOL.

Migration scripts live in `scripts/migrations/` (one file per substep that needs schema change). Backwards-compatible only — no drops, no destructive changes. Populate new columns NULL for historical rows.

## Research gate checklist

Per PLAN.md §Research Gate (mandatory for every substep before GENERATE):

- [x] ≥3 primary sources per substep (academic paper, production system write-up, or authoritative library doc)
- [x] ≥10 URLs total collected in RESEARCH.md entry (abstracts alone do not count)
- [x] At least one source read in full (not just abstract) — noted as "(read full)" in RESEARCH.md
- [x] Hypothesis is falsifiable and testable by the substep's verification command
- [x] Success criteria written into `handoff/current/4.5.N-contract.md` BEFORE generating code
- [x] Contradictory / anti-pattern sources included (what-not-to-do sources are mandatory, not optional)
- [x] For numerical methods (PSR, DSR, bootstrap CI, Sortino, Calmar): formula cited with page/equation number from a peer-reviewed source

Research-gate entries for the 11 substeps will be appended to RESEARCH.md as each substep enters RESEARCH phase. Phase-level references:
- Bailey & Lopez de Prado (2012) — *The Sharpe Ratio Efficient Frontier* (PSR formula).
- Bailey & Lopez de Prado (2014) — *The Deflated Sharpe Ratio* (DSR correction for N trials).
- TradingAgents (Xiao et al., 2024) — per-trade progressive-disclosure drawer pattern (4.5.5).
- Marcos Lopez de Prado — *Advances in Financial Machine Learning* §13 (MFE/MAE, triple-barrier, capture ratio).
- yfinance rate-limit discussion (4.5.6) — cache+gate pattern.

## Harness protocol per substep
RESEARCH → PLAN (per-step contract) → GENERATE → EVALUATE (TaskCompleted verifier) → LOG (harness_log.md).
Each substep writes its own `handoff/current/4.5.N-contract.md` before GENERATE.
