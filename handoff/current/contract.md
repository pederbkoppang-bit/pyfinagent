# Contract — phase-48.4: Live rotation bake-off SMOKE (first real validation)

**Cycle:** 15 (Priority 5 — operator said "you decide"; chosen: the safe "verify live" step). **LLM spend:** $0 (quant-only backtests; real BQ + real compute, no LLM). **Run mode:** the live smoke is BACKGROUNDED (~6-12 min, 4 real backtests).

## Research-gate summary
`researcher` `a97220b09474e710d` (gate **PASSED**): 5 sources read in full + recency scan + 11 internal files. Brief: `handoff/current/research_brief_phase_48_4_live_smoke.md`.

Decisive findings:
- **Viable window (the trap):** the engine's default 12mo-train/3mo-test needs ≥15 months → a 6-month window yields ZERO walk-forward windows + a degenerate DSR (var fallback 0.5). **Use 2022-01-01..2024-06-30 → 6 windows, T~367 ≫ the 32 floor.** BQ confirmed: `financial_reports.historical_prices` spans 2017-01-03..2026-05-28; the window has 511,960 rows / 502 tickers; the ~756-day training lookback is not starved.
- **Minimal scale:** 2 seeds (tb_baseline + qm_trend_tilt) × num_param_variants=2 = 4 real backtests; N=2 is the floor for a real per-strategy PBO (N<2 → degenerate 0.0 false-passes the gate). T~367 ≫ 32 guaranteed.
- **Runtime ~6-12 min** (cold first ~5-10 min + warm <30s each) → BACKGROUND. Macro preload is INSIDE run_backtest (no ~40min hang). `load_promoted_params` is a read.
- **Ctor:** `BigQueryClient(settings)` (takes settings, not no-args).
- **Q/A PASS bar:** dsr/pbo finite in [0,1] (pbo NOT a degenerate 0.0), sharpe finite, n_windows≥2, a verdict with selected_id/reason, ONE `rotation_log.jsonl` row at `allocation_pct=0.0` matching the verdict, zero deploy side-effects. **`no_candidate_passed_gate` is a VALID outcome** — the smoke proves the PLUMBING runs on real backtests, not that a seed won.

## Hypothesis
Running `run_rotation_bakeoff` on a real ≥15-month window with 2 seeds × 2 variants will exercise the entire 48.1-48.3 chain (full-kwarg engine → 4 real walk-forward backtests → nav→daily-returns → `generate_report` DSR + per-strategy (T×K) `compute_pbo` → producer → selector → persisted verdict) end-to-end on ACTUAL data for the first time, producing finite, sane DSR/PBO/Sharpe + a real verdict — proving the machinery works live (everything prior was $0-mock-tested), at $0 LLM, audit-only (no deploy).

## Immutable success criteria (verbatim from .claude/masterplan.json phase-48.4)
1. run_rotation_bakeoff executes on the real engine over a >=15-month window (2022-01-01..2024-06-30, >=2 walk-forward windows) with 2 seeds (tb_baseline + qm_trend_tilt) x num_param_variants=2, AUDIT-ONLY (allocation_pct=0, no deploy), $0 LLM
2. the run produces, for at least the seeds that complete, FINITE per-strategy metrics: dsr in [0,1], pbo in [0,1] that is NOT a degenerate 0.0 (i.e. the (T,N) matrix was real: T>=32 from nav_history + N>=2 variants), sharpe finite, n_windows>=2 (so DSR's per-window variance is real, not the 0.5 fallback)
3. the selector returns a verdict dict (selected_id + reason + ranked + num_trials); `no_candidate_passed_gate` / first_selection / a switch are ALL valid plumbing-proven outcomes; exactly ONE rotation_log.jsonl row is persisted at allocation_pct=0.0 matching the verdict
4. zero deploy side-effects (NO promoted_strategies MERGE, NO settings.paper_* mutation, NO live order); the captured real {dsr,pbo,sharpe} per seed + the verdict + the persisted row are recorded verbatim in experiment_results.md + live_check_48.4.md

## Plan steps
1. Write `scripts/rotation/run_smoke_bakeoff.py` (or an inline script) that imports `get_settings`, `BigQueryClient(settings)`, `backend.backtest.cache.clear_cache`, and `run_rotation_bakeoff`, and invokes it with the smoke params (2 seeds, num_param_variants=2, 2022-01-01..2024-06-30, persist=True, log to a captured path), printing the verdict + per-seed metrics as JSON.
2. Run it BACKGROUNDED (real compute ~6-12 min); capture stdout + the persisted rotation_log row.
3. Write experiment_results.md + live_check_48.4.md with the verbatim real verdict/metrics/row.
4. Spawn a fresh Q/A against the captured output + the rotation_log row (the PASS bar above).
5. On PASS: append harness_log.md, flip masterplan 48.4 → done.

## Out-of-scope (DEFERRED)
- The FULL 4-seed × 8-variant bake-off (longer run); the deployment params→settings.paper_* bridge (the keystone — touches live trading, operator-gated activation); the weekly cron; re-enabling the reverted trailing/vol-target readers; effective-N / CPCV.

## References
- `handoff/current/research_brief_phase_48_4_live_smoke.md`
- `backend/autoresearch/rotation_runner.py` (48.3), `strategy_backtest_adapter.py` (48.2), producer/registry/selector (48.1/47.6)
- `backend/backtest/backtest_engine.py` + `analytics.py`; `backend/db/bigquery_client.py:22` (ctor); `financial_reports.historical_prices` (BQ data)
