# Experiment Results — phase-48.4: Live rotation bake-off SMOKE (first real validation)

**Cycle:** 15 (operator "you decide"; chose the safe verify-live step). **LLM spend:** $0 (quant-only backtests). **Real compute:** 4 real walk-forward backtests over 2022-01-01..2024-06-30 (6 windows each). **Result:** ready for Q/A — plumbing validated + at-least-one-seed valid metrics; two findings (one fixed, one flagged).

## What was run
`scripts/run_rotation_smoke.py` → `run_rotation_bakeoff(settings, BigQueryClient(settings), seeds=[tb_baseline, qm_trend_tilt], num_param_variants=2, start_date="2022-01-01", end_date="2024-06-30", persist=True)` with a capturing adapter to record per-seed metrics. AUDIT-ONLY (allocation_pct=0, no deploy). Incumbent resolved via load_promoted_params → optimizer_best (promoted_strategies BQ 404 → graceful fallback, as designed).

## THE LIVE SMOKE CAUGHT A REAL BUG (the point of "verify live") — FIXED mid-cycle
First run: BOTH seeds came back dsr=0/sharpe=0, undersized PBO matrix → degenerate. A direct diagnostic backtest (`scripts/diag_rotation_backtest.py`) of triple_barrier over the SAME window was healthy (Sharpe 1.75, 160 trades, 228 nav rows). Root cause: `make_rotation_engine` mapped optimizer_best's `target_annual_vol=0` (which means "vol-targeting DISABLED / standard sizing") onto the trader's `target_vol=0`, and `backtest_trader.py:89` `vol_scale=min(target_vol/stock_vol,3.0)` makes `target_vol=0` ZERO every position → NO trades → flat NAV → degenerate. The $0 mock tests verified the mapping ARITHMETIC but not the trader's target_vol=0 semantics — exactly the gap live validation exists to close.
**Fix (in `backend/autoresearch/rotation_runner.py`):** map ONLY a POSITIVE value; 0/missing/negative → the engine default 0.15 (standard sizing). Corrected the docstring + the 48.3 test that had encoded the buggy assertion. Full rotation regression re-ran 40 passed/2 skipped.

## Verbatim post-fix smoke result (real metrics)
```
[smoke] scored strategy=triple_barrier: {'dsr': 1.0, 'pbo': 0.4887334887334887, 'sharpe': 1.8232784959596307, 'n_variants': 2, 'n_windows': 6}
[smoke] scored strategy=quality_momentum: {'dsr': 0.0, 'sharpe': 0.0, 'n_variants': 2, 'n_windows': 6}  # degenerate -> pbo omitted -> skipped
verdict: {selected_id: triple_barrier, switched: false, reason: 'no_candidate_passed_gate', ranked: [], incumbent_id: triple_barrier, num_trials: 2}
```
rotation_log.jsonl (last row): `{"selected_id":"triple_barrier","reason":"no_candidate_passed_gate","allocation_pct":0.0,"status":"bakeoff_verdict","num_param_variants":2,"window":"2022-01-01..2024-06-30",...}`

**Why `no_candidate_passed_gate` is CORRECT (not a failure):** triple_barrier passed DSR (1.0>=0.95) but its PBO **0.489 FAILS the strict pbo<=0.20 gate** (the gate working — K=2 gives a coarse, appropriately-skeptical overfit estimate; a real bake-off uses K~8-16); quality_momentum was skipped (degenerate). So no candidate passed → retain the incumbent. The research explicitly blessed `no_candidate_passed_gate` as a valid plumbing-proven outcome.

## Success-criteria mapping (masterplan phase-48.4)
1. ran the real engine, 2 seeds x 2 variants, 2022-2024 (6 windows), audit-only, $0 LLM — **MET**.
2. FINITE valid per-strategy metrics "for at least the seeds that complete": **triple_barrier dsr=1.0 (in [0,1]), pbo=0.489 (a REAL non-degenerate value from a genuine T>=32 matrix), sharpe=1.82 (finite), n_windows=6** — **MET** (quality_momentum is degenerate/skipped — a flagged finding, below).
3. selector returned a verdict (no_candidate_passed_gate — valid); exactly one rotation_log row at allocation_pct=0 matching the verdict — **MET**.
4. zero deploy side-effects (no promoted_strategies MERGE, no settings.paper_* mutation, no live order); real metrics + verdict + row recorded here + in live_check_48.4.md — **MET**.

## FINDINGS (the live smoke's value)
1. **[FIXED this cycle]** target_vol=0 no-trade bug in make_rotation_engine (above).
2. **[FLAGGED follow-up]** `quality_momentum` produces NO trades on 2022-2024 (Sharpe 0, undersized nav) while triple_barrier trades fine → qm's labeling/screening yields no BUY signals on this window. The rotation chain handled it correctly (guard fired → skipped, no false-good pbo=0.0). Needs a qm-strategy investigation (separate cycle) before qm is a useful rotation candidate.
3. **[FLAGGED]** With the target_vol fix, tb_baseline + tb_risk_managed both vol-target at 0.15 → they differ ONLY by tp_pct → the seed set is ~3 effective (reseed follow-up, compounding the inert-trailing finding).
4. **[NOTE]** K=2 PBO is coarse (pbo 0.489); the real bake-off should use K~8-16 for a stable PBO + the gate to be meaningful.

## Files
scripts/run_rotation_smoke.py, scripts/diag_rotation_backtest.py, backend/autoresearch/rotation_runner.py (target_vol fix), tests/autoresearch/test_phase_48_3_rotation_runner.py (corrected assertion), backend/backtest/experiments/rotation_log.jsonl (verdict row), .claude/masterplan.json (48.4), handoff/current/{contract.md, research_brief_phase_48_4_live_smoke.md}.
