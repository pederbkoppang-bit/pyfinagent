# Live Check — phase-48.2: Rotation real-engine adapter

`verification.live_check` = "n/a -- deterministic $0 mock test (mocks engine.run_backtest; runs the REAL
pure-numpy generate_report + compute_pbo; no real backtest/BQ/LLM/macro)." Verbatim output, 2026-05-29.

## Immutable command (ast + mock test) -- exit 0
```
$ python -c "import ast; ast.parse(open('backend/autoresearch/strategy_backtest_adapter.py').read()); print('ast OK')"
ast OK
$ python -m pytest tests/autoresearch/test_phase_48_2_backtest_adapter.py -q
.........s                                                               [100%]
9 passed, 1 skipped in 3.12s
```
Full rotation-suite regression (48.1 + 48.2 + selector): 32 passed, 1 skipped. No import cycle.

## The load-bearing guard works (the key correctness control)
```
short-nav adapter (n=10 -> T=9 < 32):  raw = {dsr, sharpe, n_variants, n_windows}   # NO 'pbo' key
  -> build_per_strategy_candidates([...], short_fn) == []   # producer SKIPS, no false-good pbo=0.0
healthy adapter (n=45 -> T=44 >= 32):  raw = {dsr, pbo, sharpe, ...}; pbo in [0,1]
```
This is the single most important control: compute_pbo silently returns 0.0 on an undersized matrix,
which PASSES the pbo<=0.20 gate; the adapter omits pbo instead so the strategy is skipped, not waved
through. Asserted directly in test_adapter_undersize_matrix_emits_no_pbo_so_producer_skips.

## DSR/Sharpe extraction is real (not mocked)
`_extract_dsr_sharpe(fake_result, num_trials=8)` returns the SAME `deflated_sharpe` the REAL
generate_report computes (asserted ==); sharpe == aggregate_sharpe. compute_pbo runs on a real
(T>=32, N>=2) numpy matrix. Both are pure numpy -- the mock only replaces engine.run_backtest.

## Contract + discipline verified
- strategy-name validated vs STRATEGY_REGISTRY: unknown -> raise -> producer skips (no silent
  triple_barrier fallback). test_adapter_unknown_strategy_raises_and_producer_skips.
- warm-cache: clear_cache called EXACTLY once per bake-off. test_..._clears_cache_once.
- end-to-end registry -> adapter -> producer -> select_best_strategy yields a verdict (4 seeds, all valid).
- adapter imports NO settings/BQ (engine_factory closes over them); no import cycle.

## Deferred (NOT live this cycle -- documented in the module docstring)
The mock proves WIRING, not live values. The LIVE multi-run bake-off (4 seeds x K~8 = ~32 real
backtests, tens of minutes even warm) is gated behind the @pytest.mark.skip opt-in test +
a future live-run cycle (its live_check = real per-seed {dsr,pbo,sharpe}). Also deferred: CPCV
multi-path, weekly cron, the params->settings.paper_* deployment bridge, effective-N clustering. Flagged
risk: the vanilla make_engine factory ignores tb_risk_managed's target_vol/trailing overrides on a live
run -- factory-extension is the live caller's job.
