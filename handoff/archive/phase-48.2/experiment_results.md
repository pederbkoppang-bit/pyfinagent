# Experiment Results — phase-48.2: Rotation real-engine adapter (make_engine_backtest_fn)

**Cycle:** 13 (Priority 5 follow-on #1; operator approved continuing past the 12-cycle budget). **LLM spend:** $0 (mocks engine.run_backtest; runs the REAL pure-numpy generate_report + compute_pbo on a hand-built fake; no real backtest/BQ/LLM/macro). **Result:** ready for Q/A.

## What was built (1 new module + 1 new test file)
Replaces the 48.1 producer's INJECTED `backtest_fn` with a real `BacktestEngine`-backed implementation of that exact boundary.

`backend/autoresearch/strategy_backtest_adapter.py`:
- `make_engine_backtest_fn(engine_factory, *, num_param_variants=8, param_grid_fn=None, num_trials=None, pbo_S=16, min_pbo_rows=32, clear_cache_fn=None, log=None) -> backtest_fn(params) -> {dsr, pbo, sharpe, n_variants, n_windows}`.
- Per strategy: validate name vs `STRATEGY_REGISTRY` (raise on unknown — producer skips; NO silent triple_barrier fallback) → build K competing-config variants (`_default_param_grid`, strategy categorical FIXED, risk knob jittered) → `engine_factory(variant).run_backtest(skip_cache_clear=True)` per variant (warm cache; macro preload is inside run_backtest) → DSR/Sharpe from the seed variant's `generate_report(...)["analytics"]` → `_assemble_pbo_matrix` (T×K daily-returns from nav_history) → `compute_pbo`. `clear_cache_fn` called ONCE in finally (lazily imports `cache.clear_cache` if not injected). Per-variant try/except (a bad variant drops a column, not the strategy).
- **LOAD-BEARING guard:** `_assemble_pbo_matrix` returns None when N<2 OR T<min_pbo_rows(32); the adapter then emits a dict WITHOUT `pbo` → the producer SKIPS the strategy (never a fake-good 0.0 that compute_pbo silently returns on an undersized matrix and that would false-pass the pbo<=0.20 gate).
- Imports NO settings/BQ (engine_factory closes over them) → $0-mockable, no import cycle.

`tests/autoresearch/test_phase_48_2_backtest_adapter.py` (mock-only, $0): 9 tests + 1 `@pytest.mark.skip` opt-in live integration test.

## Verbatim verification output (immutable command + regression)
```
$ python -c "import ast; ast.parse(open('backend/autoresearch/strategy_backtest_adapter.py').read()); print('ast OK')"
ast OK
$ python -m pytest tests/autoresearch/test_phase_48_2_backtest_adapter.py -q
.........s                                                               [100%]
9 passed, 1 skipped in 3.12s

# regression (full rotation suite: 48.1 + 48.2 + selector):
$ python -m pytest tests/autoresearch/test_phase_48_1_* tests/autoresearch/test_phase_48_2_* tests/autoresearch/test_strategy_selector.py -q
........................s........                                        [100%]
32 passed, 1 skipped in 2.77s
# imports OK; no cycle (adapter imports analytics+backtest_engine; producer imports registry+selector; neither imports the adapter back)
```

## Success-criteria mapping (masterplan phase-48.2)
1. adapter factory + helpers + producer-boundary shape; dsr from generate_report, pbo from a separate per-strategy (T×K) compute_pbo — **MET** (test_extract_dsr_sharpe_matches_generate_report, test_default_param_grid_*, test_adapter_emits_full_metrics_*; dsr asserted == generate_report's deflated_sharpe).
2. LOAD-BEARING undersize guard → no pbo → producer skips — **MET** (test_assemble_pbo_matrix_guard + test_adapter_undersize_matrix_emits_no_pbo_so_producer_skips: asserts "pbo" not in raw AND build_per_strategy_candidates returns []).
3. strategy-name validated (reject→skip, no silent fallback); warm-cache clear-once; no settings/BQ import / no cycle — **MET** (test_adapter_unknown_strategy_raises_and_producer_skips; test_adapter_emits_full_metrics_and_clears_cache_once asserts clear called exactly once; import-cycle check passed).
4. $0 mock test (real generate_report+compute_pbo on fake), undersize/reject/end-to-end; live test skip; ast+pytest green — **MET** (9 passed + 1 skipped; end-to-end registry→adapter→producer→selector yields a verdict; ast OK).

## Scope honesty / DEFERRED (documented in the module docstring + masterplan + contract)
The $0 mock proves the metric-extraction WIRING, not live DSR/PBO values (engine.run_backtest is mocked). DEFERRED: (a) the LIVE multi-run bake-off (4 seeds × K≈8 = ~32 real backtests, tens of minutes) — gated behind the `@pytest.mark.skip` opt-in test + a future live-run cycle whose live_check is real per-seed {dsr,pbo,sharpe}; (b) CPCV multi-path PBO upgrade (cpcv_folds exists at gate.py:42); (c) the weekly cron; (d) the deployment params→settings.paper_* bridge (best_params is NOT threaded into decide_trades — a row-flip alone changes only the heartbeat); (e) effective-N (ONC) clustering; (f) a true date-keyed matrix join. **Flagged live risk (not blocking the mock slice):** the vanilla `run_harness.make_engine` factory threads only a SUBSET of kwargs (no target_vol/trailing/blend) → a live run with it would silently ignore tb_risk_managed's risk overrides; the adapter is factory-agnostic and the factory-extension is the live caller's job (documented).

## Files
backend/autoresearch/strategy_backtest_adapter.py, tests/autoresearch/test_phase_48_2_backtest_adapter.py, .claude/masterplan.json (phase-48.2), handoff/current/{contract.md, research_brief_phase_48_2_rotation_adapter.md}.
