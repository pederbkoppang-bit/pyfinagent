# Contract — phase-48.2: Rotation real-engine adapter (make_engine_backtest_fn)

**Cycle:** 13 (Priority 5 follow-on #1 — replace the 48.1 producer's injected `backtest_fn` with the real BacktestEngine). **Operator approved continuing past the 12-cycle budget.** **LLM spend:** $0 (mock-tested wiring; the REAL generate_report+compute_pbo run on a hand-built fake — pure numpy, no backtest/BQ/LLM/macro).

## Research-gate summary
3-agent Workflow `wf_2ab3cff3-74f` → `handoff/current/research_brief_phase_48_2_rotation_adapter.md`. Gate **PASSED**: 6 external sources read in full (Bailey/Borwein/LdP/Zhu PBO+CSCV Algo 2.3, Bailey&LdP DSR, Wikipedia DSR, walk-forward 2025, 2× CPCV recipes) + recency scan; exact engine/analytics internals audit; design synthesis.

Decisive findings:
- **Per-strategy PBO crux resolved:** a single backtest's WINDOWS cannot be CSCV columns (Bailey Algo 2.3 — columns = competing CONFIGURATIONS; one series → `compute_pbo` returns 0.0 = false gate-pass). Chosen textbook-exact: **per-strategy K-variant param grid** → (T×K) daily-returns matrix from `nav_history` → existing `compute_pbo`. CPCV multi-path is the deferred next-cycle upgrade.
- `generate_report(...)["analytics"]` gives `deflated_sharpe` (=DSR) + `sharpe` but **NOT pbo** → adapter computes pbo separately.
- **Load-bearing guard:** `compute_pbo` silently returns 0.0 when N<2 OR T<32 → PASSES the `pbo<=0.20` gate → the adapter must emit NO `pbo` on an undersized matrix so the producer SKIPS (never a fake-good 0.0).
- DSR `num_trials` = K (grid size); plain count over-deflates (SAFE); effective-N deferred. Needs ≥2 windows for real V[SR_n].
- Validate `strategy` against `STRATEGY_REGISTRY` (reject→skip, never silent triple_barrier). Warm-cache: `run_backtest(skip_cache_clear=True)` per variant + ONE `cache.clear_cache()`; macro preload is inside run_backtest.

## Hypothesis
A factory `make_engine_backtest_fn(engine_factory, ...)` that, per strategy, runs K param-variants through a warm engine, assembles a guarded (T×K) PBO matrix, and reads DSR/Sharpe from `generate_report`, exactly fills the 48.1 producer's injected `backtest_fn` boundary — moving rotation from foundation toward live, verifiable at $0 by mocking only `engine.run_backtest` (the real pure-numpy `generate_report`+`compute_pbo` run on fakes). The slow live bake-off (32+ real backtests) defers behind an opt-in test.

## Immutable success criteria (verbatim from .claude/masterplan.json phase-48.2)
1. strategy_backtest_adapter.py: make_engine_backtest_fn(engine_factory, *, num_param_variants, param_grid_fn, num_trials, pbo_S, min_pbo_rows, logger) returns a backtest_fn(params)->{dsr,pbo,sharpe,...} matching the 48.1 producer's BacktestFn boundary; dsr from generate_report(...)['analytics']['deflated_sharpe'], pbo from a separate compute_pbo on a per-strategy (T x K) daily-returns matrix assembled from K param-grid variants' nav_history; pure helpers _daily_returns_from_nav / _default_param_grid / _assemble_pbo_matrix / _extract_dsr_sharpe
2. LOAD-BEARING undersize guard: _assemble_pbo_matrix returns None when N<2 OR T<min_pbo_rows(32), and the adapter then emits a dict WITHOUT a pbo key (NOT pbo=0.0) so build_per_strategy_candidates SKIPS that strategy -- preventing compute_pbo's silent 0.0 from false-passing the pbo<=0.20 gate; a test asserts this directly
3. strategy-name validated against backtest_engine.STRATEGY_REGISTRY before running (unknown -> raise -> producer skips; NO silent triple_barrier fallback); warm-cache discipline (run_backtest(skip_cache_clear=True) per variant, cache.clear_cache once in finally); adapter imports NO settings/bq (engine_factory closes over them) so it is $0-mockable and free of an import cycle
4. $0 mock test: mock engine.run_backtest -> a REAL hand-built BacktestResult (2-3 real WindowResult + ~40 synthetic nav rows) and run the REAL generate_report + compute_pbo; assert dsr/sharpe extraction, a hand-built (T>=32,N>=2) compute_pbo float, the undersize-guard skip path, strategy-name reject, and end-to-end registry->adapter->producer->select_best_strategy; the live multi-minute integration test is @pytest.mark.skip (opt-in); ast clean; pytest green

## Plan steps
1. `backend/autoresearch/strategy_backtest_adapter.py`: helpers `_daily_returns_from_nav(nav_history)` (mirror analytics.py:553-554), `_default_param_grid(seed_params, K)` (validate strategy vs STRATEGY_REGISTRY first; jitter risk knobs keeping strategy categorical), `_assemble_pbo_matrix(results, min_rows)` (truncate to shortest common length, drop empties, return None if N<2 or T<min_rows), `_extract_dsr_sharpe(seed_result, num_trials)` (wrap generate_report). Factory `make_engine_backtest_fn(...)` returning the closure; per-variant try/except; `cache.clear_cache()` in finally; on None matrix emit `{dsr,sharpe}` WITHOUT pbo.
2. `tests/autoresearch/test_phase_48_2_backtest_adapter.py` (mock-only, $0): `_make_fake_result` (real BacktestResult + 2-3 real WindowResult + ~40 nav rows); tests for dsr/sharpe extraction, hand-built (40,4) compute_pbo float, the undersize-guard→skip path, strategy-name reject, cache.clear_cache-once, and end-to-end registry→adapter→producer→selector with a mocked engine_factory; one `@pytest.mark.skip` opt-in live integration test.
3. Verify: `ast.parse` + `pytest tests/autoresearch/test_phase_48_2_backtest_adapter.py -q`.

## Out-of-scope (DEFERRED, documented in docstring + masterplan)
- The LIVE multi-run bake-off (4 seeds × K≈8 = ~32 real backtests, tens of minutes) — gated behind the `@pytest.mark.skip` opt-in + a future live-run cycle whose live_check is real per-seed {dsr,pbo,sharpe}.
- CPCV multi-path PBO upgrade (`cpcv_folds` exists at gate.py:42); the weekly cron; the params→settings.paper_* deployment bridge; effective-N (ONC) clustering; a true date-keyed matrix join.
- **Flagged risk (not blocking the mock slice):** the vanilla `run_harness.make_engine` factory threads only a SUBSET of kwargs (no target_vol/trailing/blend) → a live run with it would silently ignore `tb_risk_managed`'s risk overrides. The adapter is factory-agnostic; extending the factory is the caller's job (documented).

## References
- `handoff/current/research_brief_phase_48_2_rotation_adapter.md` (+ workflow `wf_2ab3cff3-74f`)
- `backend/autoresearch/strategy_candidate_producer.py` (48.1 — the consumer + `BacktestFn` boundary) + `strategy_registry.py`
- `backend/backtest/analytics.py` (`generate_report:536`, `compute_pbo:184`, `compute_deflated_sharpe:239`); `backend/backtest/backtest_engine.py` (`STRATEGY_REGISTRY:32`, `WindowResult:89`, `BacktestResult:110`); `scripts/harness/run_harness.py` (make_engine + DSR precedent)
