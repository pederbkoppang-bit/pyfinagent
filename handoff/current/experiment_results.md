# Experiment Results — phase-48.1: Strategy-rotation foundation (registry + per-strategy DSR/PBO producer)

**Cycle:** 12 (Priority 5 — [MONEY/NORTH-STAR]). **LLM spend:** $0 (config + pure code + fixture tests; no real backtest, no BQ, no LLM, no macro preload). **Result:** ready for Q/A.

## What was built (2 new modules + 2 new test files)
The 47.6 `select_best_strategy` was a pure fn wired NOWHERE; only one strategy config existed and nothing produced its `per_strategy` input. This cycle ships the missing PRODUCER half + a config-driven seed set, wired end-to-end.

1. `backend/autoresearch/strategy_registry.py` — `SEED_STRATEGIES` (4 seeds) + `load_seed_strategies(seeds=None, base_params=None)` + `load_base_params()`. Each seed = `param_overrides` overlaid on `optimizer_best.params`. Seeds span orthogonal strategy-TYPE axes (research finding: diversification from TYPE not param tweaks): `tb_baseline` (incumbent rail, empty overrides), `mr_short_horizon` (mean_reversion + short holding/turnover), `qm_trend_tilt` (quality_momentum + long holding), `tb_risk_managed` (triple_barrier + vol-targeting/trailing/tighter-TP — a deliberately correlated risk-axis variant). Operator-tunable (injected `seeds`) + fail-open (empty base still enumerates ids). Pure, ASCII-only.
2. `backend/autoresearch/strategy_candidate_producer.py` — pure `build_per_strategy_candidates(configs, backtest_fn)` emitting the exact verified selector/gate contract `{strategy,dsr,pbo,params,sharpe}` (id under `strategy`, dsr+pbo mandatory floats), SKIPPING+warning on `backtest_fn` raise / non-dict / missing-or-non-numeric dsr|pbo (so the gate never silently drops a malformed candidate). `run_strategy_bakeoff(backtest_fn, incumbent=None, *, seeds, base_params, num_trials)` = registry → producer → `select_best_strategy`. `backtest_fn` is the ONLY injected dependency (no engine/BQ/LLM import).
3. Tests `tests/autoresearch/test_phase_48_1_strategy_registry.py` (8) + `test_phase_48_1_candidate_producer.py` (7).

## Verbatim verification output (immutable command)
```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/autoresearch/strategy_registry.py','backend/autoresearch/strategy_candidate_producer.py']]; print('ast OK 2 files')"
ast OK 2 files
$ python -m pytest tests/autoresearch/test_phase_48_1_strategy_registry.py tests/autoresearch/test_phase_48_1_candidate_producer.py tests/autoresearch/test_strategy_selector.py -q
.......................                                                  [100%]
23 passed in 0.02s
```
(15 new tests + the 8 existing `test_strategy_selector.py` — no selector-contract regression.)

Import + end-to-end spine smoke ($0, real-base registry off the live optimizer_best.json):
```
real-base seeds: ['tb_baseline', 'mr_short_horizon', 'qm_trend_tilt', 'tb_risk_managed']
verdict: {'selected_id': 'mr_short_horizon', 'switched': True, 'reason': 'first_selection',
          'ranked': ['mr_short_horizon', ...], 'num_trials': 4}
imports OK, no cycle
```

## Success-criteria mapping (masterplan phase-48.1)
1. registry >=4 distinct seeds, params=base+overrides, >=3 strategy types (mr+qm+tb), operator-tunable, fail-open — **MET** (test_seed_set_has_at_least_four_distinct_ids, test_seeds_span_at_least_three_strategy_types, test_param_overrides_apply_on_top_of_base, test_operator_tunable_injected_seeds, test_fail_open_empty_base_still_enumerates_ids).
2. producer pure + exact contract + SKIPS malformed (raise / missing dsr|pbo / non-numeric) — **MET** (test_producer_emits_exact_selector_contract + 3 skip-guard tests; backtest_fn the only dependency, no engine/BQ/LLM import).
3. registry→producer→selector composes (first_selection top-DSR passer; gate-veto; anti-churn retain) — **MET** (test_bakeoff_first_selection_picks_top_dsr_gate_passer [tb_risk_managed vetoed], test_bakeoff_anti_churn_retains_incumbent_below_min_improvement, test_bakeoff_switches_on_material_improvement).
4. deferred work documented in both docstrings; ast clean; new pytest green; existing selector test green — **MET** (DEFERRED blocks in both modules; ast OK 2 files; 23 passed incl. the 8 existing).

## Scope honesty / DEFERRED (documented in both module docstrings + masterplan + contract)
$0 fixture-based slice — a test PASS does NOT prove live DSR/PBO are computable (the `backtest_fn` is injected). The `backtest_fn` OUT shape is a deliberate strict SUBSET of the verified `analytics.generate_report()["analytics"]` + `compute_pbo` so the next cycle's real-engine adapter is a drop-in. **DEFERRED (later cycles):** (a) the real BacktestEngine adapter (warm-cache `run_backtest` loop → `nav_history` daily_returns → `generate_report` DSR + per-strategy (T×K) `compute_pbo`); (b) the weekly rotation cron; (c) the deployment switch + the **params→settings.paper_* bridge** (deploy audit headline: `best_params` is NOT threaded into `decide_trades`/`paper_trader`, so flipping a `promoted_strategies` row alone changes only the heartbeat, not live orders); (d) effective-N clustering (plain `num_trials=N` over-deflates DSR for correlated seeds — the SAFE direction). No live rotation is implied by this cycle.

## Files
backend/autoresearch/strategy_registry.py, backend/autoresearch/strategy_candidate_producer.py, tests/autoresearch/test_phase_48_1_strategy_registry.py, tests/autoresearch/test_phase_48_1_candidate_producer.py, .claude/masterplan.json (phase-48.1), handoff/current/{contract.md, research_brief_phase_48_1_rotation_foundation.md}.
