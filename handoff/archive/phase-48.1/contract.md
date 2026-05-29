# Contract — phase-48.1: Strategy-rotation foundation (registry + per-strategy DSR/PBO producer)

**Cycle:** 12 (Priority 5 — [MONEY/NORTH-STAR] dynamic strategy rotation). **LLM spend:** $0 (config + pure code + fixture-based unit tests; no real backtest, no BQ, no LLM, no macro preload). **This is the 12th/final budget cycle** — after it the 12-cycle SOFT STOP fires on count; the real-engine adapter + cron + deployment bridge hand off to the next session.

## Research-gate summary
4-agent Workflow `wf_784c2e77-298` → `handoff/current/research_brief_phase_48_1_rotation_foundation.md`. Gate **PASSED**: 8 external sources read in full (DSR/Bailey-LdP, PBO/CSCV, effective-N, jump-model anti-churn, ensemble practice, IS/WFA/OOS) + recency scan; 2 codebase audits (backtest-engine interface, deployment/param-space); selector/gate contract re-verified directly from code.

Decisive findings:
- **Diversify on orthogonal AXES (strategy TYPE), not param tweaks** → the seed set spans `triple_barrier` / `mean_reversion` / `quality_momentum`, not just risk-knob variants.
- **Selector/gate contract (verified):** `select_best_strategy(per_strategy, incumbent=None, *, gate, min_improvement=0.01, num_trials=5)`; id under `strategy_id|strategy|trial_id`; `PromotionGate` drops any candidate missing `dsr` OR `pbo`, promotes iff `dsr>=0.95 AND pbo<=0.20`. → producer MUST emit both as floats.
- **Engine (for the deferred adapter):** `run_backtest -> BacktestResult.nav_history`; `daily_returns=np.diff(navs)/navs[:-1]`; `generate_report(...)["analytics"]` → DSR/Sharpe; `compute_pbo((T,N) matrix, S=16)`. Slow (minutes) + warm-cache pattern → isolate behind an injected `backtest_fn`.
- **Deploy (shapes a DEFERRED follow-on):** `load_promoted_params` is the switch, but `best_params` is NOT threaded into `decide_trades`/`paper_trader` — live behavior is `settings.paper_*`-driven. A real rotation must bridge params→settings. NOT built this cycle.

## Hypothesis
Shipping the missing PRODUCER half (config-driven seed registry + a pure `build_per_strategy_candidates(configs, backtest_fn)` emitting the exact selector contract, wired end-to-end via `run_strategy_bakeoff`) makes the north-star rotation mechanism fully composable and unit-testable at $0, leaving only the real-engine adapter + cron + deployment bridge as cleanly-scoped follow-ons — with the seed set config-driven so the operator can retune which strategies compete without a rebuild.

## Immutable success criteria (verbatim from .claude/masterplan.json phase-48.1)
1. strategy_registry.py: load_seed_strategies(seeds=None, base_params=None) returns >=4 distinct seeds, each {id, rationale, params} where params = optimizer_best.params overlaid with the seed's param_overrides; seeds span orthogonal axes (>=3 distinct 'strategy' types incl. mean_reversion + quality_momentum + the triple_barrier baseline); operator-tunable (honors an injected seeds list) and fail-open (empty base still enumerates ids)
2. strategy_candidate_producer.py: build_per_strategy_candidates(configs, backtest_fn) is PURE (backtest_fn the only injected dependency, no engine/BQ/LLM import) and emits one {strategy,dsr,pbo,params,sharpe} dict per config matching the verified selector/gate contract; SKIPS+warns (does not emit a partial dict) when backtest_fn raises or omits dsr/pbo so the gate never silently drops a malformed candidate
3. registry->producer->select_best_strategy composes end-to-end via run_strategy_bakeoff(backtest_fn, incumbent=None, seeds=None, num_trials=None); a fixture backtest_fn drives a first_selection (top gate-passer by DSR wins, a DSR<0.95 or PBO>0.20 seed is gate-vetoed) and an incumbent case exercises the anti-churn retain
4. deferred work (real-engine adapter, weekly cron, deployment params->settings bridge, effective-N clustering) is explicitly documented in both module docstrings; ast.parse clean on both new files; new pytest green; the existing tests/autoresearch/test_strategy_selector.py stays green (no selector-contract regression)

## Plan steps
1. `backend/autoresearch/strategy_registry.py`: `SEED_STRATEGIES` (4 entries: `tb_baseline` empty-overrides incumbent rail, `mr_short_horizon` {strategy:mean_reversion, mr_holding_days:8, holding_days:30}, `qm_trend_tilt` {strategy:quality_momentum, holding_days:120}, `tb_risk_managed` {strategy:triple_barrier, target_annual_vol:0.15, trailing_stop_enabled:true, trailing_trigger_pct:5, trailing_distance_pct:3, tp_pct:6}). `load_seed_strategies(seeds=None, base_params=None)` overlays each on optimizer_best.params (read fail-open). ASCII-only.
2. `backend/autoresearch/strategy_candidate_producer.py`: pure `build_per_strategy_candidates(configs, backtest_fn)` → list of `{strategy,dsr,pbo,params,sharpe}`; SKIP+warn on raise/missing-dsr-or-pbo. `run_strategy_bakeoff(backtest_fn, incumbent=None, *, seeds=None, num_trials=None)` = load → produce → `select_best_strategy`.
3. Both module docstrings cite the lit basis + an explicit DEFERRED block (real-engine adapter, weekly cron, deployment params→settings bridge, effective-N clustering).
4. Tests `tests/autoresearch/test_phase_48_1_strategy_registry.py` + `test_phase_48_1_candidate_producer.py`: registry distinctness/axes/tunable/fail-open; producer contract + skip-on-raise + skip-on-missing-pbo guards; end-to-end bakeoff first_selection (mr wins, a vetoed seed) + anti-churn incumbent retain.
5. Verify: ast (2 files) + pytest (2 new + the existing `test_strategy_selector.py` regression).

## Out-of-scope (DEFERRED, documented in docstrings + masterplan)
- Real BacktestEngine adapter (warm-cache `run_backtest` loop → `nav_history` daily_returns → `generate_report` DSR + per-strategy (T×K) `compute_pbo`).
- Weekly rotation cron; the deployment switch + the **params→settings.paper_* bridge** (deploy audit: required for rotation to change live orders, not just the heartbeat).
- Effective-N clustering (v1's plain `num_trials=N` over-deflates — the SAFE direction).

## References
- `handoff/current/research_brief_phase_48_1_rotation_foundation.md` (+ workflow `wf_784c2e77-298`)
- `backend/autoresearch/strategy_selector.py` (47.6, the consumer) + `gate.py` (PromotionGate)
- `backend/backtest/backtest_engine.py` / `analytics.py` (the deferred adapter target); `backend/services/autonomous_loop.py::load_promoted_params` (the deferred switch)
