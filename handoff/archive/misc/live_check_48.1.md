# Live Check — phase-48.1: Strategy-rotation foundation

`verification.live_check` = "n/a -- deterministic fixture-based unit test ($0: no real backtest, no BQ,
no LLM, no macro preload)." Verbatim output, 2026-05-29.

## Immutable command (ast 2 files + pytest incl. selector regression) -- exit 0
```
$ python -c "import ast; [ast.parse(open(f).read()) for f in ['backend/autoresearch/strategy_registry.py','backend/autoresearch/strategy_candidate_producer.py']]; print('ast OK 2 files')"
ast OK 2 files
$ python -m pytest tests/autoresearch/test_phase_48_1_strategy_registry.py tests/autoresearch/test_phase_48_1_candidate_producer.py tests/autoresearch/test_strategy_selector.py -q
.......................                                                  [100%]
23 passed in 0.02s
```
15 new tests + the 8 existing `test_strategy_selector.py` (no selector-contract regression).

## The spine composes end-to-end (import smoke, real-base registry, $0)
```
real-base seeds: ['tb_baseline', 'mr_short_horizon', 'qm_trend_tilt', 'tb_risk_managed']
verdict: {'selected_id': 'mr_short_horizon', 'switched': True, 'reason': 'first_selection', 'num_trials': 4}
imports OK, no cycle
```
registry (off the live optimizer_best.json) -> producer -> select_best_strategy returns a valid verdict.
No import cycle (producer imports selector + registry; neither imports the producer).

## Contract verified against the real code (not just the audit)
- `PromotionGate.evaluate` (gate.py:28) DROPS any candidate missing dsr OR pbo -> producer emits both as
  floats and SKIPS+warns on incomplete metrics (test_producer_skips_when_pbo_missing,
  test_producer_skips_non_numeric_metrics) so a malformed candidate never silently vanishes.
- `select_best_strategy` reads the id under strategy_id|strategy|trial_id -> producer emits the registry
  id under `strategy`; ranks DSR-desc/PBO-asc; anti-churn min_improvement=0.01 (all exercised).

## Deferred (NOT live this cycle -- documented in both module docstrings)
The $0 fixture does NOT prove live DSR/PBO computability. DEFERRED to later cycles: the real
BacktestEngine adapter (warm-cache run_backtest -> nav_history daily_returns -> generate_report DSR +
per-strategy (TxK) compute_pbo), the weekly cron, the deployment switch + the params->settings.paper_*
bridge (best_params is NOT threaded into decide_trades; flipping a promoted_strategies row alone changes
only the heartbeat, not live orders), and effective-N clustering (plain num_trials=N over-deflates -- the
SAFE direction). No live rotation is implied.
