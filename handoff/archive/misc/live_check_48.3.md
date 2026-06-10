# Live Check — phase-48.3: Live rotation runner + full-kwarg engine_factory

`verification.live_check` = "n/a -- deterministic $0 tests (monkeypatched BacktestEngine + injected stub
factory/adapter + tmp_path persistence; no real backtest/BQ/LLM)." Verbatim output, 2026-05-29.

## Immutable command (ast + 48.3 test) -- exit 0
```
$ python -c "import ast; ast.parse(open('backend/autoresearch/rotation_runner.py').read()); print('ast OK')"
ast OK
$ python -m pytest tests/autoresearch/test_phase_48_3_rotation_runner.py -q
........s                                                                [100%]
8 passed, 1 skipped in 3.25s
```
Full rotation regression (47.6 + 48.1 + 48.2 + 48.3): 40 passed, 2 skipped. No import cycle.

## The kwarg-gap fix + target_vol revival (verified at source)
make_rotation_engine threads the 8 kwargs make_engine drops; test asserts the ctor receives
market/train_window_months/test_window_months/embargo_days/starting_capital/commission_model/target_vol.
target_annual_vol -> target_vol mapping verified:
```
{target_annual_vol: 0.15} -> ctor target_vol == 0.15   (vol-targeting ON; tb_risk_managed)
{target_annual_vol: 0}    -> ctor target_vol == 0      (OFF; tb_baseline)
{target_vol: 0.2, target_annual_vol: 0.1} -> 0.2       (explicit wins)
```
So tb_risk_managed IS vol-differentiated from tb_baseline (the trailing-stop half stays inert -- WARNed).

## Dead-key honesty (verified)
make_rotation_engine WARNs "inert risk keys [...trailing_stop_enabled...]" on a seed carrying
trailing_* (caplog asserted) and does NOT write them to _strategy_params (no cargo-cult).

## Audit-only persistence (no deploy)
run_rotation_bakeoff persists ONE JSONL row at allocation_pct=0.0, status="bakeoff_verdict"
(test_seam_A: row matches the verdict's selected_id). persist=False writes nothing; a raising bq_fn is
swallowed (fail-open). NO promoted_strategies MERGE, NO settings.paper_* mutation -- the deployment bridge
is the next cycle. git-diff: zero edits to autonomous_loop/portfolio_manager/paper_trader/decide_trades.

## Incumbent resolution (verified)
_resolve_incumbent reads load_promoted_params (mocked) -> {strategy_id, strategy, params, dsr} with dsr
from optimizer_best.json; empty params -> None (selector does first_selection). NOTE (documented): the
incumbent is keyed by strategy NAME, which may not equal a seed ID -- selector treats the best seed as a
challenger to the incumbent's recorded DSR (incumbent->seed-id mapping is a follow-up).

## Deferred (NOT live this cycle)
The mock proves WIRING. The LIVE ~32-backtest bake-off is @pytest.mark.skip opt-in (its live_check =
real per-seed {dsr,pbo,sharpe} + the persisted rotation_log row, a future live-run cycle). Also deferred:
the weekly cron; the deployment params->settings.paper_* bridge (the keystone); re-enabling the reverted
trailing/vol-target readers; effective-N; CPCV. Seed-set follow-up: tb_risk_managed's trailing half is
inert engine-wide (9fbd9cd6 revert) -> ~3.5 distinct seeds until the readers are restored.
