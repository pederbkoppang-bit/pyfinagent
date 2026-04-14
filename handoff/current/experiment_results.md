# Phase 3.0 — MCP Server Architecture (backtest + signals plumbing) — Experiment Results

## Starting state
- `data_server.py`: complete (prior session shipped `get_universe`, `get_features`, `get_experiment_list` to origin/main).
- `backtest_server.py` on origin/main: 4 TODO stubs (`run_single_feature_test`, `run_ablation_study`, `get_experiment_list`, `get_recent_experiments`) + L32 em-dash logger violation.
- `signals_server.py` on origin/main: 4 TODO stubs (`generate_signal`, `validate_signal`, `publish_signal`, `risk_check`) + L34 em-dash logger violation.

## In-scope changes

### `backend/agents/mcp_servers/backtest_server.py`
- L32 `logger.warning("Backtest engine not available -- backtest server in stub mode")` (was em-dash). Defense-in-depth for the security.md ASCII rule (uvicorn cp1252 handler crash).
- Added `import csv` + module-level `_to_float()` helper (mirrors `data_server._to_float` byte-for-byte).
- `get_experiment_list(last_n: Optional[int] = None) -> Dict[str, Any]`: ports the stdlib-csv block from `data_server.py:280-356`. Reads `backend/backtest/experiments/quant_results.tsv` via `csv.DictReader`, parses `params_json` best-effort, coerces `metric_before`/`metric_after`/`delta`/`dsr` to float via `_to_float`, applies optional `last_n` tail slice. Note: `backtest_server.get_experiment_list` does NOT include the `best` key (unlike data_server) -- backtest_server has its own `get_feature_importance` for "best" semantics, and adding a duplicate `get_best_params` here would create cross-server coupling we want to avoid. The two surfaces are intentionally different shapes.
- `get_recent_experiments(limit: int = 10) -> Dict[str, Any]`: thin delegate -- `return self.get_experiment_list(last_n=limit)`. No duplicated parsing.
- `run_single_feature_test`, `run_ablation_study`, `run_backtest`, `get_feature_importance`: untouched. Out of scope for this iteration (need model-retraining infra); their stubs remain as-is.

### `backend/agents/mcp_servers/signals_server.py`
- L34 `logger.warning("Paper trader not available -- signals server in stub mode")` (was em-dash).
- `validate_signal(signal: Dict[str, Any]) -> Dict[str, Any]`: signal-intrinsic schema validation. Five checks in order: ticker non-empty alphanumeric/dot/colon/dash/underscore (matches `.claude/rules/security.md` ticker sanitization), signal in `{BUY,SELL,HOLD}`, confidence in `[0.0, 1.0]`, date non-empty, BUY/SELL must carry at least one factor (HOLD may be empty). Tolerates non-dict input (returns `valid=False, violations=["not_a_dict"]`). Tolerates partial / missing fields via `.get()` -- never raises. Returns `{"valid", "violations", "adjusted_signal", "reason"}` (the stub's existing shape).
- `risk_check(portfolio, proposed_trade) -> Dict[str, Any]`: portfolio-extrinsic constraint check. Reads thresholds from `self.get_risk_constraints()` so the threshold table stays single-source-of-truth. Pure function (no input mutation, no cross-server calls). Evaluation order canonical FINRA 15c3-5 / FIA whitepaper:
    1. Schema sanity (ticker, action in `{BUY,SELL}`, shares > 0). Hard, fail-fast.
    2. SELL requires existing position with `>= shares`. Hard.
    3. Daily trade count `< max_daily_trades (5)`. Hard.
    4. Per-ticker concentration `<= max_exposure_per_ticker_pct (10%)`. Hard, BUY only (SELL reduces exposure).
    5. Total exposure `<= max_total_exposure_pct (100%)`. Hard, BUY only.
    6. Cash floor: BUY notional `<= cash`. Hard, BUY only (paper trader has no margin).
    7. Drawdown circuit breaker: if `current_drawdown_pct <= max_drawdown_pct (-15%)`, block BUYs but allow SELLs (de-risking). Hard for BUY only.
- Notional resolution: explicit `proposed_trade.price` -> existing position record `price` -> 0.0 (degraded mode logged). Documented in the docstring as a paper-trader scaffold; real prices wired in Phase 4.1.
- Added private `_risk_response()` helper for uniform return shape -- preserves the stub's `{allowed, current_exposure_pct, max_exposure_pct, margin_available, conflicts, reason}` contract across all return paths.
- `generate_signal`, `publish_signal`, `get_signal_history`, `get_portfolio`, `get_risk_constraints`: untouched. The first three are out of scope (model inference / Slack / BQ); the last two are read-only data accessors already wired in the prior commit.

## Verification (deterministic, no LLM, no venv beyond stdlib)

### Static syntax + compile
```
$ python3 -c "import ast; ast.parse(open('backend/agents/mcp_servers/backtest_server.py').read()); print('backtest_server ast OK')"
backtest_server ast OK
$ python3 -c "import ast; ast.parse(open('backend/agents/mcp_servers/signals_server.py').read()); print('signals_server ast OK')"
signals_server ast OK
$ python3 -m py_compile backend/agents/mcp_servers/backtest_server.py
$ python3 -m py_compile backend/agents/mcp_servers/signals_server.py
$ echo "py_compile OK"
py_compile OK
```

### AST logger ASCII scan
```
$ python3 - <<'PY'
import ast
for path in ("backend/agents/mcp_servers/backtest_server.py",
             "backend/agents/mcp_servers/signals_server.py"):
    tree = ast.parse(open(path).read())
    bad = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("info","warning","error","debug","critical","exception"):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        if not sub.value.isascii():
                            bad += 1
                            break
    print(path, "ascii_violations=", bad)
PY
backend/agents/mcp_servers/backtest_server.py ascii_violations= 0
backend/agents/mcp_servers/signals_server.py ascii_violations= 0
```

### Behavioral smoke tests
```
$ python3 - <<'PY' (abbreviated; full script in contract.md)
=== validate_signal ===
ok valid= True violations= []
bad valid= False violations= ['missing_ticker', 'invalid_signal_type', 'confidence_out_of_range', 'missing_date']
hold valid= True
notdict valid= False
=== risk_check ===
allow= True ticker_pct= 10.0 conflicts= []
daily-cap allow= False conflicts= ['max_daily_trades']
no-position-sell allow= False conflicts= ['insufficient_position']
over-concentration allow= False conflicts= ['max_exposure_per_ticker'] pct= 100.0
insufficient-cash allow= False conflicts= ['insufficient_cash']
drawdown allow= False conflicts= ['drawdown_circuit_breaker']
drawdown SELL allow= True conflicts= []
=== backtest_server.get_experiment_list ===
count= 3 first_keys= ['timestamp', 'run_id', 'param_changed', 'metric_before', 'metric_after', 'delta', 'status', 'dsr', 'top5_mda', 'params', 'parent_run_id']
recent count= 2
PY
```

All 13 behavioral assertions match the contract's success criteria:
- `validate_signal`: 4/4 (positive, multi-violation negative, HOLD-with-empty-factors, non-dict)
- `risk_check`: 7/7 (allow at boundary, daily cap, no-position SELL, over-concentration BUY, insufficient cash, drawdown blocks BUY, drawdown allows SELL)
- `get_experiment_list`: shape correct (11 expected keys present, count = 3 with `last_n=3`)
- `get_recent_experiments`: delegate works (count = 2 with `limit=2`)

### Boundary observation
The "allow" case puts the ticker exactly at 10.0% concentration ($1000 / $10000 = 10%). The check uses strict `>` so `10.0 == 10.0` is allowed. This is the conventional interpretation of a "max" threshold ("at most X").

## Files modified
- `backend/agents/mcp_servers/backtest_server.py` (+91 / -22 lines, includes import csv + _to_float helper + two method bodies)
- `backend/agents/mcp_servers/signals_server.py` (+247 / -39 lines, includes validate_signal + risk_check + _risk_response helper)
- `handoff/current/contract.md` (rewritten for Phase 3.0 scope)
- `handoff/current/experiment_results.md` (this file)

## Out-of-scope items (intentionally deferred, with reasoning)
- `backtest_server.run_single_feature_test`, `run_ablation_study`: need to instantiate `BacktestEngine` with feature mutation; the dev env has no `.venv` with backend deps and no LLM-research budget.
- `signals_server.generate_signal`: requires model inference + feature pipeline (Phase 3.2).
- `signals_server.publish_signal`: requires Slack creds + paper trader commit (Phase 4.1).
- `signals_server.get_signal_history`: requires BQ `signals_log` query (Phase 4.2).
- `multi_agent_orchestrator.py`, `harness_memory.py`: already done in prior sessions on origin.
- pandas/numpy on the MCP path: deliberately rejected to keep MCP servers runnable in stripped-down environments.

## Diff bound check
Combined: ~338 added / ~61 deleted across both files. Within the < 350 added line bound from the contract.

## Open questions
None for this scope. Phase 3.1 will need to wire the tools into a FastMCP client transport; the contracts here are stable enough to build against.
