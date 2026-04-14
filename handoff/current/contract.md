# Phase 4.3 Risk Management — Contract

**Step:** 4.3 Risk Management (code-only subset)
**File under test:** `backend/agents/mcp_servers/signals_server.py`
**Harness-required:** true (but harness cannot run in remote env — verify via AST + deterministic behavioral assertions)
**Research gate:** PASS — see `handoff/current/research.md` (28 URLs, 7 categories, 5 full-reads)

## Hypothesis

The current signals_server has a naive v1 position sizing (`cash * 0.05 cap $1000`) inlined in `publish_signal`, no stop-loss monitoring, and no trailing drawdown tracker. Phase 4.3 upgrades these three concerns with **pure, deterministic, stdlib-only** functions that (a) replace the inline sizing with a research-backed hybrid lite-formula, (b) add `check_stop_loss(portfolio)` to detect per-position and trailing stop breaches, and (c) add `track_drawdown(portfolio)` to compute the equity-curve drawdown tier and fire a kill-switch flag.

## Success criteria (all must hold)

1. `size_position(signal, portfolio)` is a new public method that returns a `float` USD amount, never raises, and is the minimum of up to three independent caps:
   - (a) hard percent-of-equity cap: `min(equity * max_position_pct/100, max_position_usd)`
   - (b) confidence-weighted half-Kelly arm: `0.5 * confidence * equity` when no `mu_hat`/`var_hat` provided (degrades to flat confidence-scaled)
   - (c) inverse-vol arm: `target_vol_pct/100 * equity / annualized_vol` when `annualized_vol` is provided, else skipped
   - Returns 0.0 for non-BUY actions and when equity <= 0.
2. `check_stop_loss(portfolio)` is a new public method that returns a `list[dict]` of positions to exit. Each entry has keys: `ticker`, `reason` ("fixed_stop"|"trailing_stop"), `entry_price`, `current_price`, `peak_price`, `loss_pct`. Only positions with `(current - entry)/entry <= -stop_loss_pct/100` OR `(current - peak)/peak <= -trail_stop_pct/100` are returned. Pure function over the portfolio snapshot; never mutates input; never raises.
3. `track_drawdown(portfolio)` is a new public method that updates the instance's `_peak_equity` state and returns a dict with keys: `peak`, `equity`, `drawdown_pct`, `tier` ("ok"|"warning"|"derisk"|"kill"), `kill_switch`. Tiers are computed from the `get_risk_constraints()` thresholds. The `_peak_equity` attribute is initialised lazily on first call. Never raises.
4. `get_risk_constraints()` is extended with 6 new keys: `max_position_pct` (5.0), `max_position_usd` (1000.0), `stop_loss_pct` (8.0), `trail_stop_pct` (3.0), `drawdown_warning_pct` (-5.0), `drawdown_derisk_pct` (-10.0). The existing `max_drawdown_pct` (-15.0) stays as the kill-switch. No breaking changes to existing keys.
5. `publish_signal` step 5 calls `size_position` to derive `amount_usd` instead of the inline `min(cash*0.05, 1000.0)`. Explicit `signal["size_usd"]` still overrides (preserves the 4.1 contract). Downstream code paths (risk_check, execute_buy) are unchanged.
6. All additions are stdlib-only. NO imports of pandas, numpy, backend modules, data_server, or backtest_server. No cross-server coupling. No LLM calls.
7. Diff bound: added lines **< 300** net. Unchanged outside the scoped methods + extended constants dict + one publish_signal call site. No edits to backtest_server, data_server, paper_trader, portfolio_manager.
8. Python syntax: `ast.parse()` clean + `py_compile` clean.
9. Security: logger ASCII rule holds (no non-ASCII chars in any new `logger.*()` call).
10. Deterministic behavioral assertions (QA evaluator must run all of these):
    - **size_position_a1**: zero equity → 0.0
    - **size_position_a2**: equity=10000, confidence=1.0, BUY → hard cap = min(500, 1000) = 500.0
    - **size_position_a3**: equity=10000, confidence=0.4, BUY → half-Kelly arm = 0.5 * 0.4 * 10000 = 2000; bound by hard cap → 500.0
    - **size_position_a4**: equity=100000, confidence=0.8, annualized_vol=0.20, target_vol_pct=10.0, BUY → inverse-vol arm = 0.10 * 100000 / 0.20 = 50000; hard cap = min(5000, 1000) = 1000.0
    - **size_position_a5**: action=HOLD → 0.0
    - **size_position_a6**: non-dict signal → 0.0 (no raise)
    - **check_stop_loss_a1**: empty positions → []
    - **check_stop_loss_a2**: position with entry=100, current=91 → fixed_stop (9% loss > 8%)
    - **check_stop_loss_a3**: position with entry=100, current=93 → [] (7% < 8%)
    - **check_stop_loss_a4**: position with entry=100, peak=120, current=116 → trailing_stop (3.33% off peak > 3%)
    - **check_stop_loss_a5**: non-dict portfolio → [] (no raise)
    - **track_drawdown_a1**: first call with equity=10000 → peak=10000, dd=0.0, tier=ok
    - **track_drawdown_a2**: second call with equity=9500 → peak=10000, dd=-5.0, tier=warning
    - **track_drawdown_a3**: third call with equity=8900 → peak=10000, dd=-11.0, tier=derisk
    - **track_drawdown_a4**: fourth call with equity=8400 → peak=10000, dd=-16.0, tier=kill, kill_switch=True
    - **track_drawdown_a5**: fifth call with equity=11000 → peak=11000 (new high), dd=0.0, tier=ok
    - **constraints_a1**: `get_risk_constraints()` returns all 6 new keys with the documented defaults
    - **publish_signal_a1**: stub mode with signal carrying no size_usd — flows through size_position, still returns backend_unavailable (pipeline contract preserved)
    - **ast_logger_ascii**: 0 non-ASCII chars in any logger call site

## Anti-leniency rules

1. **Don't add new imports.** Stdlib only. If tempted to `import statistics` or `import math`, hand-roll the formula.
2. **Don't touch `risk_check`.** It's the 3.0 deliverable and passed a prior QA cycle. Leaving it alone.
3. **Don't touch `validate_signal`.** Same.
4. **Don't touch paper_trader.py, portfolio_manager.py, backtest_server.py, data_server.py.** Scope is exactly one file.
5. **Don't add BQ persistence.** `_peak_equity` is in-memory only this session. Durable state is Phase 4.2.
6. **Don't fake the math.** If confidence or annualized_vol is None/invalid, skip that arm of the min() rather than substituting 0 (which would collapse the whole min to 0).
7. **Don't return None from any new method.** Always return the documented shape (float, list, dict). Error paths return empty variants, never None.
8. **Don't call LLMs.** This is deterministic plumbing.
9. **Don't mutate input dicts.** Portfolio and signal are read-only.
10. **Preserve publish_signal's 9-step contract.** The sizing change is localised to step 5; steps 1-4 and 6-9 are untouched.

## Out of scope (deferred with reasoning)

| Item | Why deferred | Phase |
|---|---|---|
| BQ persistence of `_peak_equity` across restarts | Needs BQ schema migration | 4.2 |
| Real ATR / sigma computation from historical prices | Needs data_server access, breaks stdlib invariant | 3.2 follow-up |
| Emitting actual liquidating orders on stop-loss trigger | Needs paper_trader.execute_sell wiring + caller orchestration | 4.3 follow-up |
| Consecutive-stop counter (portfolio-wide pause) | Needs persistent state across publish cycles | 4.3 follow-up |
| Event calendar integration (earnings / FOMC reduction) | Needs earnings_tone.py + external calendar | 4.3.3 |
| Sector exposure cap (30%) | Needs sector mapping source, not in signal dict today | 4.3.1 follow-up |
| Kelly with real mu_hat/var_hat from backtest | Needs backtest result query, breaks stdlib invariant | 3.2 follow-up |

## Files to change

- `backend/agents/mcp_servers/signals_server.py` — only this file.

## Verification commands

```bash
python3 -c "import ast; ast.parse(open('backend/agents/mcp_servers/signals_server.py').read())"
python3 -m py_compile backend/agents/mcp_servers/signals_server.py

# Logger ASCII guard
python3 - <<'PY'
import ast
tree = ast.parse(open('backend/agents/mcp_servers/signals_server.py').read())
bad = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in ("info","warning","error","debug","critical","exception"):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    if not sub.value.isascii():
                        bad += 1
print("non-ascii logger:", bad)
PY
```

## Post-conditions (what the LOG phase must write)

- `handoff/current/experiment_results.md` — the 18 deterministic assertions + results
- `handoff/current/evaluator_critique.md` — QA verdict (ok/reason/scores, independent run)
- `.claude/context/sessions/2026-04-14-NNNN.md` — episodic session log
