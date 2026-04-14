# Contract: Phase 4.2.2 Signal Accuracy Tracking

**Step:** Phase 4.2 Paper Trading Evaluation -- signal accuracy tracking subset
**Target:** `backend/agents/mcp_servers/signals_server.py`
**Scope:** 4 new/extended methods + `publish_signal` side-effect -- all pure stdlib
**Research:** `handoff/current/research.md` (17 URLs/7 categories, Gate PASSED)

## Hypothesis

Adding deterministic signal accuracy tracking to `SignalsServer` (in-memory, stdlib-only, 4 methods + a `publish_signal` side-effect that appends to `signal_history`) unblocks the Phase 4.2.2 "Signal Accuracy Tracking" masterplan subset, enables a future Slack weekly accuracy report via `get_accuracy_report()`, and preserves the single-file MCP boundary invariant (no new imports beyond stdlib, no cross-server calls, no BQ, no pandas).

## In-scope changes (exactly 5)

1. **`SignalsServer.__init__`** -- add `self._signals_by_id: Dict[str, Dict] = {}` (O(1) lookup index).
2. **`SignalsServer.publish_signal`** -- add a new Step 9 at the end of the success path: append the signal record (with `signal_id`, `timestamp`, `outcome` placeholder) to `self.signal_history` and mirror into `self._signals_by_id`. Byte-identical to Steps 1-8 on all error/degraded paths.
3. **`SignalsServer.get_signal_history(limit=None, since_date=None)`** -- replace the stub. Return real history with optional tail-limit and ISO date filter.
4. **`SignalsServer.track_signal_accuracy(signal_id, exit_price, exit_date)`** -- NEW. Record an exit, compute signed forward return, classify hit/miss/neutral. Idempotent.
5. **`SignalsServer.get_accuracy_report(group_by=None, neutral_band_pct=0.20)`** -- NEW. Aggregate stats: count, scored_count, hits, misses, hit_rate, Wilson-95 CI, mean/median forward return. Optional groupby 'signal_type' or 'ticker'.

Plus one private helper:
6. **`SignalsServer._wilson_ci(hits, n, z=1.96)`** -- static method, Wilson Score CI for small-sample binomial proportion.

## Out of scope (DEFERRED, with reasoning)

- **IC / Pearson / Spearman correlation** -- needs pandas + N >= 30. Phase 4.2.4.
- **Brier score** -- needs full probability vector over {BUY, SELL, HOLD}, signals lack this shape.
- **Per-factor attribution** -- signals don't carry per-factor weights. Phase 3.2.
- **Per-sector grouping** -- no sector lookup service available in stdlib. Phase 4.2.2 follow-up.
- **Slack weekly report** -- formatter work in `slack_bot/formatters.py`. This contract only exposes `get_accuracy_report()` as the data source.
- **Durable BQ persistence** -- Phase 4.2.4 signals_log table + schema migration.
- **Cross-restart retention of signal_history** -- in-memory only, documented.
- **Modification of any other SignalsServer method** -- `validate_signal`, `risk_check`, `size_position`, `check_stop_loss`, `track_drawdown`, `get_portfolio`, `get_risk_constraints`, `generate_signal`, `_empty_response`, `_signal_id`, `_remember`, `_risk_response` must stay byte-identical.
- **Any new import** beyond what the file already imports, except `statistics` and `math` if not already present.

## Anti-leniency rules

1. **Stdlib only.** No pandas/numpy. Use `math`, `statistics`, `collections`, `datetime`, `hashlib`.
2. **Never raise.** Every public method returns a structured dict or a safe default on every failure path.
3. **No input mutation.** `track_signal_accuracy` never mutates the `exit_price`/`exit_date` arg. `publish_signal`'s append step deepcopies the signal.
4. **Idempotent accuracy tracking.** Second call with same `signal_id` updates in place and returns `{"updated": True}`; never duplicates.
5. **HOLD exclusion from hit rate.** HOLD signals are stored in history but excluded from the scored count. Assert with a test.
6. **Wilson CI bounds.** Must handle n=0, n=1, p=0.0, p=1.0 without NaN or ZeroDivisionError. Return (0.0, 0.0) for n=0.
7. **Byte-identical preservation** of the 12 unchanged public methods + 3 helpers. Verify via source-line diff.
8. **Diff budget: < 400 added lines total, < 100 net new logic lines** (rest is docstrings + research justification comments).
9. **Logger ASCII only.** All new `logger.*()` calls use `--`, `->`, plain ASCII -- per `.claude/rules/security.md`.
10. **No cross-server imports.** No `from backend.agents.mcp_servers.data_server` or `.backtest_server`.
11. **Return-shape invariant.** `get_signal_history` preserves the existing stub's `{"month", "count", "signals"}` keys as a subset of the new return shape.
12. **Hit/miss semantics documented in method docstring** and re-stated in return shape:
    - BUY hit: `forward_return_pct > +neutral_band_pct`
    - BUY miss: `forward_return_pct < -neutral_band_pct`
    - BUY neutral: `|forward_return_pct| <= neutral_band_pct`
    - SELL hit: `forward_return_pct < -neutral_band_pct`
    - SELL miss: `forward_return_pct > +neutral_band_pct`
    - SELL neutral: `|forward_return_pct| <= neutral_band_pct`
    - HOLD: always `scored=False`, unscored.

## Success criteria (for QA evaluator, 22 assertions)

**get_signal_history:**
- SC1: Called on a fresh server returns `{"month": str, "count": 0, "signals": []}` preserving the stub's shape.
- SC2: After two publish_signal successes, returns `count=2` and `signals` list has two entries in insertion order.
- SC3: `limit=1` returns the most recent entry only.
- SC4: `since_date="2099-01-01"` returns 0 entries (future filter).
- SC5: Non-string/invalid `since_date` degrades gracefully (returns unfiltered list, never raises).

**track_signal_accuracy:**
- SC6: Unknown signal_id returns `{"ok": False, "reason": "signal_not_found", "updated": False}`.
- SC7: Known BUY signal with entry $100, exit $110 records `hit=True, forward_return_pct=10.0, outcome="hit"`.
- SC8: Known BUY signal with entry $100, exit $90 records `hit=False, outcome="miss"`.
- SC9: Known BUY signal with entry $100, exit $100.10 (0.10% move) records `outcome="neutral"`, excluded from hit count.
- SC10: Second call with same signal_id returns `{"updated": True}` and does NOT duplicate the history entry.
- SC11: HOLD signal returns `outcome="unscored", scored=False`.
- SC12: Non-dict inputs (None, list, int) return error dict, never raise.

**get_accuracy_report:**
- SC13: Fresh server returns `{total_count: 0, scored_count: 0, hits: 0, misses: 0, hit_rate: 0.0, hit_rate_ci_low: 0.0, hit_rate_ci_high: 0.0, mean_forward_return_pct: 0.0, median_forward_return_pct: 0.0, groups: {}}`.
- SC14: 10 tracked signals with 7 hits, 3 misses return `hit_rate = 0.7`, Wilson CI contains 0.7 and CI width > 0.
- SC15: `group_by="signal_type"` returns a `groups` dict with keys in {"BUY", "SELL", "HOLD"} and each value has the full metric dict.

**Wilson CI helper:**
- SC16: `_wilson_ci(0, 0)` returns `(0.0, 0.0)`.
- SC17: `_wilson_ci(10, 10)` returns `(low, 1.0)` with `low > 0.7`.
- SC18: `_wilson_ci(0, 10)` returns `(0.0, high)` with `high < 0.3`.
- SC19: `_wilson_ci(5, 10)` returns `(low, high)` with `0.22 < low < 0.25` and `0.75 < high < 0.78` (Wilson 95% CI on 5/10 textbook value).

**Preservation:**
- SC20: Source ranges of `_signal_id`, `validate_signal`, `risk_check`, `size_position`, `check_stop_loss`, `track_drawdown`, `get_risk_constraints` unchanged (byte-identical source lines).
- SC21: No new non-stdlib imports (verify via `ast.walk` import scan).
- SC22: AST logger ASCII guard: 0 non-ASCII in any logger.* call on the modified file.

## Verification commands

```bash
python3 -c "import ast; ast.parse(open('backend/agents/mcp_servers/signals_server.py').read())"
python3 -m py_compile backend/agents/mcp_servers/signals_server.py
```

Plus the 22 behavioral assertions above re-run independently by the qa-evaluator subagent.

## Retry budget

Max 3 attempts. If QA fails with specific criteria, address those and resubmit.
