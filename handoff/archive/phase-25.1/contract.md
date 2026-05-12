# Sprint Contract — phase-25.1 — Wire stop-loss enforcement

**Cycle:** phase-25 cycle 1 (FIRST P0 implementation)
**Date:** 2026-05-12
**Step ID:** 25.1
**Priority:** P0
**Depends on:** none

## Research-gate
`gate_passed: true` (tier=moderate). 6 sources: arxiv 2604.27150 stop-loss param, Alpaca orders-at-alpaca + paper-trading, TradersPost stop-loss strategies, idempotency patterns, async state machines.

```json
{"tier":"moderate","external_sources_read_in_full":6,"snippet_only_sources":6,"urls_collected":12,"recency_scan_performed":true,"internal_files_inspected":13,"gate_passed":true}
```

## Hypothesis
Adding a Step 5.6 in `autonomous_loop.py` (between mark_to_market and decide_trades) that calls `check_stop_losses()` and executes sells will close the TER no-sell bug without breaking other paths.

**Researcher verdict CONFIRMED:**
- Exact insertion: line 332, right before `# -- Step 6: Decide trades` comment
- `execute_sell(ticker, quantity=None, price=None, reason, signals=None)` — pass None for full-position close + live price
- `check_stop_losses() -> list[str]` returns ticker names; iterate + call `execute_sell` per
- Naturally idempotent: `execute_sell` returns None if `get_position` returns None
- Wrap both calls in `asyncio.to_thread` per existing pattern (line 399 reference)
- Initialize `summary["stop_loss_triggered"] = []` before the loop so key is always present

## Success criteria (verbatim from masterplan step 25.1)

1. `grep_check_stop_losses_in_autonomous_loop_returns_match`
2. `unit_test_position_at_stop_triggers_execute_sell`
3. `summary_includes_stop_loss_triggered_field`

**Verifier:** `source .venv/bin/activate && python3 tests/verify_phase_25_1.py`
**Live-check:** "BQ paper_trades row with reason='stop_loss_trigger' visible after next cycle"

## Plan (NOT read-only — actual code changes)

1. **Implement Step 5.6** in `backend/services/autonomous_loop.py` at line 332 — new block before `# -- Step 6: Decide trades`. Use `await asyncio.to_thread(trader.check_stop_losses)` + per-ticker `execute_sell` with `reason="stop_loss_trigger"`. Append to `summary["stop_loss_triggered"]`. Use logger.warning (ASCII-only) on triggers. Try/except wraps to prevent one bad sell from breaking the rest.
2. **Create verifier** at `tests/verify_phase_25_1.py` — assertions:
   - `summary["steps"].append("stop_loss_enforcement")` literal present in autonomous_loop.py
   - `"stop_loss_trigger"` reason string present
   - `summary["stop_loss_triggered"]` key initialized
   - `trader.check_stop_losses` is called from autonomous_loop
   - Python AST syntax-clean on autonomous_loop.py
3. **Write experiment_results.md** with verbatim verifier output
4. **Spawn Q/A** — confirm 5/5 harness-compliance + verifier PASS + LLM-judgment legs
5. **Append harness_log Cycle 57** with PASS verdict
6. **Write live_check_25.1.md** (placeholder pre-live-cycle)
7. **Flip masterplan 25.1 to done** — auto-commit + push fires

## References

External (read in full):
- https://arxiv.org/html/2604.27150
- https://docs.alpaca.markets/us/docs/orders-at-alpaca
- https://docs.alpaca.markets/us/docs/paper-trading
- https://blog.traderspost.io/article/stop-loss-strategies-algorithmic-trading
- https://thearchitectsnotebook.substack.com/p/advanced-idempotency-in-system-design
- https://python-statemachine.readthedocs.io/en/latest/async.html

Internal anchors:
- `backend/services/autonomous_loop.py:302-330` (Step 5 + 5.5 kill-switch existing pattern)
- `backend/services/autonomous_loop.py:332` (insertion line — Step 6 boundary)
- `backend/services/autonomous_loop.py:396-409` (existing `execute_sell` + `asyncio.to_thread` pattern)
- `backend/services/paper_trader.py:414-423` (`check_stop_losses()` definition — finally a caller)
- `backend/services/paper_trader.py:224-253` (`execute_sell()` signature + idempotency via `get_position`)
- `tests/verify_phase_24_0.py` (model for new verifier)
