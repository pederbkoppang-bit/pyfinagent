# Experiment Results -- phase-30.6

**Step:** P2: Price-tolerance pre-trade gate in execute_buy.
**Date:** 2026-05-19.
**Mode:** OVERNIGHT. Autonomous loop PAUSED.

## Summary

Added `paper_price_tolerance_pct: float = Field(5.0, ...)` setting and
extended `paper_trader.execute_buy` with a price-tolerance gate that
rejects BUYs when the live fill price diverges from the analysis-time
price by more than the configured tolerance. Default 5% per SEC LULD
Tier 1 band for S&P 500 + Russell 1000 > $3 (the pyfinagent universe).

Also extended `TradeOrder` with `price_at_analysis` field and updated
`autonomous_loop` Step 7 buy-loop to ALWAYS fetch live price for fill
+ pass `price_at_analysis=order.price_at_analysis` separately. This
makes the gate's two inputs (live price + analysis-time price) cleanly
separable.

Closes phase-30.0 cross-val 6.1 / P2-4: FIA WP July 2024 Sec 1.3
canonical pre-trade gate.

## Files touched

| Path | Lines added | Lines removed |
|------|-------------|---------------|
| `backend/config/settings.py` | 14 | 0 |
| `backend/services/portfolio_manager.py` | 9 | 0 |
| `backend/services/autonomous_loop.py` | 21 | 5 |
| `backend/services/paper_trader.py` | 30 | 0 |
| `backend/tests/test_price_tolerance_gate.py` (NEW) | 245 | 0 |
| **Total** | **319** | **5** |

Non-comment LOC: ~50 (production code) + ~140 (test). Well under the
250-line code-change target.

**Scope adherence:** the audit's P2-4 named
`backend/services/paper_trader.py::execute_buy`. Implementation also
touched `settings.py` (canonical place for the new setting), and
threaded the new field through `portfolio_manager.py::TradeOrder` +
`autonomous_loop.py` Step 7. All within the audit's documented chain
(execute_buy is the destination; the others are the threading sites).

## Implementation details

### `backend/config/settings.py`

Added after `paper_default_stop_loss_pct`:

```python
paper_price_tolerance_pct: float = Field(
    5.0, ge=0.0, le=50.0,
    description="Reject BUY when live fill price diverges from analysis-time price by more than this percent. 0 = no limit (legacy). Default 5 per SEC LULD Tier 1 band for S&P 500 + Russell 1000 > $3.",
)
```

8-line provenance comment cites FIA WP Sec 1.3 + SEC LULD + arXiv
2603.10092 (non-bypassable invariants).

### `backend/services/portfolio_manager.py`

- TradeOrder dataclass: new `price_at_analysis: Optional[float] = None`
  field.
- Buy-side build (line ~258): populates `price_at_analysis=cand.get("price")`
  alongside the existing `price=cand.get("price")` -- two distinct
  fields with the same value at order-creation time. The autonomous
  loop will then overwrite `price` with the LIVE fetch while
  preserving `price_at_analysis` as the historical reference.

### `backend/services/autonomous_loop.py`

Step 7 buy-loop (line ~897-919): changed from "prefer order.price,
fall back to live" to "always fetch live for fill, fall back to
analysis if live fails (network outage)". Passes
`price_at_analysis=order.price_at_analysis` to execute_buy.

This change is the prerequisite for the gate to function: without it,
the gate has nothing to compare against (live == analysis).

### `backend/services/paper_trader.py::execute_buy`

Gate placed BETWEEN the phase-25.6 stop-loss-synthesis block (line
~115) and the portfolio fetch (line ~117). Per arXiv 2603.10092 §3.1
"non-bypassable invariants" pattern, the gate fires BEFORE the
ExecutionRouter call so it cannot be circumvented by routing.

```python
price_tolerance_pct = float(
    getattr(self.settings, "paper_price_tolerance_pct", 0.0) or 0.0
)
if (
    price_tolerance_pct > 0
    and price_at_analysis is not None
    and price_at_analysis > 0
    and price > 0
):
    divergence_pct = abs(price - price_at_analysis) / price_at_analysis * 100.0
    if divergence_pct > price_tolerance_pct:
        logger.warning(...)
        return None
```

Reject pattern matches every other guard in `execute_buy` (cash, max-
positions, idempotency): `logger.warning(...)` + `return None`.

Fail-open on `price_at_analysis is None` -- the lite-Claude path
sometimes lacks a written analysis price; failing closed would crash
trading.

### `backend/tests/test_price_tolerance_gate.py` (NEW)

6 test cases covering pass/reject/disable/None/grep:

1. `test_price_tolerance_pass_1pct_deviation` -- 1% deviation passes
   the 5% gate.
2. `test_price_tolerance_reject_live_10pct_above_analysis` -- live
   +10% over analysis -> reject.
3. `test_price_tolerance_reject_live_10pct_below_analysis` -- live
   -10% under analysis -> reject (symmetric).
4. `test_price_tolerance_zero_disables_gate` -- tolerance=0 makes the
   gate a no-op even on +100% deviation.
5. `test_price_tolerance_skipped_when_analysis_price_missing` --
   None price_at_analysis -> fail-open (gate skipped).
6. `test_price_tolerance_symbols_present_in_source` -- mirrors the
   masterplan verification grep predicate so a future refactor that
   removes the wiring breaks pytest.

Mocking: `PaperTrader(settings=..., bq_client=MagicMock())`;
`ExecutionRouter` patched via `unittest.mock.patch` to return a
synthetic fill.

## Verification

### Masterplan verification command (phase-30.6)

```bash
grep -q 'paper_price_tolerance_pct' backend/config/settings.py && \
  grep -q 'price_tolerance' backend/services/paper_trader.py
```

Result: **exit 0**.

### Test run

```
$ python -m pytest backend/tests/test_price_tolerance_gate.py -v
collected 6 items

test_price_tolerance_pass_1pct_deviation PASSED
test_price_tolerance_reject_live_10pct_above_analysis PASSED
test_price_tolerance_reject_live_10pct_below_analysis PASSED
test_price_tolerance_zero_disables_gate PASSED
test_price_tolerance_skipped_when_analysis_price_missing PASSED
test_price_tolerance_symbols_present_in_source PASSED

6 passed in 0.78s
```

### Regression sweep

```
$ python -m pytest backend/tests/test_cycle_heartbeat_alarm.py \
                   backend/tests/test_autonomous_loop_step_5_6.py \
                   backend/tests/test_observability.py \
                   tests/services/test_sector_concentration.py -q
39 passed, 1 warning in 4.08s
```

Phase-30.1 (7) + phase-30.2+30.3 (7) + observability (12) + sector
concentration (13) = 39/39 still green. No regression.

### Syntax check

`python -c "import ast; ast.parse(...)"` on settings.py,
portfolio_manager.py, autonomous_loop.py, paper_trader.py, and the
test file: OK.

## Hard guardrail attestation

- No mutating BigQuery calls -- the gate is pure pre-write math.
- No Alpaca calls.
- No frontend / `.claude/` / `.mcp.json` touched.
- Diff stays within the audit's proposed-diff scope.
- Test ships and passes deterministically (6 cases).

## Success criteria check

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `settings_field_paper_price_tolerance_pct_added_default_5` | PASS | `grep -q 'paper_price_tolerance_pct' backend/config/settings.py` exits 0; field has `default=5.0` per `Field(5.0, ...)` |
| `execute_buy_rejects_when_fill_price_diverges_by_more_than_tolerance` | PASS | Tests #2 and #3 (live +10% and -10% over 5% gate) both assert `trade is None`; gate is symmetric |
| `test_covers_both_pass_and_reject_branches` | PASS | Test #1 covers pass branch (1% deviation); tests #2 and #3 cover reject branch (10% up and down). Tests #4 (disable) and #5 (None) cover edge cases. Test #6 covers regression-guard grep symbol |
