---
step: phase-5.4
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - backend/markets/risk_engine.py (NEW, ~135 LOC) -- stateless multi-asset position sizer
  - tests/markets/test_risk_engine.py (NEW, ~165 LOC, 17 tests)
---

# Experiment Results -- phase-5.4

## What was done

Added the multi-asset risk engine to the `backend/markets/` package
established in phase-5.1. New `RiskEngine` class with a single public
method `compute_position_size(symbol, asset_class, equity, asset_vol,
*, delta=None, **kwargs)` that returns a positive USD notional sized
for equity / option / FX / future. Crypto is explicitly rejected
(owner directive 2026-04-19).

## Deliverable

### `backend/markets/risk_engine.py` (NEW, ~135 LOC)

Module-level constants:
- `DEFAULT_TARGET_VOL = 0.15` (matches `BacktestTrader` + `BacktestEngine`)
- `MAX_LEVERAGE = 3.0`
- `FX_MICRO_LOT = 1000`
- `SUPPORTED_ASSET_CLASSES = ("equity", "option", "fx", "future")`
- `MIN_ASSET_VOL = 1e-6`

Class:
- `RiskEngine.__init__(self, *, target_vol=0.15, max_leverage=3.0)` -- validates positivity at construction
- `RiskEngine._base_notional(equity, asset_vol)` -- vol-targeted formula `equity * target_vol / max(asset_vol, 1e-6)`, clamped at `max_leverage * equity`
- `RiskEngine.compute_position_size` -- normalises asset_class case, dispatches:
  - `crypto` -> ValueError (explicit owner directive)
  - unsupported -> ValueError
  - `equity` -> `base_notional`
  - `option` -> `base_notional * abs(delta)` (delta defaults to 1.0)
  - `fx` -> `max(1, round(base / 1000)) * 1000`
  - `future` -> `base_notional` placeholder (5.8 will add contract-multiplier table)

Pure module: no I/O, no env reads, no module-level side effects.
Thread-safe by design (stateless past `__init__`).

### `tests/markets/test_risk_engine.py` (NEW, ~165 LOC, 17 tests)

8 from research plan + 9 additional defensive (delta default = 1.0,
case-insensitive asset class, target_vol/max_leverage/equity validation,
default constant sanity check, inline immutable verification reproduction):

1. `test_equity_basic` -- 100000, vol=0.20 -> 75000.0
2. `test_equity_clamp_at_max_leverage` -- vol=0.001 -> exactly 3 * equity
3. `test_option_delta_half` -- delta=0.5 -> exactly half of base_notional
4. `test_option_delta_negative_same_as_positive` -- abs(delta) used
5. `test_option_default_delta_is_one` (defensive) -- no delta = full sizing
6. `test_fx_micro_lot_floor` -- tiny base -> exactly 1000
7. `test_fx_micro_lot_rounding` -- non-integer multiple rounds to nearest 1000
8. `test_fx_immutable_verification_inline` -- 100000/0.08 -> > 0, multiple of 1000
9. `test_future_returns_base_notional` (defensive) -- placeholder behavior
10. `test_no_crypto_raises` -- ValueError matches "crypto"
11. `test_unknown_asset_class_raises` -- ValueError matches "unsupported"
12. `test_case_insensitive_asset_class` (defensive) -- lower/UPPER/Mixed all equal
13. `test_target_vol_must_be_positive` (defensive) -- ValueError on 0 / negative
14. `test_max_leverage_must_be_positive` (defensive) -- ValueError on 0
15. `test_equity_must_be_positive` (defensive) -- ValueError on 0 / negative
16. `test_immutable_verification_assertions` -- exact reproduction of masterplan python -c
17. `test_default_target_vol_matches_existing_codebase` (defensive) -- DEFAULT_TARGET_VOL == 0.15

## Verification (verbatim, immutable from masterplan)

```
$ source .venv/bin/activate && python -c "from backend.markets.risk_engine import RiskEngine; r=RiskEngine(); eq=r.compute_position_size('AAPL','equity',100000,0.2); opt=r.compute_position_size('AAPL240119C00150000','option',100000,0.3,delta=0.5); fx=r.compute_position_size('EUR_USD','fx',100000,0.08); assert all(x>0 for x in [eq,opt,fx]); print('ok')"
ok

$ source .venv/bin/activate && python -m pytest tests/markets/test_risk_engine.py -v
============================== 17 passed in 0.01s ==============================
```

## Files touched

| Path | Action | Note |
|------|--------|------|
| `backend/markets/risk_engine.py` | CREATED | ~135 LOC |
| `tests/markets/test_risk_engine.py` | CREATED | ~165 LOC, 17 tests |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-5.4-research-brief.md` | created (researcher) | -- |

NO modifications to `BacktestTrader`, `portfolio_manager`,
`kelly_allocator`, `paper_trader`, or any existing service. NO new
dependencies. NO BQ schema changes. NO frontend.

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | RiskEngine.compute_position_size returns positive notional for equity/option/fx | PASS (3 inline asserts in immutable command + tests #1, #3, #6, #8) |
| 2 | Option position size is delta-adjusted (delta=0.5 ~= half) | PASS (test #3 asserts EXACT equality to half) |
| 3 | FX uses micro-lot (1000), enforces min order | PASS (tests #6, #7) |
| 4 | Regression: existing equity sizing unchanged within epsilon | PASS (no existing path modified; new class is additive; default target_vol=0.15 matches BacktestTrader) |
| 5 | No crypto branch | PASS (test #10 -- crypto raises ValueError; SUPPORTED_ASSET_CLASSES does not include crypto) |
| 6 | Immutable `python -c` returns "ok" | PASS |

## Honest disclosures

1. **17 tests vs 8-10 in plan** -- 9 extras: defensive validation tests on construction params + equity positivity + case-insensitivity + future placeholder + default-constant sanity. Floor exceeded; not a violation.

2. **Future asset class is a placeholder.** Returns `base_notional` unchanged (no contract-multiplier scaling). Properly modeled in 5.8 IBKR cycle (per masterplan dependency chain). Documented in module docstring + test #9 verifies the placeholder behavior.

3. **No regression hook into existing equity sizing.** I considered adding a `test_regression_vs_backtest_trader_equity` but chose not to: the existing `BacktestTrader.size_position(probability, vol, equity)` couples sizing to a probability-weighted multi-position framework that this single-position `RiskEngine` does not replicate. The "regression unchanged within epsilon" criterion is satisfied by NOT modifying `BacktestTrader` (verified by `git diff backend/backtest/backtest_trader.py` -- empty).

4. **No cycle-2 fix needed.** First pytest run was 17/17 PASS.

5. **Stateless + thread-safe.** Class has no mutable state past `__init__`. Same instance can serve many concurrent calls.

6. **No wiring into paper_trader.** This cycle ships the engine as a pure callable. Wiring into execution paths becomes load-bearing in 5.6+ when option/FX execution comes online.

## Closes

Net-new task #80 (UAT-5.4). Masterplan step phase-5.4.

## Next

Spawn Q/A.
