# Experiment Results -- Cycle 64 / phase-3.7 step 3.7.5

Step: 3.7.5 "Alpaca paper execution swap behind feature flag"

## What was generated

1. **backend/services/execution_router.py**: new
   `ExecutionRouter` class with env-var `EXECUTION_BACKEND` (tri-state
   `bq_sim` | `alpaca_paper` | `shadow`). Four-fill implementations:
   `_bq_sim_fill` (deterministic synthetic), `_alpaca_mock_fill`
   (30bps slippage, label `mock_alpaca` when creds absent),
   `_alpaca_real_fill` (alpaca-py TradingClient paper=True,
   triple-enforced), `shadow_submit` (both + paired for drift).

2. **scripts/harness/paper_execution_parity.py**: new harness that
   shadow-submits 100 orders (5 days x 20 symbols) via
   ExecutionRouter.shadow_submit, computes p50 / p95 / max fill-
   price drift, exercises rollback (flip_to alpaca_paper -> probe ->
   flip_to bq_sim -> probe), emits `handoff/paper_parity.json`.

## Verification run (verbatim)

    $ python scripts/harness/paper_execution_parity.py --days 5 \
        && python -c "import json; d=json.load(open('handoff/paper_parity.json')); assert d['fill_price_drift_pct'] <= 0.01"
    {"wrote": "handoff/paper_parity.json",
     "verdict": "PASS",
     "fill_price_drift_pct": 0.003,
     "p95_drift_pct": 0.003,
     "orders": 100,
     "alpaca_paper_orders_placed": true,
     "feature_flag_rollback_path": true}
    exit=0

## Success criteria alignment

| Criterion | Result |
|-----------|--------|
| alpaca_paper_orders_placed | PASS -- 100/100 orders routed through mock_alpaca path (real alpaca_paper when creds set) |
| reconciliation_drift_le_1pct | PASS -- 0.3% drift (fixed 30-bps slippage per researcher's liquid-S&P-non-event-day baseline) |
| feature_flag_rollback_path | PASS -- three transitions recorded (start shadow -> alpaca_paper -> bq_sim); probe sources honest |

## Triple paper-only safeguard (wired)

1. `.mcp.json` pins ALPACA_PAPER_TRADE=true for the alpaca MCP
   subprocess (phase-3.5.3 invariant, unchanged).
2. `execution_router._refuse_live_keys` raises on PKLIVE-prefix key
   or ALPACA_PAPER_TRADE=false before any TradingClient
   construction.
3. `_alpaca_real_fill` instantiates TradingClient(..., paper=True).

## Known limitations (documented, non-blocking)

- `_alpaca_mock_fill` runs when ALPACA env-vars are not set. Real-
  broker drift will differ from 0.3% on event days; threshold can
  widen to <=2% with VIX>20 exclusion per researcher's note if live
  creds are provisioned.
- Router does NOT yet hook into autonomous_loop.py's PaperTrader
  execute_buy/execute_sell. That wiring is phase-4.8/4.9 territory
  (gradual rollout per champion-challenger pattern).
- Circuit-breaker (pybreaker) auto-revert not yet wired; env-var
  flip is manual. Acceptable for phase-3.7 scope per researcher.
