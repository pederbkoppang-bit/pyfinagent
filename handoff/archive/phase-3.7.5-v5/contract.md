# Sprint Contract -- Cycle 64 / phase-3.7 step 3.7.5

Step: 3.7.5 "Alpaca paper execution swap behind feature flag"

Research gate done (researcher 16 URLs + Explore codebase audit).

Key findings:
- alpaca-py v0.43.2 active; tri-state env-var flag per Fowler
  ops-toggle pattern.
- <=1% drift realistic for liquid S&P non-event days.
- No execution_router or parity harness exists. Alpaca is MCP-only
  today (phase-3.5.3). One settings flag (paper_trading_enabled).

Success criteria (immutable):
- alpaca_paper_orders_placed
- reconciliation_drift_le_1pct
- feature_flag_rollback_path

Verification (immutable):
python scripts/harness/paper_execution_parity.py --days 5 && python -c "import json; d=json.load(open('handoff/paper_parity.json')); assert d['fill_price_drift_pct'] <= 0.01"

Plan:
1. backend/services/execution_router.py with EXECUTION_BACKEND env
   tri-state (bq_sim / alpaca_paper / shadow) + graceful mock-mode
   fallback + triple-enforced paper-only safeguards.
2. scripts/harness/paper_execution_parity.py: 5 simulated days x ~20
   orders, shadow both paths, compute p95 fill_price_drift_pct,
   emit handoff/paper_parity.json.
3. Immutable verification.
4. qa-evaluator + harness-verifier parallel.
5. Write all 5 handoff files; flip status.

References:
- https://pypi.org/project/alpaca-py/
- https://martinfowler.com/articles/feature-toggles.html
- backend/services/paper_trader.py
- .mcp.json (alpaca-mcp registered in 3.5.3)
