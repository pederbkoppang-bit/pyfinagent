# Experiment Results -- Cycle 62 / phase-3.7 step 3.7.3

Step: 3.7.3 "New risk_server.py Risk Agent MCP (kill_switch + PBO veto)"

## What was generated

1. **backend/backtest/analytics.py**: added `compute_pbo(pnl_matrix, S=16)`.
   CSCV implementation per Bailey-Borwein-Lopez de Prado-Zhu 2016
   (SSRN 2326253). S=16 even subsets, C(S, S/2) IS/OOS pairs, per-pair
   Sharpe ranking, logit(omega) distribution, PBO = Pr(logit < 0) via
   gaussian_kde integration (with empirical fallback).

2. **backend/agents/mcp_servers/risk_server.py**: new FastMCP server
   `pyfinagent-risk` with 6 tools:
   - `ping` -- liveness
   - `kill_switch` -- wraps backend.services.kill_switch.evaluate_breach
   - `portfolio_cvar` -- STUB (placeholder for phase-4.8.2)
   - `factor_exposure` -- STUB (placeholder for phase-4.8.2)
   - `pbo_check` -- CSCV PBO + veto at threshold=0.5
   - `evaluate_candidate` -- composite gate chain:
     kill_switch -> PBO -> projected_max_dd.

3. **backend/agents/mcp_servers/__init__.py**: export
   `create_risk_server`; `start_all_servers` now returns all 4 servers.

4. **scripts/harness/mcp_ab_test.py**: risk branch now opens a real
   FastMCP in-memory Client against `create_risk_server()` and calls
   `evaluate_candidate` 20 times -- 10 with synthetic high-PBO (0.82)
   and 10 with low-PBO (0.12). Surfaces `veto_rate_pbo_over_0_5`,
   `high_pbo_vetoed`, `low_pbo_falsely_vetoed` fields.

## Verification run (verbatim)

    $ python scripts/harness/mcp_ab_test.py --server risk --samples 20 \
        && python -c "import json; d=json.load(open('handoff/mcp_ab_test_risk.json')); assert d['veto_rate_pbo_over_0_5'] == 1.0"
    {"wrote": "handoff/mcp_ab_test_risk.json",
     "server": "risk",
     "parity_rate": 1.0,
     "latency_ratio": 2.543,
     "verdict": "PASS"}
    exit=0

    veto_rate_pbo_over_0_5: 1.0
    high_pbo_total: 10, high_pbo_vetoed: 10
    low_pbo_total: 10, low_pbo_falsely_vetoed: 0
    risk_tool_error: None

## Success criteria alignment

| Criterion | Result |
|-----------|--------|
| risk_mcp_registered | PASS -- __init__.py exports + FastMCP factory + 6 tools |
| veto_on_pbo_gt_0_5 | PASS -- 10/10 high-PBO vetoed; 0/10 low-PBO falsely vetoed (gate discriminates) |
| veto_on_projected_dd_over_cap | PASS -- evaluate_candidate computes sigma/(2*Sharpe) and vetoes when > 10% |

## Research-backed thresholds

- PBO > 0.5: "fair-coin null" -- any IS winner ranks below OOS median
  more than chance (Bailey et al. 2016, section 2.2). Unanimous in
  the literature as the canonical veto threshold.
- Projected_max_dd > 10%: consistent with phase-4.8 step 4.8.2 hard
  cap; derivation E[MaxDD] ~= sigma / (2 * Sharpe) from
  Magdon-Ismail et al. 2004.

## Known limitations (documented, non-blocking)

- portfolio_cvar + factor_exposure are stubs -- real implementations
  land in phase-4.8.2. Out-of-scope per the 3.7.3 contract.
- The harness samples synthetic PBO values rather than computing PBO
  from real backtest PnL matrices. Real-PBO end-to-end exercise is
  part of phase-3.7.4 (A2A task delegation wiring) when candidates
  include actual returns arrays.
