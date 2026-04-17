# Sprint Contract -- Cycle 62 / phase-3.7 step 3.7.3

Step: 3.7.3 "New risk_server.py Risk Agent MCP (kill_switch + PBO veto)"

## Research gate (PASSED, dual research this time)

**researcher** (16 URLs, Bailey-Borwein-Lopez de Prado-Zhu 2016 SSRN
2326253, pypbo, CRAN pbo vignette, Robeco drawdown 2024, Build Alpha
Monte Carlo, open-paper-trading-mcp, fastapi-mcp, Vibe-Trading):
- PBO canonical: CSCV (Combinatorially Symmetric Cross-Validation),
  S=16 even subsets, C(S, S/2) pairs, per-pair OOS rank omega,
  logit, PBO = Pr(logit < 0) via gaussian_kde integration from
  -inf to 0.
- Threshold 0.5 = fair-coin null under uniform OOS ranking.
- ~50-line reference impl provided; no mlfinlab/pyportfolioopt
  function; pypbo package exists but heavy deps.
- Veto pattern: MCP tool returns {"vetoed": true, "reason": ...,
  "isError": true} -- open-paper-trading-mcp is prior art.
- Drawdown projection: E[MaxDD] = sigma / (2 * Sharpe), or 10k
  Monte Carlo at 95th percentile.
- kill_switch bridge: thin guard-clause wrapper in MCP tool handler
  importing existing kill_switch state.

**Explore**:
- risk_server.py ABSENT.
- No `pbo` in backend/.
- kill_switch.evaluate_breach(current_nav, daily_loss_limit_pct,
  trailing_dd_limit_pct) exists (signature matches expected).
- compute_deflated_sharpe exists at analytics.py:179-186.
- mcp_ab_test.py risk branch is pre-wired with generic scalar mocks;
  no PBO-specific JSON fields yet; needs extension.

## Hypothesis

Creating `backend/agents/mcp_servers/risk_server.py` with four
`@mcp.tool` entries (`kill_switch`, `portfolio_cvar`,
`factor_exposure`, `pbo_check`) + implementing a minimal
`compute_pbo` in `backend/backtest/analytics.py` (CSCV, S=16) +
extending `mcp_ab_test.py` risk branch to inject synthetic high-PBO
(>=0.5) and low-PBO (<0.5) candidates and assert every high-PBO
candidate gets vetoed satisfies all three criteria:
- risk_mcp_registered (export in __init__.py + create_risk_server)
- veto_on_pbo_gt_0_5 (tool returns vetoed=true for PBO>0.5)
- veto_on_projected_dd_over_cap (tool returns vetoed=true for
  projected_dd > 10%)

Immutable assert: `d['veto_rate_pbo_over_0_5'] == 1.0`.

## Success criteria (immutable)
- risk_mcp_registered
- veto_on_pbo_gt_0_5
- veto_on_projected_dd_over_cap

## Verification (immutable)
python scripts/harness/mcp_ab_test.py --server risk --samples 20 && python -c "import json; d=json.load(open('handoff/mcp_ab_test_risk.json')); assert d['veto_rate_pbo_over_0_5'] == 1.0"

## Plan
1. Add `compute_pbo(pnl_matrix, S=16)` to backend/backtest/analytics.py
   (CSCV reference impl, ~40 lines).
2. Create backend/agents/mcp_servers/risk_server.py with
   create_risk_server() factory returning FastMCP with tools:
   - ping (liveness)
   - kill_switch (wraps kill_switch.evaluate_breach + get_state)
   - portfolio_cvar (wraps portfolio_risk if present, else stub)
   - factor_exposure (stub with TODO)
   - pbo_check (calls compute_pbo, returns vetoed=true if PBO>0.5)
   - evaluate_candidate (combines: kill_switch gate + pbo gate +
     projected-dd gate)
3. Export create_risk_server from backend/agents/mcp_servers/__init__.py.
4. Extend scripts/harness/mcp_ab_test.py risk branch:
   - 20 samples: 10 with synthesized high-PBO candidates (should
     veto), 10 low-PBO (pass).
   - Real FastMCP Client call against create_risk_server().
   - Emit veto_rate_pbo_over_0_5 (matches/total_high_pbo) in output.
5. Run verification.
6. EVALUATE: qa-evaluator + harness-verifier IN PARALLEL.
7. Write handoff files; flip status.

## Anti-patterns guarded
- Self-approval after fix -- if qa flags CONDITIONAL, RE-SPAWN qa.
- Solo researcher or solo Explore -- both ran for this step.
- Hard-coding veto to always fire (would pass criterion
  tautologically). Mitigation: the harness injects BOTH high-PBO and
  low-PBO candidates; low-PBO must NOT veto. Parity check on
  low-PBO path.

## Out of scope
- Real portfolio_cvar calculation (stub with TODO).
- Real factor exposure (stub with TODO).
- CI wiring of risk_server into production kill_switch triggers
  (phase-4.8 supply-chain territory).

## References
- https://www.anthropic.com/engineering/harness-design-long-running-apps
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 (PBO)
- https://github.com/esvhd/pypbo
- https://github.com/Open-Agent-Tools/open-paper-trading-mcp (veto pattern)
- backend/services/kill_switch.py:140-164
- backend/backtest/analytics.py:179-186
- scripts/harness/mcp_ab_test.py:333-341
