# Contract: Phase 4.4.2.3 -- Paper Max Drawdown < 15%

## Step ID
4.4.2.3

## Hypothesis
Paper trading has been running since 2026-03-20 (31 days). The portfolio's max
drawdown is -5.0% (from $10,000 to $9,499.50). The kill switch threshold is
-15.0% (hardcoded in get_risk_constraints, verified in 4.4.4.4). The kill switch
was never triggered (0 entries in risk_intervention_log). A BQ-backed drill can
verify all criteria for checklist item 4.4.2.3.

## Success Criteria (from GO_LIVE_CHECKLIST.md)
- Paper trading run never crossed the -15% drawdown line
- Kill switch never triggered during the paper trading window

## Plan
1. Query BQ for paper_portfolio + paper_portfolio_snapshots + risk_intervention_log
2. Save evidence snapshot as JSON artifact
3. Write stdlib-only drill that reads evidence + verifies kill switch code threshold
4. Run drill, confirm 9/9 PASS
5. Flip checklist item, append evidence line

## Research Gate
Waived per pure-data-verification rule. No algorithm or external knowledge needed.
BQ queries are the primary verification method per the checklist HOW recipe.

## References
- BQ: `sunny-might-477607-p8.financial_reports.paper_portfolio`
- BQ: `sunny-might-477607-p8.financial_reports.paper_portfolio_snapshots`
- BQ: `sunny-might-477607-p8.pyfinagent_data.risk_intervention_log`
- Code: `backend/agents/mcp_servers/signals_server.py` get_risk_constraints
- Prior: Cycle 9 (4.4.4.1 kill switch drill), Cycle 8 (4.4.4.4 risk limits)
