# Phase 2.7 Contract — Paper Trading Activation

## Hypothesis
Activating the paper trading engine with our best params (Sharpe 1.17) gives us real-world validation data before the May go-live. Every day of paper trading is irreplaceable evidence.

## Success Criteria
1. Paper portfolio initialized with $10,000 starting capital
2. Screener runs successfully (zero LLM cost)
3. At least one test cycle completes: Screen → Analyze → Decide → Trade
4. BQ tables written: paper_portfolio, paper_positions, paper_trades, paper_portfolio_snapshots
5. API endpoints return portfolio data
6. Daily scheduler configured (10:00 ET / 16:00 Oslo)

## Fail Conditions
- Cycle hangs or crashes
- BQ writes fail
- LLM costs exceed $10 in a single cycle
- Screener returns zero candidates

## Budget
Approved by Peder: ~$2-5/day Gemini API for daily analysis cycles.

## Started
2026-03-28 23:58 Oslo
