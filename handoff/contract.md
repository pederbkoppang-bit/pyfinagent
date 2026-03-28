# Phase 2.8 Contract — Harness Hardening

## Hypothesis
Adding seed stability tests, ablation studies, and slippage modeling will validate that our Sharpe 1.17 result is robust and not an artifact of a specific random seed or fragile feature combination.

## Success Criteria
1. Seed stability: 5 random seeds, Sharpe std < 0.1, all seeds > 0.9
2. Ablation: removing any single feature drops Sharpe < 20% (no fragile dependencies)
3. Slippage: 5 bps slippage added, Sharpe > 0.85
4. No single walk-forward window drives > 30% of total return
5. All tests scripted and repeatable via run_harness.py

## Fail Conditions
- Sharpe varies > 0.2 across seeds (strategy is seed-dependent)
- Removing one feature collapses Sharpe below 0.5 (fragile dependency)
- Slippage test drops Sharpe below 0.7

## Cost
Zero LLM cost — all tests are pure compute (ML + backtest engine).

## Started
2026-03-29 00:15 Oslo
