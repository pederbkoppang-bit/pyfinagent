# Experiment Results -- Phase 4.4.5.5 Trading Guide

**Date:** 2026-04-16
**Cycle:** 28 (continuation of Go-Live Checklist evidence series)
**Branch:** `claude/awesome-euler-J0Wc2`

## Deliverable

`docs/TRADING_GUIDE.md` -- 259 lines, pure Markdown, ASCII-only, no emojis.

## Contents

11 sections covering:
1. What pyfinAgent is (and is not)
2. Signal anatomy (every Slack field explained)
3. Confidence thresholds (ranges, interpretation, sizing link)
4. Position sizing (3-arm hybrid: hard cap, half-Kelly, inverse-vol)
5. Stop-loss execution (inclusive boundary, precedence over re-eval)
6. Risk limits (4 hardcoded limits with values)
7. When to override Ford (concrete skip/escalate scenarios)
8. Daily workflow (step-by-step morning/during/evening/weekly)
9. Key numbers to remember (parameter table)
10. Important references (rollback plan, checklist, Slack channels)
11. Disclaimer
+ Appendix: System architecture overview

## Accuracy Verification

All cited parameters verified against source:
- tp_pct=10.0, sl_pct=12.92, holding_days=90 (optimizer_best.json)
- per-ticker=10%, total=100%, drawdown=-15%, daily_trades=5 (get_risk_constraints)
- Stop-loss: current_price <= stop_loss_price (portfolio_manager.py:80)
- Sizing: min(hard_cap, half_kelly, inverse_vol) (signals_server.py:928)
- Sell-first-then-buy order (portfolio_manager.py convention)

## Lead-self Verification

23/23 SCs + 8/8 ADVs checked via pre-baked Python block before commit.
