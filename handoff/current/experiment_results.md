# Experiment Results -- Phase 4.4.5.5 Trading Guide

**Date:** 2026-04-16
**Cycle:** 27
**Branch:** claude/awesome-euler-K0ae7

## Deliverable

`docs/TRADING_GUIDE.md` -- 386 lines, 9 top-level sections, 100% ASCII.

## Sections

1. **Signal Anatomy** -- 6 core fields (ticker, signal, confidence, date, factors, reason), Slack Block Kit display format, signal_id deduplication, three signal types.
2. **Confidence Interpretation** -- continuous 0.0-1.0 scale, interpretation ranges (0.80-1.00 high, 0.60-0.79 moderate, 0.40-0.59 low, 0.00-0.39 very low), recommendation mapping (BUY/STRONG_BUY, SELL/STRONG_SELL, HOLD).
3. **Position Sizing** -- hybrid min(hard_cap, half_kelly, inverse_vol) formula, hard cap (5% equity / $1,000), half-Kelly (0.5 * confidence * equity), inverse-vol, portfolio-level risk constraints table.
4. **Stop-Loss and Drawdown Protection** -- fixed stop 8% below entry (O'Neil CAN SLIM), trailing stop 3% below peak (Chandelier-lite), drawdown tiers (ok/-5%/-10%/-15% kill), stop precedence over BUY re-eval.
5. **Daily Signal Flow** -- 9-step pipeline (screen, rank, analyze, re-eval, decide, risk check, execute, notify, track), Peder's action steps.
6. **When to Override Ford** -- earnings day, corporate events, market stress, low confidence, non-public info, always honor stop-losses, rollback trigger (Sharpe < 0.5 / 14 days), decision framework.
7. **Key Numbers Reference** -- all hardcoded limits in one table.
8. **Glossary** -- 8 key terms defined.
9. **Important Reminders** -- 5 cardinal rules.

## Checklist Update

`docs/GO_LIVE_CHECKLIST.md` item 4.4.5.5 flipped `[ ]` -> `[x]` with evidence line.

## Verification

- Lead self-verification: 34/34 SC PASS (3 Python assertion blocks)
- QA evaluator: 41/41 automated checks PASS, manual review PASS, scores 10/10/10/10/10
- No `.py` files touched. No `backend/` files touched.
- All sibling docs byte-identical.
