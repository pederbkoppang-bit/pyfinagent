# Sprint Contract -- Cycle 19: Zero-Orders Bug Fix

Generated: 2026-04-16T09:00:00+00:00

## Problem Statement

Paper trading has generated zero trades for 27+ consecutive days. Root cause
diagnosed by Cycle 18 on branch `claude/awesome-euler-ch0wi`: Gemini synthesis
returns `"Strong Buy"` / `"Strong Sell"` (space-separated, mixed case) but
`decide_trades` used `.upper()` producing `"STRONG BUY"` (with space), while
the lookup sets `_BUY_RECS` / `_SELL_RECS` use `"STRONG_BUY"` / `"STRONG_SELL"`
(underscore). All strong signals were silently dropped.

Secondary issue: outdated Claude model ID `claude-sonnet-4-20250514` in
`autonomous_loop.py` (should be `claude-sonnet-4-6`).

## Fix

1. Add `_normalize_rec(raw)` helper: `.strip().upper().replace(" ", "_")`
2. Update 3 comparison sites in `decide_trades` to use `_normalize_rec`
3. Add zero-orders diagnostic logging when no orders generated
4. Update model ID to `claude-sonnet-4-6`

## Success Criteria

### A. Scope discipline
- SC1: Exactly 2 files modified: `portfolio_manager.py`, `autonomous_loop.py`
- SC2: Zero files outside `backend/services/` touched
- SC3: No changes to `_SELL_RECS`, `_DOWNGRADE_RECS`, `_BUY_RECS` set values
- SC4: ASCII-only in both files

### B. Normalization correctness
- SC5: `_normalize_rec("Strong Buy")` == `"STRONG_BUY"`
- SC6: `_normalize_rec("Strong Sell")` == `"STRONG_SELL"`
- SC7: `_normalize_rec("BUY")` == `"BUY"` (canonical form preserved)
- SC8: `_normalize_rec("")` == `""` (empty safe)
- SC9: `_normalize_rec(" Strong Buy ")` == `"STRONG_BUY"` (whitespace stripped)
- SC10: Zero `.upper()` calls remain in `decide_trades` for recommendation processing
- SC11: >= 3 `_normalize_rec` calls in `decide_trades`

### C. Model ID
- SC12: `autonomous_loop.py` contains `"claude-sonnet-4-6"`
- SC13: `autonomous_loop.py` does NOT contain `"claude-sonnet-4-20250514"`

### D. Global invariants
- SC14: `ast.parse` clean on both files
- SC15: Stop-loss path (lines 78-85 of portfolio_manager.py) unchanged
- SC16: `TradeOrder` dataclass unchanged

### E. Diagnostic logging
- SC17: `logger.warning` call present when `not orders`
- SC18: Warning includes recommendation distribution count

## Research Gate

WAIVED. Pure bug fix against diagnosed root cause from Cycle 18. No new
research surface. The normalization pattern (`.strip().upper().replace()`) is
standard Python string canonicalization.
