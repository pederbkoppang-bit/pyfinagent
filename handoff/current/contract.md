# Contract -- Zero-Orders Bug Fix (Paper Trading)

**Cycle:** 18
**Date:** 2026-04-16
**Item:** Paper trading zero-orders bug (unblocks Phase 4.4.2)

## Problem Statement

Paper trading has been live for 27 days with zero trades executed. `decide_trades` returns empty list every cycle. Root cause: recommendation normalization mismatch between the Gemini synthesis schema (uses "Strong Buy" / "Strong Sell" with spaces) and the `_BUY_RECS` / `_SELL_RECS` lookup sets (use "STRONG_BUY" / "STRONG_SELL" with underscores). Additionally, Claude analysis fails due to missing ANTHROPIC_API_KEY, and the model ID was outdated.

## Success Criteria

### A. Normalization fix (SC1-6)
- SC1: `_normalize_rec` helper exists in `portfolio_manager.py`
- SC2: `_normalize_rec("Strong Buy")` == `"STRONG_BUY"`
- SC3: `_normalize_rec("Strong Sell")` == `"STRONG_SELL"`
- SC4: `_normalize_rec("BUY")` == `"BUY"` (Claude path unchanged)
- SC5: All 3 call sites in `decide_trades` use `_normalize_rec` (sell rec, sell old_rec, buy rec)
- SC6: No raw `.upper()` calls on recommendation strings remain in `decide_trades`

### B. Diagnostic logging (SC7-8)
- SC7: Zero-orders case emits a `logger.warning` with candidate count, rec distribution, cash, NAV
- SC8: ASCII-only log message (no Unicode)

### C. Model update (SC9-10)
- SC9: `autonomous_loop.py` uses `claude-sonnet-4-6` (not `claude-sonnet-4-20250514`)
- SC10: No other changes to `autonomous_loop.py` beyond model string

### D. Scope discipline (SC11-14)
- SC11: Exactly 2 `.py` files modified
- SC12: `ast.parse` clean on both files
- SC13: Zero new imports in either file
- SC14: `TradeOrder` dataclass unchanged
