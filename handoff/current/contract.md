# Sprint Contract -- Zero-Orders Bug Fix (Cycle 27)
Generated: 2026-04-16

## Target
Fix the recommendation normalization bug in `portfolio_manager.py` that caused paper trading to generate zero trades for 27 consecutive days. Also update the outdated Claude model ID in `autonomous_loop.py`.

## Root Cause
Gemini synthesis returns `"Strong Buy"` / `"Strong Sell"` (with spaces). `decide_trades` did `.upper()` producing `"STRONG BUY"`, but lookup sets use `"STRONG_BUY"` (underscore). "Strong Buy" signals were silently dropped -- no error, no log, just zero orders.

## Success Criteria

### A. Scope Discipline (SC1-4)
- SC1: Exactly 2 `.py` files modified: `portfolio_manager.py`, `autonomous_loop.py`
- SC2: Zero files under `scripts/go_live_drills/` touched
- SC3: Zero files under `docs/` touched
- SC4: `signals_server.py` byte-identical to HEAD

### B. Normalization Fix (SC5-10)
- SC5: `_normalize_rec` function defined at module level in `portfolio_manager.py`
- SC6: `_normalize_rec("Strong Buy")` returns `"STRONG_BUY"`
- SC7: `_normalize_rec("Strong Sell")` returns `"STRONG_SELL"`
- SC8: `_normalize_rec("BUY")` returns `"BUY"` (no regression on clean inputs)
- SC9: All 3 recommendation comparison sites in `decide_trades` use `_normalize_rec` instead of `.upper()`
- SC10: Zero bare `.upper()` calls on `recommendation` `.get()` patterns remain

### C. Lookup Sets Unchanged (SC11-13)
- SC11: `_BUY_RECS = {"BUY", "STRONG_BUY"}` unchanged
- SC12: `_SELL_RECS = {"SELL", "STRONG_SELL"}` unchanged
- SC13: `_DOWNGRADE_RECS = {"HOLD", "SELL", "STRONG_SELL"}` unchanged

### D. Model ID Fix (SC14)
- SC14: `autonomous_loop.py` contains `claude-sonnet-4-6`, not `claude-sonnet-4-20250514`

### E. Diagnostic Logging (SC15-16)
- SC15: Zero-orders case emits `logger.warning` with candidate count, recommendation distribution, cash, and NAV
- SC16: Log message is ASCII-only (no Unicode per security.md)

### F. Global Invariants (SC17-18)
- SC17: `ast.parse` clean on both modified files
- SC18: Stop-loss code path (lines 73-85 of original) untouched

## Adversarial Probes
- AP1: Does `_normalize_rec("")` return `""` without error? (edge case)
- AP2: Does `_normalize_rec("  buy  ")` return `"BUY"`? (whitespace)
- AP3: Are the 3 existing go-live drill tests unaffected? (byte-identity)
- AP4: Is the zero-orders log message ASCII-only? (Windows cp1252 safety)
